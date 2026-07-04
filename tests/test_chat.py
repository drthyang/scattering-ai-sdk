import json

import numpy as np

from scattering_ai.core.chat import ChatSession
from scattering_ai.llm.base import LLMResponse, ModelCapabilities, ToolCall


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.seen = []

    @property
    def capabilities(self):
        return ModelCapabilities(tool_use=True)

    def complete(self, messages, tools=None):
        self.seen.append(list(messages))
        return self.responses.pop(0)


def curve_file(tmp_path):
    x = np.linspace(0, 10, 1000)
    y = 1 + 6 * np.exp(-((x - 4) ** 2) / 0.03)
    path = tmp_path / "c.dat"
    np.savetxt(path, np.column_stack([x, y]))
    return str(path)


def test_chat_two_turns_with_persistent_state(tmp_path):
    path = curve_file(tmp_path)
    llm = ScriptedLLM([
        # turn 1: inspect then answer
        LLMResponse(tool_calls=[ToolCall(name="inspect_curve",
                                         arguments={"path": path}, id="a")]),
        LLMResponse(content="The file has 1000 points from 0 to 10."),
        # turn 2: fit then answer
        LLMResponse(tool_calls=[ToolCall(
            name="fit_peaks_1d", arguments={"path": path, "centers": [4.0]}, id="b")]),
        LLMResponse(content="Fitted: center 4.000."),
    ])
    session = ChatSession(llm=llm, model_id="scripted", workspace=tmp_path / "ws")

    reply1 = session.turn("What is in this file?")
    assert "1000 points" in reply1

    reply2 = session.turn("Fit the peak near 4.")
    assert "Fitted" in reply2

    # history persists: the final call saw both turns
    final_messages = llm.seen[-1]
    user_turns = [m for m in final_messages if m.role == "user"]
    assert len(user_turns) == 2
    tool_messages = [m for m in final_messages if m.role == "tool"]
    assert len(tool_messages) == 2
    fit_payload = json.loads(tool_messages[-1].content)
    assert abs(fit_payload["peaks"][0]["center"] - 4.0) < 0.01

    # tool trace + transcript on disk
    assert [t["tool"] for t in session.tool_trace] == ["inspect_curve", "fit_peaks_1d"]
    transcript = (session.workspace / "chat_transcript.md").read_text()
    assert "Fit the peak near 4." in transcript
    assert "fit_peaks_1d" in transcript


def test_chat_tool_errors_flow_back(tmp_path):
    llm = ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall(name="inspect_curve",
                                         arguments={"path": "/nope.dat"}, id="a")]),
        LLMResponse(content="That file does not exist."),
    ])
    session = ChatSession(llm=llm, workspace=tmp_path / "ws")
    reply = session.turn("Look at /nope.dat")
    assert "does not exist" in reply
    assert session.tool_trace[0]["error"]


def series_files(tmp_path):
    x = np.linspace(0, 10, 1200)
    paths = []
    for temp in np.arange(5, 100, 5.0):
        center = 5.0 + (0.0 if temp < 50 else 0.004 * (temp - 50))
        y = 1.0 + 10 * np.exp(-((x - center) ** 2) / 0.03)
        y[:30] = -3.0  # masked sentinel
        p = tmp_path / f"scan_T_base_{temp:.1f}K.dat"
        np.savetxt(p, np.column_stack([x, y]))
        paths.append(str(p))
    return paths


def test_chat_transition_question_drives_skill(tmp_path):
    """A transition question should trigger skill_scan_series_transitions on the
    whole glob, and the grounded reply reuses its verdict."""
    import pytest

    pytest.importorskip("matplotlib")
    series_files(tmp_path)  # write the series to disk
    glob = str(tmp_path / "*.dat")

    captured = {}

    def record(name, arguments, result):
        captured["name"] = name
        captured["result"] = result

    llm = ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall(
            name="skill_scan_series_transitions",
            arguments={"paths": [glob], "fwhm_guess": 0.03}, id="s")]),
        LLMResponse(content="Tracked the strongest peaks; a changepoint appears near 50 K."),
    ])
    session = ChatSession(llm=llm, model_id="scripted", workspace=tmp_path / "ws",
                          on_tool_call=record)
    reply = session.turn("Is there a phase transition across temperature?")

    assert captured["name"] == "skill_scan_series_transitions"
    assert captured["result"]["mask_value_used"] == -3.0  # auto-detected, not given
    assert captured["result"]["verdict"]["transition_detected"]
    assert "50" in reply
    # the skill ran as a single tool call, not a hand-chained sequence
    assert [t["tool"] for t in session.tool_trace] == ["skill_scan_series_transitions"]


def test_chat_files_context_in_system_prompt(tmp_path):
    llm = ScriptedLLM([LLMResponse(content="ok")])
    session = ChatSession(llm=llm, workspace=tmp_path / "ws",
                          files=["/data/a.gr", "/data/b.nxs"])
    session.turn("hi")
    system = llm.seen[0][0]
    assert system.role == "system"
    assert "/data/a.gr" in system.content and "/data/b.nxs" in system.content


def test_chat_record_correction_feeds_journal(tmp_path):
    # The chat affordance for human corrections (self-improvement P2): an explicit
    # human act — never inferred from the dialogue — lands in the shared journal.
    from scattering_ai.learning.journal import Journal

    jdir = tmp_path / "journal"
    session = ChatSession(llm=ScriptedLLM([]), model_id="scripted",
                          workspace=tmp_path / "ws", journal=jdir)
    corr = session.record_correction(
        target="transition_temperature",
        statement="GaNb4Se8 transitions are 50 K and 29 K, not 39 K",
        corrected_value="50,29", domain="series")

    assert corr.episode_id == session.session_id and corr.corrected_value == "50,29"
    corrs = Journal(jdir).corrections()
    assert len(corrs) == 1 and corrs[0].target == "transition_temperature"


def test_chat_correction_requires_journal(tmp_path):
    import pytest

    session = ChatSession(llm=ScriptedLLM([]), workspace=tmp_path / "ws")
    with pytest.raises(RuntimeError):
        session.record_correction(target="domain", statement="wrong route")


# --- E6: capture everywhere (chat-turn journaling) ----------------------------

def test_chat_journals_tool_failure_turn_redacted(tmp_path):
    from scattering_ai.learning.journal import Journal

    jdir = tmp_path / "j"
    llm = ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall(name="no_such_tool",
                                         arguments={"secretkey": "/private/data.dat"},
                                         id="a")]),
        LLMResponse(content="I couldn't run that tool."),
    ])
    session = ChatSession(llm=llm, model_id="m", workspace=tmp_path / "ws",
                          journal=jdir, files=["/private/data.dat"])
    session.turn("do the thing")

    eps = Journal(jdir).episodes()
    assert len(eps) == 1
    ep = eps[0]
    assert ep.surface == "chat" and ep.outcome == "error"
    assert any(f.diagnostic == "no_such_tool" and f.severity == "error"
               for f in ep.findings)
    # redaction: tool arguments and full data paths never enter the journal
    raw = Journal(jdir).path.read_text()
    assert "secretkey" not in raw and "/private/data.dat" not in raw
    assert "data.dat" in raw  # only the basename is kept (file_names)


def test_chat_clean_turn_is_not_journaled(tmp_path):
    from scattering_ai.learning.journal import Journal

    jdir = tmp_path / "j"
    session = ChatSession(llm=ScriptedLLM([LLMResponse(content="Here is the answer.")]),
                          model_id="m", workspace=tmp_path / "ws", journal=jdir)
    session.turn("hi")
    assert Journal(jdir).episodes() == []  # no error, has reply -> not a signal


def test_chat_dead_end_no_reply_is_captured(tmp_path):
    from scattering_ai.learning.journal import Journal

    jdir = tmp_path / "j"
    session = ChatSession(llm=ScriptedLLM([LLMResponse(content="")]),
                          model_id="m", workspace=tmp_path / "ws", journal=jdir)
    session.turn("?")
    eps = Journal(jdir).episodes()
    assert len(eps) == 1 and eps[0].interpretation_available is False


def test_chat_failure_surfaces_as_signal_not_empty_result(tmp_path):
    from scattering_ai.learning.journal import Journal
    from scattering_ai.learning.signals import signals_report

    jdir = tmp_path / "j"
    ChatSession(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall(name="no_such_tool", arguments={}, id="a")]),
            LLMResponse(content="failed")]),
        model_id="m", workspace=tmp_path / "ws", journal=jdir).turn("x")
    types = {c["type"] for c in signals_report(Journal(jdir))["clusters"]}
    assert "error_outcome" in types
    assert "empty_result" not in types  # chat turns must not trip empty_result


def test_chat_journaling_off_by_default(tmp_path):
    session = ChatSession(llm=ScriptedLLM([LLMResponse(content="")]),
                          model_id="m", workspace=tmp_path / "ws")  # no journal
    session.turn("?")
    assert not list(tmp_path.glob("**/episodes.jsonl"))


def test_chat_repeated_reformulation_is_captured(tmp_path):
    from scattering_ai.learning.journal import Journal

    jdir = tmp_path / "j"
    # two near-identical user turns, each with a clean reply -> the second is a
    # reformulation (the assistant isn't landing) and gets journaled
    session = ChatSession(
        llm=ScriptedLLM([LLMResponse(content="Here is an answer."),
                         LLMResponse(content="Here is another answer.")]),
        model_id="m", workspace=tmp_path / "ws", journal=jdir)
    session.turn("How do I fit the peak near two point six angstroms?")
    session.turn("How do I fit the peak around two point six angstroms?")

    eps = Journal(jdir).episodes()
    assert len(eps) == 1 and eps[0].outcome == "warnings"
    assert any(f.diagnostic == "repeated_reformulation" and f.severity == "warning"
               for f in eps[0].findings)


def test_chat_distinct_questions_are_not_reformulation(tmp_path):
    from scattering_ai.learning.journal import Journal

    jdir = tmp_path / "j"
    session = ChatSession(
        llm=ScriptedLLM([LLMResponse(content="A."), LLMResponse(content="B.")]),
        model_id="m", workspace=tmp_path / "ws", journal=jdir)
    session.turn("What is the first peak position?")
    session.turn("Now slice the volume along the h axis and integrate.")
    assert Journal(jdir).episodes() == []  # distinct, clean turns -> no signal
