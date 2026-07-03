import json

import numpy as np

from scattering_ai import Agent, AnalysisRequest, Confidence
from scattering_ai.llm.base import LLMResponse, ModelCapabilities, ToolCall
from scattering_ai.tools.registry import default_toolkit


def write_two_peak_curve(path):
    rng = np.random.default_rng(3)
    x = np.linspace(0, 10, 1500)
    y = 0.2 + 0.02 * x
    for h, c, w in [(8.0, 3.0, 0.25), (5.0, 7.0, 0.4)]:
        y = y + h * np.exp(-4 * np.log(2) * (x - c) ** 2 / w**2)
    y = y + rng.normal(0, 0.03, x.size)
    np.savetxt(path, np.column_stack([x, y]))
    return str(path)


class ScriptedToolLLM:
    """Replays a fixed sequence of responses; records what it was sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.transcripts = []

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(tool_use=True)

    def complete(self, messages, tools=None):
        self.transcripts.append((list(messages), tools))
        return self.responses.pop(0)


def test_registry_executes_and_reports_errors(tmp_path):
    curve_path = write_two_peak_curve(tmp_path / "curve.dat")
    registry = default_toolkit(tmp_path / "ws")

    result = registry.execute("find_peaks_1d", {"path": curve_path})
    assert result["n_peaks"] == 2

    fit = registry.execute("fit_peaks_1d", {"path": curve_path, "centers": [3.0, 7.0]})
    assert abs(fit["peaks"][0]["center"] - 3.0) < 0.01

    assert "error" in registry.execute("no_such_tool", {})
    assert "error" in registry.execute("find_peaks_1d", {"wrong_arg": 1})
    assert "error" in registry.execute("inspect_curve", {"path": "/nonexistent.dat"})


def test_agent_tool_loop_end_to_end(tmp_path):
    curve_path = write_two_peak_curve(tmp_path / "curve.dat")
    final = {
        "summary": "Two peaks fitted at 3.00 and 7.00.",
        "interpretation": ["Both peaks are well resolved."],
        "recommended_next_checks": ["Compare with the 100 K dataset."],
        "confidence": "high",
    }
    llm = ScriptedToolLLM(
        [
            LLMResponse(
                tool_calls=[ToolCall(name="find_peaks_1d", arguments={"path": curve_path},
                                     id="c1")]
            ),
            LLMResponse(
                tool_calls=[
                    ToolCall(name="fit_peaks_1d",
                             arguments={"path": curve_path, "centers": [3.0, 7.0]}, id="c2")
                ]
            ),
            LLMResponse(content=json.dumps(final)),
        ]
    )
    agent = Agent(llm=llm, model_id="scripted", workspace=tmp_path / "ws")
    report = agent.analyze(
        AnalysisRequest(
            domain="data",
            question="Fit the peaks in this pattern.",
            data={"files": [curve_path]},
        )
    )

    assert report.summary == final["summary"]
    assert report.confidence == Confidence.HIGH
    assert report.used_tools == ["find_peaks_1d", "fit_peaks_1d"]
    assert len(report.provenance.tool_calls) == 2
    assert report.provenance.prompt_version == "data_analysis/v2"

    # tool results actually flowed back into the conversation
    final_messages = llm.transcripts[-1][0]
    tool_messages = [m for m in final_messages if m.role == "tool"]
    assert len(tool_messages) == 2
    assert '"n_peaks": 2' in tool_messages[0].content
    # and the fit numbers the LLM cited came from the tool
    fit_payload = json.loads(tool_messages[1].content)
    assert abs(fit_payload["peaks"][0]["center"] - 3.0) < 0.01


def test_tool_loop_respects_round_cap(tmp_path):
    curve_path = write_two_peak_curve(tmp_path / "curve.dat")
    # LLM that calls tools forever, then would answer — cap must stop it
    responses = [
        LLMResponse(tool_calls=[ToolCall(name="inspect_curve",
                                         arguments={"path": curve_path}, id=f"c{i}")])
        for i in range(Agent.MAX_TOOL_ROUNDS)
    ] + [LLMResponse(content=json.dumps({"summary": "done", "confidence": "low"}))]
    llm = ScriptedToolLLM(responses)
    agent = Agent(llm=llm, model_id="scripted", workspace=tmp_path / "ws")
    report = agent.analyze(
        AnalysisRequest(domain="data", question="?", data={"files": [curve_path]})
    )
    assert report.summary == "done"
    assert len(report.provenance.tool_calls) == Agent.MAX_TOOL_ROUNDS


def test_missing_file_diagnostic_in_data_domain(tmp_path):
    agent = Agent(workspace=tmp_path / "ws")
    report = agent.analyze(
        AnalysisRequest(domain="data", question="?", data={"files": ["/nope.dat"]})
    )
    assert any("not found" in w for w in report.warnings)
