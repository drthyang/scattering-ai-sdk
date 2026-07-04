"""Self-improvement Phase 1: the capture journal (opt-in, redacted, read-only)."""

import numpy as np

from scattering_ai import Agent, AnalysisRequest
from scattering_ai.learning.journal import Journal, resolve_journal_dir

_EXPECTED_FIELDS = {
    "id", "timestamp", "surface", "sdk_version", "domain", "model", "input_hash",
    "n_files", "file_names", "tools", "findings", "confidence",
    "provenance_complete", "n_figures", "outcome", "duration_s",
}


def _series(tmp_path):
    x = np.linspace(0, 10, 800)
    paths = []
    for t in (5.0, 10.0, 15.0, 20.0):
        y = 1 + 5 * np.exp(-((x - 4) ** 2) / 0.03)
        p = tmp_path / f"scan_T_base_{t:.1f}K.dat"
        np.savetxt(p, np.column_stack([x, y]))
        paths.append(str(p))
    return paths


def test_journaling_is_opt_in_off_by_default(tmp_path):
    Agent().analyze(AnalysisRequest(question="?", data={"files": _series(tmp_path)}))
    # no journal configured -> nothing written anywhere under tmp
    assert not list(tmp_path.glob("**/episodes.jsonl"))


def test_journal_records_redacted_episode(tmp_path):
    jdir = tmp_path / "journal"
    Agent(journal=jdir).analyze(
        AnalysisRequest(question="?", data={"files": _series(tmp_path)}))
    j = Journal(jdir)
    eps = j.episodes()
    assert len(eps) == 1
    ep = eps[0]
    assert ep.surface == "analyze" and ep.domain == "data"
    assert ep.n_files == 4 and all(f.endswith(".dat") for f in ep.file_names)
    # only category-label fields exist — the pydantic model is the whitelist
    assert set(ep.model_dump().keys()) == _EXPECTED_FIELDS
    # findings are reduced to diagnostic + severity; no evidence values
    for f in ep.findings:
        assert set(f.model_dump().keys()) == {"diagnostic", "severity"}
    # the raw file carries no evidence / data-value keys
    raw = j.path.read_text()
    for leaked in ("evidence", "positions", "lattice", "transition_param", "r_values"):
        assert leaked not in raw


def test_summary_aggregates(tmp_path):
    jdir = tmp_path / "j"
    ag = Agent(journal=jdir)
    ag.analyze(AnalysisRequest(question="?", data={"files": _series(tmp_path)}))
    ag.analyze(AnalysisRequest(domain="data", question="?", data={"files": []}))
    s = Journal(jdir).summary()
    assert s["n_episodes"] == 2
    assert s["by_surface"]["analyze"] == 2
    assert "data" in s["by_domain"]
    assert set(s["by_outcome"]) <= {"ok", "warnings", "error"}


def test_journaling_failure_never_breaks_analysis(tmp_path):
    # a journal path that is a regular file -> Journal() mkdir fails, but analyze
    # must still return a valid report (best-effort capture)
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("i am a file")
    report = Agent(journal=blocker).analyze(
        AnalysisRequest(question="?", data={"files": _series(tmp_path)}))
    assert report.status == "ok"


def test_resolve_journal_dir_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SCATTERING_AI_JOURNAL", str(tmp_path / "envj"))
    assert resolve_journal_dir(None) == tmp_path / "envj"
    monkeypatch.delenv("SCATTERING_AI_JOURNAL")
    assert resolve_journal_dir(None) is None
    assert resolve_journal_dir(tmp_path / "explicit") == tmp_path / "explicit"


def test_cli_learn_status(tmp_path, capsys):
    import json

    from scattering_ai.cli import main

    jdir = tmp_path / "j"
    Agent(journal=jdir).analyze(
        AnalysisRequest(question="?", data={"files": _series(tmp_path)}))
    rc = main(["learn", "status", "--journal", str(jdir)])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["n_episodes"] == 1
