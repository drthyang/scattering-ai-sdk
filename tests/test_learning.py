"""Self-improvement Phase 1: the capture journal (opt-in, redacted, read-only)."""

import json
from pathlib import Path

import numpy as np

from scattering_ai import Agent, AnalysisRequest
from scattering_ai.learning.journal import Journal, resolve_journal_dir

_EXPECTED_FIELDS = {
    "id", "timestamp", "surface", "sdk_version", "domain", "model", "input_hash",
    "n_files", "file_names", "tools", "findings", "confidence",
    "provenance_complete", "interpretation_available", "n_figures", "outcome",
    "duration_s",
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


# --- Phase 2: signals + corrections -------------------------------------------

from scattering_ai.learning.journal import Episode, FindingTag  # noqa: E402
from scattering_ai.learning.signals import (  # noqa: E402
    cluster_signals,
    make_correction,
    signals_for_episode,
    signals_report,
)


def _episode(**kw) -> Episode:
    base = dict(id="ep", timestamp="t", surface="analyze", domain="pdf")
    base.update(kw)
    return Episode(**base)


def test_signals_derived_deterministically_per_type():
    err = _episode(id="e1", outcome="error",
                   findings=[FindingTag(diagnostic="peak_fit", severity="error")])
    warn = _episode(id="e2", outcome="warnings",
                    findings=[FindingTag(diagnostic="mask_sentinel", severity="warning")])
    prov = _episode(id="e3", provenance_complete=False)
    lowc = _episode(id="e4", model="claude", confidence="low")
    nointerp = _episode(id="e5", model="claude", interpretation_available=False)
    empty = _episode(id="e6")  # no findings, no figures

    types = lambda ep: {s.type for s in signals_for_episode(ep)}
    assert types(err) == {"error_outcome"}
    assert types(warn) == {"unhandled_warning"}
    assert types(prov) == {"provenance_gap", "empty_result"}
    assert "low_confidence" in types(lowc)
    assert "interpretation_unavailable" in types(nointerp)
    assert types(empty) == {"empty_result"}

    # confidence/interpretation signals require an LLM to have contributed
    assert signals_for_episode(_episode(confidence="low")) == \
        signals_for_episode(_episode())  # no model -> no low_confidence


def test_error_signal_keyed_by_diagnostic_for_clustering():
    eps = [
        _episode(id=f"e{i}", outcome="error",
                 findings=[FindingTag(diagnostic="peak_fit", severity="error")])
        for i in range(3)
    ]
    sigs = [s for ep in eps for s in signals_for_episode(ep)]
    clusters = cluster_signals(sigs)
    assert len(clusters) == 1
    top = clusters[0]
    assert top.type == "error_outcome" and top.key == "peak_fit"
    assert top.count == 3 and set(top.episodes) == {"e0", "e1", "e2"}


def test_correction_records_and_surfaces_as_signal(tmp_path):
    jdir = tmp_path / "j"
    ag = Agent(journal=jdir)
    report = ag.analyze(AnalysisRequest(question="?", data={"files": _series(tmp_path)}))
    ep = Journal(jdir).episodes()[0]

    corr = ag.record_correction(
        ep, target="transition_temperature",
        statement="GaNb4Se8 transitions are 50 K and 29 K, not 39 K",
        corrected_value="50,29")
    assert corr.episode_id == ep.id and corr.corrected_value == "50,29"

    corrs = Journal(jdir).corrections()
    assert len(corrs) == 1 and corrs[0].target == "transition_temperature"

    rep = signals_report(Journal(jdir))
    assert rep["n_corrections"] == 1
    corr_clusters = [c for c in rep["clusters"] if c["type"] == "correction"]
    assert corr_clusters and corr_clusters[0]["key"] == "transition_temperature"
    assert report.status == "ok"  # correcting never disturbs the report


def test_record_correction_requires_journal():
    import pytest

    with pytest.raises(RuntimeError):
        Agent().record_correction("ep", target="domain", statement="wrong route")


def test_high_severity_recurring_cluster_ranks_first():
    # An "ok" run with a figure + info finding trips no incidental signal, so we
    # isolate exactly the signal under test.
    info = [FindingTag(diagnostic="ok", severity="info")]
    low = [_episode(id=f"l{i}", model="c", confidence="low", n_figures=1, findings=info)
           for i in range(2)]  # low_confidence, weight 1, count 2 -> 2
    high = _episode(id="x", provenance_complete=False, n_figures=1, findings=info)  # 3
    sigs = [s for ep in low for s in signals_for_episode(ep)] + signals_for_episode(high)
    clusters = cluster_signals(sigs)
    # high-severity provenance_gap outranks a more frequent low-severity signal
    assert clusters[0].type == "provenance_gap"
    assert {c.type for c in clusters} == {"provenance_gap", "low_confidence"}


def test_cli_learn_signals_and_correct(tmp_path, capsys):
    import json

    from scattering_ai.cli import main

    jdir = tmp_path / "j"
    Agent(journal=jdir).analyze(
        AnalysisRequest(question="?", data={"files": _series(tmp_path)}))
    ep = Journal(jdir).episodes()[0]

    rc = main(["learn", "correct", "--journal", str(jdir), "--episode", ep.id,
               "--target", "domain", "--statement", "should be pdf not data"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["recorded"] == "correction"

    rc = main(["learn", "signals", "--journal", str(jdir)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_corrections"] == 1
    assert any(c["type"] == "correction" for c in out["clusters"])


# --- Phase 3: proposals + review ----------------------------------------------

from scattering_ai.learning.proposals import (  # noqa: E402
    Proposal,
    build_proposals,
    render_proposals,
)


class _FakeJournal:
    """A journal stand-in returning fixed episodes/corrections."""

    def __init__(self, episodes, corrections=()):
        self._eps, self._corrs = list(episodes), list(corrections)

    def episodes(self):
        return self._eps

    def corrections(self):
        return self._corrs


def test_recurrence_threshold_gates_non_correction_proposals():
    warn = [_episode(id=f"w{i}", outcome="warnings",
                     findings=[FindingTag(diagnostic="mask_sentinel", severity="warning")])
            for i in range(3)]
    # 2 occurrences < default threshold of 3 -> no proposal
    assert build_proposals(_FakeJournal(warn[:2])) == []
    props = build_proposals(_FakeJournal(warn))
    assert len(props) == 1
    p = props[0]
    assert p.tier == 0 and p.change_class == "next_check_rule" and p.key == "mask_sentinel"
    assert p.evidence_count == 3 and set(p.evidence_episodes) == {"w0", "w1", "w2"}


def test_correction_becomes_tier0_regression_eval_carrying_value():
    corr = make_correction("e1", target="transition_temperature",
                           statement="transitions are 50 K and 29 K, not 39 K",
                           corrected_value="50,29", domain="series")
    props = build_proposals(_FakeJournal([], [corr]))
    assert len(props) == 1  # a single correction always proposes
    p = props[0]
    assert p.tier == 0 and p.change_class == "regression_eval"
    assert p.corrected_value == "50,29" and "50,29" in p.suggested_action
    assert p.evidence_episodes == ["e1"]


def test_mis_route_correction_is_tier1_router_rule():
    corr = make_correction("e2", target="domain",
                           statement="should be pdf not data", domain="data")
    p = build_proposals(_FakeJournal([], [corr]))[0]
    assert p.tier == 1 and p.change_class == "router_rule"


def test_proposals_sorted_tier0_first():
    err = [_episode(id=f"e{i}", outcome="error",
                    findings=[FindingTag(diagnostic="peak_fit", severity="error")])
           for i in range(3)]  # -> Tier 2 task
    corr = make_correction("c1", target="d_spacing", statement="wrong",
                           corrected_value="2.5", domain="pdf")  # -> Tier 0
    props = build_proposals(_FakeJournal(err, [corr]))
    assert [p.tier for p in props] == sorted(p.tier for p in props)
    assert props[0].tier == 0 and props[-1].tier == 2


def test_prose_polish_touches_only_title_and_rationale():
    corr = make_correction("e1", target="transition_temperature",
                           statement="transitions are 50 K and 29 K",
                           corrected_value="50,29", domain="series")
    before = build_proposals(_FakeJournal([], [corr]))[0]

    def evil_polish(p: Proposal):
        # a misbehaving polish tries to smuggle in new facts; only prose applies
        return "REPHRASED TITLE", "REPHRASED RATIONALE"

    after = build_proposals(_FakeJournal([], [corr]), polish=evil_polish)[0]
    assert after.title == "REPHRASED TITLE" and after.rationale == "REPHRASED RATIONALE"
    assert after.prose_polished is True
    # every deterministic field is untouched by the polish step
    assert after.tier == before.tier
    assert after.change_class == before.change_class
    assert after.suggested_action == before.suggested_action
    assert after.corrected_value == before.corrected_value
    assert after.evidence_episodes == before.evidence_episodes


def test_polish_failure_falls_back_to_deterministic_prose():
    corr = make_correction("e1", target="x", statement="s", domain="pdf")
    def boom(p):
        raise RuntimeError("model down")
    p = build_proposals(_FakeJournal([], [corr]), polish=boom)[0]
    assert p.prose_polished is False and p.title  # original prose kept


def test_render_proposals_markdown_groups_by_tier():
    corr = make_correction("e1", target="transition_temperature",
                           statement="50 K and 29 K", corrected_value="50,29",
                           domain="series")
    md = render_proposals(build_proposals(_FakeJournal([], [corr])))
    assert "# Self-improvement proposals" in md
    assert "Tier 0" in md and "Suggested action" in md
    assert render_proposals([]).strip().endswith("threshold yet.")


def test_cli_learn_review(tmp_path, capsys):
    from scattering_ai.cli import main

    jdir = tmp_path / "j"
    Agent(journal=jdir).analyze(
        AnalysisRequest(question="?", data={"files": _series(tmp_path)}))
    ep = Journal(jdir).episodes()[0]
    main(["learn", "correct", "--journal", str(jdir), "--episode", ep.id,
          "--target", "transition_temperature", "--statement", "50 K and 29 K",
          "--value", "50,29", "--domain", "series"])
    capsys.readouterr()

    rc = main(["learn", "review", "--journal", str(jdir), "--write"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Self-improvement proposals" in out and "Tier 0" in out
    assert (jdir / "proposals.md").exists()


# --- Phase 4: guarded apply ---------------------------------------------------

import pytest  # noqa: E402

from scattering_ai.learning.apply import (  # noqa: E402
    GateResult,
    TierViolation,
    apply_proposal,
    applied_records,
    assert_not_denylisted,
    assert_tier0_target,
    pytest_gate,
)
from scattering_ai.learning.proposals import build_proposals as _bp  # noqa: E402


def _journal(tmp_path):
    return Journal(tmp_path / "j")


def _regression_proposal():
    corr = make_correction("e1", target="transition_temperature",
                           statement="transitions are 50 K and 29 K",
                           corrected_value="50,29", domain="series")
    return _bp(_FakeJournal([], [corr]))[0]  # Tier-0 regression_eval


_GREEN = lambda base: GateResult(passed=True, detail="ok")
_RED = lambda base: GateResult(passed=False, detail="1 failed")


def test_denylist_blocks_source_and_schemas():
    for bad in ("src/scattering_ai/core/schemas.py",
                "src/scattering_ai/domains/pdf/diagnostics.py",
                "src/scattering_ai/tools/curves.py"):
        with pytest.raises(TierViolation):
            assert_not_denylisted(bad)
    with pytest.raises(TierViolation):
        assert_not_denylisted("../escape.py")
    # Tier-0 must also stay inside the data allowlist
    with pytest.raises(TierViolation):
        assert_tier0_target("some/other/place.json")
    assert assert_tier0_target("tests/regressions/p.json")


def test_nothing_applies_without_approval(tmp_path):
    j = _journal(tmp_path)
    rec = apply_proposal(_regression_proposal(), journal=j, repo_root=tmp_path,
                         gate=_GREEN)  # approve defaults to False
    assert rec.outcome == "refused" and rec.gate_passed is None
    assert not (tmp_path / "tests").exists()  # no diff written
    assert applied_records(j)[0].outcome == "refused"  # but the attempt is audited


def test_tier0_lands_only_when_gate_green(tmp_path):
    j = _journal(tmp_path)
    p = _regression_proposal()
    rec = apply_proposal(p, journal=j, repo_root=tmp_path, approve=True, gate=_GREEN)
    assert rec.outcome == "applied" and rec.gate_passed is True
    written = tmp_path / "tests" / "regressions" / f"{p.id}.json"
    assert written.exists()
    case = json.loads(written.read_text())
    assert case["target"] == "transition_temperature" and case["corrected_value"] == "50,29"
    # audit links proposal -> evidence episodes -> outcome (invariant 4)
    audit = applied_records(j)[-1]
    assert audit.proposal_id == p.id and audit.evidence_episodes == ["e1"]


def test_tier0_reverts_byte_for_byte_when_gate_red(tmp_path):
    j = _journal(tmp_path)
    p = _regression_proposal()
    rec = apply_proposal(p, journal=j, repo_root=tmp_path, approve=True, gate=_RED)
    assert rec.outcome == "reverted" and rec.gate_passed is False
    # a red gate leaves the tree exactly as it was: no file, no empty dirs
    assert not (tmp_path / "tests" / "regressions" / f"{p.id}.json").exists()
    assert not (tmp_path / "tests").exists()


def test_tier0_revert_restores_prior_content(tmp_path):
    j = _journal(tmp_path)
    p = _regression_proposal()
    target = tmp_path / "tests" / "regressions" / f"{p.id}.json"
    target.parent.mkdir(parents=True)
    target.write_text("ORIGINAL")
    apply_proposal(p, journal=j, repo_root=tmp_path, approve=True, gate=_RED)
    assert target.read_text() == "ORIGINAL"  # pre-existing content untouched


def test_tier0_symlink_target_cannot_escape_into_source(tmp_path):
    # Reported by a Codex-delegated review: the allowlist/denylist check the
    # relative string, so a symlinked tests/regressions -> src/ would let a
    # Tier-0 write land on denylisted source while the string stays "allowed".
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "tests" / "regressions").symlink_to(repo / "src", target_is_directory=True)

    j = _journal(tmp_path)
    p = _regression_proposal()  # would write tests/regressions/<id>.json
    with pytest.raises(TierViolation):
        apply_proposal(p, journal=j, repo_root=repo, approve=True, gate=_GREEN)
    assert not any((repo / "src").iterdir())  # nothing reached source via the link


def test_tier1_writes_inert_draft_never_source(tmp_path):
    corr = make_correction("e2", target="domain", statement="should be pdf",
                           domain="data")
    p = _bp(_FakeJournal([], [corr]))[0]  # Tier-1 router_rule
    j = _journal(tmp_path)
    rec = apply_proposal(p, journal=j, repo_root=tmp_path, approve=True, gate=_RED)
    assert rec.outcome == "draft" and rec.gate_passed is None  # no gate for drafts
    draft = j.dir / "drafts" / f"{p.id}.md"
    assert draft.exists() and "DRAFT (not activated)" in draft.read_text()
    assert not (tmp_path / "src").exists()


def test_tier2_writes_task_with_evidence_only(tmp_path):
    eps = [_episode(id=f"e{i}", outcome="error",
                    findings=[FindingTag(diagnostic="peak_fit", severity="error")])
           for i in range(3)]
    p = _bp(_FakeJournal(eps))[0]  # Tier-2 task
    assert p.tier == 2
    j = _journal(tmp_path)
    rec = apply_proposal(p, journal=j, repo_root=tmp_path, approve=True)
    assert rec.outcome == "task"
    task = j.dir / "tasks" / f"{p.id}.md"
    assert task.exists() and "Tier 2" in task.read_text()
    assert set(rec.evidence_episodes) == {"e0", "e1", "e2"}


def test_series_correction_embeds_data_dependent_check():
    from scattering_ai.learning.apply import _writes_regression_eval, synthesize_check

    assert synthesize_check("series", "transition_temperature", "50 K and 29 K") == {
        "kind": "series_transitions", "temperatures": [50.0, 29.0], "tol": 8.0}
    assert synthesize_check("pdf", "d_spacing", "2.5")["kind"] == "value_present"

    payload = json.loads(_writes_regression_eval(_regression_proposal())[0].content)
    assert payload["check"]["kind"] == "series_transitions"
    assert sorted(payload["check"]["temperatures"]) == [29.0, 50.0]


def test_real_pytest_gate_runs_the_science_end_to_end(tmp_path):
    """The default subprocess gate runs the REAL regression test, which exercises
    the changepoint detector on the applied series correction."""
    import shutil

    repo = tmp_path / "repo"
    (repo / "tests" / "regressions").mkdir(parents=True)
    # copy the real gate so the data-dependent check actually executes
    shutil.copy(Path(__file__).parent / "test_regressions.py",
                repo / "tests" / "test_regressions.py")

    j = _journal(tmp_path)
    good = _regression_proposal()  # series / transition_temperature = 50,29
    rec = apply_proposal(good, journal=j, repo_root=repo, approve=True, gate=pytest_gate)
    assert rec.outcome == "applied", rec.gate_detail
    case = json.loads((repo / "tests" / "regressions" / f"{good.id}.json").read_text())
    assert case["check"]["kind"] == "series_transitions"


def test_cli_apply_dry_run_then_approve(tmp_path, capsys):
    from scattering_ai.cli import main

    jdir = tmp_path / "j"
    ag = Agent(journal=jdir)
    ag.analyze(AnalysisRequest(question="?", data={"files": _series(tmp_path)}))
    ep = Journal(jdir).episodes()[0]
    ag.record_correction(ep, target="d_spacing", statement="wrong",
                         corrected_value="2.5", domain="pdf")
    pid = _bp(Journal(jdir))[0].id

    # dry run: no --approve -> nothing written
    rc = main(["learn", "apply", "--journal", str(jdir), "--id", pid])
    assert rc == 0 and json.loads(capsys.readouterr().out)["dry_run"] is True

    # approve into an isolated repo root -> applied + audited
    repo = tmp_path / "repo"
    (repo / "tests" / "regressions").mkdir(parents=True)
    (repo / "tests" / "test_regressions.py").write_text(
        "def test_ok():\n    assert True\n")
    rc = main(["learn", "apply", "--journal", str(jdir), "--id", pid,
               "--approve", "--repo", str(repo)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["outcome"] == "applied" and out["proposal_id"] == pid
    assert (repo / "tests" / "regressions" / f"{pid}.json").exists()


# --- Phase E7: agent-executed improvement briefs ------------------------------

from scattering_ai.learning.briefs import (  # noqa: E402
    Brief,
    brief_from_proposal,
    render_brief,
)


def test_brief_from_tier1_proposal_carries_evidence_and_constraints():
    corr = make_correction("e9", target="domain", statement="should be pdf not data",
                           domain="data")
    p = _bp(_FakeJournal([], [corr]))[0]  # Tier-1 router_rule
    brief = brief_from_proposal(p)
    assert brief.tier == 1 and brief.branch == f"fix/{p.id}"
    assert brief.evidence_episodes == ["e9"]
    assert "router.py" in brief.affected_area
    assert any("isolated branch" in c for c in brief.constraints)
    assert any("pytest" in a for a in brief.acceptance)


def test_brief_from_tier2_task_points_at_diagnostic():
    eps = [_episode(id=f"e{i}", outcome="error",
                    findings=[FindingTag(diagnostic="peak_fit", severity="error")])
           for i in range(3)]
    p = _bp(_FakeJournal(eps))[0]  # Tier-2 task, error_outcome
    brief = brief_from_proposal(p)
    assert brief.tier == 2 and "peak_fit" in brief.affected_area
    md = render_brief(brief)
    assert "Improvement brief" in md and "Constraints" in md and "not a patch" in md


def test_brief_rejects_tier0():
    corr = make_correction("e1", target="d_spacing", statement="wrong",
                           corrected_value="2.5", domain="pdf")
    p = _bp(_FakeJournal([], [corr]))[0]  # Tier-0 regression_eval
    with pytest.raises(ValueError):
        brief_from_proposal(p)


def test_cli_learn_brief(tmp_path, capsys):
    from scattering_ai.cli import main

    jdir = tmp_path / "j"
    ag = Agent(journal=jdir)
    ag.analyze(AnalysisRequest(question="?", data={"files": _series(tmp_path)}))
    ep = Journal(jdir).episodes()[0]
    ag.record_correction(ep, target="domain", statement="should be pdf", domain="data")
    pid = _bp(Journal(jdir))[0].id

    rc = main(["learn", "brief", "--journal", str(jdir), "--id", pid, "--write"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Improvement brief" in out and "Constraints" in out
    assert (jdir / "briefs" / f"{pid}.md").exists()
