"""Self-improvement E7: one-command hand-off of a proposal to a coding agent."""

import pytest

from scattering_ai.learning.handoff import prepare_handoff, render_handoff
from scattering_ai.learning.proposals import Proposal


def _tier1():
    return Proposal(id="p-1-correction-data-domain", tier=1, change_class="router_rule",
                    signal_type="correction", domain="data", key="domain",
                    title="Draft a router rule", rationale="mis-route",
                    suggested_action="draft a router rule", evidence_episodes=["e1"])


def _tier2():
    return Proposal(id="p-2-error-pdf-peak_fit", tier=2, change_class="task",
                    signal_type="error_outcome", domain="pdf", key="peak_fit",
                    title="Investigate peak_fit errors", rationale="errors",
                    suggested_action="open a task", evidence_episodes=["e0", "e1"])


def test_handoff_writes_brief_and_builds_worktree_command(tmp_path):
    jdir = tmp_path / "journal"
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = prepare_handoff(_tier1(), jdir, repo, agent="codex")

    assert plan.branch == "fix/p-1-correction-data-domain"
    assert (jdir / "briefs" / "p-1-correction-data-domain.md").exists()
    joined = "\n".join(plan.commands)
    assert "worktree add -b fix/p-1-correction-data-domain" in joined
    assert "codex exec" in joined and "worktree remove" in joined
    md = render_handoff(plan)
    assert "Hand-off" in md and "isolated worktree" in md


def test_handoff_agent_choice_switches_command(tmp_path):
    plan = prepare_handoff(_tier2(), tmp_path / "j", tmp_path / "repo", agent="claude")
    assert any("claude -p" in c for c in plan.commands)
    with pytest.raises(ValueError):
        prepare_handoff(_tier2(), tmp_path / "j", tmp_path / "repo", agent="gpt")


def test_handoff_rejects_tier0(tmp_path):
    t0 = Proposal(id="p-0-correction-pdf-d_spacing", tier=0,
                  change_class="regression_eval", signal_type="correction",
                  domain="pdf", key="d_spacing", title="Pin eval", rationale="x",
                  suggested_action="add eval")
    with pytest.raises(ValueError):
        prepare_handoff(t0, tmp_path / "j", tmp_path / "repo")


def test_cli_learn_handoff(tmp_path, capsys):
    import numpy as np

    from scattering_ai import Agent, AnalysisRequest
    from scattering_ai.cli import main
    from scattering_ai.learning.journal import Journal
    from scattering_ai.learning.proposals import build_proposals

    jdir = tmp_path / "j"
    x = np.linspace(0, 10, 400)
    p = tmp_path / "scan_T_base_5.0K.dat"
    np.savetxt(p, np.column_stack([x, 1 + np.exp(-((x - 4) ** 2))]))
    ag = Agent(journal=jdir)
    ag.analyze(AnalysisRequest(question="?", data={"files": [str(p)]}))
    ep = Journal(jdir).episodes()[0]
    ag.record_correction(ep, target="domain", statement="should be pdf", domain="data")
    pid = next(pr.id for pr in build_proposals(Journal(jdir)) if pr.tier == 1)

    rc = main(["learn", "handoff", "--journal", str(jdir), "--id", pid,
               "--repo", str(tmp_path / "repo")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Hand-off" in out and "worktree add" in out
