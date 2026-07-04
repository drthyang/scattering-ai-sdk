"""Human-reviewed self-improvement — E7 hand-off: one command to a coding agent.

Turns a Tier-1/Tier-2 proposal's brief (`briefs.py`) into a ready-to-run
hand-off: it writes the brief and emits a single command sequence that puts a
coding agent (Codex / Claude Code) to work on an **isolated git worktree**, so
the agent's changes never touch your current checkout and come back as a branch
you review.

Per D15 this prepares the hand-off; it does **not** launch the agent or merge
anything itself. The agent runs under the human's hand, and the result is a
branch + PR for review — the loop never lands scientific/code changes.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from scattering_ai.learning.briefs import Proposal, brief_from_proposal, render_brief


class HandoffPlan(BaseModel):
    proposal_id: str
    tier: int
    agent: str
    branch: str
    worktree: str
    brief_path: str
    commands: list[str] = Field(default_factory=list)


def _agent_command(agent: str, brief_path: Path) -> str:
    """The command that runs the chosen agent with the brief as its prompt."""
    quoted = f'"$(cat \'{brief_path}\')"'
    if agent == "codex":
        return f"codex exec {quoted}"
    if agent == "claude":
        return f"claude -p {quoted}"
    raise ValueError(f"unknown agent {agent!r}; use 'codex' or 'claude'")


def prepare_handoff(proposal: Proposal, journal_dir: str | Path,
                    repo_root: str | Path, agent: str = "codex") -> HandoffPlan:
    """Write the brief and build the isolated-worktree hand-off command sequence.

    Tier-0 is rejected — it applies directly through the eval-gated ``apply``.
    """
    if proposal.tier == 0:
        raise ValueError(
            f"proposal {proposal.id!r} is Tier-0 — apply it directly via "
            "`learn apply` (eval-gated), not a coding-agent hand-off")

    brief = brief_from_proposal(proposal)
    briefs_dir = Path(journal_dir) / "briefs"
    briefs_dir.mkdir(parents=True, exist_ok=True)
    brief_path = briefs_dir / f"{proposal.id}.md"
    brief_path.write_text(render_brief(brief), encoding="utf-8")

    repo_root = Path(repo_root)
    branch = brief.branch  # fix/<proposal-id>
    worktree = repo_root.parent / f"sai-handoff-{proposal.id}"

    commands = [
        f'git -C "{repo_root}" worktree add -b {branch} "{worktree}"',
        f'cd "{worktree}"',
        _agent_command(agent, brief_path),
        f'# review:  git -C "{repo_root}" diff main...{branch}   then open a PR',
        f'# cleanup: git -C "{repo_root}" worktree remove "{worktree}"',
    ]
    return HandoffPlan(
        proposal_id=proposal.id, tier=proposal.tier, agent=agent, branch=branch,
        worktree=str(worktree), brief_path=str(brief_path), commands=commands)


def render_handoff(plan: HandoffPlan) -> str:
    """Human-readable hand-off: where the brief is and the one command to run."""
    return "\n".join([
        f"# Hand-off: `{plan.proposal_id}` (tier {plan.tier}) → {plan.agent}",
        "",
        f"- **brief:** {plan.brief_path}",
        f"- **branch:** `{plan.branch}` (isolated worktree: {plan.worktree})",
        "",
        "Run:",
        "",
        "```bash",
        *plan.commands,
        "```",
        "",
        "The agent works in an isolated worktree and returns a branch — review "
        "the diff and open a PR. Nothing is launched or merged for you.",
    ]) + "\n"
