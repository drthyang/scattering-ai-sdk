"""Human-reviewed self-improvement — Phase E7: agent-executed improvement briefs.

For a Tier-1 (draft) or Tier-2 (task) proposal the loop cannot safely auto-apply,
a **brief** packages the evidence and intent into a self-contained artifact a
coding agent (Codex / Claude Code) can act on — on an **isolated branch**, with
the result returned as a branch + PR for human review.

Per D15 (self-implementation is agent-drafted, human-gated): a brief is
**evidence + intent, never a patch**. It carries no suggested diff and cannot
authorise a merge; scientific logic, schemas, and core code still change only
through a human-reviewed diff. This is the "the agent drafts work I review"
capability, bounded by the same tier system that governs apply.

Tier-0 proposals do not get briefs — they apply directly through the eval-gated
``apply_proposal``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from scattering_ai.learning.proposals import Proposal


class Brief(BaseModel):
    """A reviewable work order for a coding agent, derived from a proposal."""

    proposal_id: str
    tier: int
    change_class: str
    title: str
    intent: str
    evidence_episodes: list[str] = Field(default_factory=list)
    corrected_value: str = ""
    affected_area: str = ""
    acceptance: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    branch: str = ""


# Best-guess starting point per change class — guidance for the agent, not a
# claim of the exact edit. The agent must diagnose before changing anything.
def _affected_area(p: Proposal) -> str:
    if p.change_class == "router_rule":
        return "src/scattering_ai/domains/router.py (routing rules)"
    if p.change_class == "prompt_version_bump":
        dom = p.domain or "<domain>"
        return (f"src/scattering_ai/domains/{dom}/prompts.py — draft a NEW "
                "PROMPT_VERSION; never edit an activated prompt in place")
    if p.signal_type == "provenance_gap":
        return "src/scattering_ai/core/ provenance handling (schemas.py is Tier-2)"
    if p.signal_type == "error_outcome":
        return (f"the '{p.key}' diagnostic in the '{p.domain or '?'}' pack — "
                "reproduce the failure first")
    if p.signal_type == "empty_result":
        return f"the '{p.domain or '?'}' pack diagnostics (a coverage gap)"
    return "investigate from the evidence episodes before editing"


_BASE_CONSTRAINTS = [
    "Work on an isolated branch; return a branch + PR for human review — do not merge.",
    "This brief is evidence and intent, not a patch. Diagnose before changing anything.",
    "Scientific logic, schemas, and core code change only through a human-reviewed diff.",
    "Keep the change minimal and add a regression test that pins the fix.",
]


def brief_from_proposal(proposal: Proposal) -> Brief:
    """Package a Tier-1/Tier-2 proposal into an agent-ready brief."""
    if proposal.tier == 0:
        raise ValueError(
            f"proposal {proposal.id!r} is Tier-0 — it applies directly via "
            "`learn apply` (eval-gated), not through a brief")

    acceptance = ["The full test suite passes (`pytest -q`)."]
    if proposal.tier == 1:
        acceptance.append("The change is drafted for review, not silently activated.")
    else:  # tier 2
        acceptance.append(
            "No schema or core-reasoning change without explicit human sign-off.")
    acceptance.append(f"A regression test pins the behavior for '{proposal.key}'.")

    return Brief(
        proposal_id=proposal.id,
        tier=proposal.tier,
        change_class=proposal.change_class,
        title=proposal.title,
        intent=f"{proposal.rationale} {proposal.suggested_action}".strip(),
        evidence_episodes=list(proposal.evidence_episodes),
        corrected_value=proposal.corrected_value,
        affected_area=_affected_area(proposal),
        acceptance=acceptance,
        constraints=list(_BASE_CONSTRAINTS),
        branch=f"fix/{proposal.id}",
    )


def render_brief(brief: Brief) -> str:
    """Render a brief as markdown suitable to hand to a coding agent verbatim."""
    def bullets(items):
        return "\n".join(f"- {i}" for i in items) if items else "- (none)"

    lines = [
        f"# Improvement brief: {brief.title}",
        "",
        f"- **proposal:** `{brief.proposal_id}`  ·  **tier {brief.tier} · "
        f"{brief.change_class}**",
        f"- **suggested branch:** `{brief.branch}`",
        f"- **evidence episodes:** {', '.join(brief.evidence_episodes) or '—'}",
    ]
    if brief.corrected_value:
        lines.append(f"- **corrected value:** `{brief.corrected_value}`")
    lines += [
        "",
        "## Intent",
        "",
        brief.intent,
        "",
        "## Where to look first",
        "",
        brief.affected_area,
        "",
        "## Acceptance",
        "",
        bullets(brief.acceptance),
        "",
        "## Constraints (non-negotiable)",
        "",
        bullets(brief.constraints),
        "",
        "> Generated by the self-improvement loop (E7). It is a work order, not a "
        "patch: a human reviews and merges the resulting branch.",
    ]
    return "\n".join(lines) + "\n"
