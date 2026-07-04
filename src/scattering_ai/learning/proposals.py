"""Human-reviewed self-improvement — Phase 3: proposals + review.

A **proposal** turns a recurring signal cluster (P2) into a reviewable
improvement suggestion. Proposals are generated **deterministically** from
templates: the tier, change class, evidence, and any suggested value come
straight from the signals — never from a model.

An LLM may *optionally* rewrite a proposal's ``title`` and ``rationale`` into
more readable English (``polish_prose``). That is the **only** thing it may
touch: it can never introduce or alter evidence, the suggested action, the
corrected value, the tier, or the change class. This is the "LLM role: prose
only" decision, enforced structurally by re-applying the deterministic fields
after any polish.

Nothing here edits ``src/``, prompts, schemas, or scientific logic. Generating
and reviewing proposals is read-only; the guarded apply step is Phase 4.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from pydantic import BaseModel, Field

from scattering_ai.learning.signals import (
    Correction,
    SignalCluster,
    cluster_signals,
    signal_from_correction,
    signals_for_episode,
)

# Default recurrence a non-correction signal must reach before it is worth a
# proposal. Corrections are exempt: a single human correction always proposes.
DEFAULT_MIN_OCCURRENCES = 3


class Proposal(BaseModel):
    """A reviewable improvement suggestion, tier-classified by risk.

    The deterministic fields (``tier``, ``change_class``, ``suggested_action``,
    ``corrected_value``, ``evidence_*``) are the contract with the apply step
    (P4). ``title`` and ``rationale`` are the only prose an LLM may rewrite.
    """

    id: str
    tier: int  # 0 data | 1 draft-only | 2 never-touched
    change_class: str  # regression_eval | next_check_rule | prompt_version_bump | router_rule | task
    signal_type: str
    domain: str = ""
    key: str = ""
    title: str
    rationale: str
    suggested_action: str
    corrected_value: str = ""
    evidence_episodes: list[str] = Field(default_factory=list)
    evidence_count: int = 0
    prose_polished: bool = False


# --- deterministic tier mapping ----------------------------------------------
# Each non-correction signal type maps to exactly one (tier, change_class).
# Corrections are handled separately because their tier depends on the target.

_SPECS: dict[str, dict[str, Any]] = {
    "unhandled_warning": {"tier": 0, "change_class": "next_check_rule"},
    "low_confidence": {"tier": 1, "change_class": "prompt_version_bump"},
    "interpretation_unavailable": {"tier": 1, "change_class": "prompt_version_bump"},
    "provenance_gap": {"tier": 2, "change_class": "task"},
    "error_outcome": {"tier": 2, "change_class": "task"},
    "empty_result": {"tier": 2, "change_class": "task"},
}


def _slug(*parts: str) -> str:
    raw = "-".join(p for p in parts if p)
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in raw).strip("-_").lower()


def _templated_text(cluster: SignalCluster) -> tuple[str, str, str]:
    """(title, rationale, suggested_action) for a non-correction cluster."""
    dom = cluster.domain or "unknown"
    n = cluster.count
    key = cluster.key
    t = cluster.type
    if t == "unhandled_warning":
        return (
            f"Add a next-check rule for the recurring '{key}' warning in '{dom}'",
            f"The '{key}' diagnostic raised a warning in {n} runs in the '{dom}' "
            "domain without a guiding next-check, so the user is left without a "
            "canned follow-up.",
            f"Add a next_check_rules entry keyed on '{key}' in the '{dom}' pack "
            "(Tier-0 data: written as a diff, landed only if evals + tests pass).",
        )
    if t == "low_confidence":
        return (
            f"Draft a prompt revision for persistent low confidence in '{dom}'",
            f"{n} '{dom}' runs finished at low confidence, suggesting the "
            "interpretation prompt under-specifies what evidence warrants a "
            "higher rating.",
            f"Draft (do not activate) a '{dom}' prompt version bump clarifying "
            "confidence criteria; a human edits and commits.",
        )
    if t == "interpretation_unavailable":
        return (
            f"Draft a prompt revision: interpretation missing in '{dom}'",
            f"The model contributed but produced no interpretation in {n} '{dom}' "
            "runs, suggesting the prompt does not reliably elicit an "
            "interpretation section.",
            f"Draft (do not activate) a '{dom}' prompt version bump that requires "
            "an interpretation section; a human edits and commits.",
        )
    if t == "provenance_gap":
        return (
            f"Investigate recurring provenance gaps ({dom})",
            f"{n} runs recorded incomplete provenance. Full attributability is a "
            "core-code invariant (D4); an attributable-but-incomplete report "
            "should never be emitted.",
            "Open a task with the evidence episodes. Tier-2: the system does not "
            "edit core code, schemas, or scientific logic.",
        )
    if t == "error_outcome":
        return (
            f"Investigate recurring '{key}' errors in '{dom}'",
            f"The '{key}' diagnostic errored in {n} runs — a recurring code or "
            "data-handling defect rather than a one-off.",
            "Open a task with the evidence episodes. Tier-2: code changes are "
            "never auto-written.",
        )
    # empty_result
    return (
        f"Investigate empty results in '{dom}'",
        f"{n} '{dom}' runs produced neither findings nor figures — a likely "
        "coverage gap in the domain diagnostics.",
        "Open a task with the evidence episodes. Tier-2: the system does not "
        "edit scientific logic.",
    )


def _templated_proposal(cluster: SignalCluster) -> Proposal:
    spec = _SPECS[cluster.type]
    title, rationale, action = _templated_text(cluster)
    return Proposal(
        id=_slug("p", str(spec["tier"]), cluster.type, cluster.domain, cluster.key),
        tier=spec["tier"],
        change_class=spec["change_class"],
        signal_type=cluster.type,
        domain=cluster.domain,
        key=cluster.key,
        title=title,
        rationale=rationale,
        suggested_action=action,
        evidence_episodes=list(cluster.episodes),
        evidence_count=cluster.count,
    )


def _correction_proposal(cluster: SignalCluster, corrections: list[Correction]) -> Proposal:
    """A correction proposal: Tier-0 regression eval, or Tier-1 router rule for
    a mis-route (``target == "domain"``)."""
    dom = cluster.domain or "auto"
    statements = "; ".join(c.statement for c in corrections) or cluster.key
    values = [c.corrected_value for c in corrections if c.corrected_value]
    corrected_value = values[0] if values else ""

    if cluster.key == "domain":  # a routing correction
        tier, change_class = 1, "router_rule"
        title = f"Draft a router rule from a human mis-route correction ({dom})"
        action = (
            "Draft (do not activate) a router rule reflecting the corrected "
            "routing; a human edits and commits.")
    else:
        tier, change_class = 0, "regression_eval"
        title = f"Pin a regression eval for '{cluster.key}' ({dom})"
        value_txt = f" asserting {cluster.key} = {corrected_value}" if corrected_value else ""
        action = (
            f"Add a pinned regression case{value_txt} from this correction "
            "(Tier-0 data: written as a diff, landed only if the new eval + the "
            "existing suite pass).")

    return Proposal(
        id=_slug("p", str(tier), "correction", cluster.domain, cluster.key),
        tier=tier,
        change_class=change_class,
        signal_type="correction",
        domain=cluster.domain,
        key=cluster.key,
        title=title,
        rationale=f"Human correction ({len(corrections)}): {statements}",
        suggested_action=action,
        corrected_value=corrected_value,
        evidence_episodes=[c.episode_id for c in corrections if c.episode_id],
        evidence_count=len(corrections),
    )


def build_proposals(journal, min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
                    polish: Callable | None = None) -> list[Proposal]:
    """Generate reviewable proposals from a journal's signals + corrections.

    ``polish`` is an optional ``(Proposal) -> (title, rationale)`` callable
    (see :func:`polish_prose`) applied to prose only.
    """
    episodes = journal.episodes()
    corrections = journal.corrections()

    signals = [s for ep in episodes for s in signals_for_episode(ep)]
    signals += [signal_from_correction(c) for c in corrections]
    clusters = cluster_signals(signals)

    corr_by_key: dict[tuple[str, str], list[Correction]] = {}
    for c in corrections:
        corr_by_key.setdefault((c.domain, c.target), []).append(c)

    proposals: list[Proposal] = []
    for cluster in clusters:
        if cluster.type == "correction":
            corrs = corr_by_key.get((cluster.domain, cluster.key), [])
            proposals.append(_correction_proposal(cluster, corrs))
        elif cluster.type in _SPECS and cluster.count >= min_occurrences:
            proposals.append(_templated_proposal(cluster))

    # Proposals are ordered by tier (Tier-0 first — the cheapest, safest wins)
    # then by evidence weight.
    proposals.sort(key=lambda p: (p.tier, -p.evidence_count, p.id))

    if polish is not None:
        for p in proposals:
            _apply_polish(p, polish)
    return proposals


def polish_prose(llm) -> Callable[[Proposal], tuple[str, str]]:
    """Build a prose-polish callable backed by an :class:`LLMClient`.

    The returned callable asks the model to rewrite a proposal's title and
    rationale as JSON. It returns ``(title, rationale)`` strings; the caller
    re-applies only those two fields, so the model can never alter evidence or
    suggested values. Any failure falls back to the original prose.
    """
    from scattering_ai.llm.base import Message

    def _polish(p: Proposal) -> tuple[str, str]:
        prompt = (
            "Rewrite the following improvement-proposal title and rationale to be "
            "clear and concise for a scientist. Do NOT invent facts, numbers, or "
            "actions; only rephrase. Return JSON: "
            '{"title": "...", "rationale": "..."}.\n\n'
            f"Title: {p.title}\nRationale: {p.rationale}"
        )
        resp = llm.complete([Message(role="user", content=prompt)])
        data = json.loads(resp.content)
        return str(data["title"]), str(data["rationale"])

    return _polish


def _apply_polish(p: Proposal, polish: Callable[[Proposal], tuple[str, str]]) -> None:
    """Re-apply prose from ``polish`` to a proposal, prose fields only.

    Best-effort and strictly bounded: even a misbehaving polish function can
    only change ``title`` and ``rationale``; everything the apply step relies on
    is untouched.
    """
    try:
        title, rationale = polish(p)
    except Exception:
        return
    if title and rationale:
        p.title = str(title)
        p.rationale = str(rationale)
        p.prose_polished = True


_TIER_LABELS = {
    0: "Tier 0 — data (diff → eval-gated → human commits)",
    1: "Tier 1 — draft only (human edits & commits)",
    2: "Tier 2 — never touched (task hand-off with evidence)",
}


def render_proposals(proposals: list[Proposal],
                     title: str = "Self-improvement proposals") -> str:
    """Render proposals as reviewable markdown, grouped by tier."""
    lines = [f"# {title}", ""]
    if not proposals:
        lines.append("No proposals: no correction and no signal has recurred "
                     "past the threshold yet.")
        return "\n".join(lines) + "\n"

    lines.append(f"{len(proposals)} proposal(s), lowest-risk tier first. "
                 "Nothing here is applied — review and approve explicitly.")
    lines.append("")
    for tier in (0, 1, 2):
        group = [p for p in proposals if p.tier == tier]
        if not group:
            continue
        lines += [f"## {_TIER_LABELS[tier]}", ""]
        for p in group:
            lines += [
                f"### {p.title}",
                "",
                f"- **id:** `{p.id}`",
                f"- **change class:** {p.change_class}",
                f"- **signal:** {p.signal_type}"
                + (f" · domain `{p.domain}`" if p.domain else "")
                + (f" · key `{p.key}`" if p.key else ""),
                f"- **evidence:** {p.evidence_count} occurrence(s)"
                + (f", episodes {', '.join(p.evidence_episodes[:10])}"
                   if p.evidence_episodes else ""),
            ]
            if p.corrected_value:
                lines.append(f"- **corrected value:** `{p.corrected_value}`")
            lines += [
                "",
                f"**Rationale.** {p.rationale}",
                "",
                f"**Suggested action.** {p.suggested_action}",
                "",
            ]
    return "\n".join(lines).rstrip() + "\n"
