"""Human-reviewed self-improvement — Phase 2: signals + corrections.

Signals are typed observations derived **deterministically** from the redacted
episode journal (P1) — no LLM, no data values, only category labels and counts.
They are the bridge from raw capture to reviewable proposals (P3): a signal that
recurs across many episodes is exactly the evidence a proposal will cite.

Corrections are the highest-value signal and the one thing the system will never
infer. A **human** states that a past result was wrong (and optionally the right
value); the correction is recorded verbatim, references the episode it corrects,
and later seeds a pinned regression eval (P5). *Example:* "GaNb4Se8 transitions
are 50 K and 29 K, not 39 K."

Everything here is read-only w.r.t. the SDK: it observes and clusters. Nothing
in this module edits code, prompts, schemas, or scientific logic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from scattering_ai.learning.journal import Episode

# Ranking weight per severity — clusters are ordered by weight × recurrence so
# the review surface (P3) sees the highest-value, most-recurrent patterns first.
_WEIGHT = {"high": 3, "medium": 2, "low": 1}


class Signal(BaseModel):
    """One typed observation about a single episode (or correction).

    ``key`` is the clustering key *within* a type — a diagnostic name, a
    correction target, or the domain — so recurrence is counted per pattern
    rather than lumping every signal of a type together.
    """

    type: str
    severity: str = "medium"  # high | medium | low
    episode_id: str = ""
    domain: str = ""
    key: str = ""


class Correction(BaseModel):
    """A human-stated correction of a past result. Never inferred."""

    id: str
    timestamp: str
    episode_id: str = ""
    domain: str = ""
    target: str  # what was wrong: "domain", "transition_temperature", ...
    statement: str  # plain English: "transitions are 50 K and 29 K, not 39 K"
    corrected_value: str = ""  # optional machine-usable value, as a string


def make_correction(episode_id: str, target: str, statement: str,
                    corrected_value: str = "", domain: str = "") -> Correction:
    """Construct a :class:`Correction` with an id/timestamp filled in."""
    stamp = datetime.now(timezone.utc)
    return Correction(
        id=(episode_id or "manual") + "-" + stamp.strftime("%H%M%S%f")[:9],
        timestamp=stamp.isoformat(timespec="seconds"),
        episode_id=episode_id,
        domain=domain,
        target=target,
        statement=statement,
        corrected_value=corrected_value,
    )


def signals_for_episode(ep: Episode) -> list[Signal]:
    """Derive every deterministic signal an episode carries (may be empty)."""
    out: list[Signal] = []

    def add(kind: str, severity: str, key: str) -> None:
        out.append(Signal(type=kind, severity=severity, episode_id=ep.id,
                          domain=ep.domain, key=key))

    errors = [f for f in ep.findings if f.severity == "error"]
    warnings = [f for f in ep.findings if f.severity == "warning"]

    # An error outcome, keyed by the diagnostic that raised it so recurring
    # failure modes cluster; fall back to a placeholder when unattributed.
    if errors:
        for f in errors:
            add("error_outcome", "high", f.diagnostic)
    elif ep.outcome == "error":
        add("error_outcome", "high", "unattributed")

    for f in warnings:
        add("unhandled_warning", "medium", f.diagnostic)

    if not ep.provenance_complete:
        add("provenance_gap", "high", "provenance")

    domain_key = ep.domain or "unknown"
    # Confidence/interpretation signals only make sense once an LLM contributed;
    # a purely deterministic run has no model and legitimately no interpretation.
    if ep.model and ep.confidence == "low":
        add("low_confidence", "low", domain_key)
    if ep.model and not ep.interpretation_available:
        add("interpretation_unavailable", "medium", domain_key)

    # "Empty result" is an analysis concept — a diagnostics run that produced
    # nothing. A chat turn legitimately has no findings/figures, so don't fire.
    if ep.surface == "analyze" and not ep.findings and ep.n_figures == 0:
        add("empty_result", "medium", domain_key)

    return out


def signal_from_correction(c: Correction) -> Signal:
    """A correction is itself a high-value signal, keyed by its target."""
    return Signal(type="correction", severity="high", episode_id=c.episode_id,
                  domain=c.domain, key=c.target)


class SignalCluster(BaseModel):
    """Recurring signals of one (type, domain, key), ready for review (P3)."""

    type: str
    domain: str = ""
    key: str = ""
    severity: str = "medium"
    count: int = 0
    episodes: list[str] = Field(default_factory=list)


def cluster_signals(signals: list[Signal], max_episodes: int = 10) -> list[SignalCluster]:
    """Group signals by (type, domain, key), ranked by weight × recurrence."""
    groups: dict[tuple[str, str, str], SignalCluster] = {}
    for s in signals:
        gk = (s.type, s.domain, s.key)
        g = groups.get(gk)
        if g is None:
            g = SignalCluster(type=s.type, domain=s.domain, key=s.key,
                              severity=s.severity)
            groups[gk] = g
        g.count += 1
        if s.episode_id and s.episode_id not in g.episodes:
            g.episodes.append(s.episode_id)

    clusters = list(groups.values())
    clusters.sort(key=lambda g: (-_WEIGHT.get(g.severity, 0) * g.count, g.type, g.key))
    for g in clusters:
        del g.episodes[max_episodes:]
    return clusters


def signals_report(journal) -> dict[str, Any]:
    """The ``learn signals`` view: every signal, clustered, plus totals.

    This is the P2→P3 handoff — a proposal is generated from a cluster whose
    recurrence clears a threshold.
    """
    episodes = journal.episodes()
    corrections = journal.corrections()

    signals = [s for ep in episodes for s in signals_for_episode(ep)]
    signals += [signal_from_correction(c) for c in corrections]
    clusters = cluster_signals(signals)

    by_type: dict[str, int] = {}
    for s in signals:
        by_type[s.type] = by_type.get(s.type, 0) + 1

    return {
        "n_episodes": len(episodes),
        "n_corrections": len(corrections),
        "n_signals": len(signals),
        "by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
        "clusters": [c.model_dump() for c in clusters],
    }
