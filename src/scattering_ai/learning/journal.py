"""Human-reviewed self-improvement — Phase 1: the episode journal.

An append-only, **local**, **redacted** record of what happened in each analysis
run, so recurring failure modes, warnings, and low-confidence outcomes can later
be reviewed and turned into improvement proposals (Phase 3+).

Safety, by construction:
- **Read-only w.r.t. the SDK.** This subsystem observes; it never edits code,
  prompts, schemas, or scientific logic. Later phases add a human-approved,
  eval-gated, diff-only apply step for a small tier of *data* changes only.
- **Redacted.** An episode stores category labels (domain, diagnostic names,
  severities), identifiers (input hash, file *basenames*), and outcomes — never
  data arrays, evidence values, cell parameters, or message text.
- **Local-first.** Nothing is transmitted anywhere; the journal is a plain
  JSONL file in a directory you choose.
- **Opt-in.** Journaling only happens when a journal directory is configured
  (``Agent(journal=...)`` or the ``SCATTERING_AI_JOURNAL`` env var); the default
  is off, and a journaling failure never affects the analysis.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

JOURNAL_ENV = "SCATTERING_AI_JOURNAL"
_FILE = "episodes.jsonl"
_CORR_FILE = "corrections.jsonl"


class ToolEvent(BaseModel):
    name: str


class FindingTag(BaseModel):
    """A finding reduced to safe category labels — no evidence values."""

    diagnostic: str
    severity: str


class Episode(BaseModel):
    """One analysis run, redacted to category labels + identifiers."""

    id: str
    timestamp: str
    surface: str  # "analyze" | "chat"
    sdk_version: str = ""
    domain: str = ""
    model: str = ""
    input_hash: str = ""
    n_files: int = 0
    file_names: list[str] = Field(default_factory=list)  # basenames only
    tools: list[ToolEvent] = Field(default_factory=list)
    findings: list[FindingTag] = Field(default_factory=list)
    confidence: str = ""
    provenance_complete: bool = True
    interpretation_available: bool = True  # LLM produced interpretation prose
    n_figures: int = 0
    outcome: str = "ok"  # "ok" | "warnings" | "error"
    duration_s: float | None = None


class Journal:
    """Append-only JSONL episode log in a local directory."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / _FILE
        self.corr_path = self.dir / _CORR_FILE

    def record(self, episode: Episode) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(episode.model_dump_json() + "\n")

    def episodes(self) -> list[Episode]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(Episode.model_validate_json(line))
        return out

    def record_correction(self, correction) -> None:
        """Append a human-stated correction (P2).

        Unlike an episode, a correction is a *deliberate human act*, not passive
        capture — so it is stored verbatim (it carries the corrected value on
        purpose) in a separate ``corrections.jsonl``. Still local-first; nothing
        is transmitted.
        """
        with self.corr_path.open("a", encoding="utf-8") as fh:
            fh.write(correction.model_dump_json() + "\n")

    def corrections(self) -> list:
        from scattering_ai.learning.signals import Correction

        if not self.corr_path.exists():
            return []
        out = []
        for line in self.corr_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(Correction.model_validate_json(line))
        return out

    def summary(self) -> dict[str, Any]:
        """Aggregate view of the journal — the basis for the ``learn status``
        report and, later, signal clustering."""
        eps = self.episodes()
        if not eps:
            return {"n_episodes": 0}

        def counter(items) -> dict[str, int]:
            out: dict[str, int] = {}
            for it in items:
                out[it] = out.get(it, 0) + 1
            return dict(sorted(out.items(), key=lambda kv: -kv[1]))

        warnings = [f for e in eps for f in e.findings if f.severity == "warning"]
        errors = [f for e in eps for f in e.findings if f.severity == "error"]
        durations = [e.duration_s for e in eps if e.duration_s is not None]
        return {
            "n_episodes": len(eps),
            "by_surface": counter(e.surface for e in eps),
            "by_domain": counter(e.domain for e in eps if e.domain),
            "by_outcome": counter(e.outcome for e in eps),
            "n_provenance_incomplete": sum(1 for e in eps if not e.provenance_complete),
            "n_low_confidence": sum(1 for e in eps if e.confidence == "low" and e.model),
            "top_warnings": dict(list(counter(f.diagnostic for f in warnings).items())[:8]),
            "top_errors": dict(list(counter(f.diagnostic for f in errors).items())[:8]),
            "top_tools": dict(list(counter(t.name for e in eps for t in e.tools).items())[:10]),
            "avg_duration_s": round(sum(durations) / len(durations), 2) if durations else None,
        }


def resolve_journal_dir(explicit: str | Path | None) -> Path | None:
    """The configured journal directory, or None when journaling is off."""
    import os

    directory = explicit or os.environ.get(JOURNAL_ENV)
    return Path(directory) if directory else None


def episode_from_analysis(report, findings, files, surface: str = "analyze",
                          duration: float | None = None) -> Episode:
    """Build a redacted episode from a completed analysis.

    ``findings`` are the deterministic diagnostic findings (tagged to
    diagnostic + severity only); ``report`` supplies the resolved domain,
    provenance, confidence and figure count.
    """
    prov = report.provenance
    if any(f.severity.value == "error" for f in findings):
        outcome = "error"
    elif report.warnings:
        outcome = "warnings"
    else:
        outcome = "ok"
    return Episode(
        id=(prov.input_hash.replace("sha256:", "")[:12] or "nohash")
        + "-" + datetime.now(timezone.utc).strftime("%H%M%S%f")[:9],
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        surface=surface,
        sdk_version=prov.sdk_version,
        domain=report.domain,
        model=prov.model,
        input_hash=prov.input_hash,
        n_files=len(files or []),
        file_names=[Path(f).name for f in (files or [])][:20],
        tools=[ToolEvent(name=t) for t in report.used_tools],
        findings=[FindingTag(diagnostic=f.diagnostic, severity=f.severity.value)
                  for f in findings],
        confidence=report.confidence.value,
        provenance_complete=(prov.missing_fields(requires_model=bool(prov.model)) == []),
        interpretation_available=bool(report.interpretation),
        n_figures=len(report.figures),
        outcome=outcome,
        duration_s=round(duration, 3) if duration is not None else None,
    )
