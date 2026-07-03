"""Markdown renderer for analysis reports.

Follows the roadmap report structure: Conclusion, Evidence, Interpretation,
Warnings, Recommended next checks, Knowledge used, Confidence, Provenance.
"""

from __future__ import annotations

from scattering_ai.core.schemas import AnalysisReport


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- None."


def render(report: AnalysisReport, title: str = "Analysis Report") -> str:
    citations = [
        f"{c.source}#{c.section}" if c.section else c.source for c in report.citations
    ]
    provenance = report.provenance
    lines = [
        f"# {title}",
        "",
        "## Conclusion",
        "",
        report.summary or "No conclusion available.",
        "",
        "## Evidence (observations)",
        "",
        _bullets(report.observations),
        "",
        "## Scientific interpretation",
        "",
        _bullets(report.interpretation),
        "",
        "## Warnings",
        "",
        _bullets(report.warnings),
        "",
        "## Recommended next checks",
        "",
        _bullets(report.recommended_next_checks),
        "",
        "## Knowledge used",
        "",
        _bullets(citations),
        "",
        f"**Confidence:** {report.confidence.value}",
        "",
        "## Provenance",
        "",
        f"- SDK version: {provenance.sdk_version or 'n/a'}",
        f"- Model: {provenance.model or 'none (deterministic-only mode)'}",
        f"- Prompt version: {provenance.prompt_version or 'n/a'}",
        f"- Input hash: {provenance.input_hash or 'n/a'}",
        f"- Generated: {provenance.timestamp or 'n/a'}",
        "",
    ]
    return "\n".join(lines)
