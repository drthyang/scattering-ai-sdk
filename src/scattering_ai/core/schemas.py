"""Request/report envelope schemas (schema_version 1).

These Pydantic models are the single source of truth for the SDK's public
data contract. Within a major schema version, changes must be additive only
(new optional fields). Domain packs extend ``AnalysisData`` with typed
sub-schemas; the envelope stays stable.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1"


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AnalysisOptions(BaseModel):
    use_rag: bool = True
    use_tools: bool = True
    report_format: str = "markdown"


class AnalysisData(BaseModel):
    """Generic scientific state container.

    Domain packs define typed models for their slice of this data; the
    envelope accepts anything JSON-serializable so applications can send
    state before a typed schema exists for it.
    """

    run_summary: dict[str, Any] = Field(default_factory=dict)
    r_values: list[float] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    plots: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalysisRequest(BaseModel):
    schema_version: str = SCHEMA_VERSION
    project: str = ""
    domain: str = "auto"  # "auto" routes to the best pack from the input
    question: str
    data: AnalysisData = Field(default_factory=AnalysisData)
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)


class Citation(BaseModel):
    source: str
    section: str = ""


class ToolCallRecord(BaseModel):
    tool: str
    args_hash: str = ""


class ReportValidationError(ValueError):
    """Raised when a report fails output validation (roadmap D4).

    Reproducibility is the SDK's scientific-credibility claim, so a report the
    pipeline cannot fully attribute is rejected rather than returned.
    """


# Always fillable by the pipeline and the core of reproducibility.
REQUIRED_PROVENANCE_FIELDS = ("sdk_version", "schema_version", "input_hash", "timestamp")


class Provenance(BaseModel):
    """Reproducibility record embedded in every report.

    Output validation (roadmap D4) rejects reports whose provenance is
    incomplete: the fields in ``REQUIRED_PROVENANCE_FIELDS`` are always
    mandatory, and ``model``/``prompt_version`` become mandatory once an LLM
    contributes interpretation (a report claiming model reasoning it cannot
    attribute is not reproducible).
    """

    sdk_version: str = ""
    schema_version: str = SCHEMA_VERSION
    input_hash: str = ""
    model: str = ""
    prompt_version: str = ""
    retrieved_chunks: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    timestamp: str = ""

    def missing_fields(self, requires_model: bool = False) -> list[str]:
        """Provenance fields that are absent or malformed (empty = complete)."""
        missing = {f for f in REQUIRED_PROVENANCE_FIELDS if not getattr(self, f)}
        if self.input_hash and not self.input_hash.startswith("sha256:"):
            missing.add("input_hash")  # present but not a recognizable hash
        if requires_model:
            missing.update(f for f in ("model", "prompt_version") if not getattr(self, f))
        return sorted(missing)


class AnalysisReport(BaseModel):
    schema_version: str = SCHEMA_VERSION
    status: str = "ok"
    domain: str = ""  # the pack that handled this (resolved from "auto")
    summary: str = ""
    observations: list[str] = Field(default_factory=list)
    interpretation: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    recommended_next_checks: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    used_tools: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW
    provenance: Provenance = Field(default_factory=Provenance)

    def assert_valid(self, requires_model: bool = False) -> AnalysisReport:
        """Reject the report if its provenance is incomplete (roadmap D4).

        ``requires_model`` is set when an LLM contributed interpretation, so
        the model identity and prompt version must be recorded. Returns self
        when valid, to allow ``return report.assert_valid(...)``.
        """
        missing = self.provenance.missing_fields(requires_model)
        if missing:
            raise ReportValidationError(
                "report rejected: incomplete provenance, missing/invalid "
                f"{missing}. Every report must be fully attributable."
            )
        return self

    @property
    def markdown(self) -> str:
        from scattering_ai.reports.markdown import render

        return render(self)
