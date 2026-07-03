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
    domain: str
    question: str
    data: AnalysisData = Field(default_factory=AnalysisData)
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)


class Citation(BaseModel):
    source: str
    section: str = ""


class ToolCallRecord(BaseModel):
    tool: str
    args_hash: str = ""


class Provenance(BaseModel):
    """Reproducibility record embedded in every report.

    Reports without full provenance will be rejected by output validation
    once enforcement lands (roadmap D4); until then fields default to empty
    so partial pipelines can still emit reports during development.
    """

    sdk_version: str = ""
    schema_version: str = SCHEMA_VERSION
    input_hash: str = ""
    model: str = ""
    prompt_version: str = ""
    retrieved_chunks: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    timestamp: str = ""


class AnalysisReport(BaseModel):
    schema_version: str = SCHEMA_VERSION
    status: str = "ok"
    summary: str = ""
    observations: list[str] = Field(default_factory=list)
    interpretation: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    recommended_next_checks: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    used_tools: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW
    provenance: Provenance = Field(default_factory=Provenance)

    @property
    def markdown(self) -> str:
        from scattering_ai.reports.markdown import render

        return render(self)
