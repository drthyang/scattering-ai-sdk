"""Typed RMC run state (the rmc domain's slice of AnalysisData).

Applications can either populate ``AnalysisData.metadata["rmc"]`` with this
structure, or send plain ``r_values`` which are treated as a single unlabeled
series. ``RMCRunState.from_analysis_data`` handles both.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from scattering_ai.core.schemas import AnalysisData


class RMCSeries(BaseModel):
    """One R-value (or chi^2) history for one dataset."""

    label: str
    kind: Literal["bragg", "pdf", "total", "other"] = "other"
    radiation: Literal["neutron", "xray", "other"] = "other"
    values: list[float]


class RMCRunState(BaseModel):
    series: list[RMCSeries] = Field(default_factory=list)
    expected_files: list[str] = Field(default_factory=list)
    present_files: list[str] = Field(default_factory=list)
    log_tail: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_analysis_data(cls, data: AnalysisData) -> RMCRunState:
        rmc = data.metadata.get("rmc")
        if rmc:
            state = cls.model_validate(rmc)
        else:
            state = cls()
        if not state.series and data.r_values:
            state.series = [RMCSeries(label="overall", kind="total", values=data.r_values)]
        if not state.present_files and data.files:
            state.present_files = list(data.files)
        return state
