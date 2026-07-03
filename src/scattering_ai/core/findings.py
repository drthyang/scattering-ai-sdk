"""Diagnostic finding model shared by all domain packs.

Findings are produced deterministically, before any LLM reasoning. The
``evidence`` dict must contain the numbers a finding is based on, so the
report layer can cite them and the LLM never has to (re)compute anything.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Finding(BaseModel):
    diagnostic: str
    severity: Severity = Severity.INFO
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)
