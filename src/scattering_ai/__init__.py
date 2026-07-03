"""scattering-ai-sdk: AI SDK for scattering science.

Only symbols exported here are covered by semver; submodules may change
between minor versions until 1.0 (see ROADMAP.md, "Extensibility Architecture").
"""

from scattering_ai.core.schemas import (
    AnalysisReport,
    AnalysisRequest,
    Citation,
    Confidence,
    Provenance,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "AnalysisReport",
    "AnalysisRequest",
    "Citation",
    "Confidence",
    "Provenance",
    "__version__",
]
