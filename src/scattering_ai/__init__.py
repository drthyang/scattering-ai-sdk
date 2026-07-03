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

__version__ = "0.1.0"

from scattering_ai.core.agent import Agent, analyze  # noqa: E402  (needs __version__)

__all__ = [
    "Agent",
    "AnalysisReport",
    "AnalysisRequest",
    "Citation",
    "Confidence",
    "Provenance",
    "analyze",
    "__version__",
]
