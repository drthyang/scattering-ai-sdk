"""Built-in agent skills — composite workflows validated on real data.

Skills are organized by category, one module each, so the set stays legible as
it grows:

- ``patterns``  — 1D pattern fitting
- ``series``    — temperature/field scans and transitions
- ``slices``    — 2D slice characterization
- ``structure`` — crystal-structure (CIF/mCIF) visualization + symmetry overview
- ``volumes``   — 3D volumes / 3D-ΔPDF
- ``magnetic``  — magnetic structures (mPDF, spin correlations, magnetic I(Q))

Each module exposes ``skills(registry) -> list[AgentTool]`` with a ``category``;
``register_skills`` collects them onto a registry as ``skill_*`` tools. Domain
packs and third parties can add their own modules the same way.
"""

from __future__ import annotations

from scattering_ai.skills import (
    magnetic,
    patterns,
    series,
    slices,
    structure,
    volumes,
)
from scattering_ai.tools.registry import ToolRegistry

_SKILL_MODULES = (patterns, series, slices, structure, volumes, magnetic)


def register_skills(registry: ToolRegistry) -> ToolRegistry:
    """Expose every built-in skill on a registry as a ``skill_*`` tool."""
    for module in _SKILL_MODULES:
        for tool in module.skills(registry):
            registry.add(tool)
    return registry
