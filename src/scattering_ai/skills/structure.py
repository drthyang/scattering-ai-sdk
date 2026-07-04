"""Skills for crystal structures (CIF / mCIF visualization)."""

from __future__ import annotations

from typing import Any

from scattering_ai.skills.base import SkillRun
from scattering_ai.tools.registry import AgentTool, ToolRegistry, _params

CATEGORY = "structure"


def visualize_structure(registry: ToolRegistry, path: str, bonds: bool = True) -> dict[str, Any]:
    """Render a CIF/mCIF: describe the structure (formula, magnetism) and draw
    the unit cell with atoms, bonds, and magnetic moments."""
    run = SkillRun(registry)
    plotted = run.call("plot_structure", path=path, bonds=bonds)
    if "error" in plotted:
        return run.finish(error=plotted["error"])
    return run.finish(
        n_atoms=plotted.get("n_atoms"),
        n_magnetic=plotted.get("n_magnetic"),
        species=plotted.get("species"),
        space_group_cif=plotted.get("space_group_cif"),
        plots={"structure": plotted.get("saved")},
        assessment={
            "is_magnetic": bool(plotted.get("n_magnetic")),
            "note": "quick embedded view; use VESTA/Vesta-like tools for "
            "publication figures and detailed coordination analysis",
        },
    )


def skills(registry: ToolRegistry) -> list[AgentTool]:
    return [AgentTool(
        "skill_visualize_structure",
        "SKILL (composite workflow): render a crystal structure from a CIF or "
        "mCIF — the unit cell with element-coloured atoms, bonds, and (for a "
        "magnetic structure) the moment arrows — and summarize it. Prefer this "
        "for 'show/draw/visualize this structure' questions.",
        _params(
            {"path": {"type": "string"}, "bonds": {"type": "boolean"}},
            ["path"],
        ),
        lambda **kw: visualize_structure(registry, **kw),
        category=CATEGORY,
    )]
