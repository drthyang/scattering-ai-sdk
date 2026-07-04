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


def symmetry_overview(registry: ToolRegistry, path: str, max_index: int = 6) -> dict[str, Any]:
    """Full symmetry picture of a structure in one call: space group + Wyckoff,
    maximal subgroups (transition pathways), pseudosymmetric parent, and the
    reflection conditions."""
    run = SkillRun(registry)
    fs = run.call("find_symmetry", path=path)
    if "error" in fs:
        return run.finish(error=fs["error"])
    tree = run.call("subgroup_tree", path=path)
    pseudo = run.call("pseudosymmetry_scan", path=path)
    absences = run.call("systematic_absences", path=path, max_index=max_index)

    subgroups = tree.get("subgroups", []) if "error" not in tree else []
    sub_labels = ", ".join(
        f"{s['international']} (i{s['index']}"
        + (f"×{s['n_variants']}" if s.get("n_variants", 1) > 1 else "") + ")"
        for s in subgroups)
    parent = pseudo.get("highest_symmetry") if "error" not in pseudo else None
    pseudo_note = (
        f" Pseudosymmetric parent at looser tolerance: {parent['international']} "
        f"(#{parent['number']})." if pseudo.get("pseudosymmetric") and parent else "")
    summary = (
        f"{fs['international']} (#{fs['number']}, {fs['crystal_system']}). "
        f"{len(subgroups)} maximal subgroup(s) — transition pathways: "
        f"{sub_labels or 'none'}.{pseudo_note}")
    return run.finish(
        summary=summary,
        space_group={k: fs[k] for k in ("number", "international", "crystal_system",
                                        "point_group", "wyckoff_sites")},
        maximal_subgroups=subgroups,
        pseudosymmetric=bool(pseudo.get("pseudosymmetric")),
        pseudosymmetric_parent=parent,
        centering=absences.get("centering") if "error" not in absences else None,
        n_allowed_reflections=absences.get("n_allowed_unique") if "error" not in absences else None,
    )


def skills(registry: ToolRegistry) -> list[AgentTool]:
    string = {"type": "string"}
    return [
        AgentTool(
            "skill_visualize_structure",
            "SKILL (composite workflow): render a crystal structure from a CIF or "
            "mCIF — the unit cell with element-coloured atoms, bonds, and (for a "
            "magnetic structure) the moment arrows — and summarize it. Prefer this "
            "for 'show/draw/visualize this structure' questions.",
            _params({"path": string, "bonds": {"type": "boolean"}}, ["path"]),
            lambda **kw: visualize_structure(registry, **kw),
            category=CATEGORY,
        ),
        AgentTool(
            "skill_symmetry_overview",
            "SKILL (composite workflow): the full symmetry picture of a structure "
            "(CIF) in one call — space group + Wyckoff sites, maximal subgroups "
            "(phase-transition pathways, with a tree figure), pseudosymmetric "
            "parent, and reflection conditions. Prefer this for 'what is the "
            "symmetry / what transitions are possible?' questions.",
            _params({"path": string, "max_index": {"type": "integer"}}, ["path"]),
            lambda **kw: symmetry_overview(registry, **kw),
            category=CATEGORY,
        ),
    ]
