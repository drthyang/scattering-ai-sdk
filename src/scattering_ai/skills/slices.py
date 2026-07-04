"""Skills for 2D slices (first-look characterization)."""

from __future__ import annotations

from typing import Any

from scattering_ai.skills.base import SkillRun
from scattering_ai.tools.registry import AgentTool, ToolRegistry, _params

CATEGORY = "2D slices"


def characterize_slice(
    registry: ToolRegistry,
    path: str,
    axis: int | None = None,
    center: float | None = None,
    thickness: float | None = None,
    slice_path: str = "",
) -> dict[str, Any]:
    """Full first-look at a 2D slice: cut (if needed) → peaks → rings → plot."""
    run = SkillRun(registry)
    if not slice_path:
        if axis is None or center is None or thickness is None:
            return {"error": "provide either slice_path or path+axis+center+thickness"}
        cut = run.call("slice_volume", path=path, axis=axis, center=center,
                       thickness=thickness)
        if "error" in cut:
            return run.finish(error=cut["error"])
        slice_path = cut["saved"]

    peaks = run.call("find_peaks_2d", slice_path=slice_path)
    rings = run.call("detect_rings_2d", slice_path=slice_path)
    plot = run.call("plot_slice_2d", slice_path=slice_path, log=True, mark_peaks=True)

    contaminated = rings.get("phase_match_counts", {})
    return run.finish(
        slice_path=slice_path,
        n_bragg_peaks=peaks.get("n_peaks"),
        strongest_peaks=peaks.get("strongest", [])[:10],
        ring_candidates=rings.get("rings", [])[:10],
        contaminant_matches=contaminated,
        plots={"slice": plot.get("saved"), "ring_profile": rings.get("plot")},
        assessment={
            "has_contaminant_rings": bool(contaminated),
            "note": "ring candidates with low completeness are weak evidence; "
            "confirm in the most isotropic plane",
        },
    )


def skills(registry: ToolRegistry) -> list[AgentTool]:
    opt_number = {"type": ["number", "null"]}
    return [AgentTool(
        "skill_characterize_slice",
        "SKILL (composite workflow): full first-look at a 2D slice — cut from "
        "a volume (or use an existing slice_path), detect Bragg peaks, hunt "
        "contaminant rings, render a log-scale plot. Prefer this over manual "
        "chaining for 'what is in this slice?' questions.",
        _params(
            {"path": {"type": "string"}, "axis": {"type": ["integer", "null"]},
             "center": opt_number, "thickness": opt_number,
             "slice_path": {"type": "string"}},
            ["path"],
        ),
        lambda **kw: characterize_slice(registry, **kw),
        category=CATEGORY,
    )]
