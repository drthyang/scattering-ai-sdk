"""Skills for 3D volumes (diffuse scattering / 3D-ΔPDF)."""

from __future__ import annotations

from typing import Any

from scattering_ai.skills.base import SkillRun
from scattering_ai.tools.registry import AgentTool, ToolRegistry, _params

CATEGORY = "3D volumes"


def delta_pdf_workflow(registry: ToolRegistry, path: str, apodization: str = "hann",
                       punch_sigma: float = 6.0) -> dict[str, Any]:
    """Compute the 3D-ΔPDF of a diffuse volume and pull out its correlations:
    punch Bragg → apodize → centred FFT, then find the strongest positive
    correlation peaks in the central plane."""
    run = SkillRun(registry)
    dp = run.call("delta_pdf", path=path, apodization=apodization, punch_sigma=punch_sigma)
    if "error" in dp:
        return run.finish(error=dp["error"])
    peaks = run.call("find_peaks_2d", slice_path=dp["saved_slice"])
    correlations = peaks.get("strongest", [])[:8]
    summary = (
        f"3D-ΔPDF computed ({apodization} apodization). The central plane has "
        f"{len(correlations)} positive correlation maxima; ΔPDF ranges "
        f"{dp['max_negative']:g} … {dp['max_positive']:g} (positive = pair "
        "distances more common than the average structure).")
    return run.finish(
        summary=summary,
        correlations=correlations,
        extremes={"max_positive": dp["max_positive"], "max_negative": dp["max_negative"]},
        plots={"delta_pdf": dp.get("plot")},
        assessment={
            "note": "low-|Δr| features near the centre can be Bragg-subtraction "
            "residue; confirm against the raw diffuse volume",
        },
    )


def skills(registry: ToolRegistry) -> list[AgentTool]:
    return [AgentTool(
        "skill_delta_pdf",
        "SKILL (composite workflow): compute the 3D difference PDF (3D-ΔPDF) of a "
        "diffuse-scattering volume (.nxs) — Bragg punch, apodize, centred FFT — "
        "and report the strongest real-space correlations with a central-plane "
        "plot. Prefer this for '3D-ΔPDF' / 'real-space correlations' questions.",
        _params(
            {"path": {"type": "string"}, "apodization": {"type": "string"},
             "punch_sigma": {"type": "number"}},
            ["path"],
        ),
        lambda **kw: delta_pdf_workflow(registry, **kw),
        category=CATEGORY,
    )]
