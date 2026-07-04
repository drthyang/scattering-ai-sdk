"""Skills for 1D patterns (peak fitting)."""

from __future__ import annotations

from typing import Any

from scattering_ai.skills.base import SkillRun
from scattering_ai.tools.registry import AgentTool, ToolRegistry, _params

CATEGORY = "1D patterns"


def fit_pattern_peaks(
    registry: ToolRegistry,
    path: str,
    xmin: float | None = None,
    xmax: float | None = None,
    max_peaks: int = 6,
    fwhm_guess: float | None = None,
) -> dict[str, Any]:
    """Find the strongest peaks in a 1D pattern and fit them all at once,
    with a fit plot and residual for judgment."""
    run = SkillRun(registry)
    found = run.call("find_peaks_1d", path=path)
    if "error" in found:
        return run.finish(error=found["error"])
    peaks = [
        p for p in found.get("peaks", [])
        if (xmin is None or p["x"] >= xmin) and (xmax is None or p["x"] <= xmax)
    ][:max_peaks]
    if not peaks:
        return run.finish(error="no peaks found in the requested range")

    centers = sorted(p["x"] for p in peaks)
    fit = run.call("plot_fit_1d", path=path, centers=centers, fwhm_guess=fwhm_guess)
    if "error" in fit:
        return run.finish(error=fit["error"], detected_peaks=peaks)

    table = [
        {"center": p["center"], "center_err": p["center_err"],
         "fwhm": p["fwhm"], "fwhm_err": p["fwhm_err"], "height": p["height"]}
        for p in fit["peaks"]
    ]
    return run.finish(
        peak_table=table,
        rwp=fit["rwp"],
        reduced_chi2=fit["reduced_chi2"],
        flags=fit["flags"],
        plots={"fit": fit.get("plot")},
        assessment={
            "fit_trustworthy": not (fit["flags"]["at_bounds"] or fit["flags"]["poor_fit"]),
            "note": "inspect the residual panel for unfitted features before "
            "trusting the table",
        },
    )


def skills(registry: ToolRegistry) -> list[AgentTool]:
    opt_number = {"type": ["number", "null"]}
    return [AgentTool(
        "skill_fit_pattern_peaks",
        "SKILL (composite workflow): find the strongest peaks in a 1D pattern "
        "(optionally within [xmin, xmax]) and fit them all with uncertainties, "
        "quality flags, and a fit+residual plot. Prefer this over manual "
        "find/fit chaining for 'fit the peaks' questions.",
        _params(
            {"path": {"type": "string"}, "xmin": opt_number, "xmax": opt_number,
             "max_peaks": {"type": "integer"}, "fwhm_guess": opt_number},
            ["path"],
        ),
        lambda **kw: fit_pattern_peaks(registry, **kw),
        category=CATEGORY,
    )]
