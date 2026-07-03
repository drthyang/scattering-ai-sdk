"""The first three skills, distilled from workflows validated on real data
(FeCoSn G(r) fitting, CORELLI slice characterization, GaNb4Se8 transition
scan). Each returns structured evidence plus plot paths for visual judgment.
"""

from __future__ import annotations

from typing import Any

from scattering_ai.skills.base import SkillRun
from scattering_ai.tools.registry import AgentTool, ToolRegistry, _params


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


def scan_series_transitions(
    registry: ToolRegistry,
    paths: list[str],
    mask_value: float | None = None,
    n_peaks: int = 4,
    fwhm_guess: float | None = None,
) -> dict[str, Any]:
    """Track the strongest peaks across a parametric series and vote on a
    transition: waterfall + per-peak tracking plots included."""
    run = SkillRun(registry)
    info = run.call("inspect_series", paths=paths, mask_value=mask_value)
    if "error" in info:
        return run.finish(error=info["error"])
    waterfall = run.call("plot_series_files", paths=paths, mask_value=mask_value)

    candidates = info.get("peaks_in_first_curve", [])[:n_peaks]
    per_peak = []
    detections = []
    for peak in candidates:
        tracked = run.call("track_peak_series", paths=paths, center=peak["x"],
                           fwhm_guess=fwhm_guess, mask_value=mask_value)
        if "error" in tracked:
            per_peak.append({"center": peak["x"], "error": tracked["error"]})
            continue
        det_c = tracked.get("transition_on_center") or {}
        det_f = tracked.get("transition_on_fwhm") or {}
        per_peak.append({
            "center": peak["x"],
            "n_good_fits": tracked.get("n_good_fits"),
            "transition_on_center": det_c,
            "transition_on_fwhm": det_f,
            "plot": tracked.get("plot"),
        })
        for det in (det_c, det_f):
            if det.get("detected"):
                detections.append(det["transition_param"])

    verdict: dict[str, Any] = {"transition_detected": bool(detections)}
    if detections:
        import numpy as np

        verdict.update(
            transition_estimate=round(float(np.median(detections)), 2),
            detections_range=[min(detections), max(detections)],
            n_supporting_trends=len(detections),
            note="median of changepoints across tracked peaks; confirm against "
            "the waterfall and tracking plots",
        )
    return run.finish(
        series=info.get("params"),
        tracked_peaks=per_peak,
        verdict=verdict,
        plots={"waterfall": waterfall.get("saved")},
    )


def register_skills(registry: ToolRegistry) -> ToolRegistry:
    """Expose the built-in skills on a registry as skill_* tools."""
    opt_number = {"type": ["number", "null"]}
    string = {"type": "string"}

    skills = [
        AgentTool(
            "skill_characterize_slice",
            "SKILL (composite workflow): full first-look at a 2D slice — cut from "
            "a volume (or use an existing slice_path), detect Bragg peaks, hunt "
            "contaminant rings, render a log-scale plot. Prefer this over manual "
            "chaining for 'what is in this slice?' questions.",
            _params(
                {"path": string, "axis": {"type": ["integer", "null"]},
                 "center": opt_number, "thickness": opt_number,
                 "slice_path": string},
                ["path"],
            ),
            lambda **kw: characterize_slice(registry, **kw),
        ),
        AgentTool(
            "skill_fit_pattern_peaks",
            "SKILL (composite workflow): find the strongest peaks in a 1D pattern "
            "(optionally within [xmin, xmax]) and fit them all with uncertainties, "
            "quality flags, and a fit+residual plot. Prefer this over manual "
            "find/fit chaining for 'fit the peaks' questions.",
            _params(
                {"path": string, "xmin": opt_number, "xmax": opt_number,
                 "max_peaks": {"type": "integer"}, "fwhm_guess": opt_number},
                ["path"],
            ),
            lambda **kw: fit_pattern_peaks(registry, **kw),
        ),
        AgentTool(
            "skill_scan_series_transitions",
            "SKILL (composite workflow): track the strongest peaks across a "
            "temperature/field series, run changepoint detection on every trend, "
            "and vote on a transition estimate, with waterfall and tracking plots. "
            "Prefer this for 'is there a transition?' questions.",
            _params(
                {"paths": {"type": "array", "items": string},
                 "mask_value": opt_number, "n_peaks": {"type": "integer"},
                 "fwhm_guess": opt_number},
                ["paths"],
            ),
            lambda **kw: scan_series_transitions(registry, **kw),
        ),
    ]
    for skill in skills:
        registry.add(skill)
    return registry
