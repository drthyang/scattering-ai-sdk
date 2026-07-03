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


def _monitor_change(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Low-parameter vs high-parameter change of a tracked peak, from the good
    fits only. This is the 'what moved across the scan' evidence, reported
    whether or not a sharp changepoint is found."""
    good = [r for r in rows if r.get("ok") and r.get("center") is not None]
    if len(good) < 2:
        return {"n_good_fits": len(good)}
    lo, hi = good[0], good[-1]

    def delta(key: str) -> float | None:
        a, b = lo.get(key), hi.get(key)
        return None if a is None or b is None else round(float(b - a), 4)

    return {
        "n_good_fits": len(good),
        "param_span": [lo["param"], hi["param"]],
        "center_start": round(float(lo["center"]), 4),
        "center_end": round(float(hi["center"]), 4),
        "center_shift": delta("center"),
        "fwhm_change": delta("fwhm"),
        "height_change": delta("height"),
    }


def scan_series_transitions(
    registry: ToolRegistry,
    paths: list[str],
    mask_value: float | None = None,
    n_peaks: int = 4,
    fwhm_guess: float | None = None,
) -> dict[str, Any]:
    """Monitor the strongest peaks across a parametric (T/field) series: track
    each peak's center/FWHM/height over the whole range, report how each moved,
    and vote on a transition. Waterfall + per-peak tracking plots included.

    Robust by design: peaks are chosen from the mean over the scan (not one
    curve), and the masked-region sentinel is auto-detected when not given."""
    import numpy as np

    run = SkillRun(registry)
    info = run.call("inspect_series", paths=paths, mask_value=mask_value)
    if "error" in info:
        return run.finish(error=info["error"])
    # Honor the sentinel inspect_series actually used (given or auto-detected)
    # so every downstream fit sees the same masked data.
    mask_used = info.get("mask_value_used", mask_value)
    waterfall = run.call("plot_series_files", paths=paths, mask_value=mask_used)

    candidates = info.get("strongest_peaks") or info.get("peaks_in_first_curve", [])
    candidates = candidates[:n_peaks]
    per_peak = []
    detections = []
    for peak in candidates:
        tracked = run.call("track_peak_series", paths=paths, center=peak["x"],
                           fwhm_guess=fwhm_guess, mask_value=mask_used)
        if "error" in tracked:
            per_peak.append({"center": peak["x"], "error": tracked["error"]})
            continue
        det_c = tracked.get("transition_on_center") or {}
        det_f = tracked.get("transition_on_fwhm") or {}
        per_peak.append({
            "center": peak["x"],
            "n_good_fits": tracked.get("n_good_fits"),
            "monitored": _monitor_change(tracked.get("rows", [])),
            "transition_on_center": det_c,
            "transition_on_fwhm": det_f,
            "plot": tracked.get("plot"),
        })
        for det in (det_c, det_f):
            if det.get("detected"):
                detections.append(det["transition_param"])

    n_tracked = sum(1 for p in per_peak if not p.get("error"))
    verdict: dict[str, Any] = {"transition_detected": bool(detections)}
    if detections:
        verdict.update(
            transition_estimate=round(float(np.median(detections)), 2),
            detections_range=[min(detections), max(detections)],
            n_supporting_trends=len(detections),
            note="median of changepoints across tracked peaks; confirm against "
            "the waterfall and tracking plots",
        )
    params = info.get("params") or []
    span = f"{min(params):g}–{max(params):g}" if params else "?"
    if not per_peak:
        summary = "No trackable peaks found in the series."
    elif detections:
        summary = (
            f"Tracked {n_tracked} peak(s) across {len(params)} curves "
            f"({info.get('param_label', 'param')} {span}). "
            f"{len(detections)} trend(s) show a changepoint near "
            f"{verdict['transition_estimate']:g} "
            f"(range {min(detections):g}–{max(detections):g}); likely a "
            "transition. Confirm against the waterfall and tracking plots."
        )
    else:
        summary = (
            f"Tracked {n_tracked} peak(s) across {len(params)} curves "
            f"({info.get('param_label', 'param')} {span}); no sharp changepoint "
            "stood out. See the per-peak 'monitored' changes and the plots — a "
            "gradual shift is not flagged as a transition."
        )
    return run.finish(
        summary=summary,
        series=params,
        mask_value_used=mask_used,
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
            "SKILL (composite workflow): monitor the strongest peaks across a "
            "temperature/field series — track each peak's center/FWHM/height over "
            "the whole range, report how much each moved, run changepoint "
            "detection on every trend, and vote on a transition estimate, with "
            "waterfall and tracking plots. Peaks are picked from the mean over "
            "the scan and the masked-region sentinel is auto-detected, so you "
            "only need to pass the files. THE right tool for 'is there a "
            "transition?' / 'trace the transition' / 'what changes with "
            "temperature?' — prefer it over manual peak-by-peak chaining.",
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
