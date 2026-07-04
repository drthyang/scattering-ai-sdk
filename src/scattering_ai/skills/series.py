"""Skills for parametric series (temperature / field scans, transitions)."""

from __future__ import annotations

from typing import Any

from scattering_ai.skills.base import SkillRun
from scattering_ai.tools.registry import AgentTool, ToolRegistry, _params

CATEGORY = "series & transitions"


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


def skills(registry: ToolRegistry) -> list[AgentTool]:
    opt_number = {"type": ["number", "null"]}
    string = {"type": "string"}
    return [AgentTool(
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
        category=CATEGORY,
    )]
