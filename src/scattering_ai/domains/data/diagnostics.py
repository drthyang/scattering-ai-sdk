"""Deterministic diagnostics for the generic ``data`` domain.

Beyond the missing-file check, this detects a **parametric series** (a set of
temperature/field-scan files) and, when it finds one, tracks the strongest
peaks across the whole range and reports any **phase transition** — with a
waterfall and per-peak tracking figure the interpretation can point at. This is
the "observe a phase transition" path for a plain diffraction/PDF scan series.
"""

from __future__ import annotations

from pathlib import Path

from scattering_ai.core.findings import Finding, Severity

MIN_SERIES = 4  # fewer files than this is not worth a series scan
N_TRACK = 4  # strongest peaks to follow across the scan

NEXT_CHECK_RULES: dict[str, str] = {
    "phase_transition": "Confirm the changepoint against the waterfall and "
    "tracking figures; decide if the shift is continuous (2nd-order) or "
    "discontinuous (1st-order), and correlate with known sample transitions.",
    "series_no_transition": "If a transition is expected, track weaker/other "
    "peaks or a narrower range; a smooth drift alone is not a transition.",
    "missing_files": "Locate or regenerate the missing files before trusting "
    "the analysis.",
}


def _is_series(files: list[str]) -> bool:
    from scattering_ai.tools.series import extract_param

    params = [extract_param(Path(f).name) for f in files]
    good = [p for p in params if p is not None]
    return len(good) >= MIN_SERIES and len(good) >= 0.6 * len(files)


def _series_figures(series, tracked_all, workspace) -> list[str]:
    figures: list[str] = []
    try:
        from scattering_ai.tools.plotting import plot_series, plot_tracking
    except ImportError:
        return []
    ws = Path(workspace)
    try:
        figures.append(plot_series(series.curves, series.params,
                                   ws / "series_waterfall.png",
                                   param_label=series.param_label))
    except Exception:
        pass
    if tracked_all:  # tracking panel for the peak with the clearest transition
        _, tracked, det_c, det_f = max(
            tracked_all,
            key=lambda t: (bool(t[2].get("detected")) + bool(t[3].get("detected")),
                           t[1]["n_good_fits"]),
        )
        try:
            figures.append(plot_tracking(tracked, ws / "series_tracking.png",
                                         transitions={"center": det_c, "fwhm": det_f}))
        except Exception:
            pass
    return figures


def _series_transition(files: list[str], workspace) -> list[Finding]:
    if not _is_series(files):
        return []
    import numpy as np

    from scattering_ai.tools.curves import find_peaks
    from scattering_ai.tools.series import (
        detect_transition,
        load_series,
        stack_series,
        track_peak,
    )

    try:
        series = load_series(files, mask_value="auto")
    except Exception:
        return []
    label = series.param_label
    findings = [Finding(
        diagnostic="series_detected", severity=Severity.INFO,
        message=f"Parametric series: {len(series.curves)} curves over {label} "
        f"{min(series.params):g}–{max(series.params):g}"
        + (f"; masked sentinel {series.meta['mask_value_used']:g} auto-detected"
           if series.meta.get("mask_value_used") is not None else "") + ".",
        evidence={"param_label": label, "params": series.params},
    )]

    peaks = find_peaks(stack_series(series), subtract_background=True)[:N_TRACK]
    detections: list[float] = []
    tracked_all = []
    for pk in peaks:
        tracked = track_peak(series, center=pk["x"])
        good = [r for r in tracked["rows"] if r.get("ok")]
        if len(good) < 6:
            continue
        params = [r["param"] for r in good]
        det_c = detect_transition(params, [r["center"] for r in good],
                                  [r["center_err"] for r in good])
        det_f = detect_transition(params, [r["fwhm"] for r in good],
                                  [r["fwhm_err"] for r in good])
        tracked_all.append((pk["x"], tracked, det_c, det_f))
        detections += [d["transition_param"] for d in (det_c, det_f) if d.get("detected")]

    figures = _series_figures(series, tracked_all, workspace)
    if detections:
        t_c = round(float(np.median(detections)), 2)
        findings.append(Finding(
            diagnostic="phase_transition", severity=Severity.WARNING,
            message=f"Phase transition near {label} = {t_c} — {len(detections)} "
            f"peak trend(s) show a changepoint (range {min(detections):g}–"
            f"{max(detections):g}). Confirm against the figures.",
            evidence={"transition_param": t_c, "n_supporting_trends": len(detections),
                      "detections_range": [min(detections), max(detections)],
                      "n_peaks_tracked": len(tracked_all), "figures": figures},
        ))
    else:
        findings.append(Finding(
            diagnostic="series_no_transition", severity=Severity.INFO,
            message=f"Tracked {len(tracked_all)} peak(s) across the series; no "
            "sharp changepoint stood out (a smooth drift is not flagged).",
            evidence={"n_peaks_tracked": len(tracked_all), "figures": figures},
        ))
    return findings


def run_all(files: list[str], workspace=None) -> list[Finding]:
    findings: list[Finding] = []
    present = []
    for f in files:
        if Path(f).exists():
            present.append(f)
        else:
            findings.append(Finding(
                diagnostic="missing_files", severity=Severity.ERROR,
                message=f"Input file not found: {f}", evidence={"path": f}))
    findings.extend(_series_transition(present, workspace))
    return findings
