"""Parametric series of 1D curves (temperature, field, time scans).

A series is an ordered stack of curves with a control-parameter value each
(roadmap B3: temperature/field dependence is a series axis over 1D data,
not a separate toolkit). Tools here track fitted features across the axis
and detect changepoints — with uncertainties propagated from the fits.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from scattering_ai.tools.curves import fit_peaks
from scattering_ai.tools.io import load_curve
from scattering_ai.tools.models import Curve1D

# "T_base_5.0K", "T=5K", "5.0K", "_005K_" — first match wins
_PARAM_PATTERNS = [
    re.compile(r"[Tt]_?base_?(\d+\.?\d*)\s*K"),
    re.compile(r"[TtHh]\s*=?\s*(\d+\.?\d*)\s*[KT]"),
    re.compile(r"[_\b](\d+\.?\d*)\s*K[_\b.]"),
]


def extract_param(name: str) -> float | None:
    for pattern in _PARAM_PATTERNS:
        match = pattern.search(name)
        if match:
            return float(match.group(1))
    return None


@dataclass
class Series1D:
    params: list[float]  # sorted ascending
    curves: list[Curve1D]
    param_label: str = "T (K)"
    meta: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "n_curves": len(self.curves),
            "param_label": self.param_label,
            "param_range": [min(self.params), max(self.params)] if self.params else None,
            "params": self.params,
            "x_range": [float(self.curves[0].x.min()), float(self.curves[0].x.max())]
            if self.curves
            else None,
            "source": self.meta.get("source", ""),
        }


def load_series(
    paths: list[str | Path],
    param_label: str = "T (K)",
    mask_value: float | None = None,
) -> Series1D:
    """Load curves and their parameter values (parsed from filenames).

    ``mask_value``: exact sentinel used for masked points (e.g. -3.0 in
    detector-masked regions); occurrences become NaN. Pass the string
    ``"auto"`` to detect a repeated flat sentinel from the data itself.
    """
    entries = []
    for p in paths:
        p = Path(p)
        value = extract_param(p.name)
        if value is None:
            raise ValueError(f"Cannot extract parameter value from filename: {p.name}")
        entries.append((value, load_curve(p)))
    entries.sort(key=lambda e: e[0])
    series = Series1D(
        params=[e[0] for e in entries],
        curves=[e[1] for e in entries],
        param_label=param_label,
        meta={"source": str(Path(paths[0]).parent) if paths else ""},
    )
    if mask_value == "auto":
        mask_value = auto_mask_value(series.curves)
        series.meta["mask_value_detected"] = mask_value
    if mask_value is not None:
        apply_mask(series, float(mask_value))
        series.meta["mask_value_used"] = float(mask_value)
    return series


def apply_mask(series: Series1D, value: float) -> None:
    """Turn every point exactly equal to ``value`` into NaN, in place."""
    for curve in series.curves:
        curve.y = np.where(curve.y == value, np.nan, curve.y)


def auto_mask_value(curves: list[Curve1D]) -> float | None:
    """Detect a repeated flat sentinel (e.g. -3.0 in detector-masked regions).

    The tell is not magnitude but *repetition*: real scattering intensities are
    essentially unique, so a single exact value repeated hundreds of times in
    the low tail of the data is a masking flag, not signal. A candidate must
    (a) sit at or below the 2nd percentile, (b) be repeated many times, and
    (c) appear in most curves. Returns None when nothing stands out, so it is
    safe on clean data. Detection only — nothing is masked here.
    """
    arrays = [np.asarray(c.y, dtype=float).ravel() for c in curves]
    finite = np.concatenate(arrays)
    finite = finite[np.isfinite(finite)]
    if finite.size < 50:
        return None
    low_tail = float(np.percentile(finite, 2.0))
    min_count = max(10, int(0.01 * finite.size))
    values, counts = np.unique(finite, return_counts=True)
    best: tuple[int, float] | None = None
    for value, count in zip(values, counts, strict=True):
        if value > low_tail or count < min_count:
            continue
        n_curves = sum(1 for arr in arrays if np.any(arr == value))
        if n_curves < max(2, len(arrays) // 2):
            continue
        if best is None or count > best[0]:
            best = (int(count), float(value))
    return best[1] if best else None


def stack_series(series: Series1D) -> Curve1D:
    """Average a series onto its common grid (a mean over the parameter axis).

    Used to pick tracking candidates: a peak that persists across the whole
    scan survives the average, while noise present in only one curve does not.
    Falls back to the first curve if the curves are not on a shared grid.
    """
    ref = series.curves[0]
    same_grid = all(
        c.x.size == ref.x.size and np.allclose(c.x, ref.x, equal_nan=True)
        for c in series.curves
    )
    if not same_grid:
        return ref
    stack = np.vstack([c.y for c in series.curves])
    with warnings.catch_warnings():  # all-NaN columns (fully masked) -> NaN, expected
        warnings.simplefilter("ignore", RuntimeWarning)
        y_mean = np.nanmean(stack, axis=0)
    return Curve1D(
        x=ref.x.copy(), y=y_mean, xlabel=ref.xlabel, ylabel=ref.ylabel,
        meta={"stacked_from": len(series.curves)},
    )


def track_peak(
    series: Series1D,
    center: float,
    window: float | None = None,
    fwhm_guess: float | None = None,
    follow: bool = True,
) -> dict[str, Any]:
    """Fit one peak in every curve of the series and track its parameters.

    ``follow=True`` seeds each fit with the previous curve's fitted center,
    so a drifting peak stays captured. Failed or flagged fits are recorded,
    not silently dropped.
    """
    rows: list[dict[str, Any]] = []
    current_center = float(center)
    for value, curve in zip(series.params, series.curves, strict=True):
        row: dict[str, Any] = {"param": value}
        try:
            result = fit_peaks(
                curve, centers=[current_center], window=window, fwhm_guess=fwhm_guess
            )
            peak = result["peaks"][0]
            row.update(
                center=peak["center"],
                center_err=peak["center_err"],
                fwhm=peak["fwhm"],
                fwhm_err=peak["fwhm_err"],
                height=peak["height"],
                height_err=peak["height_err"],
                rwp=result["rwp"],
                flags=result["flags"],
                ok=not (result["flags"]["at_bounds"] or result["flags"]["poor_fit"]),
            )
            if follow and row["ok"]:
                current_center = peak["center"]
        except Exception as exc:
            row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        rows.append(row)

    good = [r for r in rows if r.get("ok")]
    return {
        "tracked_center_start": float(center),
        "n_points": len(rows),
        "n_good_fits": len(good),
        "rows": rows,
        "param_label": series.param_label,
    }


def detect_transition(
    params: list[float],
    values: list[float],
    errors: list[float] | None = None,
) -> dict[str, Any]:
    """Two-segment changepoint detection on a tracked quantity.

    Fits a straight line to each side of every candidate split and compares
    the best two-segment residual against the single-line fit. Returns the
    candidate transition, its improvement ratio, and a significance flag
    (improvement > 3x beats a smooth trend). Deterministic and crude by
    design — a candidate generator for interpretation, not a verdict.
    """
    x = np.asarray(params, dtype=float)
    y = np.asarray(values, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if errors is not None:
        w = 1.0 / np.clip(np.asarray(errors, dtype=float)[keep], 1e-12, None)
    else:
        w = np.ones_like(x)
    n = x.size
    if n < 6:
        return {"detected": False, "reason": f"only {n} valid points (need >= 6)"}

    def wrss(xs, ys, ws) -> float:
        coeffs = np.polyfit(xs, ys, 1, w=ws)
        return float(np.sum((ws * (ys - np.polyval(coeffs, xs))) ** 2))

    single = wrss(x, y, w)
    # If a single line already fits at numerical precision, there is no
    # transition to find (guards the 0/0 ratio on noiseless linear data).
    scale = float(np.sum((w * (y - y.mean())) ** 2))
    if single <= 1e-12 * max(scale, 1e-300):
        return {"detected": False, "reason": "single line fits at numerical precision",
                "n_points": int(n)}

    best_split, best_rss = None, np.inf
    for i in range(3, n - 2):  # at least 3 points per segment
        rss = wrss(x[:i], y[:i], w[:i]) + wrss(x[i:], y[i:], w[i:])
        if rss < best_rss:
            best_split, best_rss = i, rss

    improvement = single / max(best_rss, 1e-300)
    t_c = float((x[best_split - 1] + x[best_split]) / 2)
    return {
        "detected": bool(improvement > 3.0),
        "transition_param": round(t_c, 3),
        "improvement_ratio": round(float(improvement), 2),
        "single_segment_wrss": round(single, 4),
        "two_segment_wrss": round(float(best_rss), 4),
        "n_points": int(n),
        "note": "changepoint candidate; confirm against the raw curves",
    }
