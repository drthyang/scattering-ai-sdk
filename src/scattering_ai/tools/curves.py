"""Deterministic 1D operations: crop, rebin, background, peaks, fitting,
Fourier transform.

Design rules (roadmap B3): every function is pure and testable without an
LLM; fit results carry uncertainties and explicit quality flags the agent
can reason about instead of trusting a single number.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize, signal

from scattering_ai.tools.models import Curve1D


def crop(curve: Curve1D, xmin: float | None = None, xmax: float | None = None) -> Curve1D:
    lo = -np.inf if xmin is None else xmin
    hi = np.inf if xmax is None else xmax
    keep = (curve.x >= lo) & (curve.x <= hi)
    return Curve1D(
        x=curve.x[keep],
        y=curve.y[keep],
        e=None if curve.e is None else curve.e[keep],
        xlabel=curve.xlabel,
        ylabel=curve.ylabel,
        meta={**curve.meta, "cropped": [float(lo), float(hi)]},
    )


def rebin(curve: Curve1D, dx: float) -> Curve1D:
    """Rebin onto a regular grid of width ``dx`` by averaging within bins.

    Errors propagate as sqrt(sum(e^2))/n per bin when present.
    """
    if dx <= 0:
        raise ValueError("dx must be positive")
    edges = np.arange(curve.x.min(), curve.x.max() + dx, dx)
    idx = np.digitize(curve.x, edges) - 1
    n_bins = len(edges) - 1
    counts = np.bincount(idx.clip(0, n_bins - 1), minlength=n_bins).astype(float)
    valid = counts > 0

    y_sum = np.bincount(idx.clip(0, n_bins - 1), weights=curve.y, minlength=n_bins)
    y_new = np.full(n_bins, np.nan)
    y_new[valid] = y_sum[valid] / counts[valid]

    e_new = None
    if curve.e is not None:
        e2_sum = np.bincount(idx.clip(0, n_bins - 1), weights=curve.e**2, minlength=n_bins)
        e_new = np.full(n_bins, np.nan)
        e_new[valid] = np.sqrt(e2_sum[valid]) / counts[valid]
        e_new = e_new[valid]

    centers = (edges[:-1] + edges[1:]) / 2
    return Curve1D(
        x=centers[valid],
        y=y_new[valid],
        e=e_new,
        xlabel=curve.xlabel,
        ylabel=curve.ylabel,
        meta={**curve.meta, "rebinned_dx": float(dx)},
    )


def estimate_background(curve: Curve1D, degree: int = 3, n_iter: int = 30) -> np.ndarray:
    """Iteratively clipped polynomial background.

    Fit a polynomial, clip points above fit + sigma (peaks), refit. Robust
    for peaks-on-smooth-background patterns; not meant for PDFs, where the
    baseline is physics, not background.
    """
    x, y = curve.x, curve.y
    finite = np.isfinite(y)
    keep = finite.copy()
    for _ in range(n_iter):
        coeffs = np.polynomial.polynomial.polyfit(x[keep], y[keep], degree)
        bg = np.polynomial.polynomial.polyval(x, coeffs)
        resid = y - bg
        # Estimate noise from negative residuals only: peaks push residuals
        # positive, so the negative side is noise-only. Using both sides (or
        # re-shrinking sigma each iteration) biases the fit toward the lower
        # noise envelope.
        negative = resid[keep & (resid < 0)]
        sigma = 1.4826 * float(np.median(np.abs(negative))) if negative.size else float(
            np.nanstd(resid[keep])
        )
        new_keep = finite & (resid < 2.5 * sigma)
        if new_keep.sum() < degree + 2 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return bg


def noise_level(curve: Curve1D) -> float:
    """Point-to-point noise estimate via median absolute successive difference."""
    diffs = np.diff(curve.y[np.isfinite(curve.y)])
    return float(np.median(np.abs(diffs)) / np.sqrt(2) * 1.4826)


def find_peaks(
    curve: Curve1D,
    min_prominence: float | None = None,
    subtract_background: bool = False,
    max_peaks: int = 20,
) -> list[dict[str, Any]]:
    """Find peaks; prominence threshold defaults to 8x the noise level.

    A minimum width of 3 samples rejects single-point noise spikes; pass an
    explicit ``min_prominence`` when hunting weak features.
    """
    y = curve.y - estimate_background(curve) if subtract_background else curve.y.copy()
    prominence = min_prominence if min_prominence is not None else 8 * noise_level(curve)
    indices, props = signal.find_peaks(y, prominence=prominence, width=3)
    peaks = [
        {
            "x": float(curve.x[i]),
            "y": float(curve.y[i]),
            "prominence": float(props["prominences"][j]),
            "fwhm_estimate": float(
                props["widths"][j] * np.median(np.diff(curve.x))
            ),
        }
        for j, i in enumerate(indices)
    ]
    peaks.sort(key=lambda p: p["prominence"], reverse=True)
    return peaks[:max_peaks]


def _pseudo_voigt(x: np.ndarray, height: float, center: float, fwhm: float, eta: float):
    dx2 = (x - center) ** 2
    gauss = np.exp(-4 * np.log(2) * dx2 / fwhm**2)
    lorentz = 1 / (1 + 4 * dx2 / fwhm**2)
    return height * (eta * lorentz + (1 - eta) * gauss)


def fit_peaks(
    curve: Curve1D,
    centers: list[float],
    window: float | None = None,
    fwhm_guess: float | None = None,
) -> dict[str, Any]:
    """Fit pseudo-Voigt peaks plus a linear background around ``centers``.

    Returns fitted parameters with 1-sigma uncertainties and quality flags:
    ``at_bounds`` (parameters pinned to limits), ``high_uncertainty``
    (stderr > 50% of value), and ``rwp``/``reduced_chi2`` for the region.
    """
    centers = sorted(float(c) for c in centers)
    dx = float(np.median(np.diff(curve.x)))
    if fwhm_guess is None:
        fwhm_guess = 10 * dx
    if window is None:
        window = 6 * fwhm_guess
    region = crop(curve, centers[0] - window, centers[-1] + window)
    if region.x.size < 5 * len(centers) + 2:
        raise ValueError("Not enough points in fit window")
    x, y = region.x, region.y

    def model(x, *params):
        slope, intercept = params[0], params[1]
        out = slope * x + intercept
        for k in range(len(centers)):
            height, center, fwhm, eta = params[2 + 4 * k : 6 + 4 * k]
            out = out + _pseudo_voigt(x, height, center, fwhm, eta)
        return out

    baseline = float(np.median(y))
    p0: list[float] = [0.0, baseline]
    lower: list[float] = [-np.inf, -np.inf]
    upper: list[float] = [np.inf, np.inf]
    span = x.max() - x.min()
    for c in centers:
        height0 = max(float(np.interp(c, x, y) - baseline), 10 * dx)
        p0 += [height0, c, fwhm_guess, 0.5]
        lower += [0.0, c - window / 2, dx, 0.0]
        upper += [np.inf, c + window / 2, span, 1.0]

    sigma = region.e if region.e is not None else None
    popt, pcov = optimize.curve_fit(
        model, x, y, p0=p0, bounds=(lower, upper), sigma=sigma, maxfev=20000
    )
    perr = np.sqrt(np.diag(pcov))

    residual = y - model(x, *popt)
    rwp = float(np.sqrt(np.sum(residual**2) / np.sum(y**2)))
    dof = max(x.size - len(popt), 1)
    noise = noise_level(region) or 1.0
    reduced_chi2 = float(np.sum((residual / noise) ** 2) / dof)

    result_peaks = []
    at_bounds, high_uncertainty = [], []
    names = ["height", "center", "fwhm", "eta"]
    for k in range(len(centers)):
        entry: dict[str, Any] = {}
        for j, name in enumerate(names):
            i = 2 + 4 * k + j
            value, err = float(popt[i]), float(perr[i])
            entry[name] = value
            entry[f"{name}_err"] = err
            near_low = np.isfinite(lower[i]) and abs(value - lower[i]) < 1e-8 * max(1, abs(value))
            near_high = np.isfinite(upper[i]) and abs(value - upper[i]) < 1e-8 * max(1, abs(value))
            if (near_low or near_high) and name != "eta":
                at_bounds.append(f"peak{k}.{name}")
            if err > 0.5 * abs(value) and name in ("height", "fwhm"):
                high_uncertainty.append(f"peak{k}.{name}")
        result_peaks.append(entry)

    return {
        "peaks": result_peaks,
        "background": {"slope": float(popt[0]), "intercept": float(popt[1])},
        "rwp": rwp,
        "reduced_chi2": reduced_chi2,
        "n_points": int(x.size),
        "window": [float(x.min()), float(x.max())],
        "flags": {
            "at_bounds": at_bounds,
            "high_uncertainty": high_uncertainty,
            # High-count data legitimately gives chi2 >> 1 from tiny model-shape
            # imperfections, so "poor" requires meaningful misfit (rwp) too.
            "poor_fit": rwp > 0.10 and reduced_chi2 > 10,
        },
    }


def detect_sq_convention(curve: Curve1D) -> str:
    """Detect what a "S(Q)" file actually stores from its high-Q limit.

    Returns "sq" (tail → 1), "sq_minus_1" (tail → 0), or "unknown".
    Facility files are inconsistent: NOMAD SofQ exports often store S(Q)-1
    under an S(Q) name.
    """
    q, y = curve.x, curve.y
    tail = y[np.isfinite(y) & (q >= 0.8 * q.max())]
    if tail.size < 10:
        return "unknown"
    mean = float(tail.mean())
    if abs(mean - 1.0) < 0.1:
        return "sq"
    if abs(mean) < 0.1:
        return "sq_minus_1"
    return "unknown"


def sq_to_gr(
    curve: Curve1D,
    qmin: float | None = None,
    qmax: float | None = None,
    rmax: float = 30.0,
    dr: float = 0.01,
    input_kind: str = "auto",
) -> Curve1D:
    """Sine Fourier transform of total scattering data to the PDF G(r).

    G(r) = (2/pi) * integral of Q[S(Q)-1] sin(Qr) dQ over [qmin, qmax].

    ``input_kind``: "sq" (y = S(Q)), "sq_minus_1" (y = S(Q)-1), "fq"
    (y = F(Q) = Q[S(Q)-1]), or "auto" to detect from the high-Q limit
    (raises if ambiguous — better to fail than silently transform the
    wrong quantity).
    """
    if input_kind == "auto":
        input_kind = detect_sq_convention(curve)
        if input_kind == "unknown":
            raise ValueError(
                "Cannot auto-detect input convention (high-Q tail is neither ~0 "
                "nor ~1); pass input_kind explicitly."
            )
    q, y = curve.x, curve.y
    keep = np.isfinite(y)
    if qmin is not None:
        keep &= q >= qmin
    if qmax is not None:
        keep &= q <= qmax
    q, y = q[keep], y[keep]
    if input_kind == "fq":
        fq = y
    elif input_kind == "sq_minus_1":
        fq = q * y
    elif input_kind == "sq":
        fq = q * (y - 1.0)
    else:
        raise ValueError(f"Unknown input_kind: {input_kind!r}")

    r = np.arange(dr, rmax + dr, dr)
    integrand = fq[None, :] * np.sin(np.outer(r, q))
    gr = (2 / np.pi) * np.trapezoid(integrand, q, axis=1)
    return Curve1D(
        x=r,
        y=gr,
        xlabel="r(Å)",
        ylabel="G(Å$^{-2}$)",
        meta={
            **curve.meta,
            "transform": "sq_to_gr",
            "qrange": [float(q.min()), float(q.max())],
        },
    )
