"""Deterministic 2D operations on slices: background/noise, peak detection,
azimuthal profiles with powder-ring identification, and line cuts to 1D.

All functions tolerate NaN (masked / no-coverage) pixels. Reciprocal-space
math assumes orthogonal cells for now (|Q| from h/a, k/b, l/c); generalize
with the B matrix when non-orthogonal data lands in data/.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from scattering_ai.tools.models import Curve1D, Slice2D

# d-spacings (Å) of common sample-environment / contaminant phases.
# Backed by knowledge/scattering/contaminant_signals.md.
CONTAMINANT_D_SPACINGS: dict[str, list[float]] = {
    "Al": [2.338, 2.025, 1.432, 1.221, 1.169, 1.012, 0.929, 0.905],
    "Cu": [2.087, 1.808, 1.278, 1.090, 1.044, 0.904, 0.829, 0.808],
    "steel(bcc Fe)": [2.027, 1.433, 1.170, 1.013, 0.906, 0.827],
    "V(can)": [2.141, 1.514, 1.236, 1.070, 0.957, 0.874],
}


def noise_level_2d(slice2d: Slice2D) -> float:
    """Robust pixel noise: scaled MAD of the median-filtered residual."""
    data = slice2d.data
    finite = np.isfinite(data)
    filled = np.where(finite, data, np.nanmedian(data))
    smooth = ndimage.median_filter(filled, size=5)
    resid = (data - smooth)[finite]
    return float(1.4826 * np.median(np.abs(resid)))


def estimate_background_2d(slice2d: Slice2D, size: int = 21) -> np.ndarray:
    """Smooth background via large-window median filtering (NaN-safe).

    Peaks and rings are narrow compared to ``size``; the median suppresses
    them while following broad diffuse variation.
    """
    data = slice2d.data
    finite = np.isfinite(data)
    filled = np.where(finite, data, np.nanmedian(data))
    background = ndimage.median_filter(filled, size=size)
    background[~finite] = np.nan
    return background


def find_peaks_2d(
    slice2d: Slice2D,
    min_snr: float = 8.0,
    min_separation: int = 3,
    max_peaks: int = 200,
) -> list[dict[str, Any]]:
    """Detect local maxima above background + min_snr * noise.

    Returns peaks sorted by intensity: {x, y, intensity, snr} in axis
    coordinates.
    """
    data = slice2d.data
    background = estimate_background_2d(slice2d)
    noise = noise_level_2d(slice2d) or 1e-12
    net = data - background

    finite = np.isfinite(net)
    filled = np.where(finite, net, -np.inf)
    local_max = ndimage.maximum_filter(filled, size=2 * min_separation + 1) == filled
    candidates = local_max & finite & (net > min_snr * noise)

    ys, xs = np.nonzero(candidates)
    peaks = [
        {
            "x": float(slice2d.x_centers[j]),
            "y": float(slice2d.y_centers[i]),
            "intensity": float(data[i, j]),
            "snr": float(net[i, j] / noise),
        }
        for i, j in zip(ys, xs, strict=True)
    ]
    peaks.sort(key=lambda p: p["intensity"], reverse=True)
    return peaks[:max_peaks]


def _q_geometry(
    slice2d: Slice2D, x_scale: float | None, y_scale: float | None
) -> tuple[np.ndarray, float]:
    """(|Q| per pixel in Å⁻¹, pixel area in Q-space).

    Prefers the exact B-matrix path when the slice carries its HKL
    orientation and lattice (slices cut from a Volume3D do); falls back to
    per-axis scales, which assume an orthogonal cell and axis-aligned axes.
    """
    from scattering_ai.tools.lattice import Lattice

    meta = slice2d.meta
    lattice = Lattice.from_dict(meta["lattice"]) if meta.get("lattice") else None
    if lattice is not None and all(k in meta for k in ("hkl_origin", "hkl_u", "hkl_v")):
        origin = np.asarray(meta["hkl_origin"], dtype=float)
        u_vec = np.asarray(meta["hkl_u"], dtype=float)
        v_vec = np.asarray(meta["hkl_v"], dtype=float)
        hkl = (
            origin[None, None, :]
            + slice2d.x_centers[None, :, None] * u_vec[None, None, :]
            + slice2d.y_centers[:, None, None] * v_vec[None, None, :]
        )
        q = lattice.q_magnitude(hkl)
        dx = abs(float(np.median(np.diff(slice2d.x_centers))))
        dy = abs(float(np.median(np.diff(slice2d.y_centers))))
        pixel_area = float(
            np.linalg.norm(np.cross(lattice.q_cartesian(u_vec) * dx,
                                    lattice.q_cartesian(v_vec) * dy))
        )
        return q, pixel_area
    if x_scale is None or y_scale is None:
        raise ValueError(
            "Slice carries no lattice/orientation metadata; pass x_scale and "
            "y_scale (axis units -> 1/Angstrom, orthogonal-cell approximation)."
        )
    qx = slice2d.x_centers * x_scale
    qy = slice2d.y_centers * y_scale
    q = np.sqrt(qx[None, :] ** 2 + qy[:, None] ** 2)
    dqx = abs(float(np.median(np.diff(slice2d.x_centers)))) * x_scale
    dqy = abs(float(np.median(np.diff(slice2d.y_centers)))) * y_scale
    return q, dqx * dqy


def azimuthal_profile(
    slice2d: Slice2D,
    x_scale: float | None = None,
    y_scale: float | None = None,
    dq: float = 0.02,
    statistic: str | float = "median",
) -> Curve1D:
    """Azimuthal intensity vs |Q| (Å⁻¹), robust to single-crystal Bragg peaks.

    Powder rings fill their whole annulus while Bragg peaks are pointlike
    outliers within it, so a robust statistic keeps rings and rejects Bragg:
    "median", "mean", or a float percentile (e.g. 25 — stricter, since a
    ring elevates even the lower quartile but scattered Bragg spots do not).

    The result meta carries ``completeness``: the fraction of each annulus
    actually covered by data. Partial annuli (corners of the field of view,
    anisotropic slices) weaken the ring/Bragg discrimination — treat
    low-completeness candidates with suspicion.
    """
    data = slice2d.data
    q, pixel_area = _q_geometry(slice2d, x_scale, y_scale)
    finite = np.isfinite(data)
    q_flat, v_flat = q[finite], data[finite]
    edges = np.arange(0, q_flat.max() + dq, dq)
    idx = np.digitize(q_flat, edges) - 1

    order = np.argsort(idx)
    idx_sorted, v_sorted = idx[order], v_flat[order]
    boundaries = np.searchsorted(idx_sorted, np.arange(len(edges)))
    centers = (edges[:-1] + edges[1:]) / 2

    if isinstance(statistic, str):
        reduce = np.median if statistic == "median" else np.mean
        label = statistic
    else:
        percentile = float(statistic)
        reduce = lambda v: float(np.percentile(v, percentile))  # noqa: E731
        label = f"p{statistic:g}"

    xs, ys, completeness = [], [], []
    for b in range(len(edges) - 1):
        lo, hi = boundaries[b], boundaries[b + 1]
        if hi - lo > 10:
            expected = 2 * np.pi * centers[b] * dq / pixel_area
            xs.append(centers[b])
            ys.append(float(reduce(v_sorted[lo:hi])))
            completeness.append(min(float((hi - lo) / max(expected, 1)), 1.0))
    return Curve1D(
        x=np.asarray(xs),
        y=np.asarray(ys),
        xlabel="|Q| (Å$^{-1}$)",
        ylabel=f"azimuthal {label} intensity",
        meta={
            **slice2d.meta,
            "operation": "azimuthal_profile",
            "dq": dq,
            "completeness": completeness,
        },
    )


def detect_rings(
    slice2d: Slice2D,
    x_scale: float | None = None,
    y_scale: float | None = None,
    match_tolerance: float = 0.06,
    min_prominence_snr: float = 4.0,
    min_completeness: float = 0.0,
    max_rings: int = 30,
) -> dict[str, Any]:
    """Find powder-ring **candidates** and match them to known contaminants.

    Method: azimuthal lower-quartile profile (a ring elevates even the 25th
    percentile of its annulus; isolated Bragg spots do not), restricted to
    annuli with coverage ≥ ``min_completeness``. Each candidate |Q| is
    matched to 2π/d of known contaminant d-spacings (Al, Cu, steel, V can)
    within ``match_tolerance`` (Å⁻¹).

    |Q| uses the exact B matrix when the slice carries lattice/orientation
    metadata (slices from Volume3D do); x_scale/y_scale are the orthogonal
    fallback for bare slices.

    This is a candidate generator, not a verdict: sample Bragg rows crossing
    short annulus arcs produce false candidates, especially in anisotropic
    slices. ``completeness`` per ring quantifies that risk; interpretation
    (and the researcher) decides.
    """
    from scipy import signal as _signal

    from scattering_ai.tools.curves import estimate_background, noise_level

    profile = azimuthal_profile(slice2d, x_scale, y_scale, statistic=25)
    completeness = np.asarray(profile.meta["completeness"])
    net = profile.y - estimate_background(profile, degree=5)
    noise = noise_level(profile) or 1e-12
    indices, props = _signal.find_peaks(net, prominence=min_prominence_snr * noise, width=1)

    rings = []
    for j, i in enumerate(indices):
        if completeness[i] < min_completeness:
            continue
        q_ring = float(profile.x[i])
        matches = [
            {"phase": phase, "d": d, "q_expected": round(2 * np.pi / d, 4)}
            for phase, d_list in CONTAMINANT_D_SPACINGS.items()
            for d in d_list
            if abs(q_ring - 2 * np.pi / d) < match_tolerance
        ]
        rings.append(
            {
                "q": round(q_ring, 4),
                "d": round(2 * np.pi / q_ring, 4),
                "prominence": round(float(props["prominences"][j]), 4),
                "snr": round(float(net[i] / noise), 1),
                "completeness": round(float(completeness[i]), 3),
                "candidates": matches,
            }
        )
    rings.sort(key=lambda r: r["prominence"], reverse=True)
    rings = rings[:max_rings]

    identified: dict[str, int] = {}
    for ring in rings:
        for match in ring["candidates"]:
            identified[match["phase"]] = identified.get(match["phase"], 0) + 1
    return {"rings": rings, "phase_match_counts": identified, "n_rings": len(rings)}


def line_cut(
    slice2d: Slice2D,
    start: tuple[float, float],
    end: tuple[float, float],
    width: float = 0.0,
    n_points: int | None = None,
) -> Curve1D:
    """Extract a 1D cut along a segment, averaging over ``width``
    perpendicular to it. Coordinates are in axis units. NaN-safe.
    """
    x0, y0 = start
    x1, y1 = end
    length = float(np.hypot(x1 - x0, y1 - y0))
    if length == 0:
        raise ValueError("start and end coincide")
    dx_pix = float(np.median(np.diff(slice2d.x_centers)))
    dy_pix = float(np.median(np.diff(slice2d.y_centers)))
    if n_points is None:
        n_points = max(int(length / min(dx_pix, dy_pix)), 2)

    t = np.linspace(0, 1, n_points)
    xs = x0 + t * (x1 - x0)
    ys = y0 + t * (y1 - y0)

    # perpendicular unit vector (axis units)
    px, py = -(y1 - y0) / length, (x1 - x0) / length
    n_perp = max(int(width / min(dx_pix, dy_pix)), 1) if width > 0 else 1
    offsets = np.linspace(-width / 2, width / 2, n_perp) if width > 0 else np.array([0.0])

    data = slice2d.data
    finite = np.isfinite(data)
    filled = np.where(finite, data, 0.0)

    def to_index(x_vals, y_vals):
        col = np.interp(x_vals, slice2d.x_centers, np.arange(len(slice2d.x_centers)))
        row = np.interp(y_vals, slice2d.y_centers, np.arange(len(slice2d.y_centers)))
        return np.vstack([row, col])

    value_sum = np.zeros(n_points)
    weight_sum = np.zeros(n_points)
    for offset in offsets:
        coords = to_index(xs + offset * px, ys + offset * py)
        value_sum += ndimage.map_coordinates(filled, coords, order=1, mode="nearest")
        weight_sum += ndimage.map_coordinates(
            finite.astype(float), coords, order=1, mode="nearest"
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(weight_sum > 0.5, value_sum / np.maximum(weight_sum, 1e-12), np.nan)

    return Curve1D(
        x=t * length,
        y=values,
        xlabel=f"distance along cut ({slice2d.xlabel} units)",
        ylabel="intensity",
        meta={
            **slice2d.meta,
            "operation": "line_cut",
            "start": [float(x0), float(y0)],
            "end": [float(x1), float(y1)],
            "width": float(width),
        },
    )
