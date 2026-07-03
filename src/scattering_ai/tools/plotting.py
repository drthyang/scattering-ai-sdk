"""Plotters for judging results: curves, fits, slices, series, tracking.

Every function renders to a PNG path and returns it, so plots work the same
from Python, the CLI (``scattering-ai plot``), the agent tool registry, and
the MCP/HTTP servers. Requires matplotlib (``pip install
scattering-ai-sdk[plots]``); imported lazily so the base install stays light.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scattering_ai.tools.models import Curve1D, Slice2D


def _plt():
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Plotting requires matplotlib: pip install scattering-ai-sdk[plots]"
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _finish(fig, out: str | Path) -> str:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    _plt().close(fig)
    return str(out)


def plot_curve(
    curve: Curve1D,
    out: str | Path,
    peaks: list[dict[str, Any]] | None = None,
    title: str = "",
    logy: bool = False,
) -> str:
    """Data with optional detected-peak markers."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(curve.x, curve.y, lw=0.8, color="#1f5fa8")
    if curve.e is not None:
        ax.fill_between(curve.x, curve.y - curve.e, curve.y + curve.e,
                        alpha=0.25, color="#1f5fa8", lw=0)
    for p in peaks or []:
        ax.axvline(p["x"], color="crimson", ls=":", alpha=0.7)
        ax.annotate(f"{p['x']:.3g}", (p["x"], p["y"]), fontsize=7,
                    rotation=90, va="bottom", ha="right", color="crimson")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(curve.xlabel)
    ax.set_ylabel(curve.ylabel)
    ax.set_title(title or Path(str(curve.meta.get("source", ""))).name)
    return _finish(fig, out)


def plot_fit(
    curve: Curve1D,
    fit_result: dict[str, Any],
    out: str | Path,
    title: str = "",
) -> str:
    """Data + fitted model + per-peak components + residual panel."""
    from scattering_ai.tools.curves import _pseudo_voigt, evaluate_fit

    plt = _plt()
    lo, hi = fit_result["window"]
    mask = (curve.x >= lo) & (curve.x <= hi)
    x, y = curve.x[mask], curve.y[mask]
    model = evaluate_fit(x, fit_result)

    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(9, 5.5), sharex=True, height_ratios=[3, 1]
    )
    ax.plot(x, y, ".", ms=3, color="#333", label="data")
    ax.plot(x, model, lw=1.4, color="crimson", label="fit")
    background = fit_result["background"]
    bg = background["slope"] * x + background["intercept"]
    ax.plot(x, bg, ls="--", lw=0.9, color="gray", label="background")
    for p in fit_result["peaks"]:
        ax.plot(x, bg + _pseudo_voigt(x, p["height"], p["center"], p["fwhm"], p["eta"]),
                lw=0.8, alpha=0.7)
        ax.annotate(
            f"{p['center']:.4g}±{p['center_err']:.2g}\nfwhm {p['fwhm']:.3g}±{p['fwhm_err']:.2g}",
            (p["center"], p["height"] + bg[np.argmin(abs(x - p["center"]))]),
            fontsize=7, ha="center", va="bottom",
        )
    flags = fit_result["flags"]
    flag_text = ", ".join(k for k, v in flags.items() if v) or "clean"
    ax.set_title(
        (title or "Peak fit")
        + f"  |  rwp={fit_result['rwp']:.3g}  χ²ᵣ={fit_result['reduced_chi2']:.3g}  [{flag_text}]"
    )
    ax.set_ylabel(curve.ylabel)
    ax.legend(fontsize=8)

    ax_res.axhline(0, color="gray", lw=0.7)
    ax_res.plot(x, y - model, lw=0.7, color="#1f5fa8")
    ax_res.set_ylabel("residual")
    ax_res.set_xlabel(curve.xlabel)
    return _finish(fig, out)


def plot_slice(
    slice2d: Slice2D,
    out: str | Path,
    log: bool = False,
    vmax_percentile: float = 99.0,
    peaks: list[dict[str, Any]] | None = None,
    title: str = "",
) -> str:
    """2D map, NaN-aware, optional detected-peak overlay."""
    plt = _plt()
    from matplotlib.colors import LogNorm

    data = slice2d.data
    finite = np.isfinite(data)
    vmax = float(np.nanpercentile(data[finite], vmax_percentile)) if finite.any() else 1.0
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if log:
        positive = data[finite & (data > 0)]
        vmin = float(np.percentile(positive, 5)) if positive.size else 1e-3
        norm = LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, 2e-12))
        mesh = ax.pcolormesh(slice2d.x_centers, slice2d.y_centers, data,
                             norm=norm, cmap="viridis")
    else:
        mesh = ax.pcolormesh(slice2d.x_centers, slice2d.y_centers, data,
                             vmin=0, vmax=vmax, cmap="viridis")
    for p in peaks or []:
        ax.plot(p["x"], p["y"], "o", ms=8, mfc="none", mec="crimson", mew=1.2)
    ax.set_xlabel(slice2d.xlabel)
    ax.set_ylabel(slice2d.ylabel)
    ax.set_title(title or _slice_title(slice2d))
    fig.colorbar(mesh, ax=ax, label="intensity")
    return _finish(fig, out)


def _slice_title(slice2d: Slice2D) -> str:
    meta = slice2d.meta
    if "slice_axis" in meta:
        return (f"{meta['slice_axis']} = {meta.get('center')} "
                f"± {meta.get('thickness', 0) / 2}")
    if "hkl_origin" in meta:
        return f"plane o={meta['hkl_origin']} u={meta.get('hkl_u')} v={meta.get('hkl_v')}"
    return Path(str(meta.get("source", ""))).name


def plot_series(
    curves: list[Curve1D],
    params: list[float],
    out: str | Path,
    param_label: str = "T (K)",
    offset: float | None = None,
    xmin: float | None = None,
    xmax: float | None = None,
) -> str:
    """Waterfall of a parametric series, colored by parameter value."""
    plt = _plt()
    from matplotlib import colormaps

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = colormaps["coolwarm"]
    span = max(params) - min(params) or 1.0
    if offset is None:
        scales = [np.nanpercentile(np.abs(c.y[np.isfinite(c.y)]), 95) for c in curves]
        offset = 0.8 * float(np.median(scales))
    for i, (value, curve) in enumerate(zip(params, curves, strict=True)):
        mask = np.ones_like(curve.x, dtype=bool)
        if xmin is not None:
            mask &= curve.x >= xmin
        if xmax is not None:
            mask &= curve.x <= xmax
        ax.plot(curve.x[mask], curve.y[mask] + i * offset, lw=0.7,
                color=cmap((value - min(params)) / span))
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(min(params), max(params))
    )
    fig.colorbar(sm, ax=ax, label=param_label)
    ax.set_xlabel(curves[0].xlabel)
    ax.set_ylabel(f"{curves[0].ylabel} (offset per curve)")
    ax.set_title(f"series: {len(curves)} curves, {param_label} "
                 f"{min(params):g}–{max(params):g}")
    return _finish(fig, out)


def plot_tracking(
    tracking: dict[str, Any],
    out: str | Path,
    transitions: dict[str, dict] | None = None,
) -> str:
    """Tracked peak center/FWHM/height vs parameter, with error bars and
    detected-transition markers."""
    plt = _plt()
    rows = [r for r in tracking["rows"] if r.get("ok")]
    bad = [r for r in tracking["rows"] if not r.get("ok")]
    param = [r["param"] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
    specs = [("center", "center"), ("fwhm", "FWHM"), ("height", "height")]
    for ax, (key, label) in zip(axes, specs, strict=True):
        ax.errorbar(param, [r[key] for r in rows],
                    yerr=[r.get(f"{key}_err", 0) for r in rows],
                    marker="o", ms=3.5, lw=0.8, color="#1f5fa8")
        ax.set_ylabel(label)
        for r in bad:
            ax.axvline(r["param"], color="orange", alpha=0.25, lw=3)
    for name, det in (transitions or {}).items():
        if det and det.get("detected"):
            for ax in axes:
                ax.axvline(det["transition_param"], color="crimson", ls="--", lw=1)
            axes[0].annotate(
                f"{name}: {det['transition_param']:g} (x{det['improvement_ratio']:g})",
                (det["transition_param"], axes[0].get_ylim()[1]),
                fontsize=8, color="crimson", va="top", ha="left",
            )
    axes[-1].set_xlabel(tracking.get("param_label", "parameter"))
    axes[0].set_title(
        f"peak @{tracking['tracked_center_start']:g}: "
        f"{tracking['n_good_fits']}/{tracking['n_points']} good fits "
        f"(orange = failed/flagged)"
    )
    return _finish(fig, out)


def plot_profile(
    profile: Curve1D,
    out: str | Path,
    rings: list[dict[str, Any]] | None = None,
    title: str = "",
) -> str:
    """Azimuthal profile with ring candidates and contaminant matches."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(profile.x, profile.y, lw=0.8, color="#1f5fa8")
    for ring in rings or []:
        matched = ring.get("candidates")
        color = "crimson" if matched else "gray"
        ax.axvline(ring["q"], color=color, ls=":", alpha=0.8)
        label = ",".join(m["phase"] for m in matched) if matched else "?"
        ax.annotate(f"{label}\nc={ring.get('completeness', 0):.2f}",
                    (ring["q"], ax.get_ylim()[1]), fontsize=6, color=color,
                    va="top", ha="center")
    ax.set_xlabel(profile.xlabel)
    ax.set_ylabel(profile.ylabel)
    ax.set_title(title or "azimuthal profile (red = contaminant match)")
    return _finish(fig, out)
