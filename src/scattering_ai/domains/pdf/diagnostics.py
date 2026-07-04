"""Deterministic diagnostics for PDF / total-scattering data files.

Everything here is computable without an LLM. Each diagnostic reads a G(r) or
S(Q)/F(Q) curve and returns findings whose ``evidence`` carries the numbers,
so downstream reasoning cites instead of recomputing.

The checks encode two real-data gotchas the user has actually hit:
- an inverted / mis-normalized G(r) whose baseline does not fall as -4*pi*rho*r
  (the "inverted neutron .gr" mystery);
- a NOMAD "SofQ" file that stores S(Q)-1 under an S(Q) name.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scattering_ai.core.findings import Finding, Severity
from scattering_ai.tools.curves import crop, detect_sq_convention, find_peaks
from scattering_ai.tools.io import load_curve
from scattering_ai.tools.models import Curve1D

# Shortest physical interatomic distance; |G(r)| below this is an artifact
# (termination ripple / bad normalization), not a bond.
MIN_BOND_R = 1.0  # Angstrom
# Low-r |G| above this fraction of the G(r) amplitude is flagged.
LOW_R_ARTIFACT_FRACTION = 0.2
# S(Q)-name hints: a file called S(Q) whose tail -> 0 really holds S(Q)-1.
_SQ_NAME_HINTS = ("sofq", "s(q)", "sq.dat", "_sq", "s_q")


def is_r_space(curve: Curve1D) -> bool:
    """Is this a real-space PDF G(r) (vs reciprocal-space S(Q)/F(Q))?"""
    label = curve.xlabel.strip().lower()
    if label.startswith("r"):
        return True
    if label.startswith("q") or label.startswith("k"):
        return False
    # Fallback: PDFs start near r=0 and rarely extend past ~100; a Q grid that
    # looks like S(Q) (tail -> 0 or 1) is reciprocal space.
    return detect_sq_convention(curve) == "unknown"


def check_low_r_artifact(curve: Curve1D) -> list[Finding]:
    x, y = curve.x, curve.y
    phys = np.isfinite(y) & (x >= MIN_BOND_R)
    low = np.isfinite(y) & (x > 0) & (x < MIN_BOND_R)
    if not phys.any() or not low.any():
        return []
    amplitude = float(np.max(np.abs(y[phys])))
    if amplitude <= 0:
        return []
    i_low = np.argmax(np.abs(y[low]))
    max_low = float(np.abs(y[low])[i_low])
    r_at_max = float(x[low][i_low])
    ratio = max_low / amplitude
    if ratio <= LOW_R_ARTIFACT_FRACTION:
        return []
    return [
        Finding(
            diagnostic="pdf_low_r_artifact",
            severity=Severity.WARNING,
            message=f"Significant structure below {MIN_BOND_R:g} Å "
            f"(|G|={max_low:.3g} at r={r_at_max:.3g} Å, {ratio:.0%} of the G(r) "
            "amplitude) — likely termination ripple or a normalization error, "
            "not a bond.",
            evidence={"r_min": MIN_BOND_R, "max_abs_G_below_r_min": round(max_low, 4),
                      "r_at_max": round(r_at_max, 4), "amplitude": round(amplitude, 4),
                      "ratio": round(ratio, 4)},
        )
    ]


def check_baseline_slope(curve: Curve1D) -> list[Finding]:
    """Near the origin a physical G(r) falls as -4*pi*rho*r (negative slope).

    A non-negative slope means the file is not standard G(r): it may be an RDF
    g(r), a differential PDF, or sign-inverted.
    """
    peaks = _physical_peaks(curve)
    r_hi = min(peaks[0]["x"] * 0.9, 1.5) if peaks else 1.5
    region = np.isfinite(curve.y) & (curve.x > 0.1) & (curve.x <= r_hi)
    if region.sum() < 4:
        return []
    slope = float(np.polyfit(curve.x[region], curve.y[region], 1)[0])
    if slope < 0:
        return [
            Finding(
                diagnostic="pdf_baseline_ok",
                severity=Severity.INFO,
                message=f"G(r) baseline falls with a negative slope near the origin "
                f"({slope:+.3g}), consistent with the -4πρr limit of a standard PDF.",
                evidence={"slope": round(slope, 4), "r_range": [0.1, round(r_hi, 3)]},
            )
        ]
    return [
        Finding(
            diagnostic="pdf_baseline_slope",
            severity=Severity.WARNING,
            message=f"G(r) does not fall as -4πρr near the origin (slope {slope:+.3g} "
            "is not negative); the file may store an RDF g(r), a differential PDF, "
            "or a sign-inverted G(r) rather than standard G(r).",
            evidence={"slope": round(slope, 4), "r_range": [0.1, round(r_hi, 3)]},
        )
    ]


def _physical_peaks(curve: Curve1D) -> list[dict]:
    phys = crop(curve, MIN_BOND_R, None)
    if phys.x.size < 5:
        return []
    return find_peaks(phys, subtract_background=False)


def check_first_peak(curve: Curve1D) -> list[Finding]:
    peaks = _physical_peaks(curve)
    if not peaks:
        return []
    first = min(peaks, key=lambda p: p["x"])
    return [
        Finding(
            diagnostic="pdf_first_peak",
            severity=Severity.INFO,
            message=f"First PDF peak at r = {first['x']:.3g} Å "
            "(nearest-neighbour distance candidate).",
            evidence={"r": round(first["x"], 4), "height": round(first["y"], 4),
                      "fwhm_estimate": round(first["fwhm_estimate"], 4)},
        )
    ]


def check_sq_convention(curve: Curve1D, file_name: str) -> list[Finding]:
    conv = detect_sq_convention(curve)
    tail = curve.y[np.isfinite(curve.y) & (curve.x >= 0.8 * curve.x.max())]
    tail_mean = float(tail.mean()) if tail.size else float("nan")
    name = file_name.lower()
    looks_like_sq = any(h in name for h in _SQ_NAME_HINTS)
    if conv == "sq_minus_1" and looks_like_sq:
        return [
            Finding(
                diagnostic="sq_convention_mismatch",
                severity=Severity.WARNING,
                message=f"'{Path(file_name).name}' is named like S(Q) but its high-Q "
                f"tail → {tail_mean:.3g} (≈0), so it stores S(Q)-1 (a common NOMAD "
                "convention). Transform with input_kind='sq_minus_1', not 'sq'.",
                evidence={"detected_convention": conv, "high_q_tail_mean": round(tail_mean, 4)},
            )
        ]
    return [
        Finding(
            diagnostic="sq_convention",
            severity=Severity.INFO,
            message=f"S(Q) convention detected as '{conv}' (high-Q tail ≈ {tail_mean:.3g}).",
            evidence={"detected_convention": conv, "high_q_tail_mean": round(tail_mean, 4)},
        )
    ]


def check_q_range(curve: Curve1D) -> list[Finding]:
    q = curve.x[np.isfinite(curve.x)]
    if q.size < 2:
        return []
    qmax = float(q.max())
    ripple = 2 * np.pi / qmax if qmax > 0 else float("nan")
    return [
        Finding(
            diagnostic="sq_q_range",
            severity=Severity.INFO,
            message=f"Q from {float(q.min()):.3g} to {qmax:.3g} Å⁻¹; Qmax sets the "
            f"termination-ripple period ≈ 2π/Qmax = {ripple:.3g} Å in the PDF.",
            evidence={"qmin": round(float(q.min()), 4), "qmax": round(qmax, 4),
                      "ripple_period": round(ripple, 4)},
        )
    ]


def diagnose_file(path: str) -> list[Finding]:
    if not Path(path).exists():
        return [Finding(diagnostic="missing_files", severity=Severity.ERROR,
                        message=f"Input file not found: {path}", evidence={"path": path})]
    try:
        curve = load_curve(path)
    except Exception as exc:
        return [Finding(diagnostic="pdf_unreadable", severity=Severity.ERROR,
                        message=f"Could not read {Path(path).name}: {type(exc).__name__}: {exc}",
                        evidence={"path": path})]

    if is_r_space(curve):
        return (check_baseline_slope(curve) + check_low_r_artifact(curve)
                + check_first_peak(curve))
    return check_sq_convention(curve, path) + check_q_range(curve)


def _summary_figure(files: list[str], workspace) -> list[Finding]:
    """Overview plot of the first readable curve, peaks marked — the figure the
    interpretation points at."""
    if workspace is None:
        return []
    for path in files:
        try:
            curve = load_curve(path)
        except Exception:
            continue
        try:
            from pathlib import Path

            from scattering_ai.tools.plotting import plot_curve

            peaks = find_peaks(curve, subtract_background=not is_r_space(curve))
            out = Path(workspace) / "pdf_overview.png"
            saved = plot_curve(curve, out, peaks=peaks, logy=not is_r_space(curve))
        except ImportError:
            return []
        except Exception:
            return []
        kind = "G(r)" if is_r_space(curve) else "S(Q)/F(Q)"
        return [Finding(
            diagnostic="pdf_summary_figure", severity=Severity.INFO,
            message=f"Overview plot of the {kind} with detected peaks marked.",
            evidence={"figures": [saved]},
        )]
    return []


def _model_comparison(files: list[str], workspace) -> list[Finding]:
    """When a CIF accompanies a measured G(r), fit the structure model to it
    (scale, broadening, lattice scale) and report Rw with an overlay figure."""
    cifs = [f for f in files if f.lower().endswith((".cif", ".mcif"))]
    grs = []
    for f in files:
        if f.lower().endswith((".cif", ".mcif")) or not Path(f).exists():
            continue
        try:
            curve = load_curve(f)
        except Exception:
            continue
        if is_r_space(curve):
            grs.append((f, curve))
    if not cifs or not grs:
        return []
    try:
        from scattering_ai.tools.cif import read_structure
        from scattering_ai.tools.gr_model import fit_gr

        s = read_structure(cifs[0])
        gr_path, curve = grs[0]
        fit = fit_gr(curve.x, curve.y, s["lattice"], s["positions"], s["species"])
        if "error" in fit:
            return []
        figures = []
        if workspace is not None:
            try:
                from scattering_ai.tools.plotting import plot_gr_fit

                figures.append(plot_gr_fit(fit, Path(workspace) / "gr_model_fit.png"))
            except ImportError:
                pass
        severity = Severity.INFO if fit["rw"] < 0.4 else Severity.WARNING
        return [Finding(
            diagnostic="pdf_model_fit", severity=severity,
            message=f"Structure model {Path(cifs[0]).name} vs {Path(gr_path).name}: "
            f"Rw = {fit['rw']:.3f} ({fit['assessment']}), peak width "
            f"{fit['sigma']:.3f} Å, lattice scale ×{fit['lattice_scale']:.4f}.",
            evidence={k: fit[k] for k in ("rw", "scale", "sigma", "lattice_scale",
                                          "fit_range", "assessment")} | {"figures": figures},
        )]
    except Exception as exc:
        return [Finding(diagnostic="pdf_model_fit", severity=Severity.WARNING,
                        message=f"Model comparison failed: {type(exc).__name__}: {exc}",
                        evidence={})]


def run_all(files: list[str], workspace=None) -> list[Finding]:
    findings: list[Finding] = []
    for f in files:
        if f.lower().endswith((".cif", ".mcif")):
            continue  # the CIF is the model, not data to diagnose
        findings.extend(diagnose_file(f))
    findings.extend(_model_comparison(files, workspace))
    findings.extend(_summary_figure(files, workspace))
    return findings


# Deterministic next-check rules for PDF findings (decision D10: technique
# content lives on the domain pack, not in core).
NEXT_CHECK_RULES: dict[str, str] = {
    "pdf_low_r_artifact": "Check the S(Q) normalization, background subtraction, and "
    "Qmin; strong low-r structure usually means an over-/under-subtracted background "
    "or a wrong number-density scale.",
    "pdf_baseline_slope": "Confirm the file is a standard G(r) (pdfgetx/PDFgui "
    "convention) and not an RDF g(r) or a sign-flipped export; compare its low-r "
    "slope against -4πρ from the known number density.",
    "sq_convention_mismatch": "Re-run the S(Q)→G(r) transform with input_kind="
    "'sq_minus_1' and verify the resulting G(r) baseline and first-peak position.",
    "pdf_unreadable": "Confirm the file format and column layout; the reader "
    "expected a two-column ASCII or pdfgetx .gr/.fq file.",
    "pdf_model_fit": "If Rw is poor, check the radiation type (neutron vs x-ray "
    "weighting), the fit range, and whether the structure is the right phase; "
    "for full refinement (ADPs, occupancies) move to PDFgui/diffpy-CMI.",
    "missing_files": "Locate or regenerate the missing files before trusting the analysis.",
}
