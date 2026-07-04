"""Magnetic pair distribution function (mPDF) from an ordered spin structure.

The mPDF (Frandsen, Yang & Billinge, Acta Cryst. A70 (2014) 3) is the
real-space counterpart of magnetic neutron scattering: for spin pairs at
separation r_ij with unit separation vector r̂,

    f(r) = C/N · Σ_{i≠j} [ (A_ij / r) δ(r − r_ij) + B_ij · r / r_ij³ · Θ(r_ij − r) ]

with the longitudinal/transverse decomposition (a_i = S_i·r̂, t_i = S_i − a_i r̂)

    A_ij = t_i · t_j          (transverse correlation → delta peaks)
    B_ij = 2 a_i a_j − t_i · t_j   (the slowly varying r < r_ij tail)

so an antiferromagnetic nearest-neighbour pair gives a **negative** first peak
and a ferromagnetic one a positive peak. This computes the ideal (unnormalized
scale) mPDF of a given moment arrangement — quantitative refinement against
data, with form factors and paramagnetic corrections, is diffpy.mpdf territory.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def simulate_mpdf(lattice, positions, spins, rmax: float = 20.0,
                  rstep: float = 0.02, sigma: float = 0.1) -> dict[str, Any]:
    """Ideal mPDF f(r) of an ordered magnetic structure.

    ``spins``: one (3,) moment vector per atom (crystal-axis components, as an
    mCIF stores them); atoms with zero moment are ignored. ``sigma`` broadens
    the delta contributions (Gaussian, Å).
    """
    from scattering_ai.tools.symmetry import cell_matrix

    lattice = np.asarray(lattice, dtype=float)
    cell = lattice if lattice.shape == (3, 3) else cell_matrix(*lattice)
    frac = np.asarray(positions, dtype=float)
    spin_frac = np.asarray(spins, dtype=float)
    if spin_frac.shape != frac.shape:
        raise ValueError("spins must be an (N, 3) array matching positions")
    # moments to Cartesian (crystal-axis components scale with the basis)
    spin_cart = spin_frac @ (cell / np.linalg.norm(cell, axis=1)[:, None])

    magnetic = np.linalg.norm(spin_cart, axis=1) > 1e-8
    if magnetic.sum() < 2:
        return {"error": "need at least two atoms with non-zero moments"}
    frac, spin_cart = frac[magnetic], spin_cart[magnetic]
    n_mag = len(frac)

    n_rep = [int(np.ceil(rmax / np.linalg.norm(cell[i]))) + 1 for i in range(3)]
    shifts = np.array([[i, j, k]
                       for i in range(-n_rep[0], n_rep[0] + 1)
                       for j in range(-n_rep[1], n_rep[1] + 1)
                       for k in range(-n_rep[2], n_rep[2] + 1)], dtype=float)
    cart0 = frac @ cell
    all_cart = (frac[None, :, :] + shifts[:, None, :]).reshape(-1, 3) @ cell
    all_spin = np.tile(spin_cart, (len(shifts), 1))

    pad = 5 * sigma + rstep
    r_ext = np.arange(rstep, rmax + pad + rstep, rstep)
    delta_part = np.zeros_like(r_ext)
    tail_part = np.zeros_like(r_ext)
    for i in range(n_mag):
        vec = all_cart - cart0[i]
        d = np.linalg.norm(vec, axis=1)
        keep = (d > 1e-6) & (d < rmax + pad)
        if not keep.any():
            continue
        vec, d = vec[keep], d[keep]
        s_j = all_spin[keep]
        r_hat = vec / d[:, None]
        a_i = r_hat @ spin_cart[i]                # longitudinal components
        a_j = np.einsum("ij,ij->i", r_hat, s_j)
        t_dot = s_j @ spin_cart[i] - a_i * a_j    # t_i·t_j
        A = t_dot
        B = 2 * a_i * a_j - t_dot
        idx = np.clip(((d - rstep) / rstep).round().astype(int), 0, r_ext.size - 1)
        np.add.at(delta_part, idx, A / np.maximum(d, 1e-6))
        # B tail: each pair adds B/r_ij^3 to every bin below its own —
        # a suffix sum over per-bin buckets (vectorized step profile)
        buckets = np.zeros_like(tail_part)
        np.add.at(buckets, idx, B / d**3)
        suffix = np.cumsum(buckets[::-1])[::-1]
        tail_part += suffix - buckets             # strictly r < r_ij

    if sigma > 0:
        half = int(np.ceil(4 * sigma / rstep))
        kx = np.arange(-half, half + 1) * rstep
        kernel = np.exp(-0.5 * (kx / sigma) ** 2)
        kernel /= kernel.sum()
        delta_part = np.convolve(delta_part, kernel, mode="same")

    n_keep = int(round(rmax / rstep))
    r = r_ext[:n_keep]
    f = (delta_part[:n_keep] / rstep + tail_part[:n_keep] * r) / n_mag
    return {"r": r, "f": f, "n_magnetic": n_mag, "sigma": sigma,
            "note": "ideal mPDF (arbitrary scale); negative peaks = "
            "antiferromagnetically correlated pair distances"}


def spin_correlations(lattice, positions, spins, rmax: float = 12.0,
                      shell_tol: float = 0.05) -> dict[str, Any]:
    """Normalized spin-pair correlations ⟨Ŝ_i·Ŝ_j⟩ per neighbour shell.

    The real-space fingerprint of a magnetic configuration (what
    spinvert-style analyses report): +1 = shell fully ferromagnetically
    correlated, −1 = antiferromagnetic, 0 = uncorrelated. Works for ordered
    (mCIF) or disordered (RMC-style) spin sets.
    """
    from scattering_ai.tools.symmetry import cell_matrix

    lattice = np.asarray(lattice, dtype=float)
    cell = lattice if lattice.shape == (3, 3) else cell_matrix(*lattice)
    frac = np.asarray(positions, dtype=float)
    spin = np.asarray(spins, dtype=float) @ (cell / np.linalg.norm(cell, axis=1)[:, None])
    norms = np.linalg.norm(spin, axis=1)
    magnetic = norms > 1e-8
    if magnetic.sum() < 2:
        return {"error": "need at least two atoms with non-zero moments"}
    frac, unit = frac[magnetic], spin[magnetic] / norms[magnetic][:, None]
    n_mag = len(frac)

    n_rep = [int(np.ceil(rmax / np.linalg.norm(cell[i]))) + 1 for i in range(3)]
    shifts = np.array([[i, j, k]
                       for i in range(-n_rep[0], n_rep[0] + 1)
                       for j in range(-n_rep[1], n_rep[1] + 1)
                       for k in range(-n_rep[2], n_rep[2] + 1)], dtype=float)
    cart0 = frac @ cell
    all_cart = (frac[None, :, :] + shifts[:, None, :]).reshape(-1, 3) @ cell
    all_unit = np.tile(unit, (len(shifts), 1))

    dists: list[float] = []
    dots: list[float] = []
    for i in range(n_mag):
        d = np.linalg.norm(all_cart - cart0[i], axis=1)
        keep = (d > 1e-6) & (d < rmax)
        dists += list(d[keep])
        dots += list(all_unit[keep] @ unit[i])
    dists = np.asarray(dists)
    dots = np.asarray(dots)

    order = np.argsort(dists)
    shells: list[dict[str, Any]] = []
    start = 0
    ds, cs = dists[order], dots[order]
    for k in range(1, len(ds) + 1):
        if k == len(ds) or ds[k] - ds[start] > shell_tol:
            shells.append({
                "r": round(float(ds[start:k].mean()), 4),
                "correlation": round(float(cs[start:k].mean()), 4),
                "multiplicity": round((k - start) / n_mag, 2),
            })
            start = k
    return {"n_magnetic": n_mag, "shells": shells[:30],
            "note": "⟨Ŝ_i·Ŝ_j⟩ per shell: +1 FM, −1 AFM, 0 uncorrelated"}


# <j0> magnetic form-factor coefficients (A, a, B, b, C, c, D) — the dipole
# approximation f(s) = A e^{-a s²}+B e^{-b s²}+C e^{-c s²}+D, s = Q/4π (Å⁻¹).
# Standard values (International Tables Vol. C / P. J. Brown); A+B+C+D ≈ 1.
_J0_FF = {
    "Mn2": (0.4220, 17.684, 0.5948, 6.005, 0.0043, -0.609, -0.0219),
    "Mn3": (0.4198, 14.283, 0.6054, 5.469, 0.9241, -0.0088, -0.9498),
    "Mn4": (0.3760, 12.566, 0.6602, 5.133, -0.0372, 0.563, 0.0011),
    "Fe2": (0.0263, 34.960, 0.3668, 15.943, 0.6188, 5.594, -0.0119),
    "Fe3": (0.3972, 13.244, 0.6295, 4.903, -0.0314, 0.350, 0.0044),
    "Co2": (0.4332, 14.355, 0.5857, 4.608, -0.0382, 0.134, 0.0179),
    "Ni2": (0.0163, 35.883, 0.3916, 13.223, 0.6052, 4.339, -0.0133),
    "Cu2": (0.0232, 34.969, 0.4023, 11.564, 0.5882, 3.843, -0.0137),
    "Cr3": (0.3025, 20.548, 0.5875, 6.145, 0.0075, -0.407, -0.0300),
    "V3": (0.4085, 23.853, 0.6091, 8.246, -0.1676, 9.087, 0.1496),
    "Gd3": (0.0186, 25.387, 0.2895, 11.142, 0.7135, 3.752, -0.0217),
    "Tb3": (0.0177, 25.510, 0.2921, 10.577, 0.7133, 3.512, -0.0231),
}
# element -> a common magnetic valence key when the ion is not given explicitly
_DEFAULT_ION = {"Mn": "Mn2", "Fe": "Fe3", "Co": "Co2", "Ni": "Ni2", "Cu": "Cu2",
                "Cr": "Cr3", "V": "V3", "Gd": "Gd3", "Tb": "Tb3"}


def magnetic_form_factor(q: np.ndarray, ion: str | None) -> np.ndarray:
    """<j0> magnetic form factor f(Q); 1.0 (point dipole) for an unknown ion."""
    coeffs = _J0_FF.get(ion or "")
    if coeffs is None:
        return np.ones_like(np.asarray(q, dtype=float))
    a1, a2, b1, b2, c1, c2, d = coeffs
    s2 = (np.asarray(q, dtype=float) / (4 * np.pi)) ** 2
    return a1 * np.exp(-a2 * s2) + b1 * np.exp(-b2 * s2) + c1 * np.exp(-c2 * s2) + d


def _j0(x: np.ndarray) -> np.ndarray:
    return np.where(x < 1e-8, 1.0, np.sin(x) / np.where(x < 1e-8, 1.0, x))


def _j2(x: np.ndarray) -> np.ndarray:
    xx = np.where(x < 1e-4, 1.0, x)
    val = (3 / xx**3 - 1 / xx) * np.sin(xx) - (3 / xx**2) * np.cos(xx)
    return np.where(x < 1e-4, 0.0, val)


def powder_magnetic_iq(lattice, positions, spins, qmin: float = 0.1,
                       qmax: float = 6.0, qstep: float = 0.02,
                       ion: str | None = None, species: list | None = None,
                       rmax: float = 15.0) -> dict[str, Any]:
    """Powder-averaged magnetic diffuse scattering I(Q) from a spin structure.

    Blech–Averbach spherical average of |M_⊥(Q)|²:

        I(Q) = f(Q)² Σ_{i,j} [ (2/3) S_i·S_j j0(Qr) +
                               (1/3)(3(S_i·r̂)(S_j·r̂) − S_i·S_j) j2(Qr) ]

    (self term (2/3)|S_i|²). Guaranteed non-negative. This is the forward
    calculation at the heart of Scatty/spinvert; ``ion`` (or ``species``)
    selects the <j0> magnetic form factor.
    """
    from scattering_ai.tools.symmetry import cell_matrix

    lattice = np.asarray(lattice, dtype=float)
    cell = lattice if lattice.shape == (3, 3) else cell_matrix(*lattice)
    frac = np.asarray(positions, dtype=float)
    spin = np.asarray(spins, dtype=float) @ (cell / np.linalg.norm(cell, axis=1)[:, None])
    mag = np.linalg.norm(spin, axis=1) > 1e-8
    if mag.sum() < 1:
        return {"error": "no atoms with non-zero moments"}
    frac, spin = frac[mag], spin[mag]
    n_mag = len(frac)
    if ion is None and species is not None:
        elems = [s for s, m in zip(species, mag, strict=False) if m]
        ion = _DEFAULT_ION.get(elems[0]) if elems else None

    n_rep = [int(np.ceil(rmax / np.linalg.norm(cell[i]))) + 1 for i in range(3)]
    shifts = np.array([[i, j, k]
                       for i in range(-n_rep[0], n_rep[0] + 1)
                       for j in range(-n_rep[1], n_rep[1] + 1)
                       for k in range(-n_rep[2], n_rep[2] + 1)], dtype=float)
    cart0 = frac @ cell
    all_cart = (frac[None, :, :] + shifts[:, None, :]).reshape(-1, 3) @ cell
    all_spin = np.tile(spin, (len(shifts), 1))

    q = np.arange(qmin, qmax + qstep, qstep)
    pair_iq = np.zeros_like(q)
    for i in range(n_mag):
        vec = all_cart - cart0[i]
        d = np.linalg.norm(vec, axis=1)
        keep = (d > 1e-6) & (d < rmax)
        if not keep.any():
            continue
        vec, d, s_j = vec[keep], d[keep], all_spin[keep]
        r_hat = vec / d[:, None]
        p = s_j @ spin[i]                                     # S_i·S_j
        a_lon = (r_hat @ spin[i]) * np.einsum("ij,ij->i", r_hat, s_j)  # (S_i·r̂)(S_j·r̂)
        c0 = (2.0 / 3.0) * p
        c2 = (1.0 / 3.0) * (3 * a_lon - p)
        qd = np.outer(q, d)                                   # (nq, npair)
        pair_iq += _j0(qd) @ c0 + _j2(qd) @ c2

    self_term = (2.0 / 3.0) * float(np.sum(np.einsum("ij,ij->i", spin, spin)))
    ff = magnetic_form_factor(q, ion)
    intensity = ff**2 * (self_term + pair_iq) / n_mag
    return {"q": q, "i": np.clip(intensity, 0.0, None), "n_magnetic": n_mag,
            "ion": ion, "form_factor": "point dipole" if ion is None or ion not in _J0_FF
            else f"<j0> {ion}",
            "note": "powder magnetic diffuse I(Q); ordered structures give "
            "broadened magnetic Bragg peaks (broadening set by rmax)"}


# High-symmetry propagation-vector candidates scanned when k is not given.
_K_CANDIDATES = [
    (0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5),
    (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.5, 0.5, 0.5),
    (1.0, 1.0, 1.0), (1.0, 0.0, 0.0),
]


def kvector_consistency(lattice, positions, spins, k=None, rmax: float = 10.0,
                        shell_tol: float = 0.05) -> dict[str, Any]:
    """Test whether shell correlations match a propagation vector's ideal
    pattern cos(2π k·ΔR) — or find the best-matching high-symmetry k.

    A single-k (collinear) magnetic structure has ⟨Ŝ_i·Ŝ_j⟩ = cos(2π k·ΔR_frac)
    for every pair. Ordered structures match one k almost exactly; geometrically
    frustrated / short-range-ordered configurations match NO k and their
    correlations decay with distance — the signature this flags.
    """
    from scattering_ai.tools.symmetry import cell_matrix

    lattice = np.asarray(lattice, dtype=float)
    cell = lattice if lattice.shape == (3, 3) else cell_matrix(*lattice)
    frac = np.asarray(positions, dtype=float)
    spin = np.asarray(spins, dtype=float) @ (cell / np.linalg.norm(cell, axis=1)[:, None])
    norms = np.linalg.norm(spin, axis=1)
    magnetic = norms > 1e-8
    if magnetic.sum() < 2:
        return {"error": "need at least two atoms with non-zero moments"}
    frac, unit = frac[magnetic], spin[magnetic] / norms[magnetic][:, None]
    n_mag = len(frac)

    n_rep = [int(np.ceil(rmax / np.linalg.norm(cell[i]))) + 1 for i in range(3)]
    shifts = np.array([[i, j, kk]
                       for i in range(-n_rep[0], n_rep[0] + 1)
                       for j in range(-n_rep[1], n_rep[1] + 1)
                       for kk in range(-n_rep[2], n_rep[2] + 1)], dtype=float)
    all_frac = (frac[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    all_cart = all_frac @ cell
    all_unit = np.tile(unit, (len(shifts), 1))
    cart0 = frac @ cell

    dists, dots, dfrac = [], [], []
    for i in range(n_mag):
        d = np.linalg.norm(all_cart - cart0[i], axis=1)
        keep = (d > 1e-6) & (d < rmax)
        dists += list(d[keep])
        dots += list(all_unit[keep] @ unit[i])
        dfrac += list(all_frac[keep] - frac[i])
    dists, dots, dfrac = np.asarray(dists), np.asarray(dots), np.asarray(dfrac)
    order = np.argsort(dists)
    dists, dots, dfrac = dists[order], dots[order], dfrac[order]

    # shell boundaries
    bounds = [0]
    for m in range(1, len(dists)):
        if dists[m] - dists[bounds[-1]] > shell_tol:
            bounds.append(m)
    bounds.append(len(dists))

    def score(kvec) -> tuple[float, list[dict[str, Any]]]:
        ideal_pair = np.cos(2 * np.pi * (dfrac @ np.asarray(kvec, dtype=float)))
        shells, sq = [], []
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            measured = float(dots[a:b].mean())
            ideal = float(ideal_pair[a:b].mean())
            shells.append({"r": round(float(dists[a:b].mean()), 4),
                           "measured": round(measured, 4), "ideal": round(ideal, 4)})
            sq.append((measured - ideal) ** 2)
        return float(np.sqrt(np.mean(sq))), shells

    if k is not None:
        rms, shells = score(k)
        best_k, best_rms, best_shells = tuple(k), rms, shells
        scanned = None
    else:
        scanned = []
        best_k, best_rms, best_shells = None, np.inf, []
        for cand in _K_CANDIDATES:
            rms, shells = score(cand)
            scanned.append({"k": list(cand), "rms": round(rms, 4)})
            if rms < best_rms:
                best_k, best_rms, best_shells = cand, rms, shells

    # decay of |correlation| with r — short-range signature
    absc = [abs(s["measured"]) for s in best_shells]
    n_half = max(len(absc) // 2, 1)
    decaying = bool(np.mean(absc[:n_half]) > 2 * np.mean(absc[n_half:]) + 0.05)
    ordered = best_rms < 0.15
    return {
        "k": list(best_k) if best_k is not None else None,
        "rms": round(best_rms, 4),
        "ordered": ordered,
        "short_range": bool(not ordered and decaying),
        "shells": best_shells[:12],
        "k_scan": scanned,
        "note": "ordered: shell correlations match cos(2πk·ΔR); short_range: no "
        "k fits and |correlation| decays with r (frustration signature)",
    }
