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
