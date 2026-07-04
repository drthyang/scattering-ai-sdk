"""PDF model calculation and fitting: G(r) from a crystal structure.

The community's single most-used PDF workflow (PDFgui / diffpy-CMI territory):
compute the reduced pair distribution function

    G(r) = R(r)/(N·r) − 4π·r·ρ0,   R(r) = Σ_{i≠j} (w_i w_j /⟨w⟩²) δ(r − r_ij)

from a structure (pair sums over a supercell, scattering-length weights,
Gaussian thermal broadening), and fit scale / peak width / a lattice-scale
factor against a measured G(r) with an Rw quality metric. This is a compact
independent implementation for model *comparison* — full structural refinement
(ADPs per site, shape functions, instrument damping) remains PDFgui's job.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Neutron coherent scattering lengths b (fm, Sears tables) for common elements;
# fall back to Z (x-ray-like weight) for anything missing.
NEUTRON_B = {
    "H": -3.74, "D": 6.67, "Li": -1.90, "B": 5.30, "C": 6.646, "N": 9.36,
    "O": 5.803, "F": 5.654, "Na": 3.63, "Mg": 5.375, "Al": 3.449, "Si": 4.149,
    "P": 5.13, "S": 2.847, "Cl": 9.577, "K": 3.67, "Ca": 4.70, "Ti": -3.438,
    "V": -0.382, "Cr": 3.635, "Mn": -3.73, "Fe": 9.45, "Co": 2.49, "Ni": 10.3,
    "Cu": 7.718, "Zn": 5.680, "Ga": 7.288, "Ge": 8.185, "As": 6.58, "Se": 7.970,
    "Br": 6.795, "Sr": 7.02, "Y": 7.75, "Zr": 7.16, "Nb": 7.054, "Mo": 6.715,
    "Ag": 5.922, "Cd": 4.87, "In": 4.065, "Sn": 6.225, "Sb": 5.57, "Te": 5.80,
    "I": 5.28, "Cs": 5.42, "Ba": 5.07, "La": 8.24, "Ce": 4.84, "Nd": 7.69,
    "Gd": 6.5, "Tb": 7.38, "Ta": 6.91, "W": 4.86, "Pt": 9.60, "Au": 7.63,
    "Pb": 9.405, "Bi": 8.532,
}
_Z = {  # atomic numbers for the x-ray weighting fallback
    "H": 1, "Li": 3, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Na": 11, "Mg": 12,
    "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "K": 19, "Ca": 20, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Sr": 38, "Y": 39, "Zr": 40,
    "Nb": 41, "Mo": 42, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52,
    "I": 53, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Nd": 60, "Gd": 64, "Tb": 65,
    "Ta": 73, "W": 74, "Pt": 78, "Au": 79, "Pb": 82, "Bi": 83,
}


def _weights(species: list[str], radiation: str) -> np.ndarray:
    if radiation == "neutron":
        return np.array([NEUTRON_B.get(s, float(_Z.get(s, 10))) for s in species])
    return np.array([float(_Z.get(s, 10)) for s in species])


def simulate_gr(lattice, positions, species, rmax: float = 20.0,
                rstep: float = 0.02, sigma: float = 0.1,
                radiation: str = "neutron") -> dict[str, Any]:
    """Compute the reduced PDF G(r) of a structure.

    ``sigma`` is a uniform Gaussian pair-broadening (thermal motion, Å);
    ``radiation`` picks neutron scattering lengths or x-ray Z weighting.
    """
    from scattering_ai.tools.symmetry import cell_matrix

    lattice = np.asarray(lattice, dtype=float)
    cell = lattice if lattice.shape == (3, 3) else cell_matrix(*lattice)
    frac = np.asarray(positions, dtype=float)
    n_atoms = len(frac)
    w = _weights(list(species), radiation)
    w_mean = float(np.mean(w))
    volume = float(abs(np.linalg.det(cell)))
    rho0 = n_atoms / volume

    # supercell big enough that every pair within rmax is included
    n_rep = [int(np.ceil(rmax / np.linalg.norm(cell[i]))) + 1 for i in range(3)]
    shifts = np.array([[i, j, k]
                       for i in range(-n_rep[0], n_rep[0] + 1)
                       for j in range(-n_rep[1], n_rep[1] + 1)
                       for k in range(-n_rep[2], n_rep[2] + 1)], dtype=float)
    cart0 = frac @ cell                                   # central-cell atoms
    all_cart = (frac[None, :, :] + shifts[:, None, :]).reshape(-1, 3) @ cell
    all_w = np.tile(w, len(shifts))

    # Internal axis extends past rmax by the kernel support so that (a) pairs
    # just beyond rmax are NOT piled into the last bin (a spurious edge spike)
    # and (b) the convolution has no edge droop inside the returned range.
    pad = 5 * sigma + rstep
    r_ext = np.arange(rstep, rmax + pad + rstep, rstep)
    hist = np.zeros_like(r_ext)
    for i in range(n_atoms):                              # pair histogram
        d = np.linalg.norm(all_cart - cart0[i], axis=1)
        keep = (d > 1e-6) & (d < rmax + pad)
        if not keep.any():
            continue
        idx = np.clip(((d[keep] - rstep) / rstep).round().astype(int), 0, r_ext.size - 1)
        np.add.at(hist, idx, w[i] * all_w[keep] / w_mean**2)

    # Gaussian broadening via convolution on the regular r grid
    if sigma > 0:
        half = int(np.ceil(4 * sigma / rstep))
        kx = np.arange(-half, half + 1) * rstep
        kernel = np.exp(-0.5 * (kx / sigma) ** 2)
        kernel /= kernel.sum()
        hist = np.convolve(hist, kernel, mode="same")

    n_keep = int(round(rmax / rstep))
    r_axis, hist = r_ext[:n_keep], hist[:n_keep]
    g = hist / (n_atoms * r_axis * rstep) - 4 * np.pi * r_axis * rho0
    return {"r": r_axis, "g": g, "rho0": rho0, "n_atoms": n_atoms,
            "radiation": radiation, "sigma": sigma}


def _rw(obs: np.ndarray, calc: np.ndarray) -> float:
    return float(np.sqrt(np.sum((obs - calc) ** 2) / max(np.sum(obs**2), 1e-300)))


def fit_gr(r_obs, g_obs, lattice, positions, species, radiation: str = "neutron",
           rmin: float = 1.0, rmax: float | None = None) -> dict[str, Any]:
    """Fit a structure model to a measured G(r): scale (analytic), Gaussian
    peak width sigma, and a lattice-scale factor (uniform expansion), with Rw.

    A comparison fit, not a refinement — it answers "does this model explain
    the data, and at what lattice scale/broadening?".
    """
    from scipy import optimize

    r_obs = np.asarray(r_obs, dtype=float)
    g_obs = np.asarray(g_obs, dtype=float)
    keep = np.isfinite(g_obs) & (r_obs >= rmin)
    if rmax is not None:
        keep &= r_obs <= rmax
    r_fit, g_fit = r_obs[keep], g_obs[keep]
    if r_fit.size < 20:
        return {"error": "too few points in the fit range"}
    r_top = float(r_fit.max())

    from scattering_ai.tools.symmetry import cell_matrix

    base_cell = np.asarray(lattice, dtype=float)
    base_cell = base_cell if base_cell.shape == (3, 3) else cell_matrix(*base_cell)

    def model_on(r_grid: np.ndarray, sigma: float, r_scale: float) -> np.ndarray:
        # simulate with the uniformly scaled lattice (not a stretched r-axis),
        # so the -4πρr baseline stays exactly self-consistent
        sim = simulate_gr(base_cell * r_scale, positions, species,
                          rmax=r_top + 1, rstep=0.02, sigma=sigma,
                          radiation=radiation)
        return np.interp(r_grid, sim["r"], sim["g"])

    def objective(theta) -> float:
        sigma, r_scale = theta
        if not (0.01 <= sigma <= 0.5 and 0.9 <= r_scale <= 1.1):
            return 1e6
        calc = model_on(r_fit, sigma, r_scale)
        scale = float(np.dot(g_fit, calc) / max(np.dot(calc, calc), 1e-300))
        return _rw(g_fit, scale * calc)

    res = optimize.minimize(objective, x0=[0.1, 1.0], method="Nelder-Mead",
                            options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 120})
    sigma, r_scale = res.x
    calc = model_on(r_fit, sigma, r_scale)
    scale = float(np.dot(g_fit, calc) / max(np.dot(calc, calc), 1e-300))
    rw = _rw(g_fit, scale * calc)
    return {
        "rw": round(rw, 4),
        "scale": round(scale, 4),
        "sigma": round(float(sigma), 4),
        "lattice_scale": round(float(r_scale), 5),
        "fit_range": [rmin, rmax or r_top],
        "n_points": int(r_fit.size),
        "radiation": radiation,
        "r": r_fit, "g_obs": g_fit, "g_calc": scale * calc,
        "assessment": "good" if rw < 0.2 else ("fair" if rw < 0.4 else "poor"),
        "note": "comparison fit (scale, uniform broadening, lattice scale); "
        "full refinement of ADPs/occupancies is PDFgui/diffpy territory",
    }
