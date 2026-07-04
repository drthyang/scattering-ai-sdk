"""Crystal-structure visualization (CIF / mCIF).

Renders the unit cell, atoms (element colours and sizes), bonds, and — for a
magnetic structure — the moment vectors, to a PNG. Uses only matplotlib (the
``plots`` extra), so there is no heavy 3D dependency; it is a quick, embeddable
figure for reports and chat, not a replacement for VESTA.
"""

from __future__ import annotations

import numpy as np

from scattering_ai.tools.symmetry import cell_matrix

# Compact CPK-like colours and covalent radii (Å) for common elements; anything
# else falls back to a palette colour and a default radius.
_COLORS = {
    "H": "#ffffff", "Li": "#cc80ff", "B": "#ffb5b5", "C": "#909090", "N": "#3050f8",
    "O": "#ff0d0d", "F": "#90e050", "Na": "#ab5cf2", "Mg": "#8aff00", "Al": "#bfa6a6",
    "Si": "#f0c8a0", "P": "#ff8000", "S": "#ffff30", "Cl": "#1ff01f", "K": "#8f40d4",
    "Ca": "#3dff00", "Ti": "#bfc2c7", "V": "#a6a6ab", "Cr": "#8a99c7", "Mn": "#9c7ac7",
    "Fe": "#e06633", "Co": "#f090a0", "Ni": "#50d050", "Cu": "#c88033", "Zn": "#7d80b0",
    "Ga": "#c28f8f", "Ge": "#668f8f", "As": "#bd80e3", "Se": "#ffa100", "Sr": "#00ff00",
    "Nb": "#73c2c9", "Mo": "#54b5b5", "Ba": "#00c900", "Ta": "#4da6ff", "W": "#2194d6",
    "Tb": "#30ffc7", "Bi": "#9e4fb5",
}
_RADII = {
    "H": 0.31, "Li": 1.28, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02,
    "K": 2.03, "Ca": 1.76, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39, "Fe": 1.32,
    "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20, "As": 1.19,
    "Se": 1.20, "Sr": 1.95, "Nb": 1.64, "Mo": 1.54, "Ba": 2.15, "Ta": 1.70, "W": 1.62,
    "Tb": 1.94, "Bi": 1.48,
}
_PALETTE = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860", "#da8bc3"]


def _cell(lattice) -> np.ndarray:
    lattice = np.asarray(lattice, dtype=float)
    return lattice if lattice.shape == (3, 3) else cell_matrix(*lattice)


def plot_structure(lattice, positions, species, out, moments=None,
                   bonds: bool = True, max_bond: float = 3.2,
                   view=(22, -60), title: str = "") -> str:
    """Render a structure to ``out`` (PNG). ``moments`` (N×3, crystal-axis
    components) draws moment arrows for a magnetic structure."""
    import os

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M = _cell(lattice)
    frac = np.asarray(positions, dtype=float)
    cart = frac @ M
    uniq = list(dict.fromkeys(species))
    color = {s: _COLORS.get(s, _PALETTE[i % len(_PALETTE)]) for i, s in enumerate(uniq)}
    radius = {s: _RADII.get(s, 1.4) for s in uniq}

    fig = plt.figure(figsize=(7, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    # unit-cell frame
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) @ M
    for a in range(8):
        for b in range(a + 1, 8):
            if bin(a ^ b).count("1") == 1:  # edges differ in one corner bit
                ax.plot(*zip(corners[a], corners[b], strict=True), color="#bbbbbb", lw=0.8)

    for s in uniq:
        idx = [i for i, sp in enumerate(species) if sp == s]
        ax.scatter(cart[idx, 0], cart[idx, 1], cart[idx, 2],
                   s=180 * radius[s], c=color[s], edgecolors="#333", linewidths=0.5,
                   depthshade=True, label=s)

    if bonds:
        for i in range(len(cart)):
            for j in range(i + 1, len(cart)):
                cutoff = min(1.25 * (radius[species[i]] + radius[species[j]]), max_bond)
                if np.linalg.norm(cart[i] - cart[j]) <= cutoff:
                    ax.plot(*zip(cart[i], cart[j], strict=True), color="#777", lw=1.3)

    if moments is not None:
        moments = np.asarray(moments, dtype=float)
        mcart = moments @ M
        scale = 0.55 * float(np.max(np.linalg.norm(M, axis=1))) / (
            np.max(np.linalg.norm(mcart, axis=1)) or 1.0)
        mag = np.linalg.norm(moments, axis=1) > 1e-6
        if mag.any():
            ax.quiver(cart[mag, 0], cart[mag, 1], cart[mag, 2],
                      mcart[mag, 0] * scale, mcart[mag, 1] * scale, mcart[mag, 2] * scale,
                      color="crimson", linewidth=2.0, arrow_length_ratio=0.25)

    ax.set_box_aspect(np.linalg.norm(M, axis=1))
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.set_title(title or "Crystal structure"
                 + ("" if moments is None else " (moments in red)"), fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out)
