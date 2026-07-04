"""RMCProfile file readers: the ``.rmc6f`` configuration and R-value CSV logs.

Adapted from the MIT-licensed rmc-toolkits (github.com/drthyang/rmc-toolkits) by
the same author, so the SDK can ingest real RMCProfile output — the average unit
cell, supercell, composition, and refinement histories — instead of only a
pre-digested JSON summary.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from scattering_ai.tools.symmetry import _cell_params


def read_rmc6f(path: str | Path) -> dict[str, Any]:
    """Read an RMCProfile ``.rmc6f`` configuration.

    Returns the average **unit cell** (lattice vectors / supercell), the
    supercell multipliers, the composition, and every atom (element + its
    fractional coordinate within its unit cell). RMC configurations are
    disordered snapshots, so the folded positions describe an average site
    occupancy, not an idealized structure.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    supercell: np.ndarray | None = None
    lattice_vectors: np.ndarray | None = None
    title = ""
    atoms_start = None
    for i, line in enumerate(text):
        parts = line.split()
        if not parts:
            continue
        low = line.lower()
        if low.startswith("title") or low.startswith("metadata material"):
            title = line.split(":", 1)[-1].strip() or title
        elif parts[0] == "Supercell" or low.startswith("supercell dimensions"):
            supercell = np.asarray(parts[-3:], dtype=float)
        elif parts[0] == "Lattice" or low.startswith("cell (ang/deg)"):
            if parts[0] == "Lattice":
                lattice_vectors = np.asarray(
                    [text[i + 1].split(), text[i + 2].split(), text[i + 3].split()],
                    dtype=float)
        elif parts[0] == "Atoms:":
            atoms_start = i + 1
            break

    if supercell is None or lattice_vectors is None or atoms_start is None:
        raise ValueError(f"{path} is not a complete .rmc6f (need Supercell, "
                         "Lattice vectors, and an Atoms: section)")

    unit_vectors = lattice_vectors / supercell[:, None]
    elements: list[str] = []
    positions: list[list[float]] = []
    for line in text[atoms_start:]:
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            coords = np.asarray(parts[3:6], dtype=float)
        except ValueError:
            continue
        elements.append(parts[1].capitalize())
        positions.append(list(np.mod(coords * supercell, 1.0)))  # fold into unit cell

    if not elements:
        raise ValueError(f"{path} has no atoms in its Atoms: section")

    return {
        "title": title,
        "cell": _cell_params(unit_vectors),
        "supercell": [int(round(s)) for s in supercell],
        "n_atoms": len(elements),
        "composition": dict(sorted(Counter(elements).items())),
        "elements": elements,
        "positions": positions,
        "lattice_vectors": lattice_vectors.round(6).tolist(),
    }


def rmc_density_slab(path: str | Path, axis: int = 2, center: float = 0.5,
                     thickness: float = 0.25, element: str | None = None,
                     grid: int = 100) -> dict[str, Any]:
    """Kernel-density map of an RMC configuration folded into the average unit
    cell — a slab perpendicular to ``axis`` projected onto the other two axes.
    Reveals split sites and local disorder that the average structure hides.
    (KDE approach adapted from the MIT-licensed rmc-toolkits.)"""
    from scipy.stats import gaussian_kde

    r = read_rmc6f(path)
    pos = np.asarray(r["positions"], dtype=float)
    if element is not None:
        keep = [i for i, e in enumerate(r["elements"]) if e == element]
        if not keep:
            return {"error": f"no '{element}' atoms; have {list(r['composition'])}"}
        pos = pos[keep]
    # periodic slab selection along the chosen axis
    dist = np.abs(((pos[:, axis] - center + 0.5) % 1.0) - 0.5)
    slab = pos[dist <= thickness / 2]
    other = [i for i in range(3) if i != axis]
    if len(slab) < 5:
        return {"error": f"only {len(slab)} atoms in the slab; widen thickness"}
    kde = gaussian_kde(slab[:, other].T)
    lin = np.linspace(0.0, 1.0, grid)
    xx, yy = np.meshgrid(lin, lin)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(grid, grid)
    axis_names = ["a", "b", "c"]
    return {
        "density": density,
        "x_centers": lin,
        "y_centers": lin,
        "xlabel": f"{axis_names[other[0]]} (frac)",
        "ylabel": f"{axis_names[other[1]]} (frac)",
        "n_points": int(len(slab)),
        "element": element or "all",
        "slab": {"axis": axis_names[axis], "center": center, "thickness": thickness},
    }


def read_rmc_csv(path: str | Path) -> dict[str, Any]:
    """Read an RMCProfile CSV log (e.g. R-values / chi² vs step) into named
    columns."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path} is empty")
    labels = [c.strip() for c in lines[0].split(",")]
    rows = []
    for line in lines[1:]:
        vals = [v.strip() for v in line.split(",") if v.strip()]
        if len(vals) == len(labels):
            try:
                rows.append([float(v) for v in vals])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"{path} has no numeric rows")
    data = np.asarray(rows, dtype=float)
    return {"labels": labels, "columns": {labels[i]: data[:, i].tolist()
                                          for i in range(len(labels))},
            "n_rows": len(rows)}
