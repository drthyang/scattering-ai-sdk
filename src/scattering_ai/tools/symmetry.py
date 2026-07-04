"""Crystallographic symmetry analysis (spglib-backed).

Deterministic symmetry tools for structural work:

- ``find_symmetry`` — FINDSYM-like: the space group, point group, and Wyckoff
  sites of a structure at a given tolerance.
- ``maximal_subgroups`` — the maximal translationengleiche subgroups of that
  space group (the group-subgroup steps a structural phase transition can take).
- ``pseudosymmetry_scan`` — raise the tolerance to find a higher-symmetry parent
  a slightly distorted structure sits under.
- ``magnetic_symmetry`` — the magnetic (Shubnikov) space group of an ordered
  magnetic structure.

Everything is computed by spglib (the standard library used across the field),
so results are trustworthy and reproducible. Needs the ``symmetry`` extra
(``pip install scattering-ai-sdk[symmetry]``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Space-group-number ranges -> crystal system (International Tables).
_SYSTEM_BOUNDS = [
    (2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"),
    (142, "tetragonal"), (167, "trigonal"), (194, "hexagonal"), (230, "cubic"),
]


def _spglib():
    try:
        import spglib

        return spglib
    except ImportError as exc:  # pragma: no cover - exercised via error path
        raise ImportError(
            "Symmetry analysis needs spglib: pip install scattering-ai-sdk[symmetry]"
        ) from exc


def crystal_system(number: int) -> str:
    for bound, name in _SYSTEM_BOUNDS:
        if number <= bound:
            return name
    return "unknown"


def cell_matrix(a: float, b: float, c: float,
                alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Real-space basis vectors (rows) from cell parameters (Å, degrees)."""
    al, be, ga = (math.radians(x) for x in (alpha, beta, gamma))
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return np.array([
        [a, 0.0, 0.0],
        [b * math.cos(ga), b * math.sin(ga), 0.0],
        [cx, cy, cz],
    ])


def _to_cell(lattice, positions, species):
    """Build an spglib cell (lattice, fractional positions, integer type ids).

    ``lattice`` is a 3x3 basis-vector matrix or ``(a, b, c, alpha, beta, gamma)``.
    spglib distinguishes atoms only by equality of the type id, so arbitrary
    distinct integers per species are enough — no element table needed.
    """
    lattice = np.asarray(lattice, dtype=float)
    if lattice.shape == (6,):
        lattice = cell_matrix(*lattice)
    elif lattice.shape != (3, 3):
        raise ValueError("lattice must be a 3x3 matrix or 6 cell parameters")
    order: dict[Any, int] = {}
    numbers = [order.setdefault(s, len(order)) for s in species]
    labels = {i: s for s, i in order.items()}
    return (lattice, np.asarray(positions, dtype=float), numbers), labels


def find_symmetry(lattice, positions, species, symprec: float = 1e-3,
                  angle_tolerance: float = -1.0) -> dict[str, Any]:
    """Detect the space group and Wyckoff sites of a structure (FINDSYM-like)."""
    spglib = _spglib()
    cell, labels = _to_cell(lattice, positions, species)
    ds = spglib.get_symmetry_dataset(cell, symprec=symprec,
                                     angle_tolerance=angle_tolerance)
    if ds is None:
        return {"error": f"spglib could not determine symmetry at symprec={symprec}"}

    wyckoffs = list(ds.wyckoffs)
    site_syms = list(ds.site_symmetry_symbols)
    orbits = list(ds.crystallographic_orbits)
    numbers = cell[2]
    sites: dict[int, dict] = {}
    for i, orbit in enumerate(orbits):
        site = sites.setdefault(orbit, {
            "wyckoff": wyckoffs[i], "site_symmetry": site_syms[i],
            "species": labels[numbers[i]], "multiplicity": 0,
        })
        site["multiplicity"] += 1
    return {
        "number": int(ds.number),
        "international": ds.international,
        "hall_number": int(ds.hall_number),
        "point_group": ds.pointgroup,
        "crystal_system": crystal_system(int(ds.number)),
        "n_operations": len(ds.rotations),
        "wyckoff_sites": sorted(sites.values(), key=lambda s: s["wyckoff"]),
        "symprec": symprec,
    }


# ------------------------------------------------------------- subgroups


def _multiplication_table(rotations: np.ndarray, translations: np.ndarray):
    n = len(rotations)

    def key(r, t):
        return (r.tobytes(), tuple(np.round(np.mod(t, 1.0), 4)))

    index = {key(rotations[i], translations[i]): i for i in range(n)}
    mult = np.empty((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            r = rotations[i] @ rotations[j]
            t = np.mod(rotations[i] @ translations[j] + translations[i], 1.0)
            mult[i, j] = index[key(r, t)]
    identity = next(
        i for i in range(n)
        if np.array_equal(rotations[i], np.eye(3, dtype=rotations.dtype))
        and np.allclose(np.mod(translations[i], 1.0), 0.0)
    )
    return mult, identity


def _closure(seed: set[int], mult: np.ndarray) -> frozenset[int]:
    group = set(seed)
    frontier = list(seed)
    while frontier:
        a = frontier.pop()
        for b in list(group):
            for x in (mult[a, b], mult[b, a]):
                if x not in group:
                    group.add(int(x))
                    frontier.append(int(x))
    return frozenset(group)


def _maximal_subgroups(mult: np.ndarray, identity: int, n: int) -> list[frozenset[int]]:
    """All maximal proper subgroups, by enumerating subgroups (closure BFS)."""
    subgroups = {frozenset([identity])}
    frontier = [frozenset([identity])]
    while frontier:
        h = frontier.pop()
        for g in range(n):
            if g in h:
                continue
            k = _closure(set(h) | {g}, mult)
            if k not in subgroups:
                subgroups.add(k)
                frontier.append(k)
    proper = [h for h in subgroups if 1 < len(h) < n]
    return [h for h in proper if not any(h < k for k in proper)]


def maximal_subgroups(lattice, positions, species, symprec: float = 1e-3,
                      max_index: int = 8) -> dict[str, Any]:
    """Maximal translationengleiche subgroups of the structure's space group.

    These are the group-subgroup steps available to a displacive/ordering phase
    transition. Subgroups are computed in the primitive setting (so the group
    order stays small), typed by spglib, and grouped by (type, index) with the
    number of symmetry-equivalent conjugate variants. Cell-multiplying
    (klassengleiche) subgroups are not enumerated.
    """
    spglib = _spglib()
    cell, _ = _to_cell(lattice, positions, species)
    parent = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if parent is None:
        return {"error": f"could not determine the parent space group at symprec={symprec}"}

    prim = spglib.standardize_cell(cell, to_primitive=True, symprec=symprec)
    prim_lat = np.asarray(prim[0], dtype=float)
    sym = spglib.get_symmetry(prim, symprec=symprec)
    R, T = np.asarray(sym["rotations"]), np.asarray(sym["translations"])
    n = len(R)
    mult, identity = _multiplication_table(R, T)

    grouped: dict[tuple, dict] = {}
    for h in _maximal_subgroups(mult, identity, n):
        index = n // len(h)
        if index > max_index:
            continue
        idx = sorted(h)
        st = spglib.get_spacegroup_type_from_symmetry(R[idx], T[idx],
                                                      lattice=prim_lat, symprec=symprec)
        if st is None:
            continue
        # translationengleiche if it keeps every pure lattice translation
        pure = {i for i in range(n) if np.array_equal(R[i], np.eye(3, dtype=R.dtype))}
        kind = "t" if pure <= h else "k"
        row = grouped.setdefault((int(st.number), index, kind), {
            "number": int(st.number), "international": st.international_short,
            "index": index, "kind": kind, "n_variants": 0,
            "crystal_system": crystal_system(int(st.number)),
        })
        row["n_variants"] += 1

    subgroups = sorted(grouped.values(), key=lambda r: (r["index"], r["number"]))
    return {
        "parent": {"number": int(parent.number), "international": parent.international,
                   "crystal_system": crystal_system(int(parent.number))},
        "n_maximal_subgroups": len(subgroups),
        "subgroups": subgroups,
        "note": "maximal translationengleiche subgroups (transition pathways); "
        "n_variants = symmetry-equivalent domains; klassengleiche subgroups omitted",
    }


# --------------------------------------------------------- pseudosymmetry


def pseudosymmetry_scan(lattice, positions, species,
                        symprecs=(1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2)) -> dict[str, Any]:
    """Detect a higher-symmetry parent by relaxing the position tolerance.

    A structure distorted slightly from a parent phase snaps to the parent's
    space group once the tolerance exceeds the distortion — the pseudosymmetry a
    phase transition breaks."""
    spglib = _spglib()
    cell, _ = _to_cell(lattice, positions, species)
    scan = []
    for sp in symprecs:
        ds = spglib.get_symmetry_dataset(cell, symprec=sp)
        scan.append({"symprec": sp,
                     "number": int(ds.number) if ds else None,
                     "international": ds.international if ds else None,
                     "n_operations": len(ds.rotations) if ds else 0})
    valid = [s for s in scan if s["number"]]
    highest = max(valid, key=lambda s: s["n_operations"]) if valid else None
    base = valid[0] if valid else None
    return {
        "scan": scan,
        "tightest": base,
        "highest_symmetry": highest,
        "pseudosymmetric": bool(highest and base and highest["number"] != base["number"]),
        "note": "if the space group rises as the tolerance loosens, the tighter "
        "structure is a distorted subgroup of the looser (parent) one",
    }


# ------------------------------------------------------------- magnetic


def magnetic_symmetry(lattice, positions, species, magmoms,
                      symprec: float = 1e-3) -> dict[str, Any]:
    """Magnetic (Shubnikov) space group of an ordered magnetic structure.

    ``magmoms`` is one collinear moment per atom, or an (N, 3) array of moment
    vectors. Determines the magnetic space group the given moment arrangement
    already has; deriving the *maximal* magnetic groups allowed by a propagation
    vector (representation analysis) is out of scope.
    """
    spglib = _spglib()
    (lat, pos, numbers), _ = _to_cell(lattice, positions, species)
    moments = np.asarray(magmoms, dtype=float)
    ds = spglib.get_magnetic_symmetry_dataset((lat, pos, numbers, moments),
                                              symprec=symprec)
    if ds is None:
        return {"error": "spglib could not determine the magnetic symmetry"}
    n_ops = len(ds.rotations) if hasattr(ds, "rotations") else None
    n_tr = int(np.sum(ds.time_reversals)) if hasattr(ds, "time_reversals") else None
    return {
        "uni_number": int(ds.uni_number),
        "msg_type": int(ds.msg_type),
        "n_operations": n_ops,
        "n_time_reversals": n_tr,
        "msg_type_meaning": {
            1: "type I (colorless, no time reversal)",
            2: "type II (grey, paramagnetic)",
            3: "type III (black-white, translation-preserving)",
            4: "type IV (black-white, anti-translation)",
        }.get(int(ds.msg_type), "unknown"),
    }


# --------------------------------------------------------------- figure


def subgroup_tree_figure(result: dict[str, Any], out) -> str:
    """Render the parent space group and its maximal subgroups as a tree."""
    import os

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    subs = result.get("subgroups", [])
    parent = result.get("parent", {})
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * max(len(subs), 1)), 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, title, sub, color):
        w, h = 0.17, 0.13
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.01", linewidth=1.2,
                                    edgecolor="#1f5fa8", facecolor=color))
        ax.text(x, y + 0.018, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x, y - 0.03, sub, ha="center", va="center", fontsize=8, color="#444")

    px, py = 0.5, 0.82
    box(px, py, parent.get("international", "?"),
        f"#{parent.get('number','?')} · {parent.get('crystal_system','')}", "#dbe9f6")
    n = len(subs) or 1
    for i, s in enumerate(subs):
        x = (i + 0.5) / n
        y = 0.22
        box(x, y, s["international"], f"#{s['number']} · i{s['index']}"
            + (f" ×{s['n_variants']}" if s.get("n_variants", 1) > 1 else ""), "#eef6ee")
        ax.add_patch(FancyArrowPatch((px, py - 0.07), (x, y + 0.07),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color="#888", lw=0.9))
        ax.text((px + x) / 2, (py + y) / 2, f"[{s['index']}]", fontsize=7, color="#888")
    ax.set_title(f"Maximal subgroups of {parent.get('international','?')} "
                 f"(#{parent.get('number','?')})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(out)
