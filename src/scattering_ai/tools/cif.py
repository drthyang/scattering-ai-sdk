"""Dependency-free CIF reading (roadmap C6): lattice, symmetry, formula.

Deliberately minimal: the SDK needs the unit cell (to feed the B matrix and
predict Bragg positions), not a full structure parser. Values like
``10.4128(3)`` have their standard uncertainties stripped.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from scattering_ai.tools.lattice import Lattice

_NUMBER = re.compile(r"^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?:\(\d+\))?$")

_CELL_KEYS = {
    "_cell_length_a": "a",
    "_cell_length_b": "b",
    "_cell_length_c": "c",
    "_cell_angle_alpha": "alpha",
    "_cell_angle_beta": "beta",
    "_cell_angle_gamma": "gamma",
}
_TEXT_KEYS = {
    "_symmetry_space_group_name_h-m": "space_group",
    "_space_group_name_h-m_alt": "space_group",
    "_symmetry_int_tables_number": "space_group_number",
    "_space_group_it_number": "space_group_number",
    "_chemical_formula_sum": "formula",
    "_chemical_formula_structural": "formula_structural",
}


def read_cif(path: str | Path) -> dict:
    """Extract cell parameters and identity fields from the first data block."""
    out: dict = {}
    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not parts[0].startswith("_"):
            continue
        key, value = parts[0].lower(), parts[1].strip().strip("'\"")
        if key in _CELL_KEYS and _CELL_KEYS[key] not in out:
            match = _NUMBER.match(value)
            if match:
                out[_CELL_KEYS[key]] = float(match.group(1))
        elif key in _TEXT_KEYS and _TEXT_KEYS[key] not in out:
            out[_TEXT_KEYS[key]] = value
    if not all(k in out for k in ("a", "b", "c")):
        raise ValueError(f"No complete unit cell found in {path}")
    for angle in ("alpha", "beta", "gamma"):
        out.setdefault(angle, 90.0)
    return out


def lattice_from_cif(path: str | Path) -> Lattice:
    cell = read_cif(path)
    return Lattice(a=cell["a"], b=cell["b"], c=cell["c"],
                   alpha=cell["alpha"], beta=cell["beta"], gamma=cell["gamma"])


def _split_cif(tokens_line: str) -> list[str]:
    """Quote-aware split of one CIF data line (values may be 'quoted strings')."""
    out, i, n = [], 0, len(tokens_line)
    while i < n:
        while i < n and tokens_line[i].isspace():
            i += 1
        if i >= n:
            break
        if tokens_line[i] in "'\"":
            quote = tokens_line[i]
            j = tokens_line.find(quote, i + 1)
            j = n if j == -1 else j
            out.append(tokens_line[i + 1:j])
            i = j + 1
        else:
            j = i
            while j < n and not tokens_line[j].isspace():
                j += 1
            out.append(tokens_line[i:j])
            i = j
    return out


def _iter_loops(lines: list[str]):
    """Yield (headers, rows) for each ``loop_`` block; rows are token lists."""
    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip() != "loop_":
            i += 1
            continue
        i += 1
        headers = []
        while i < n and lines[i].strip().startswith("_"):
            headers.append(lines[i].strip().lower())
            i += 1
        rows = []
        while i < n:
            line = lines[i].strip()
            if not line or line.startswith(("_", "loop_", "data_", "#")):
                break
            rows.append(_split_cif(line))
            i += 1
        yield headers, rows


def _eval_symop_raw(op: str, xyz) -> np.ndarray:
    """Evaluate a symop string (e.g. '-x,1/2+y,z') at ``xyz`` without wrapping."""
    x, y, z = xyz
    out = []
    for part in op.split(",")[:3]:
        expr = part.strip().lower().replace(" ", "")
        expr = re.sub(r"(\d)/(\d)", r"(\1/\2)", expr)          # 1/2 -> (1/2)
        expr = re.sub(r"(?<![\d.])x", "*X", expr).replace("*X", f"*({x})")
        expr = re.sub(r"(?<![\d.])y", "*Y", expr).replace("*Y", f"*({y})")
        expr = re.sub(r"(?<![\d.])z", "*Z", expr).replace("*Z", f"*({z})")
        expr = expr.lstrip("*").replace("+*", "+").replace("-*", "-")
        out.append(eval(expr, {"__builtins__": {}}))          # noqa: S307 - constrained arithmetic
    return np.array(out, dtype=float)


def _symop_matrix(op: str) -> tuple[np.ndarray, np.ndarray]:
    """Rotation matrix and translation of a CIF symmetry-operation string."""
    t = _eval_symop_raw(op, (0.0, 0.0, 0.0))
    cols = [_eval_symop_raw(op, e) - t for e in np.eye(3)]
    return np.array(cols).T, t


def _apply_symop(op: str, xyz: np.ndarray) -> np.ndarray:
    """Fractional coordinate after a symop, wrapped into the unit cell."""
    r, t = _symop_matrix(op)
    return np.mod(r @ np.asarray(xyz, dtype=float) + t, 1.0)


def _read_moments(lines: list[str]) -> dict[str, list[float]]:
    """Magnetic moments per atom label from an mCIF _atom_site_moment loop."""
    moments: dict[str, list[float]] = {}
    for headers, rows in _iter_loops(lines):
        if not any("moment" in h and ("crystalaxis" in h or h[-2:] in ("_x", ".x"))
                   for h in headers):
            continue
        def col(ax: str, hs: list[str] = headers) -> int | None:
            return next((i for i, h in enumerate(hs)
                         if h.endswith((f"crystalaxis_{ax}", f".{ax}", f"moment_{ax}"))), None)

        lab = next((i for i, h in enumerate(headers) if h.endswith(("label", ".label"))), None)
        mx, my, mz = col("x"), col("y"), col("z")
        if lab is None or None in (mx, my, mz):
            continue
        for r in rows:
            if len(r) > max(lab, mx, my, mz):
                moments[r[lab]] = [_strip_su(r[mx]), _strip_su(r[my]), _strip_su(r[mz])]
    return moments


def read_structure(path: str | Path) -> dict:
    """Read a full structure from a CIF: cell + all atoms (asymmetric unit
    expanded by the symmetry operations if the CIF lists them).

    Returns ``lattice`` (a,b,c,alpha,beta,gamma), ``positions`` (fractional),
    ``species``, and provenance counts. Ready to feed the symmetry tools.
    """
    lines = Path(path).read_text(errors="replace").splitlines()
    cell = read_cif(path)
    lattice = [cell[k] for k in ("a", "b", "c", "alpha", "beta", "gamma")]

    asym_pos, asym_species, asym_labels, symops = [], [], [], []
    for headers, rows in _iter_loops(lines):
        if any(h.startswith("_atom_site_fract_x") for h in headers):
            cx = headers.index("_atom_site_fract_x")
            cy = headers.index("_atom_site_fract_y")
            cz = headers.index("_atom_site_fract_z")
            csym = next((headers.index(h) for h in
                         ("_atom_site_type_symbol", "_atom_site_label") if h in headers), None)
            clab = headers.index("_atom_site_label") if "_atom_site_label" in headers else csym
            for r in rows:
                if len(r) <= max(cx, cy, cz):
                    continue
                asym_pos.append([_strip_su(r[cx]), _strip_su(r[cy]), _strip_su(r[cz])])
                raw = r[csym] if csym is not None else "X"
                asym_species.append(re.sub(r"[\d+\-].*$", "", raw) or raw)
                asym_labels.append(r[clab] if clab is not None else raw)
        xyz_col = next((h for h in headers
                        if h.endswith("_xyz") or h.endswith("_as_xyz")), None)
        if xyz_col:
            c = headers.index(xyz_col)
            symops += [r[c] for r in rows if len(r) > c and "," in r[c]]

    if not asym_pos:
        raise ValueError(f"No atom sites found in {path}")

    moment_by_label = _read_moments(lines)
    ops = symops or ["x,y,z"]
    op_mats = [_symop_matrix(op) for op in ops]
    positions, species, moments = [], [], []
    for base, sp, label in zip(asym_pos, asym_species, asym_labels, strict=True):
        base = np.array(base, dtype=float)
        m0 = np.array(moment_by_label.get(label, [0.0, 0.0, 0.0]), dtype=float)
        for r, t in op_mats:
            p = np.mod(r @ base + t, 1.0)
            if any(np.allclose(p, q, atol=1e-4) for q in positions):
                continue
            positions.append(p)
            species.append(sp)
            moments.append(round(float(np.linalg.det(r))) * (r @ m0))  # axial vector
    has_moments = bool(moment_by_label) and any(np.linalg.norm(m) > 1e-6 for m in moments)
    return {
        "lattice": lattice,
        "positions": [list(p) for p in positions],
        "species": species,
        "moments": [list(m) for m in moments] if has_moments else None,
        "space_group_cif": cell.get("space_group", ""),
        "n_asymmetric": len(asym_pos),
        "n_atoms": len(positions),
        "n_magnetic": int(sum(np.linalg.norm(m) > 1e-6 for m in moments)),
        "n_symops": len(ops),
    }


def _strip_su(value: str) -> float:
    match = _NUMBER.match(value.strip())
    return float(match.group(1)) if match else float(value)


def predicted_d_spacings(lattice: Lattice, d_min: float = 1.0,
                         max_index: int = 8, top: int = 30) -> list[dict]:
    """Geometrically allowed d-spacings (no structure factors — a checklist
    for matching observed peaks, not an intensity prediction)."""
    seen: dict[float, tuple] = {}
    for h in range(-max_index, max_index + 1):
        for k in range(-max_index, max_index + 1):
            for l in range(-max_index, max_index + 1):  # noqa: E741
                if (h, k, l) == (0, 0, 0):
                    continue
                d = lattice.d_spacing((h, k, l))
                if d < d_min:
                    continue
                key = round(d, 4)
                if key not in seen or _hkl_rank((h, k, l)) < _hkl_rank(seen[key]):
                    seen[key] = (h, k, l)
    rows = [
        {"d": d, "q": round(2 * np.pi / d, 4), "hkl": list(hkl)}
        for d, hkl in sorted(seen.items(), reverse=True)
    ]
    return rows[:top]


def _hkl_rank(hkl) -> tuple:
    return (sum(abs(i) for i in hkl), tuple(-abs(i) for i in hkl))
