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
