"""Crystallographic lattice math: the B matrix (Busing-Levy convention).

Converts Miller indices (h, k, l) to Cartesian momentum transfer Q (Å⁻¹),
correctly for arbitrary (including non-orthogonal) unit cells:

    Q_cartesian = 2π · B · (h, k, l)

This replaces the orthogonal-cell approximation (|Q| from h/a, k/b, l/c)
used by the early 2D ring tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class Lattice:
    a: float
    b: float
    c: float
    alpha: float = 90.0  # degrees
    beta: float = 90.0
    gamma: float = 90.0

    @classmethod
    def from_dict(cls, d: dict) -> Lattice | None:
        try:
            return cls(**{k: float(d[k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")})
        except (KeyError, TypeError, ValueError):
            return None

    @cached_property
    def volume(self) -> float:
        ca, cb, cg = (np.cos(np.radians(x)) for x in (self.alpha, self.beta, self.gamma))
        return float(
            self.a * self.b * self.c
            * np.sqrt(1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg)
        )

    @cached_property
    def b_matrix(self) -> np.ndarray:
        """Busing-Levy B (crystallographic convention, no 2π folded in)."""
        ca, cb, cg = (np.cos(np.radians(x)) for x in (self.alpha, self.beta, self.gamma))
        sa = np.sin(np.radians(self.alpha))
        volume = self.volume
        # reciprocal cell
        a_star = self.b * self.c * sa / volume
        b_star = self.a * self.c * np.sin(np.radians(self.beta)) / volume
        c_star = self.a * self.b * np.sin(np.radians(self.gamma)) / volume
        cos_beta_star = (ca * cg - cb) / (sa * np.sin(np.radians(self.gamma)))
        cos_gamma_star = (ca * cb - cg) / (sa * np.sin(np.radians(self.beta)))
        sin_beta_star = np.sqrt(1 - cos_beta_star**2)
        # Busing & Levy (1967) eq. 3; note B[1,2] uses the DIRECT cell alpha.
        return np.array(
            [
                [a_star, b_star * cos_gamma_star, c_star * cos_beta_star],
                [0.0, b_star * np.sqrt(1 - cos_gamma_star**2),
                 -c_star * sin_beta_star * ca],
                [0.0, 0.0, 1.0 / self.c],
            ]
        )

    def q_cartesian(self, hkl: np.ndarray) -> np.ndarray:
        """Cartesian Q (Å⁻¹) for hkl of shape (..., 3)."""
        hkl = np.asarray(hkl, dtype=float)
        return 2 * np.pi * hkl @ self.b_matrix.T

    def q_magnitude(self, hkl: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.q_cartesian(hkl), axis=-1)

    def d_spacing(self, hkl) -> float:
        q = float(self.q_magnitude(np.asarray(hkl, dtype=float)))
        if q == 0:
            raise ValueError("d-spacing undefined for (0,0,0)")
        return 2 * np.pi / q
