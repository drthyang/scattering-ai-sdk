"""In-memory data objects passed between tools.

These are numpy-backed dataclasses, not Pydantic models: they live inside
tool chains, while Pydantic handles the agent/report boundary. Tools return
JSON-safe summary dicts alongside these objects so the agent can reason
about results without touching arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Curve1D:
    x: np.ndarray
    y: np.ndarray
    e: np.ndarray | None = None
    xlabel: str = "x"
    ylabel: str = "y"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.e is not None:
            self.e = np.asarray(self.e, dtype=float)
        if self.x.shape != self.y.shape:
            raise ValueError(f"x and y shapes differ: {self.x.shape} vs {self.y.shape}")

    def summary(self) -> dict[str, Any]:
        """JSON-safe description for agent consumption."""
        return {
            "n_points": int(self.x.size),
            "x_range": [float(self.x.min()), float(self.x.max())],
            "y_range": [float(np.nanmin(self.y)), float(np.nanmax(self.y))],
            "xlabel": self.xlabel,
            "ylabel": self.ylabel,
            "has_errors": self.e is not None,
            "source": self.meta.get("source", ""),
        }


@dataclass
class Slice2D:
    data: np.ndarray  # shape (ny, nx); NaN = masked / no coverage
    x_centers: np.ndarray
    y_centers: np.ndarray
    xlabel: str = "x"
    ylabel: str = "y"
    meta: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        finite = np.isfinite(self.data)
        return {
            "shape": list(self.data.shape),
            "x_range": [float(self.x_centers.min()), float(self.x_centers.max())],
            "y_range": [float(self.y_centers.min()), float(self.y_centers.max())],
            "xlabel": self.xlabel,
            "ylabel": self.ylabel,
            "coverage": round(float(finite.mean()), 4),
            "signal_range": [
                float(np.nanmin(self.data)) if finite.any() else None,
                float(np.nanmax(self.data)) if finite.any() else None,
            ],
            "source": self.meta.get("source", ""),
        }
