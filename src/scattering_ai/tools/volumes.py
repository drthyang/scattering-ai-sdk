"""3D reciprocal-space volumes: Mantid MDHistoWorkspace NeXus loader and
axis-aligned slab slicing to 2D.

Wraps the common reduction operations (roadmap B3: do not rebuild Mantid).
Requires h5py: ``pip install scattering-ai-sdk[volumes]``. Slabs are read
as HDF5 hyperslabs, so slicing a multi-GB volume never loads it fully.

MDHisto layout notes (verified on CORELLI output):
- ``data/signal`` has shape (nD2, nD1, nD0) — logical axes reversed.
- ``data/D0..D2`` are bin **edges** (length n+1), with name/units attrs.
- ``data/mask`` is int8 with 1 = masked; signal also uses NaN for
  no-coverage bins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from scattering_ai.tools.models import Slice2D

_GROUP = "MDHistoWorkspace"


@dataclass
class AxisInfo:
    name: str
    units: str
    edges: np.ndarray

    @property
    def centers(self) -> np.ndarray:
        return (self.edges[:-1] + self.edges[1:]) / 2

    @property
    def n_bins(self) -> int:
        return len(self.edges) - 1


@dataclass
class Volume3D:
    path: Path
    axes: list[AxisInfo]  # logical order [D0, D1, D2]
    lattice: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "source": str(self.path),
            "axes": [
                {
                    "name": ax.name,
                    "units": ax.units,
                    "range": [float(ax.edges[0]), float(ax.edges[-1])],
                    "n_bins": ax.n_bins,
                }
                for ax in self.axes
            ],
            "lattice": self.lattice,
        }

    def slice(self, axis: int | str, center: float, thickness: float) -> Slice2D:
        """Average an axis-aligned slab into a 2D slice.

        ``axis`` is the logical axis (0/1/2 or its name) integrated over
        ``[center - thickness/2, center + thickness/2]``. Masked and
        no-coverage bins are excluded per-pixel.
        """
        import h5py

        i_axis = self._axis_index(axis)
        ax = self.axes[i_axis]
        lo, hi = center - thickness / 2, center + thickness / 2
        # bins whose centers fall inside the slab
        in_slab = np.where((ax.centers >= lo) & (ax.centers <= hi))[0]
        if in_slab.size == 0:
            raise ValueError(
                f"Slab [{lo}, {hi}] contains no bins on axis '{ax.name}' "
                f"(range {ax.edges[0]}..{ax.edges[-1]})"
            )
        numpy_axis = 2 - i_axis  # signal is stored (D2, D1, D0)
        selector: list[Any] = [slice(None)] * 3
        selector[numpy_axis] = slice(int(in_slab[0]), int(in_slab[-1]) + 1)

        with h5py.File(self.path, "r") as f:
            data_group = f[f"{_GROUP}/data"]
            signal = data_group["signal"][tuple(selector)].astype(float)
            if "mask" in data_group:
                masked = data_group["mask"][tuple(selector)] == 1
                signal[masked] = np.nan

        import warnings

        with warnings.catch_warnings():
            # all-NaN pixel columns (no coverage anywhere in the slab) are
            # expected and correctly become NaN in the slice
            warnings.simplefilter("ignore", category=RuntimeWarning)
            averaged = np.nanmean(signal, axis=numpy_axis)

        remaining = [i for i in range(3) if i != i_axis]  # logical, ascending
        x_ax, y_ax = self.axes[remaining[0]], self.axes[remaining[1]]
        # after removing numpy_axis, result is (y, x) = (higher, lower logical)
        return Slice2D(
            data=averaged,
            x_centers=x_ax.centers,
            y_centers=y_ax.centers,
            xlabel=f"{x_ax.name} ({x_ax.units})",
            ylabel=f"{y_ax.name} ({y_ax.units})",
            meta={
                "source": str(self.path),
                "slice_axis": ax.name,
                "center": float(center),
                "thickness": float(thickness),
                "n_bins_integrated": int(in_slab.size),
            },
        )

    def oblique_slice(
        self,
        origin: tuple[float, float, float],
        u_axis: tuple[float, float, float],
        v_axis: tuple[float, float, float],
        u_range: tuple[float, float],
        v_range: tuple[float, float],
        thickness: float = 0.0,
        du: float | None = None,
        dv: float | None = None,
        n_layers: int = 5,
    ) -> Slice2D:
        """Cut an arbitrary plane: points = origin + s*u_axis + t*v_axis.

        All vectors are in the volume's logical (HKL / r.l.u.) coordinates,
        Mantid-BinMD style: the slab is defined in index space, and
        ``thickness`` is measured along the u x v cross product normalized in
        the r.l.u. metric — for axis-aligned cuts this matches ``slice()``.
        Only the bounding box of the requested grid is read from disk.
        """
        import h5py

        origin_v = np.asarray(origin, dtype=float)
        u_vec = np.asarray(u_axis, dtype=float)
        v_vec = np.asarray(v_axis, dtype=float)
        normal = np.cross(u_vec, v_vec)
        norm = np.linalg.norm(normal)
        if norm == 0:
            raise ValueError("u_axis and v_axis are parallel")
        normal /= norm

        steps = np.array([float(np.median(np.diff(ax.centers))) for ax in self.axes])
        s_vals = np.arange(u_range[0], u_range[1] + 1e-12,
                           du or float(np.min(steps / np.maximum(np.abs(u_vec), 1e-12))))
        t_vals = np.arange(v_range[0], v_range[1] + 1e-12,
                           dv or float(np.min(steps / np.maximum(np.abs(v_vec), 1e-12))))
        offsets = (
            np.linspace(-thickness / 2, thickness / 2, n_layers) if thickness > 0
            else np.array([0.0])
        )

        # hkl sample points, shape (n_layers, nt, ns, 3)
        grid = (
            origin_v[None, None, None, :]
            + s_vals[None, None, :, None] * u_vec[None, None, None, :]
            + t_vals[None, :, None, None] * v_vec[None, None, None, :]
            + offsets[:, None, None, None] * normal[None, None, None, :]
        )

        # logical axis -> fractional bin index (uniform grids)
        firsts = np.array([ax.centers[0] for ax in self.axes])
        frac = (grid - firsts) / steps  # index along logical axes 0,1,2

        # bounding box in storage order (D2, D1, D0), padded for interpolation
        lo, hi = [], []
        n_bins = [ax.n_bins for ax in self.axes]
        for logical in (2, 1, 0):
            lo.append(int(np.clip(np.floor(frac[..., logical].min()) - 1, 0,
                                  n_bins[logical] - 1)))
            hi.append(int(np.clip(np.ceil(frac[..., logical].max()) + 2, 1,
                                  n_bins[logical])))
        with h5py.File(self.path, "r") as f:
            data_group = f[f"{_GROUP}/data"]
            sub = data_group["signal"][lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(float)
            if "mask" in data_group:
                masked = data_group["mask"][lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] == 1
                sub[masked] = np.nan

        from scipy import ndimage

        coords = np.stack(
            [frac[..., 2] - lo[0], frac[..., 1] - lo[1], frac[..., 0] - lo[2]]
        ).reshape(3, -1)
        finite = np.isfinite(sub)
        filled = np.where(finite, sub, 0.0)
        values = ndimage.map_coordinates(filled, coords, order=1, mode="constant", cval=np.nan)
        weights = ndimage.map_coordinates(
            finite.astype(float), coords, order=1, mode="constant", cval=0.0
        )
        shape = grid.shape[:3]
        values = values.reshape(shape)
        weights = weights.reshape(shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            sampled = np.where(weights > 0.5, values / np.maximum(weights, 1e-12), np.nan)
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                averaged = np.nanmean(sampled, axis=0)  # over layers -> (nt, ns)

        def fmt(vec) -> str:
            return "[" + ",".join(f"{x:g}" for x in vec) + "]"

        return Slice2D(
            data=averaged,
            x_centers=s_vals,
            y_centers=t_vals,
            xlabel=f"s along {fmt(u_vec)} (r.l.u.)",
            ylabel=f"t along {fmt(v_vec)} (r.l.u.)",
            meta={
                "source": str(self.path),
                "origin": [float(x) for x in origin_v],
                "u_axis": [float(x) for x in u_vec],
                "v_axis": [float(x) for x in v_vec],
                "thickness": float(thickness),
                "n_layers": int(len(offsets)),
            },
        )

    def _axis_index(self, axis: int | str) -> int:
        if isinstance(axis, int):
            if axis not in (0, 1, 2):
                raise ValueError("axis index must be 0, 1, or 2")
            return axis
        for i, ax in enumerate(self.axes):
            if ax.name == axis:
                return i
        raise ValueError(f"Unknown axis '{axis}'; have {[a.name for a in self.axes]}")


def load_volume(path: str | Path) -> Volume3D:
    """Load MDHistoWorkspace metadata (axes, lattice); signal stays on disk."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Volume support requires h5py: pip install scattering-ai-sdk[volumes]"
        ) from exc

    path = Path(path)
    with h5py.File(path, "r") as f:
        data_group = f[f"{_GROUP}/data"]
        axes = []
        for i in range(3):
            dataset = data_group[f"D{i}"]
            name = dataset.attrs.get("name", f"dim{i}")
            units = dataset.attrs.get("units", "")
            name = name.decode() if isinstance(name, bytes) else str(name)
            units = units.decode() if isinstance(units, bytes) else str(units)
            axes.append(AxisInfo(name=name, units=units, edges=dataset[:]))

        lattice: dict[str, float] = {}
        lattice_group = f.get(f"{_GROUP}/experiment0/sample/oriented_lattice")
        if lattice_group is not None:
            for key in ("a", "b", "c", "alpha", "beta", "gamma"):
                ds = lattice_group.get(f"unit_cell_{key}")
                if ds is not None:
                    lattice[key] = float(ds[0])

        signal_shape = data_group["signal"].shape

    expected = tuple(ax.n_bins for ax in reversed(axes))
    if signal_shape != expected:
        raise ValueError(
            f"Unexpected signal shape {signal_shape}; expected {expected} from axes"
        )
    return Volume3D(path=path, axes=axes, lattice=lattice, meta={"signal_shape": signal_shape})
