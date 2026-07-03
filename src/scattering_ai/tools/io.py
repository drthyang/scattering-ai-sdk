"""Readers for 1D scattering data formats.

Formats are added as real files land in ``data/``. Currently:

- diffpy.pdfgetx ``.fq``/``.gr``/``.sq``: INI-style metadata header,
  ``#### start data`` marker, ``#L`` column-label line, two columns.
  The header (qmin/qmax, corrections, input files) is preserved in
  ``Curve1D.meta["pdfgetx"]`` — it is reduction provenance.
- Generic column ASCII with ``#`` comments (e.g. NOMAD S(Q) exports):
  2 columns = x, y; 3+ columns = x, y, error.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from scattering_ai.tools.models import Curve1D

_PDFGETX_MARKER = "#### start data"
_KEY_VALUE = re.compile(r"^(\w+)\s*=\s*(.*)$")
# #L r(Å)  G(Å$^{-2}$)  → labels ["r(Å)", "G(Å$^{-2}$)"]
_LABEL_SPLIT = re.compile(r"\s{2,}|\t")


def load_curve(path: str | Path) -> Curve1D:
    """Load a 1D curve, sniffing the format from content."""
    path = Path(path)
    with path.open("rb") as f:
        magic = f.read(8)
    if magic.startswith(b"\x89HDF"):
        return _load_nexus_curve(path)
    text = path.read_text()
    if _PDFGETX_MARKER in text:
        return _load_pdfgetx(path, text)
    return _load_columns(path, text)


def _load_nexus_curve(path: Path) -> Curve1D:
    """1D signal from a NeXus/HDF5 file via NXdata conventions.

    Follows the file's ``default`` attributes when present, otherwise takes
    the first NXdata group found; ``signal``/``axes`` attributes name the
    datasets. Requires h5py (``[volumes]`` extra).
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Reading NeXus/HDF5 curves requires h5py: "
            "pip install scattering-ai-sdk[volumes]"
        ) from exc

    def as_str(value) -> str:
        return value.decode() if isinstance(value, bytes) else str(value)

    with h5py.File(path, "r") as f:

        def find_nxdata(group):
            default = group.attrs.get("default")
            if default is not None and as_str(default) in group:
                child = group[as_str(default)]
                if as_str(child.attrs.get("NX_class", "")) == "NXdata":
                    return child
                return find_nxdata(child)
            for child in group.values():
                if isinstance(child, h5py.Group):
                    if as_str(child.attrs.get("NX_class", "")) == "NXdata":
                        return child
                    found = find_nxdata(child)
                    if found is not None:
                        return found
            return None

        nxdata = find_nxdata(f)
        if nxdata is None:
            raise ValueError(f"No NXdata group found in {path}")
        signal_name = as_str(nxdata.attrs.get("signal", "data"))
        if signal_name not in nxdata:
            raise ValueError(f"NXdata signal '{signal_name}' missing in {path}")
        y = np.asarray(nxdata[signal_name][()], dtype=float).squeeze()
        if y.ndim != 1:
            raise ValueError(
                f"{path}: NXdata signal is {y.ndim}D; load_curve handles 1D "
                "(use load_volume for MDHisto volumes)"
            )
        axes_attr = nxdata.attrs.get("axes")
        x_name = None
        if axes_attr is not None:
            first = axes_attr[0] if isinstance(axes_attr, (list, np.ndarray)) else axes_attr
            x_name = as_str(first)
        if x_name and x_name != "." and x_name in nxdata:
            x = np.asarray(nxdata[x_name][()], dtype=float).squeeze()
            if x.size == y.size + 1:  # bin edges
                x = (x[:-1] + x[1:]) / 2
            xlabel = x_name
            units = nxdata[x_name].attrs.get("units")
            if units is not None:
                xlabel = f"{x_name} ({as_str(units)})"
        else:
            x, xlabel = np.arange(y.size, dtype=float), "index"
        errors_name = f"{signal_name}_errors"
        e = (np.asarray(nxdata[errors_name][()], dtype=float).squeeze()
             if errors_name in nxdata else None)
        return Curve1D(
            x=x, y=y, e=e, xlabel=xlabel, ylabel=signal_name,
            meta={"source": str(path), "format": "nexus", "nxdata": nxdata.name},
        )


def _load_pdfgetx(path: Path, text: str) -> Curve1D:
    header, _, body = text.partition(_PDFGETX_MARKER)
    meta: dict = {}
    for line in header.splitlines():
        match = _KEY_VALUE.match(line.strip())
        if match and match.group(2):
            meta[match.group(1)] = match.group(2)

    xlabel, ylabel = "x", "y"
    rows: list[tuple[float, float]] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#L"):
            labels = [s for s in _LABEL_SPLIT.split(line[2:].strip()) if s]
            if len(labels) >= 2:
                xlabel, ylabel = labels[0], labels[1]
            continue
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            rows.append((float(parts[0]), float(parts[1])))

    data = np.asarray(rows)
    return Curve1D(
        x=data[:, 0],
        y=data[:, 1],
        xlabel=xlabel,
        ylabel=ylabel,
        meta={"source": str(path), "format": "pdfgetx", "pdfgetx": meta},
    )


def _load_columns(path: Path, text: str) -> Curve1D:
    rows: list[list[float]] = []
    n_cols = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//", ";")):
            continue
        parts = line.split()
        try:
            values = [float(p) for p in parts]
        except ValueError:
            continue  # stray text line (some formats embed titles)
        if len(values) >= 2:
            n_cols = max(n_cols, len(values))
            rows.append(values)
    if not rows:
        raise ValueError(f"No numeric columns found in {path}")

    data = np.asarray([r[: min(3, n_cols)] + [np.nan] * (min(3, n_cols) - len(r)) for r in rows])
    errors = data[:, 2] if data.shape[1] >= 3 and np.isfinite(data[:, 2]).all() else None
    return Curve1D(
        x=data[:, 0],
        y=data[:, 1],
        e=errors,
        meta={"source": str(path), "format": "columns"},
    )
