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
    text = path.read_text()
    if _PDFGETX_MARKER in text:
        return _load_pdfgetx(path, text)
    return _load_columns(path, text)


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
