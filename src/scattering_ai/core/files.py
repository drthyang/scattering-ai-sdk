"""Resolve user-supplied file entries into concrete paths.

A single entry may be an exact file, a glob pattern, or a directory. Expanding
these uniformly is what lets ``analyze(data={"files": ["scans/"]})`` and
``--file 'scans/*.dat'`` both run a whole series — the same convenience the
series tools already offer. Plain paths (including ones that do not exist yet,
e.g. RMC bookkeeping names) pass through untouched so callers keep full control.
"""

from __future__ import annotations

import glob as _glob
from pathlib import Path

# Sidecar files that live next to data but are never data themselves.
_SKIP_SUFFIXES = {".md", ".png", ".json"}


def expand_files(entries: list[str]) -> list[str]:
    expanded: list[str] = []
    for entry in entries:
        # An existing literal path always wins over glob interpretation:
        # real facility filenames contain brackets ("[h,0,0]", "(0,k,l)") that
        # glob would otherwise read as character classes and match nothing.
        if Path(entry).is_file():
            expanded.append(entry)
        elif any(ch in entry for ch in "*?["):
            expanded += sorted(_glob.glob(entry))
        elif Path(entry).is_dir():
            expanded += sorted(
                str(f) for f in Path(entry).iterdir()
                if f.is_file() and not f.name.startswith(".")
                and f.suffix.lower() not in _SKIP_SUFFIXES
            )
        else:
            expanded.append(entry)
    return list(dict.fromkeys(expanded))  # dedup, preserve order
