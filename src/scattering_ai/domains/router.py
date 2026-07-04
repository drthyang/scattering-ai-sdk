"""Domain auto-detection: pick the right technique pack from the input.

One entry point should "just work" — a user points the SDK at their file and
gets the right analysis without naming a domain. Detection is deterministic and
cheap (extension + a light content sniff), and it always resolves to a real
pack: ``data`` (generic tool-driven) is the safe fallback, never an error.

Priority: rmc (structured run state) > diffuse (reciprocal-space volume/slice)
> pdf (total-scattering curve) > data.
"""

from __future__ import annotations

from pathlib import Path

from scattering_ai.core.schemas import AnalysisRequest

AUTO = "auto"
_SYMMETRY_SUFFIXES = {".cif", ".mcif"}
_PDF_SUFFIXES = {".gr", ".fq", ".sq"}
_SLICE_SUFFIXES = {".npz"}
_VOLUME_SUFFIXES = {".nxs", ".h5", ".hdf5", ".nx5", ".nexus"}
# 1D text formats worth sniffing when the extension is ambiguous.
_AMBIGUOUS_1D = {".dat", ".txt", ".csv", ".xy", ".chi", ""}
# Filename hints for total-scattering quantities (S(Q)/F(Q)/G(r)); a bgsub
# powder-diffraction pattern has none of these and falls through to `data`.
_TS_NAME_HINTS = ("sofq", "s(q)", "_sq", "sq_", "fofq", "f(q)", "gofr", "g(r)", "_gr", "pdf")


def _is_hdf5(path: Path) -> bool:
    if path.suffix.lower() in _VOLUME_SUFFIXES:
        return True
    try:
        with path.open("rb") as f:
            return f.read(8).startswith(b"\x89HDF")
    except OSError:
        return False


def _looks_like_total_scattering(path: Path) -> bool:
    """Sniff an ambiguous 1D file: total scattering (S(Q)/F(Q)/G(r)) vs a plain
    diffraction pattern. Name hints decide first (SofQ, S(Q), _gr, ...); failing
    that, only a genuine real-space r-axis counts — a Q-space tail near 0 or 1
    is too weak to distinguish S(Q) from a background-subtracted pattern.
    """
    if any(h in path.name.lower() for h in _TS_NAME_HINTS):
        return True
    if not path.exists() or path.suffix.lower() not in _AMBIGUOUS_1D:
        return False
    try:
        from scattering_ai.tools.io import load_curve

        curve = load_curve(path)
    except Exception:
        return False
    return curve.xlabel.strip().lower().startswith("r")


def detect_domain(request: AnalysisRequest) -> tuple[str, str]:
    """Return (domain, human-readable reason) for an ``auto`` request."""
    data = request.data
    if data.metadata.get("rmc") or data.r_values or data.run_summary:
        return "rmc", "structured RMC run state (run_summary / r_values / metadata.rmc)"

    files = [Path(f) for f in data.files]
    for p in files:
        if p.suffix.lower() == ".rmc6f":
            return "rmc", f"RMCProfile configuration ({p.name})"
    for p in files:
        if p.suffix.lower() in _SYMMETRY_SUFFIXES:
            return "symmetry", f"crystal structure ({p.name})"
    for p in files:
        if _is_hdf5(p) or p.suffix.lower() in _SLICE_SUFFIXES:
            return "diffuse", f"reciprocal-space volume/slice ({p.name})"
    for p in files:
        if p.suffix.lower() in _PDF_SUFFIXES or _looks_like_total_scattering(p):
            return "pdf", f"total-scattering curve ({p.name})"
    if files:
        return "data", f"generic tool-driven analysis ({len(files)} file(s))"
    return "data", "no files or run state; generic tool-driven analysis"


def resolve_domain(request: AnalysisRequest) -> tuple[str, str]:
    """Resolve the effective domain: the request's, or auto-detected."""
    if request.domain and request.domain != AUTO:
        return request.domain, "explicit"
    return detect_domain(request)
