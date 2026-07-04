"""Deterministic diagnostics for diffuse-scattering data.

Diffuse scattering lives in reciprocal-space volumes and the 2D slices cut
from them. These checks are computable without an LLM and stay cheap: volume
checks read only axis/lattice metadata (not the full array); slice checks run
on the small 2D array. They flag the things that most often mislead a diffuse
analysis — powder-ring contaminants, Bragg-punch/mask gaps, and anisotropic
sampling — and report the Bragg-vs-diffuse character as grounding.
"""

from __future__ import annotations

from pathlib import Path

from scattering_ai.core.findings import Finding, Severity

# Below this covered fraction, masked / Bragg-punched pixels bias averages.
MIN_COVERAGE = 0.6
# Axis sampled this many times coarser than the finest = anisotropic sampling.
ANISOTROPY_RATIO = 3.0
_VOLUME_SUFFIXES = (".nxs", ".h5", ".hdf5", ".nx5", ".nexus")


def _is_hdf5(path: Path) -> bool:
    if path.suffix.lower() in _VOLUME_SUFFIXES:
        return True
    try:
        with path.open("rb") as f:
            return f.read(8).startswith(b"\x89HDF")
    except OSError:
        return False


def diagnose_volume(path: str) -> list[Finding]:
    from scattering_ai.tools.volumes import load_volume

    try:
        summary = load_volume(path).summary()
    except ImportError:
        return [Finding(diagnostic="diffuse_reader_missing", severity=Severity.ERROR,
                        message="Reading volumes needs h5py: pip install "
                        "scattering-ai-sdk[volumes].", evidence={"path": path})]
    except Exception as exc:
        return [Finding(diagnostic="diffuse_unreadable", severity=Severity.ERROR,
                        message=f"Could not read volume {Path(path).name}: "
                        f"{type(exc).__name__}: {exc}", evidence={"path": path})]

    axes = summary["axes"]
    widths = {}
    for ax in axes:
        lo, hi = ax["range"]
        widths[ax["name"]] = (hi - lo) / ax["n_bins"] if ax["n_bins"] else float("nan")
    findings = [
        Finding(
            diagnostic="diffuse_volume",
            severity=Severity.INFO,
            message="Reciprocal-space volume: "
            + ", ".join(f"{ax['name']} ∈ [{ax['range'][0]:g}, {ax['range'][1]:g}] "
                        f"({ax['n_bins']} bins)" for ax in axes)
            + (f"; cell {summary['lattice']}" if summary.get("lattice") else ""),
            evidence={"axes": axes, "bin_widths": {k: round(v, 5) for k, v in widths.items()},
                      "lattice": summary.get("lattice", {})},
        )
    ]
    finite = [w for w in widths.values() if w == w]  # drop NaN
    if finite and max(finite) / min(finite) >= ANISOTROPY_RATIO:
        coarse = max(widths, key=lambda k: widths[k])
        findings.append(Finding(
            diagnostic="diffuse_anisotropic_sampling",
            severity=Severity.INFO,
            message=f"Sampling is anisotropic: '{coarse}' is binned "
            f"{max(finite) / min(finite):.1f}× coarser than the finest axis. "
            "Diffuse features look broader along coarsely-sampled directions — "
            "compare planes at matched resolution before reading anisotropy.",
            evidence={"bin_widths": {k: round(v, 5) for k, v in widths.items()}},
        ))
    return findings


def diagnose_slice(path: str) -> list[Finding]:
    from scattering_ai.tools.registry import load_slice
    from scattering_ai.tools.slices import detect_rings, find_peaks_2d

    try:
        s = load_slice(path)
    except Exception as exc:
        return [Finding(diagnostic="diffuse_unreadable", severity=Severity.ERROR,
                        message=f"Could not read slice {Path(path).name}: "
                        f"{type(exc).__name__}: {exc}", evidence={"path": path})]

    findings: list[Finding] = []
    summary = s.summary()
    coverage = summary["coverage"]
    if coverage < MIN_COVERAGE:
        findings.append(Finding(
            diagnostic="bragg_punch_coverage",
            severity=Severity.WARNING,
            message=f"Only {coverage:.0%} of the slice has data; Bragg-punched or "
            "masked regions and detector gaps bias azimuthal and diffuse averages. "
            "Restrict analysis to covered annuli.",
            evidence={"coverage": coverage, "shape": summary["shape"]},
        ))
    else:
        findings.append(Finding(
            diagnostic="diffuse_slice_coverage", severity=Severity.INFO,
            message=f"Slice coverage {coverage:.0%}, signal range {summary['signal_range']}.",
            evidence={"coverage": coverage, "signal_range": summary["signal_range"]},
        ))

    # Ring detection needs |Q| geometry: slices cut from a volume carry lattice
    # metadata; a bare externally-produced slice does not, so skip it there.
    try:
        rings = detect_rings(s)
    except ValueError:
        rings = {}
    matched = rings.get("phase_match_counts", {})
    if matched:
        findings.append(Finding(
            diagnostic="contaminant_rings",
            severity=Severity.WARNING,
            message=f"Powder-ring candidates match known contaminants {matched}. "
            "These are sample-environment artifacts (Al can, Cu, steel), not "
            "diffuse scattering — exclude the annuli before analysis. Low "
            "completeness = partial annulus = weak evidence.",
            evidence={"phase_match_counts": matched,
                      "rings": rings.get("rings", [])[:5]},
        ))

    peaks = find_peaks_2d(s)
    findings.append(Finding(
        diagnostic="diffuse_bragg_character", severity=Severity.INFO,
        message=f"{len(peaks)} sharp (Bragg-like) peaks detected; structured "
        "intensity between them is the diffuse signal to characterize.",
        evidence={"n_sharp_peaks": len(peaks),
                  "strongest": peaks[:5]},
    ))
    return findings


def diagnose_file(path: str) -> list[Finding]:
    p = Path(path)
    if not p.exists():
        return [Finding(diagnostic="missing_files", severity=Severity.ERROR,
                        message=f"Input file not found: {path}", evidence={"path": path})]
    if _is_hdf5(p):
        return diagnose_volume(path)
    if p.suffix.lower() == ".npz":
        return diagnose_slice(path)
    return [Finding(diagnostic="diffuse_unsupported", severity=Severity.WARNING,
                    message=f"{p.name}: diffuse analysis expects a reciprocal-space "
                    "volume (.nxs) or a saved 2D slice (.npz).", evidence={"path": path})]


def run_all(files: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    for f in files:
        findings.extend(diagnose_file(f))
    return findings


NEXT_CHECK_RULES: dict[str, str] = {
    "bragg_punch_coverage": "Check the Bragg-punch / mask radius and detector gaps; "
    "confirm diffuse averages use only covered annuli, and compare against a plane "
    "with better coverage.",
    "contaminant_rings": "Confirm each ring in the most isotropic plane and against "
    "the sample environment (Al can, Cu, steel, V); exclude the matched annuli before "
    "interpreting diffuse features.",
    "diffuse_anisotropic_sampling": "Re-slice at matched bin widths, or note the "
    "resolution difference, before attributing anisotropy to the sample.",
    "diffuse_unreadable": "Confirm the file is a Mantid MDHisto NeXus volume or a "
    "slice saved by the SDK's slicing tools.",
    "missing_files": "Locate or regenerate the missing files before trusting the analysis.",
}
