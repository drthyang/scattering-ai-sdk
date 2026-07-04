"""Diffuse-scattering domain pack: the second technique pack (B4).

Slice diagnostics run on synthetic slices in CI; the volume path is exercised
on the real CORELLI data when present (needs h5py).
"""

from pathlib import Path

import numpy as np
import pytest

from scattering_ai import analyze
from scattering_ai.core.findings import Severity
from scattering_ai.domains.diffuse import diagnostics as dd
from scattering_ai.domains.registry import get_domain
from scattering_ai.tools.models import Slice2D
from scattering_ai.tools.registry import _save_slice

VOLUME = next((Path(__file__).parents[1] / "data" / "3d" / "volumes").glob("*.nxs"), None)


def _slice(nan_fraction: float = 0.0) -> Slice2D:
    """Synthetic diffuse map: broad background + a few sharp Bragg peaks."""
    n = 120
    ax = np.linspace(-6, 6, n)
    xx, yy = np.meshgrid(ax, ax)
    data = 2.0 + np.exp(-(xx**2 + yy**2) / 40.0)  # broad diffuse-ish background
    rng = np.random.default_rng(3)
    for cx, cy in [(-3, -3), (0, 0), (3, 2), (-2, 4)]:
        data += 50 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.08)
    data += rng.normal(0, 0.05, data.shape)
    if nan_fraction:
        mask = rng.random(data.shape) < nan_fraction
        data[mask] = np.nan
    return Slice2D(data=data, x_centers=ax, y_centers=ax, xlabel="k (rlu)", ylabel="l (rlu)")


def _saved(tmp_path, **kw) -> str:
    path = tmp_path / "slice.npz"
    _save_slice(_slice(**kw), path)
    return str(path)


# ----------------------------------------------------------- slice diagnostics


def test_full_coverage_slice_reports_info(tmp_path):
    kinds = {f.diagnostic: f for f in dd.diagnose_slice(_saved(tmp_path))}
    assert kinds["diffuse_slice_coverage"].severity == Severity.INFO
    # sharp Bragg peaks are found and separated from the diffuse background
    assert kinds["diffuse_bragg_character"].evidence["n_sharp_peaks"] >= 4


def test_low_coverage_flags_bragg_punch(tmp_path):
    findings = {f.diagnostic: f for f in dd.diagnose_slice(_saved(tmp_path, nan_fraction=0.5))}
    assert "bragg_punch_coverage" in findings
    assert findings["bragg_punch_coverage"].severity == Severity.WARNING
    assert findings["bragg_punch_coverage"].evidence["coverage"] < dd.MIN_COVERAGE


def test_file_routing_and_errors(tmp_path):
    assert dd.diagnose_file("/no/such/vol.nxs")[0].diagnostic == "missing_files"
    txt = tmp_path / "notes.txt"
    txt.write_text("hi")
    assert dd.diagnose_file(str(txt))[0].diagnostic == "diffuse_unsupported"


def test_is_hdf5_by_suffix_and_magic(tmp_path):
    assert dd._is_hdf5(Path("x.nxs"))
    hdf = tmp_path / "y.dat"
    hdf.write_bytes(b"\x89HDF\r\n\x1a\n....")
    assert dd._is_hdf5(hdf)
    plain = tmp_path / "z.npz"
    plain.write_bytes(b"PK\x03\x04")
    assert not dd._is_hdf5(plain)


# --------------------------------------------------------------- architecture


def test_diffuse_pack_registered():
    pack = get_domain("diffuse")
    assert pack.name == "diffuse"
    assert pack.prompt_version == "diffuse_interpret/v1"
    assert "contaminant_rings" in pack.next_check_rules
    # four built-in domains now coexist
    from scattering_ai.domains.registry import _BUILTIN

    assert {"rmc", "data", "pdf", "diffuse"} <= set(_BUILTIN)


def test_analyze_routes_to_diffuse_and_is_attributable(tmp_path):
    report = analyze(domain="diffuse", question="What's in this slice?",
                     data={"files": [_saved(tmp_path)]})
    assert report.provenance.missing_fields() == []  # D4: offline is complete
    assert any("sharp" in o for o in report.observations)


# ----------------------------------------------------------------- real data


@pytest.mark.skipif(VOLUME is None, reason="CORELLI volume not present")
def test_real_volume_metadata_and_anisotropy():
    pytest.importorskip("h5py")
    findings = {f.diagnostic: f for f in dd.diagnose_volume(str(VOLUME))}
    assert "diffuse_volume" in findings
    # CORELLI sampling is fine in-plane, coarse out-of-plane
    assert "diffuse_anisotropic_sampling" in findings
    widths = findings["diffuse_volume"].evidence["bin_widths"]
    assert max(widths.values()) / min(widths.values()) >= dd.ANISOTROPY_RATIO
