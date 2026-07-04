"""Domain auto-routing: one entry point picks the right technique pack."""

from pathlib import Path

import numpy as np
import pytest

from scattering_ai import AnalysisRequest, analyze
from scattering_ai.domains.router import detect_domain, resolve_domain
from scattering_ai.tools.models import Slice2D
from scattering_ai.tools.registry import _save_slice

DATA = Path(__file__).parents[1] / "data"


def _req(files=None, **data):
    return AnalysisRequest(domain="auto", question="?",
                           data={"files": files or [], **data})


def test_rmc_state_routes_to_rmc():
    assert detect_domain(_req(r_values=[9.0, 8.0, 7.0]))[0] == "rmc"
    assert detect_domain(_req(run_summary={"steps": 100}))[0] == "rmc"
    assert detect_domain(_req(metadata={"rmc": {"series": []}}))[0] == "rmc"


def test_extension_routing(tmp_path):
    for name, expected in [("x.gr", "pdf"), ("x.fq", "pdf"), ("x.sq", "pdf"),
                           ("x.nxs", "diffuse"), ("x.npz", "diffuse")]:
        p = tmp_path / name
        p.write_bytes(b"\x89HDF" if name.endswith(".nxs") else b"stub")
        assert detect_domain(_req([str(p)]))[0] == expected, name


def test_hdf5_magic_routes_to_diffuse(tmp_path):
    p = tmp_path / "volume.data"  # misleading extension, real HDF5 content
    p.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 16)
    assert detect_domain(_req([str(p)]))[0] == "diffuse"


def test_total_scattering_name_hint_beats_extension(tmp_path):
    p = tmp_path / "Neutron_sample_SofQ.dat"  # .dat, but S(Q) by name
    np.savetxt(p, np.column_stack([np.linspace(1, 20, 50), np.ones(50)]))
    assert detect_domain(_req([str(p)]))[0] == "pdf"


def test_plain_diffraction_pattern_falls_through_to_data(tmp_path):
    """A background-subtracted powder pattern is not total scattering -> data."""
    p = tmp_path / "sample_tth_bgsub.dat"
    x = np.linspace(1, 12, 400)
    y = np.zeros_like(x)
    y[150:155] = 50.0  # a Bragg peak on ~0 background
    np.savetxt(p, np.column_stack([x, y]))
    assert detect_domain(_req([str(p)]))[0] == "data"


def test_no_input_falls_back_to_data():
    assert detect_domain(_req([]))[0] == "data"


def test_expand_files_keeps_bracketed_filenames(tmp_path):
    """Regression: real facility names contain glob chars ('[h,0,0]', '(0,k,l)');
    an existing literal path must win over glob interpretation."""
    from scattering_ai.core.files import expand_files

    p = tmp_path / "TbTi3Bi4_(0,k,l)_[h,0,0]_[-12.0,12.0].nxs"
    p.write_bytes(b"\x89HDF")
    assert expand_files([str(p)]) == [str(p)]
    report_files = _req([str(p)])
    assert detect_domain(report_files)[0] == "diffuse"


def test_analyze_expands_directory_like_the_cli(tmp_path):
    """A directory or glob in files is expanded by the Python API, not only the
    CLI, so analyze(data={'files': [dir]}) runs a whole series."""
    from scattering_ai import analyze
    from scattering_ai.core.files import expand_files

    for t in (5.0, 10.0, 15.0, 20.0):
        (tmp_path / f"scan_T_base_{t:.1f}K.dat").write_text(
            "\n".join(f"{x} {1.0}" for x in range(50)))
    (tmp_path / "notes.md").write_text("skip")
    assert len(expand_files([str(tmp_path)])) == 4  # .md excluded
    assert len(expand_files([str(tmp_path / "*.dat")])) == 4

    report = analyze(data={"files": [str(tmp_path)]})
    assert report.domain == "data"  # a folder of scans routed and analysed


def test_resolve_respects_explicit_domain():
    assert resolve_domain(AnalysisRequest(domain="pdf", question="?"))[0] == "pdf"
    assert resolve_domain(AnalysisRequest(domain="auto", question="?"))[0] == "data"


def test_analyze_records_resolved_domain_and_note(tmp_path):
    slc = tmp_path / "s.npz"
    _save_slice(Slice2D(data=np.ones((10, 10)), x_centers=np.arange(10.0),
                        y_centers=np.arange(10.0)), slc)
    report = analyze(data={"files": [str(slc)]})  # no domain
    assert report.domain == "diffuse"
    assert report.observations[0].startswith("Auto-routed to the 'diffuse' domain")


def test_explicit_domain_has_no_routing_note():
    report = analyze(domain="data", question="?", data={"files": []})
    assert report.domain == "data"
    assert not any(o.startswith("Auto-routed") for o in report.observations)


@pytest.mark.skipif(not (DATA / "1d" / "curves" / "XRAY_FeCoSn_100K_converted.gr").exists(),
                    reason="real data not present")
def test_real_gr_auto_routes_to_pdf():
    gr = DATA / "1d" / "curves" / "XRAY_FeCoSn_100K_converted.gr"
    report = analyze(data={"files": [str(gr)]})
    assert report.domain == "pdf"
