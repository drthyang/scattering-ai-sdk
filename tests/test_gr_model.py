"""PDF model calculation + fitting (G(r) from a structure) — known answers."""

from pathlib import Path

import numpy as np
import pytest

from scattering_ai import Agent, AnalysisRequest
from scattering_ai.tools.gr_model import fit_gr, simulate_gr

PEROV = dict(
    lattice=[3.905, 3.905, 3.905, 90, 90, 90],
    positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    species=["Sr", "Ti", "O", "O", "O"],
)

PEROV_CIF = """data_SrTiO3
_cell_length_a 3.905
_cell_length_b 3.905
_cell_length_c 3.905
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Sr 0.0 0.0 0.0
Ti 0.5 0.5 0.5
O 0.5 0.5 0.0
O 0.5 0.0 0.5
O 0.0 0.5 0.5
"""


def _g_at(sim, r):
    return float(sim["g"][np.argmin(np.abs(sim["r"] - r))])


def test_simulate_gr_known_pair_distances():
    sim = simulate_gr(**PEROV, rmax=8, sigma=0.08)
    # Sr-O / O-O shell at a/sqrt(2) = 2.761 Å: strong positive peak
    assert _g_at(sim, 2.761) > 5
    # cell repeat at 3.905 Å: positive peak
    assert _g_at(sim, 3.905) > 5
    # nothing below the first bond
    assert abs(_g_at(sim, 1.2) + 4 * np.pi * 1.2 * sim["rho0"] * 0) < 5  # near baseline


def test_neutron_vs_xray_weighting():
    """Ti has a negative neutron b: the Ti-O shell at 1.95 Å must be a NEGATIVE
    dip with neutrons and a positive peak with x-rays."""
    n = simulate_gr(**PEROV, rmax=6, sigma=0.08, radiation="neutron")
    x = simulate_gr(**PEROV, rmax=6, sigma=0.08, radiation="xray")
    assert _g_at(n, 1.9525) < -2
    assert _g_at(x, 1.9525) > 0.5


def test_fit_gr_self_consistency():
    """Simulate 'data' with known scale/sigma/lattice; the fit must recover all
    three with Rw ~ 0."""
    data = simulate_gr([3.905 * 1.01] * 3 + [90] * 3, PEROV["positions"],
                       PEROV["species"], rmax=15, sigma=0.12)
    fit = fit_gr(data["r"], 2.5 * data["g"], **PEROV)
    assert fit["rw"] < 0.02
    assert abs(fit["scale"] - 2.5) < 0.05
    assert abs(fit["sigma"] - 0.12) < 0.01
    assert abs(fit["lattice_scale"] - 1.01) < 0.002
    assert fit["assessment"] == "good"


def test_fit_gr_rejects_wrong_model():
    data = simulate_gr(**PEROV, rmax=12, sigma=0.1)
    bad = fit_gr(data["r"], data["g"], [4.5] * 3 + [90] * 3, [[0, 0, 0]], ["Cu"])
    assert bad["rw"] > 0.5 and bad["assessment"] == "poor"


def test_registry_tools_and_pdf_pack(tmp_path):
    pytest.importorskip("matplotlib")
    from scattering_ai.tools.registry import default_toolkit

    cif = tmp_path / "perov.cif"
    cif.write_text(PEROV_CIF)
    sim = simulate_gr(**PEROV, rmax=12, sigma=0.1)
    gr = tmp_path / "measured.gr"
    gr.write_text("outputtype = gr\n#### start data\n#L r  G(r)\n"
                  + "\n".join(f"{a} {1.5 * b}" for a, b in
                              zip(sim["r"], sim["g"], strict=True)))

    reg = default_toolkit(tmp_path / "ws")
    model = reg.execute("simulate_gr_from_cif", {"path": str(cif), "rmax": 10.0})
    assert Path(model["saved"]).exists() and model["first_peaks"]
    fit = reg.execute("fit_gr_model", {"gr_path": str(gr), "cif_path": str(cif)})
    assert fit["rw"] < 0.02 and fit["plot"].endswith(".png")

    # CIF + G(r) auto-routes to pdf and reports the model fit with a figure
    report = Agent(workspace=tmp_path / "ws2").analyze(
        AnalysisRequest(question="?", data={"files": [str(gr), str(cif)]}))
    assert report.domain == "pdf"
    assert any("Rw" in o for o in report.observations)
    assert any(f.endswith("gr_model_fit.png") for f in report.figures)
