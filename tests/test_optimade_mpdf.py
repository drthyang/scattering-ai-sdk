"""OPTIMADE structure lookup + magnetic PDF — offline known answers.

The OPTIMADE parsing is tested against a canned payload (hermetic); the live
COD query runs only with SCATTERING_AI_LIVE_NET=1.
"""

import os

import numpy as np
import pytest

from scattering_ai.tools.mpdf import simulate_mpdf
from scattering_ai.tools.optimade import build_filter, parse_structures

# ------------------------------------------------------------- OPTIMADE


def test_build_filter():
    assert build_filter(["Sr", "Ti", "O"]) == 'elements HAS ALL "Sr", "Ti", "O"'
    assert "HAS ONLY" in build_filter(["Fe"], exclusive=True)
    assert 'chemical_formula_reduced="GaNb4Se8"' in build_filter(formula="GaNb4Se8")
    with pytest.raises(ValueError):
        build_filter()


def test_parse_structures_canned():
    payload = {"data": [{"id": "1008802", "attributes": {
        "chemical_formula_reduced": "GaNb4Se8",
        "elements": ["Ga", "Nb", "Se"], "nelements": 3,
        "_cod_sg": "F -4 3 m",
        "lattice_vectors": [[10.42, 0, 0], [0, 10.42, 0], [0, 0, 10.42]],
    }}], "meta": {"data_returned": 1}}
    out = parse_structures(payload)
    assert out[0]["formula"] == "GaNb4Se8"
    assert out[0]["space_group"] == "F -4 3 m"
    assert out[0]["cell_lengths"] == [10.42, 10.42, 10.42]


@pytest.mark.skipif(os.environ.get("SCATTERING_AI_LIVE_NET") != "1",
                    reason="set SCATTERING_AI_LIVE_NET=1 for the live COD query")
def test_live_cod_finds_ganb4se8():
    from scattering_ai.tools.optimade import query_structures

    r = query_structures(elements=["Ga", "Nb", "Se"], max_results=3)
    assert "error" not in r
    assert any(c["formula"] == "GaNb4Se8" for c in r["candidates"])


# ------------------------------------------------------------------ mPDF


CSCL = dict(lattice=[4.0] * 3 + [90] * 3, positions=[[0, 0, 0], [0.5, 0.5, 0.5]])
R_NN = 4.0 * np.sqrt(3) / 2  # corner -> body centre


def _f_at(sim, r):
    return float(sim["f"][np.argmin(np.abs(sim["r"] - r))])


def test_mpdf_afm_vs_fm_first_peak_sign():
    afm = simulate_mpdf(spins=[[0, 0, 3], [0, 0, -3]], rmax=10, sigma=0.08, **CSCL)
    fm = simulate_mpdf(spins=[[0, 0, 3], [0, 0, 3]], rmax=10, sigma=0.08, **CSCL)
    assert _f_at(afm, R_NN) < -5      # antiparallel nn -> negative peak
    assert _f_at(fm, R_NN) > 5        # parallel nn -> positive peak
    # same-sublattice neighbours (a = 4.0) are parallel in both -> positive
    assert _f_at(afm, 4.0) > 5


def test_mpdf_requires_moments():
    out = simulate_mpdf(spins=[[0, 0, 0], [0, 0, 0]], rmax=8, **CSCL)
    assert "error" in out


def test_mpdf_tool_from_mcif(tmp_path):
    mcif = tmp_path / "afm.mcif"
    mcif.write_text("""data_MnAFM
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Mn1 Mn 0.0 0.0 0.0
Mn2 Mn 0.5 0.5 0.5
loop_
_atom_site_moment.label
_atom_site_moment.crystalaxis_x
_atom_site_moment.crystalaxis_y
_atom_site_moment.crystalaxis_z
Mn1 0.0 0.0 4.0
Mn2 0.0 0.0 -4.0
""")
    from pathlib import Path

    from scattering_ai.tools.registry import default_toolkit

    reg = default_toolkit(tmp_path / "ws")
    out = reg.execute("simulate_mpdf_from_mcif", {"path": str(mcif), "rmax": 10.0})
    assert out["n_magnetic"] == 2
    # strongest AFM shell: the 24-neighbour inter-sublattice shell at
    # sqrt(11)/2 * a = 6.633 (multiplicity beats the 1/r weighting of the
    # 8-neighbour nn shell at 3.464)
    assert abs(out["strongest_afm_distance"] - 6.633) < 0.15
    assert Path(out["saved"]).exists()


def test_spin_correlations_shells():
    from scattering_ai.tools.mpdf import spin_correlations

    r = spin_correlations(spins=[[0, 0, 3], [0, 0, -3]], rmax=7, **CSCL)
    shells = {round(s["r"], 2): s for s in r["shells"]}
    assert shells[3.46]["correlation"] == -1.0      # inter-sublattice nn: AFM
    assert shells[3.46]["multiplicity"] == 8.0
    assert shells[4.0]["correlation"] == 1.0        # same sublattice: FM
    assert shells[6.63]["correlation"] == -1.0
    assert shells[6.63]["multiplicity"] == 24.0     # why it dominates the mPDF


def test_powder_magnetic_iq_nonnegative_and_fm_vs_afm():
    from scattering_ai.tools.mpdf import powder_magnetic_iq

    afm = powder_magnetic_iq(spins=[[0, 0, 3], [0, 0, -3]], qmin=0.05, qmax=5, **CSCL)
    fm = powder_magnetic_iq(spins=[[0, 0, 3], [0, 0, 3]], qmin=0.05, qmax=5, **CSCL)
    # |M_perp|^2 average is non-negative by construction (validates the formula)
    assert (afm["i"] >= -1e-9).all() and (fm["i"] >= -1e-9).all()
    low = afm["q"] < 0.6
    assert fm["i"][low].sum() > 3 * afm["i"][low].sum()  # FM forward scattering


def test_magnetic_form_factor_decays():
    from scattering_ai.tools.mpdf import magnetic_form_factor

    assert abs(magnetic_form_factor(np.array([0.0]), "Mn2")[0] - 1.0) < 0.01
    assert magnetic_form_factor(np.array([8.0]), "Mn2")[0] < 0.5
    # unknown ion -> point dipole (flat 1.0)
    assert magnetic_form_factor(np.array([5.0]), "Xx")[0] == 1.0


def test_skill_magnetic_diffuse(tmp_path):
    pytest.importorskip("matplotlib")
    from scattering_ai.tools.registry import default_toolkit

    mcif = tmp_path / "afm.mcif"
    mcif.write_text("""data_MnAFM
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Mn1 Mn 0.0 0.0 0.0
Mn2 Mn 0.5 0.5 0.5
loop_
_atom_site_moment.label
_atom_site_moment.crystalaxis_x
_atom_site_moment.crystalaxis_y
_atom_site_moment.crystalaxis_z
Mn1 0.0 0.0 4.0
Mn2 0.0 0.0 -4.0
""")
    reg = default_toolkit(tmp_path / "ws")
    out = reg.execute("skill_magnetic_diffuse", {"path": str(mcif)})
    assert out["step_errors"] == 0
    assert "antiferromagnetic" in out["summary"]
    assert out["form_factor"] == "<j0> Mn2"
    assert len([f for f in out["figures"] if f.endswith(".png")]) == 2
    cats = reg.skills_by_category()
    assert cats["magnetic"] == ["skill_magnetic_diffuse"]
