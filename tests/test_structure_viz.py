"""Structure visualization (CIF/mCIF) + skill organization."""

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from scattering_ai.tools.registry import default_toolkit  # noqa: E402

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
"""

AFM_MCIF = """data_MnAFM
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
"""


def test_plot_structure_runs(tmp_path):
    from scattering_ai.tools.structure_viz import plot_structure

    out = plot_structure([4, 4, 4, 90, 90, 90], [[0, 0, 0], [0.5, 0.5, 0.5]],
                         ["Na", "Cl"], tmp_path / "s" / "nacl.png")
    assert (tmp_path / "s" / "nacl.png").exists() and out.endswith(".png")


def test_mcif_moments_parsed_and_transformed(tmp_path):
    from scattering_ai.tools.cif import read_structure

    p = tmp_path / "afm.mcif"
    p.write_text(AFM_MCIF)
    s = read_structure(str(p))
    assert s["n_magnetic"] == 2
    assert s["moments"][0][2] == 4.0 and s["moments"][1][2] == -4.0


def test_symop_matrix_roundtrip():
    import numpy as np

    from scattering_ai.tools.cif import _symop_matrix

    r, t = _symop_matrix("1/2-x, y, 1/2+z")
    assert np.allclose(r, [[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert np.allclose(t, [0.5, 0.0, 0.5])


def test_plot_structure_tool(tmp_path):
    p = tmp_path / "perov.cif"
    p.write_text(PEROV_CIF)
    reg = default_toolkit(tmp_path / "ws")
    out = reg.execute("plot_structure", {"path": str(p)})
    assert out["n_atoms"] == 3 and Path(out["saved"]).exists()


def test_visualize_structure_skill_magnetic(tmp_path):
    p = tmp_path / "afm.mcif"
    p.write_text(AFM_MCIF)
    reg = default_toolkit(tmp_path / "ws")
    out = reg.execute("skill_visualize_structure", {"path": str(p)})
    assert out["n_magnetic"] == 2 and out["assessment"]["is_magnetic"]
    assert out["figures"] and out["figures"][0].endswith(".png")


def test_skills_are_categorized(tmp_path):
    reg = default_toolkit(tmp_path / "ws")
    cats = reg.skills_by_category()
    assert cats["structure"] == ["skill_symmetry_overview", "skill_visualize_structure"]
    assert cats["series & transitions"] == ["skill_scan_series_transitions"]
    assert cats["1D patterns"] == ["skill_fit_pattern_peaks"]
    assert cats["2D slices"] == ["skill_characterize_slice"]
    assert cats["3D volumes"] == ["skill_delta_pdf"]
