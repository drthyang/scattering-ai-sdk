"""Symmetry technique pack — validated against known crystallographic answers.

All tests need spglib (the `symmetry` extra); skipped if absent.
"""

import numpy as np
import pytest

pytest.importorskip("spglib")

from scattering_ai import Agent, AnalysisRequest  # noqa: E402
from scattering_ai.domains.router import detect_domain  # noqa: E402
from scattering_ai.tools import symmetry as sym  # noqa: E402
from scattering_ai.tools.cif import read_structure  # noqa: E402
from scattering_ai.tools.registry import default_toolkit  # noqa: E402

# Cubic perovskite SrTiO3, Pm-3m (#221)
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

BCC_CIF = """data_W
_cell_length_a 3.16
_cell_length_b 3.16
_cell_length_c 3.16
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_space_group_symop_operation_xyz
x,y,z
1/2+x,1/2+y,1/2+z
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
W1 0.0 0.0 0.0
"""


# ------------------------------------------------------------- find_symmetry


def test_find_symmetry_perovskite():
    fs = sym.find_symmetry(**PEROV)
    assert fs["number"] == 221 and fs["international"] == "Pm-3m"
    assert fs["crystal_system"] == "cubic" and fs["point_group"] == "m-3m"
    assert fs["n_operations"] == 48
    sites = {s["species"]: s["wyckoff"] for s in fs["wyckoff_sites"]}
    assert sites == {"Sr": "a", "Ti": "b", "O": "c"}


def test_cell_matrix_orthogonal_and_hexagonal():
    m = sym.cell_matrix(3, 4, 5, 90, 90, 90)
    assert np.allclose(m, np.diag([3, 4, 5]))
    hexm = sym.cell_matrix(3, 3, 5, 90, 90, 120)
    assert abs(np.linalg.norm(hexm[0]) - 3) < 1e-9
    assert abs(np.dot(hexm[0], hexm[1]) / 9 + 0.5) < 1e-9  # 120 deg


# --------------------------------------------------------- maximal subgroups


def test_maximal_subgroups_of_pm3m():
    """Known answer (International Tables A1): the maximal subgroups of Pm-3m."""
    tree = sym.maximal_subgroups(**PEROV)
    assert tree["parent"]["number"] == 221
    by_number = {s["number"]: s for s in tree["subgroups"]}
    # index-2: Pm-3 (200), P432 (207), P-43m (215)
    assert {200, 207, 215} <= set(by_number)
    assert all(by_number[n]["index"] == 2 for n in (200, 207, 215))
    # index-3 tetragonal P4/mmm (123) with 3 domain variants
    assert by_number[123]["index"] == 3 and by_number[123]["n_variants"] == 3
    # index-4 rhombohedral R-3m (166) with 4 variants
    assert by_number[166]["index"] == 4 and by_number[166]["n_variants"] == 4
    assert all(s["kind"] == "t" for s in tree["subgroups"])


def test_subgroup_tree_figure(tmp_path):
    tree = sym.maximal_subgroups(**PEROV)
    out = sym.subgroup_tree_figure(tree, tmp_path / "nested" / "tree.png")
    assert (tmp_path / "nested" / "tree.png").exists() and out.endswith(".png")


# --------------------------------------------------------- pseudosymmetry


def test_pseudosymmetry_finds_cubic_parent():
    distorted = dict(PEROV)
    distorted["positions"] = [[0, 0, 0], [0.5, 0.5, 0.52],  # Ti off-centre -> P4mm
                              [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    ps = sym.pseudosymmetry_scan(**distorted)
    assert ps["pseudosymmetric"]
    assert ps["tightest"]["number"] != 221
    assert ps["highest_symmetry"]["number"] == 221  # recovers the cubic parent


# --------------------------------------------------------- magnetic


def test_magnetic_symmetry_runs():
    mg = sym.magnetic_symmetry(magmoms=[0, 1.0, 0, 0, 0], **PEROV)
    assert isinstance(mg["uni_number"], int)
    assert mg["msg_type"] in (1, 2, 3, 4)
    assert "type" in mg["msg_type_meaning"]


# --------------------------------------------------------- CIF reading


def test_read_structure_p1(tmp_path):
    p = tmp_path / "perov.cif"
    p.write_text(PEROV_CIF)
    s = read_structure(str(p))
    assert s["n_atoms"] == 5 and s["n_symops"] == 1
    assert sym.find_symmetry(s["lattice"], s["positions"], s["species"])["number"] == 221


def test_read_structure_expands_symops(tmp_path):
    p = tmp_path / "bcc.cif"
    p.write_text(BCC_CIF)
    s = read_structure(str(p))
    assert s["n_asymmetric"] == 1 and s["n_symops"] == 2 and s["n_atoms"] == 2
    assert sym.find_symmetry(s["lattice"], s["positions"], s["species"])["number"] == 229


def test_apply_symop():
    from scattering_ai.tools.cif import _apply_symop

    assert np.allclose(_apply_symop("x,y,z", np.array([0.1, 0.2, 0.3])), [0.1, 0.2, 0.3])
    assert np.allclose(_apply_symop("-x,-y,z", np.array([0.1, 0.2, 0.3])), [0.9, 0.8, 0.3])
    assert np.allclose(_apply_symop("1/2+x,1/2-y,z", np.array([0.1, 0.2, 0.3])), [0.6, 0.3, 0.3])


# --------------------------------------------------------- pack + routing


def test_cif_routes_to_symmetry():
    req = AnalysisRequest(domain="auto", question="?", data={"files": ["structure.cif"]})
    assert detect_domain(req)[0] == "symmetry"


def test_analyze_cif_end_to_end(tmp_path):
    p = tmp_path / "perov.cif"
    p.write_text(PEROV_CIF)
    report = Agent(workspace=tmp_path / "ws").analyze(
        AnalysisRequest(question="Symmetry and transitions?", data={"files": [str(p)]}))
    assert report.domain == "symmetry"
    assert any("Pm-3m" in o for o in report.observations)
    assert any("maximal subgroup" in o for o in report.observations)
    names = {f.rsplit("/", 1)[-1] for f in report.figures}
    assert {"structure.png", "subgroup_tree.png"} <= names
    assert report.provenance.missing_fields() == []


def test_registry_symmetry_tools(tmp_path):
    p = tmp_path / "perov.cif"
    p.write_text(PEROV_CIF)
    reg = default_toolkit(tmp_path / "ws")
    fs = reg.execute("find_symmetry", {"path": str(p)})
    assert fs["international"] == "Pm-3m"
    tree = reg.execute("subgroup_tree", {"path": str(p)})
    assert 123 in {s["number"] for s in tree["subgroups"]}
    assert tree["plot"].endswith(".png")
