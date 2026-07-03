import numpy as np
import pytest

from scattering_ai.tools.cif import lattice_from_cif, predicted_d_spacings, read_cif
from scattering_ai.tools.io import load_curve

CIF_TEXT = """\
data_Al
_chemical_formula_sum            'Al4'
_symmetry_space_group_name_H-M   'F m -3 m'
_symmetry_Int_Tables_number      225
_cell_length_a                   4.0495(2)
_cell_length_b                   4.0495(2)
_cell_length_c                   4.0495(2)
_cell_angle_alpha                90
_cell_angle_beta                 90
_cell_angle_gamma                90
loop_
_atom_site_label
_atom_site_fract_x
Al1 0.0
"""


def test_read_cif_and_lattice(tmp_path):
    path = tmp_path / "al.cif"
    path.write_text(CIF_TEXT)
    info = read_cif(path)
    assert info["a"] == 4.0495  # uncertainty stripped
    assert info["space_group"] == "F m -3 m"
    assert info["formula"] == "Al4"

    lattice = lattice_from_cif(path)
    assert lattice.d_spacing((1, 1, 1)) == pytest.approx(2.338, abs=0.001)


def test_predicted_d_spacings(tmp_path):
    path = tmp_path / "al.cif"
    path.write_text(CIF_TEXT)
    rows = predicted_d_spacings(lattice_from_cif(path), d_min=1.0)
    d_values = [r["d"] for r in rows]
    assert d_values == sorted(d_values, reverse=True)
    # geometric list (no structure factors): (100) and (111) both present
    assert any(abs(d - 4.0495) < 0.001 for d in d_values)
    assert any(abs(d - 2.338) < 0.001 for d in d_values)


def test_cif_without_cell_rejected(tmp_path):
    path = tmp_path / "bad.cif"
    path.write_text("data_x\n_chemical_formula_sum 'X'\n")
    with pytest.raises(ValueError, match="unit cell"):
        read_cif(path)


def test_inspect_cif_registry_tool(tmp_path):
    from scattering_ai.tools.registry import default_toolkit

    path = tmp_path / "al.cif"
    path.write_text(CIF_TEXT)
    result = default_toolkit(tmp_path / "ws").execute("inspect_cif", {"path": str(path)})
    assert result["cell"]["a"] == 4.0495
    assert result["predicted_d_spacings"]


def test_nexus_1d_curve(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "curve.nxs"
    x = np.linspace(0.5, 10, 300)
    y = np.sin(x) + 2
    with h5py.File(path, "w") as f:
        f.attrs["default"] = "entry"
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["default"] = "data"
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data.attrs["signal"] = "counts"
        data.attrs["axes"] = ["Q"]
        data.create_dataset("counts", data=y)
        q_ds = data.create_dataset("Q", data=x)
        q_ds.attrs["units"] = "1/angstrom"

    curve = load_curve(path)
    assert curve.x.size == 300
    assert curve.xlabel == "Q (1/angstrom)"
    assert curve.ylabel == "counts"
    assert curve.meta["format"] == "nexus"
    assert np.allclose(curve.y, y)


def test_nexus_3d_signal_rejected(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "vol.nxs"
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data.attrs["signal"] = "counts"
        data.create_dataset("counts", data=np.ones((4, 5, 6)))
    with pytest.raises(ValueError, match="3D"):
        load_curve(path)
