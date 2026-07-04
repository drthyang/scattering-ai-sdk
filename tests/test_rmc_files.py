"""RMCProfile file readers (.rmc6f config + R-value CSV), adapted from rmc-toolkits."""

import numpy as np

from scattering_ai import analyze
from scattering_ai.tools.rmc_files import read_rmc6f, read_rmc_csv

RMC6F = """(Version 6f format configuration file)
Metadata material: TestNaCl
Number of atoms:          16
Supercell dimensions:            2   2   2
Cell (Ang/deg):    8.0 8.0 8.0 90.0 90.0 90.0
Lattice vectors (Ang):
    8.0   0.0   0.0
    0.0   8.0   0.0
    0.0   0.0   8.0
Atoms:
1 Na Na 0.0 0.0 0.0 1 0 0 0
2 Cl Cl 0.25 0.25 0.25 2 0 0 0
3 Na Na 0.5 0.0 0.0 1 1 0 0
4 Cl Cl 0.75 0.25 0.25 2 1 0 0
"""

CSV = "step,Rwp_neutron,Rwp_xray\n0,25.3,30.1\n1,18.2,24.5\n2,15.1,20.2\n"


def test_read_rmc6f(tmp_path):
    p = tmp_path / "c.rmc6f"
    p.write_text(RMC6F)
    r = read_rmc6f(str(p))
    assert r["cell"][:3] == [4.0, 4.0, 4.0]  # unit cell = lattice / supercell
    assert r["supercell"] == [2, 2, 2]
    assert r["n_atoms"] == 4
    assert r["composition"] == {"Cl": 2, "Na": 2}
    assert r["title"] == "TestNaCl"
    # positions folded into the unit cell
    assert all(0 <= c < 1 for pos in r["positions"] for c in pos)


def test_read_rmc_csv(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text(CSV)
    c = read_rmc_csv(str(p))
    assert c["labels"] == ["step", "Rwp_neutron", "Rwp_xray"]
    assert np.allclose(c["columns"]["Rwp_neutron"], [25.3, 18.2, 15.1])
    assert c["n_rows"] == 3


def test_rmc6f_routes_and_reads(tmp_path):
    p = tmp_path / "c.rmc6f"
    p.write_text(RMC6F)
    report = analyze(data={"files": [str(p)]})
    assert report.domain == "rmc"
    assert any("RMCProfile configuration" in o and "TestNaCl" in o for o in report.observations)


def test_registry_rmc_tools(tmp_path):
    from scattering_ai.tools.registry import default_toolkit

    p = tmp_path / "c.rmc6f"
    p.write_text(RMC6F)
    reg = default_toolkit(tmp_path / "ws")
    out = reg.execute("read_rmc6f", {"path": str(p)})
    assert out["composition"] == {"Cl": 2, "Na": 2} and "positions" not in out  # trimmed
