from pathlib import Path

import numpy as np
import pytest

from scattering_ai.tools.lattice import Lattice

VOLUMES = Path(__file__).parents[1] / "data" / "3d" / "volumes"
REAL = sorted(VOLUMES.glob("*.nxs"))


def general_d_spacing(lat: Lattice, h, k, l):  # noqa: E741
    """Reference: general triclinic d-spacing via the reciprocal metric."""
    ca, cb, cg = (np.cos(np.radians(x)) for x in (lat.alpha, lat.beta, lat.gamma))
    sa, sb, sg = (np.sin(np.radians(x)) for x in (lat.alpha, lat.beta, lat.gamma))
    v2 = (1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg)
    s11 = (sa / lat.a) ** 2
    s22 = (sb / lat.b) ** 2
    s33 = (sg / lat.c) ** 2
    s12 = 2 * (ca * cb - cg) / (lat.a * lat.b)
    s13 = 2 * (cg * ca - cb) / (lat.a * lat.c)
    s23 = 2 * (cb * cg - ca) / (lat.b * lat.c)
    inv_d2 = (s11 * h**2 + s22 * k**2 + s33 * l**2
              + s12 * h * k + s13 * h * l + s23 * k * l) / v2
    return 1 / np.sqrt(inv_d2)


def test_cubic_known_values():
    al = Lattice(4.0495, 4.0495, 4.0495)
    assert al.d_spacing((1, 1, 1)) == pytest.approx(2.338, abs=0.001)
    assert al.d_spacing((2, 0, 0)) == pytest.approx(2.0248, abs=0.001)
    assert al.q_magnitude(np.array([1.0, 1.0, 1.0])) == pytest.approx(2 * np.pi / 2.338,
                                                                      rel=1e-3)


def test_orthorhombic_matches_simple_formula():
    lat = Lattice(5.0, 7.0, 11.0)
    for hkl in [(1, 0, 0), (0, 2, 0), (1, 1, 1), (3, 2, 1)]:
        h, k, l = hkl  # noqa: E741
        expected = 2 * np.pi * np.sqrt((h / 5.0) ** 2 + (k / 7.0) ** 2 + (l / 11.0) ** 2)
        assert lat.q_magnitude(np.array(hkl, dtype=float)) == pytest.approx(expected,
                                                                            rel=1e-10)


def test_triclinic_against_metric_formula():
    lat = Lattice(5.2, 6.3, 7.9, alpha=83.0, beta=97.5, gamma=112.0)
    for hkl in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (2, -1, 3), (-1, 2, -2)]:
        expected = general_d_spacing(lat, *hkl)
        assert lat.d_spacing(hkl) == pytest.approx(expected, rel=1e-9)


def test_from_dict_and_invalid():
    assert Lattice.from_dict({"a": 1, "b": 2, "c": 3, "alpha": 90, "beta": 90,
                              "gamma": 90}) is not None
    assert Lattice.from_dict({"a": 1}) is None
    assert Lattice.from_dict({}) is None


def test_batched_q_magnitude_shape():
    lat = Lattice(4.0, 5.0, 6.0)
    hkl = np.zeros((7, 9, 3))
    hkl[..., 0] = 1.0
    q = lat.q_magnitude(hkl)
    assert q.shape == (7, 9)
    assert np.allclose(q, 2 * np.pi / 4.0)


@pytest.mark.skipif(not REAL, reason="real CORELLI volume not present")
def test_real_slice_carries_lattice_and_rings_work_without_scales():
    from scattering_ai.tools.slices import detect_rings
    from scattering_ai.tools.volumes import load_volume

    vol = load_volume(REAL[0])
    s = vol.slice(axis=0, center=0.0, thickness=0.5)
    assert s.meta["lattice"]["b"] == pytest.approx(10.41, abs=0.1)
    assert s.meta["hkl_u"] == [0.0, 1.0, 0.0]
    assert s.meta["hkl_v"] == [0.0, 0.0, 1.0]

    # No scales passed: |Q| comes from the B matrix (angles are ~90.3-90.7 deg
    # here, so the exact path matters at the ~1% level)
    result = detect_rings(s)
    assert result["n_rings"] >= 1
    assert result["phase_match_counts"].get("Al", 0) >= 1


def test_bare_slice_without_metadata_requires_scales():
    from scattering_ai.tools.models import Slice2D
    from scattering_ai.tools.slices import azimuthal_profile

    s = Slice2D(data=np.ones((20, 20)), x_centers=np.linspace(-1, 1, 20),
                y_centers=np.linspace(-1, 1, 20))
    with pytest.raises(ValueError, match="x_scale"):
        azimuthal_profile(s)
