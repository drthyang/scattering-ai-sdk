from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from scattering_ai.tools.volumes import load_volume  # noqa: E402

from .test_volumes import make_synthetic_mdhisto  # noqa: E402

VOLUMES = Path(__file__).parents[1] / "data" / "3d" / "volumes"
REAL = sorted(VOLUMES.glob("*.nxs"))


def test_oblique_reproduces_axis_aligned(tmp_path):
    """An oblique cut with axis-aligned basis must match slice()."""
    vol = load_volume(make_synthetic_mdhisto(tmp_path / "mini.nxs"))
    aligned = vol.slice(axis=2, center=0.25, thickness=0.4)  # one D2 bin

    oblique = vol.oblique_slice(
        origin=(0.0, 0.0, 0.25),
        u_axis=(1, 0, 0),
        v_axis=(0, 1, 0),
        u_range=(-1.75, 1.75),  # D0 bin centers
        v_range=(-2.5, 2.5),  # D1 bin centers
        thickness=0.0,
    )
    # sample oblique at the aligned grid: signal = d0 index in both
    expected = np.broadcast_to(np.arange(8, dtype=float), (6, 8))
    interp = np.array(
        [np.interp(aligned.x_centers, oblique.x_centers,
                   oblique.data[np.argmin(abs(oblique.y_centers - yc))])
         for yc in aligned.y_centers]
    )
    assert np.allclose(interp, expected, atol=1e-6)


def test_oblique_diagonal_plane(tmp_path):
    """A 45-degree plane through a linear field samples the analytic value."""
    vol = load_volume(make_synthetic_mdhisto(tmp_path / "mini.nxs"))
    # signal = d0 index; d0 index = (h - first_center)/step
    s = vol.oblique_slice(
        origin=(0.0, 0.0, 0.0),
        u_axis=(1, 1, 0),  # diagonal in (D0, D1)
        v_axis=(0, 0, 1),
        u_range=(-1.5, 1.5),
        v_range=(-0.5, 0.5),
        thickness=0.0,
    )
    first, step = -1.75, 0.5
    expected_d0_index = (s.x_centers - first) / step  # h coordinate = s here
    row = s.data[len(s.y_centers) // 2]
    valid = np.isfinite(row)
    assert valid.sum() > 3
    assert np.allclose(row[valid], expected_d0_index[valid], atol=1e-6)


def test_oblique_rejects_parallel_axes(tmp_path):
    vol = load_volume(make_synthetic_mdhisto(tmp_path / "mini.nxs"))
    with pytest.raises(ValueError, match="parallel"):
        vol.oblique_slice(
            origin=(0, 0, 0), u_axis=(1, 0, 0), v_axis=(2, 0, 0),
            u_range=(-1, 1), v_range=(-1, 1),
        )


@pytest.mark.skipif(not REAL, reason="real CORELLI volume not present")
def test_real_oblique_matches_axis_aligned_slice():
    vol = load_volume(REAL[0])
    aligned = vol.slice(axis=0, center=0.0, thickness=0.5)

    oblique = vol.oblique_slice(
        origin=(0.0, 0.0, 0.0),
        u_axis=(0, 1, 0),
        v_axis=(0, 0, 1),
        u_range=(-29.5, 29.5),
        v_range=(-4.9, 4.9),
        thickness=0.5,
        du=0.15,
        dv=1.0 / 30,
        n_layers=9,
    )
    # compare on a coarse common grid in the well-covered center
    k_test = np.arange(-20, 20, 2.5)
    l_test = np.arange(-4, 4.5, 1.0)
    a_vals, o_vals = [], []
    for lv in l_test:
        ai = np.argmin(abs(aligned.y_centers - lv))
        oi = np.argmin(abs(oblique.y_centers - lv))
        a_vals.append(np.interp(k_test, aligned.x_centers, aligned.data[ai]))
        o_vals.append(np.interp(k_test, oblique.x_centers, oblique.data[oi]))
    a_vals, o_vals = np.concatenate(a_vals), np.concatenate(o_vals)
    good = np.isfinite(a_vals) & np.isfinite(o_vals)
    corr = np.corrcoef(a_vals[good], o_vals[good])[0, 1]
    assert corr > 0.95
