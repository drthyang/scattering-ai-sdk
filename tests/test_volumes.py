from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from scattering_ai.tools.volumes import load_volume  # noqa: E402

VOLUMES = Path(__file__).parents[1] / "data" / "3d" / "volumes"
REAL = sorted(VOLUMES.glob("*.nxs"))


def make_synthetic_mdhisto(path):
    """Small MDHisto-like file: 4x6x8 bins, known values, one masked plane."""
    edges = [
        np.linspace(-2, 2, 9),  # D0: 8 bins
        np.linspace(-3, 3, 7),  # D1: 6 bins
        np.linspace(-1, 1, 5),  # D2: 4 bins
    ]
    # signal[d2, d1, d0] = d0 index (so averaging over D2 or D1 keeps it)
    signal = np.broadcast_to(np.arange(8, dtype=float), (4, 6, 8)).copy()
    mask = np.zeros((4, 6, 8), dtype=np.int8)
    mask[0, :, :] = 1  # first D2 plane masked
    signal[1, 0, 0] = np.nan  # one no-coverage bin

    with h5py.File(path, "w") as f:
        group = f.create_group("MDHistoWorkspace")
        group.attrs["NX_class"] = "NXentry"
        data = group.create_group("data")
        for i, e in enumerate(edges):
            ds = data.create_dataset(f"D{i}", data=e)
            ds.attrs["name"] = f"[{'hkl'[i]},0,0]"
            ds.attrs["units"] = "r.l.u."
        data.create_dataset("signal", data=signal)
        data.create_dataset("mask", data=mask)
    return path


def test_synthetic_load_and_slice(tmp_path):
    vol = load_volume(make_synthetic_mdhisto(tmp_path / "mini.nxs"))
    assert [ax.n_bins for ax in vol.axes] == [8, 6, 4]
    assert vol.axes[0].name == "[h,0,0]"

    s = vol.slice(axis=2, center=0.0, thickness=2.0)  # average all D2
    assert s.data.shape == (6, 8)  # (D1, D0)
    # signal = d0 index, independent of d1/d2 → each column equals its index
    expected = np.broadcast_to(np.arange(8, dtype=float), (6, 8))
    assert np.allclose(s.data, expected, equal_nan=True)

    s0 = vol.slice(axis="[h,0,0]", center=0.25, thickness=0.5)  # one D0 bin
    assert s0.data.shape == (4, 6)  # (D2, D1)
    assert s0.meta["n_bins_integrated"] == 1


def test_synthetic_mask_is_excluded(tmp_path):
    vol = load_volume(make_synthetic_mdhisto(tmp_path / "mini.nxs"))
    # Slab covering only the masked D2 plane → everything NaN
    masked_slice = vol.slice(axis=2, center=-0.75, thickness=0.4)
    assert np.isnan(masked_slice.data).all()


def test_slice_outside_range_raises(tmp_path):
    vol = load_volume(make_synthetic_mdhisto(tmp_path / "mini.nxs"))
    with pytest.raises(ValueError, match="no bins"):
        vol.slice(axis=0, center=10.0, thickness=0.1)


@pytest.mark.skipif(not REAL, reason="real CORELLI volume not present")
def test_real_corelli_volume():
    vol = load_volume(REAL[0])
    assert [ax.n_bins for ax in vol.axes] == [401, 401, 301]
    assert abs(vol.lattice["b"] - 10.41) < 0.1  # TbTi3Bi4

    s = vol.slice(axis=0, center=0.0, thickness=0.5)  # (0,k,l) plane
    assert s.data.shape == (301, 401)
    finite = np.isfinite(s.data)
    assert finite.mean() > 0.9
    # Bragg peaks dominate: max far above the median diffuse level
    assert np.nanmax(s.data) > 50 * np.nanmedian(s.data[finite])
