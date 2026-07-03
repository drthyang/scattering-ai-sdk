import numpy as np
import pytest

from scattering_ai.tools.models import Slice2D
from scattering_ai.tools.slices import (
    azimuthal_profile,
    detect_rings,
    estimate_background_2d,
    find_peaks_2d,
    line_cut,
)

RNG = np.random.default_rng(7)
AL_220_Q = 2 * np.pi / 1.432  # 4.388 1/A


def synthetic_slice():
    """Flat background + noise, 3 point peaks, one Al(220) ring, NaN patch.

    Axes are directly in 1/A (scales = 1) spanning +-6.
    """
    x = np.linspace(-6, 6, 401)
    y = np.linspace(-6, 6, 401)
    xx, yy = np.meshgrid(x, y)
    data = 1.0 + RNG.normal(0, 0.05, xx.shape)
    for height, x0, y0 in [(20.0, 2.0, 1.0), (15.0, -3.0, -2.0), (10.0, 0.0, 4.0)]:
        data += height * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 0.08**2))
    q = np.sqrt(xx**2 + yy**2)
    data += 1.5 * np.exp(-((q - AL_220_Q) ** 2) / (2 * 0.05**2))  # ring
    data[10:40, 10:40] = np.nan
    return Slice2D(data=data, x_centers=x, y_centers=y, xlabel="qx", ylabel="qy")


def test_find_peaks_2d_finds_planted_peaks():
    s = synthetic_slice()
    peaks = find_peaks_2d(s, min_snr=20)
    assert len(peaks) >= 3
    found = {(round(p["x"], 1), round(p["y"], 1)) for p in peaks[:3]}
    assert found == {(2.0, 1.0), (-3.0, -2.0), (0.0, 4.0)}
    # ring must not appear as the top point peaks
    for p in peaks[:3]:
        assert abs(np.hypot(p["x"], p["y"]) - AL_220_Q) > 0.3


def test_background_2d_is_flat_here():
    s = synthetic_slice()
    bg = estimate_background_2d(s)
    finite = np.isfinite(bg)
    assert abs(np.nanmedian(bg[finite]) - 1.0) < 0.05


def test_azimuthal_profile_shows_ring_and_completeness():
    s = synthetic_slice()
    prof = azimuthal_profile(s, x_scale=1.0, y_scale=1.0)
    ring_region = (prof.x > AL_220_Q - 0.1) & (prof.x < AL_220_Q + 0.1)
    off_region = (prof.x > 2.0) & (prof.x < 3.0)
    assert prof.y[ring_region].max() > prof.y[off_region].mean() + 1.0
    completeness = np.asarray(prof.meta["completeness"])
    # full annuli well inside the field of view
    inside = prof.x < 5.0
    assert np.median(completeness[inside]) > 0.8


def test_detect_rings_identifies_aluminum():
    s = synthetic_slice()
    result = detect_rings(s, x_scale=1.0, y_scale=1.0)
    assert result["n_rings"] >= 1
    top = result["rings"][0]
    assert abs(top["q"] - AL_220_Q) < 0.05
    assert any(m["phase"] == "Al" for m in top["candidates"])
    assert result["phase_match_counts"].get("Al", 0) >= 1
    assert top["completeness"] > 0.8


def test_line_cut_through_peak():
    s = synthetic_slice()
    cut = line_cut(s, start=(-6.0, 1.0), end=(6.0, 1.0), width=0.1)
    assert cut.x.size > 100
    peak_pos = cut.x[np.nanargmax(cut.y)]
    assert abs(peak_pos - 8.0) < 0.1  # peak at x=2 → distance 8 from x=-6

    # NaN patch region must not fabricate values: cut through it
    cut2 = line_cut(s, start=(-5.9, -5.9), end=(-5.0, -5.0), width=0.0)
    assert np.isnan(cut2.y).any()


def test_line_cut_rejects_zero_length():
    s = synthetic_slice()
    with pytest.raises(ValueError):
        line_cut(s, start=(1.0, 1.0), end=(1.0, 1.0))
