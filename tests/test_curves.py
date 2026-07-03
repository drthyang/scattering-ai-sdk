import numpy as np
import pytest

from scattering_ai.tools.curves import (
    crop,
    estimate_background,
    find_peaks,
    fit_peaks,
    rebin,
    sq_to_gr,
)
from scattering_ai.tools.models import Curve1D

RNG = np.random.default_rng(42)


def synthetic_pattern():
    """Two pseudo-Voigt-ish peaks on a sloping background with noise."""
    x = np.linspace(0, 10, 2000)
    y = 0.5 + 0.05 * x
    for height, center, fwhm in [(10.0, 3.0, 0.3), (6.0, 7.0, 0.5)]:
        y = y + height * np.exp(-4 * np.log(2) * (x - center) ** 2 / fwhm**2)
    y = y + RNG.normal(0, 0.05, x.size)
    return Curve1D(x=x, y=y, xlabel="Q", ylabel="I")


def test_crop_and_rebin():
    curve = synthetic_pattern()
    cropped = crop(curve, 2.0, 8.0)
    assert cropped.x.min() >= 2.0 and cropped.x.max() <= 8.0

    binned = rebin(curve, dx=0.05)
    assert binned.x.size < curve.x.size
    # Rebinning preserves the integral to good approximation
    integral = np.trapezoid(curve.y, curve.x)
    integral_binned = np.trapezoid(binned.y, binned.x)
    assert abs(integral - integral_binned) / integral < 0.01


def test_background_estimation_ignores_peaks():
    curve = synthetic_pattern()
    bg = estimate_background(curve, degree=1)
    truth = 0.5 + 0.05 * curve.x
    off_peak = (abs(curve.x - 3.0) > 1.0) & (abs(curve.x - 7.0) > 1.5)
    # The estimator must be essentially unbiased off-peak (noise sigma 0.05).
    assert np.abs(bg[off_peak] - truth[off_peak]).mean() < 0.02


def test_find_peaks_finds_both_and_only_both():
    curve = synthetic_pattern()
    peaks = find_peaks(curve, subtract_background=True)
    assert len(peaks) == 2
    positions = sorted(p["x"] for p in peaks)
    assert abs(positions[0] - 3.0) < 0.05
    assert abs(positions[1] - 7.0) < 0.05


def test_fit_peaks_recovers_parameters():
    curve = synthetic_pattern()
    result = fit_peaks(curve, centers=[3.0, 7.0], fwhm_guess=0.4)
    p1, p2 = result["peaks"]
    assert abs(p1["center"] - 3.0) < 0.01
    assert abs(p2["center"] - 7.0) < 0.01
    assert abs(p1["fwhm"] - 0.3) < 0.05
    assert abs(p2["fwhm"] - 0.5) < 0.05
    assert p1["center_err"] < 0.01
    assert not result["flags"]["at_bounds"]
    assert result["rwp"] < 0.05


def test_fit_peaks_flags_garbage_input():
    x = np.linspace(0, 10, 500)
    noise_only = Curve1D(x=x, y=RNG.normal(0, 1.0, x.size))
    result = fit_peaks(noise_only, centers=[5.0], fwhm_guess=0.5)
    assert (
        result["flags"]["high_uncertainty"]
        or result["flags"]["at_bounds"]
        or result["peaks"][0]["height"] < 3.0
    )


def test_sq_to_gr_peak_at_known_distance():
    """S(Q) of a single well-defined distance produces a G(r) peak there."""
    r0 = 2.5
    q = np.arange(0.02, 30.0, 0.02)
    # F(Q) for a Gaussian-broadened shell at r0 (sigma small)
    sigma = 0.1
    fq = np.sin(q * r0) * np.exp(-(q**2) * sigma**2 / 2)
    sq = 1 + fq / q
    curve = Curve1D(x=q, y=sq, xlabel="Q", ylabel="S(Q)")
    gr = sq_to_gr(curve, qmax=30.0, rmax=6.0, dr=0.01)
    peak_r = gr.x[np.argmax(gr.y)]
    assert abs(peak_r - r0) < 0.03


def test_rebin_propagates_errors():
    x = np.linspace(0, 10, 1000)
    curve = Curve1D(x=x, y=np.ones_like(x), e=np.full_like(x, 0.1))
    binned = rebin(curve, dx=0.1)
    assert binned.e is not None
    # Averaging ~10 points with sigma 0.1 → sigma ≈ 0.1/sqrt(10)
    assert pytest.approx(0.1 / np.sqrt(10), rel=0.3) == float(np.median(binned.e))
