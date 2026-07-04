"""3D-ΔPDF core (independent implementation of the standard windowed-FFT method).

The algorithm is validated numerically; the registry tool is exercised on the
real CORELLI volume when present (heavy — needs h5py).
"""

from pathlib import Path

import numpy as np
import pytest

from scattering_ai.tools.delta_pdf import (
    central_slices,
    compute_delta_pdf,
    delta_pdf_summary,
    punch_bragg,
)

VOLUME = next((Path(__file__).parents[1] / "data" / "3d" / "volumes").glob("*.nxs"), None)


def test_delta_pdf_recovers_modulation_period():
    """A cos(2π m i/N) modulation must give a correlation peak at offset m."""
    n, m = 32, 4
    mod = 1 + 0.5 * np.cos(2 * np.pi * m * np.arange(n) / n)
    data = np.ones((n, n, n)) * mod[:, None, None]
    dpdf = compute_delta_pdf(data, apodization="none")
    profile = np.abs(dpdf[:, n // 2, n // 2])
    profile[n // 2] = 0  # ignore residual r=0
    assert abs(int(np.argmax(profile)) - n // 2) == m


def test_delta_pdf_centrosymmetric():
    # a real I(Q) gives a real, centrosymmetric ΔPDF (Re of a Hermitian FFT is
    # even); use odd lengths so [::-1] is the exact centre reflection.
    rng = np.random.default_rng(0)
    dpdf = compute_delta_pdf(rng.random((15, 15, 15)))
    assert np.allclose(dpdf, dpdf[::-1, ::-1, ::-1], atol=1e-9)


def test_punch_bragg_removes_sharp_peaks():
    data = np.ones((16, 16, 16))
    data[8, 8, 8] = 1000.0
    punched = punch_bragg(data, radius=0)
    assert np.isnan(punched[8, 8, 8])
    assert np.isfinite(punched[0, 0, 0])


def test_central_slices_and_summary():
    dpdf = compute_delta_pdf(np.random.default_rng(1).random((12, 10, 8)))
    slices = central_slices(dpdf)
    assert slices["xy"].shape == (12, 10)
    assert slices["xz"].shape == (12, 8)
    s = delta_pdf_summary(dpdf)
    assert s["shape"] == [12, 10, 8] and s["max_positive"] >= s["max_negative"]


def test_delta_pdf_skill_is_registered_in_volumes_category(tmp_path):
    from scattering_ai.tools.registry import default_toolkit

    cats = default_toolkit(tmp_path / "ws").skills_by_category()
    assert cats["3D volumes"] == ["skill_delta_pdf"]


@pytest.mark.skipif(VOLUME is None, reason="CORELLI volume not present")
def test_delta_pdf_tool_on_real_volume(tmp_path):
    pytest.importorskip("h5py")
    pytest.importorskip("matplotlib")
    from scattering_ai.tools.registry import default_toolkit

    reg = default_toolkit(tmp_path / "ws")
    out = reg.execute("delta_pdf", {"path": str(VOLUME), "punch_sigma": 8.0})
    assert "error" not in out
    assert Path(out["saved_slice"]).exists() and Path(out["plot"]).exists()
    assert out["max_positive"] >= out["max_negative"]


@pytest.mark.skipif(VOLUME is None, reason="CORELLI volume not present")
def test_delta_pdf_skill_on_real_volume(tmp_path):
    pytest.importorskip("h5py")
    pytest.importorskip("matplotlib")
    from scattering_ai.tools.registry import default_toolkit

    reg = default_toolkit(tmp_path / "ws")
    out = reg.execute("skill_delta_pdf", {"path": str(VOLUME), "punch_sigma": 8.0})
    assert out["step_errors"] == 0
    assert "3D-ΔPDF" in out["summary"]
    assert out["figures"] and out["figures"][0].endswith(".png")
    assert "max_positive" in out["extremes"]


def test_fill_punched_recovers_smooth_field():
    """Backfilling punched holes must reconstruct a smooth diffuse field far
    better than leaving zeros (the punch lattice imprint)."""
    from scattering_ai.tools.delta_pdf import fill_punched

    n = 24
    i = np.arange(n)
    truth = 5 + np.cos(2 * np.pi * 3 * i / n)[:, None, None] * np.ones((n, n, n))
    punched = truth.copy()
    punched[4:7, 4:7, 4:7] = np.nan          # a punched Bragg hole
    filled = fill_punched(punched)
    assert np.isfinite(filled).all()
    hole_err = np.abs(filled[4:7, 4:7, 4:7] - truth[4:7, 4:7, 4:7]).max()
    zero_err = np.abs(0.0 - truth[4:7, 4:7, 4:7]).max()
    assert hole_err < 0.35 * zero_err        # much closer than zero-filling
