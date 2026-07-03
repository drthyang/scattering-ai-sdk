from pathlib import Path

import numpy as np
import pytest

from scattering_ai.tools.series import (
    detect_transition,
    extract_param,
    load_series,
    track_peak,
)

RNG = np.random.default_rng(11)
REAL = sorted((Path(__file__).parents[1] / "data" / "1d" / "series").glob("GaNb4Se8_*.dat"))


def test_extract_param_patterns():
    assert extract_param("GaNb4Se8_101_T_base_5.0K_stuff.dat") == 5.0
    assert extract_param("scan_T=45K.dat") == 45.0
    assert extract_param("sample_120.5K_.dat") == 120.5
    assert extract_param("no_temperature_here.dat") is None


def make_series_files(tmp_path, transition=50.0):
    """Peak drifts slowly below the transition, faster above; fwhm jumps."""
    paths = []
    x = np.linspace(0, 10, 1200)
    for temp in np.arange(5, 100, 5.0):
        if temp < transition:
            center, fwhm = 5.0 + 0.0004 * temp, 0.30
        else:
            center, fwhm = 5.0 + 0.0004 * transition + 0.004 * (temp - transition), 0.42
        y = 1.0 + 10 * np.exp(-4 * np.log(2) * (x - center) ** 2 / fwhm**2)
        y += RNG.normal(0, 0.02, x.size)
        y[:30] = -3.0  # masked sentinel region
        p = tmp_path / f"synth_T_base_{temp:.1f}K.dat"
        np.savetxt(p, np.column_stack([x, y]))
        paths.append(str(p))
    return paths


def test_load_series_sorted_and_masked(tmp_path):
    paths = make_series_files(tmp_path)
    series = load_series(paths[::-1], mask_value=-3.0)  # shuffled input order
    assert series.params == sorted(series.params)
    assert np.isnan(series.curves[0].y[:30]).all()


def test_track_peak_and_detect_transition(tmp_path):
    series = load_series(make_series_files(tmp_path, transition=50.0), mask_value=-3.0)
    tracked = track_peak(series, center=5.0, fwhm_guess=0.3)
    assert tracked["n_good_fits"] == tracked["n_points"]

    good = tracked["rows"]
    temps = [r["param"] for r in good]
    detection = detect_transition(
        temps, [r["center"] for r in good], [r["center_err"] for r in good]
    )
    assert detection["detected"]
    assert abs(detection["transition_param"] - 50.0) < 7.5

    fwhm_detection = detect_transition(
        temps, [r["fwhm"] for r in good], [r["fwhm_err"] for r in good]
    )
    assert fwhm_detection["detected"]


def test_no_transition_on_smooth_trend():
    temps = list(np.arange(5, 100, 5.0))
    values = [1.0 + 0.001 * t for t in temps]  # perfectly linear
    detection = detect_transition(temps, values)
    assert not detection["detected"]


def test_detect_transition_needs_enough_points():
    result = detect_transition([1, 2, 3], [1.0, 2.0, 3.0])
    assert not result["detected"]
    assert "need" in result["reason"]


@pytest.mark.skipif(len(REAL) < 10, reason="GaNb4Se8 series not present")
def test_real_ganb4se8_transition():
    """Known answer: GaNb4Se8 structural transition in the 30-55 K range.

    Peak-center trends (lattice expansion anomaly) must show a changepoint
    there for the strong Bragg peaks.
    """
    series = load_series([str(p) for p in REAL], mask_value=-3.0)
    assert len(series.curves) == 20
    assert series.params[0] == 5.0 and series.params[-1] == 99.1

    detections = []
    for center in (3.666, 4.494, 4.764, 5.186):
        tracked = track_peak(series, center=center, fwhm_guess=0.03)
        good = [r for r in tracked["rows"] if r.get("ok")]
        assert len(good) >= 18  # robust tracking across the whole series
        det = detect_transition(
            [r["param"] for r in good],
            [r["center"] for r in good],
            [r["center_err"] for r in good],
        )
        detections.append(det)

    found = [d for d in detections if d["detected"] and 30 <= d["transition_param"] <= 55]
    assert len(found) >= 3
