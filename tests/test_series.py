from pathlib import Path

import numpy as np
import pytest

from scattering_ai.tools.series import (
    auto_mask_value,
    detect_transition,
    extract_param,
    load_series,
    stack_series,
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


def test_auto_mask_detects_repeated_sentinel(tmp_path):
    series = load_series(make_series_files(tmp_path))  # unmasked load
    assert auto_mask_value(series.curves) == -3.0
    # "auto" string routes through detection and masks in one call
    masked = load_series(make_series_files(tmp_path), mask_value="auto")
    assert masked.meta["mask_value_detected"] == -3.0
    assert np.isnan(masked.curves[0].y[:30]).all()


def test_auto_mask_none_on_clean_data():
    from scattering_ai.tools.models import Curve1D

    x = np.linspace(0, 10, 1200)
    curves = [
        Curve1D(x=x, y=1.0 + 5 * np.exp(-((x - 5) ** 2) / 0.1) + RNG.normal(0, 0.02, x.size))
        for _ in range(6)
    ]
    assert auto_mask_value(curves) is None  # must not fire on unmasked data


def test_stack_series_picks_persistent_peaks(tmp_path):
    """A peak present in only one curve must not survive the mean stack."""
    from scattering_ai.tools.curves import find_peaks

    series = load_series(make_series_files(tmp_path), mask_value=-3.0)
    # inject a one-off peak (same height as the real one) into a single curve
    spike_curve = series.curves[3]
    sel = np.abs(spike_curve.x - 8.5) < 0.25
    spike_curve.y[sel] += 10.0

    stack = stack_series(series)
    stacked = find_peaks(stack, subtract_background=True)
    top = max(stacked, key=lambda p: p["prominence"])
    # stacking makes the persistent peak (~5, in every curve) dominate; the
    # one-off at 8.5 is diluted by the number of curves and never wins
    assert abs(top["x"] - 5.0) < 0.3
    spike = [p for p in stacked if abs(p["x"] - 8.5) < 0.3]
    assert not spike or spike[0]["prominence"] < top["prominence"] / 5


def test_detect_two_transitions():
    """Two kinks (e.g. structural at 50 K + magnetic at 29 K) must be found as
    TWO transitions, not one averaged changepoint between them."""
    from scattering_ai.tools.series import cluster_transitions, detect_transitions

    temps = list(np.arange(5, 100, 5.0))
    values = []
    for t in temps:  # piecewise-linear with kinks at 29 and 50
        v = 5.0 + 0.0002 * t
        if t > 29:
            v += 0.004 * (t - 29)
        if t > 50:
            v -= 0.007 * (t - 50)
        values.append(v + RNG.normal(0, 1e-4))
    det = detect_transitions(temps, values)
    assert det["detected"] and len(det["transitions"]) == 2
    found = sorted(t["param"] for t in det["transitions"])
    assert abs(found[0] - 29) <= 5 and abs(found[1] - 50) <= 5
    # clustering keeps them distinct
    clusters = cluster_transitions(found, temps)
    assert len(clusters) == 2


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
    """Known answer (user ground truth): GaNb4Se8 has transitions at ~50 K and
    ~29 K. The strong Bragg-peak center trends must yield a changepoint cluster
    bracketing 50 K; detections must NOT be collapsed into one fictitious
    average (the old single-median behaviour reported "39 K")."""
    from scattering_ai.tools.series import cluster_transitions

    series = load_series([str(p) for p in REAL], mask_value=-3.0)
    assert len(series.curves) == 20
    assert series.params[0] == 5.0 and series.params[-1] == 99.1

    candidates = []
    for center in (3.666, 4.494, 4.764, 5.186):
        tracked = track_peak(series, center=center, fwhm_guess=0.03)
        good = [r for r in tracked["rows"] if r.get("ok")]
        assert len(good) >= 18  # robust tracking across the whole series
        det = detect_transition(
            [r["param"] for r in good],
            [r["center"] for r in good],
            [r["center_err"] for r in good],
        )
        candidates += [t["param"] for t in det.get("transitions", [])]

    assert candidates
    clusters = cluster_transitions(candidates, series.params)
    # the ~50 K structural transition must appear as its own cluster
    # (data steps are ~5 K: the 49.1/54.3 bracket)
    assert any(45 <= c["param"] <= 57 for c in clusters)
