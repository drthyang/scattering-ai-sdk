import numpy as np
import pytest

pytest.importorskip("matplotlib")

from scattering_ai.tools import plotting  # noqa: E402
from scattering_ai.tools.curves import evaluate_fit, find_peaks, fit_peaks  # noqa: E402
from scattering_ai.tools.models import Curve1D, Slice2D  # noqa: E402

RNG = np.random.default_rng(5)


def two_peak_curve():
    x = np.linspace(0, 10, 1500)
    y = 0.5 + 0.02 * x
    for h, c, w in [(9.0, 3.0, 0.25), (5.0, 7.0, 0.4)]:
        y = y + h * np.exp(-4 * np.log(2) * (x - c) ** 2 / w**2)
    return Curve1D(x=x, y=y + RNG.normal(0, 0.03, x.size), xlabel="Q", ylabel="I")


def png_ok(path, min_bytes=8000):
    from pathlib import Path

    p = Path(path)
    assert p.exists() and p.stat().st_size > min_bytes
    assert p.read_bytes()[:4] == b"\x89PNG"


def test_evaluate_fit_reconstructs_model():
    curve = two_peak_curve()
    fit = fit_peaks(curve, centers=[3.0, 7.0])
    lo, hi = fit["window"]
    mask = (curve.x >= lo) & (curve.x <= hi)
    model = evaluate_fit(curve.x[mask], fit)
    residual = curve.y[mask] - model
    assert np.std(residual) < 0.05  # noise-level residual


def test_plot_curve_and_fit(tmp_path):
    curve = two_peak_curve()
    peaks = find_peaks(curve, subtract_background=True)
    png_ok(plotting.plot_curve(curve, tmp_path / "c.png", peaks=peaks))

    fit = fit_peaks(curve, centers=[3.0, 7.0])
    png_ok(plotting.plot_fit(curve, fit, tmp_path / "f.png"))


def test_plot_slice_with_nans(tmp_path):
    x = np.linspace(-3, 3, 200)
    data = RNG.normal(1, 0.1, (200, 200))
    data[:40, :40] = np.nan
    s = Slice2D(data=data, x_centers=x, y_centers=x)
    png_ok(plotting.plot_slice(s, tmp_path / "s.png"))
    png_ok(plotting.plot_slice(s, tmp_path / "s_log.png", log=True))


def test_plot_series_and_tracking(tmp_path):
    curves, params = [], []
    x = np.linspace(0, 10, 400)
    for temp in np.arange(10, 100, 10.0):
        center = 5.0 + 0.002 * temp
        curves.append(Curve1D(x=x, y=np.exp(-((x - center) ** 2) / 0.05)))
        params.append(float(temp))
    png_ok(plotting.plot_series(curves, params, tmp_path / "w.png"))

    tracking = {
        "tracked_center_start": 5.0,
        "n_points": 9,
        "n_good_fits": 8,
        "param_label": "T (K)",
        "rows": [
            {"param": p, "center": 5 + 0.002 * p, "center_err": 1e-4,
             "fwhm": 0.2, "fwhm_err": 1e-3, "height": 1.0, "height_err": 0.01,
             "ok": p != 50.0}
            for p in params
        ],
    }
    transitions = {"center": {"detected": True, "transition_param": 55.0,
                              "improvement_ratio": 12.0}}
    png_ok(plotting.plot_tracking(tracking, tmp_path / "t.png", transitions=transitions))


def test_registry_plot_tools(tmp_path):
    from scattering_ai.tools.registry import default_toolkit

    curve = two_peak_curve()
    path = tmp_path / "c.dat"
    np.savetxt(path, np.column_stack([curve.x, curve.y]))
    registry = default_toolkit(tmp_path / "ws")

    r1 = registry.execute("plot_1d", {"path": str(path)})
    png_ok(r1["saved"])
    assert r1["n_peaks_marked"] == 2

    r2 = registry.execute("plot_fit_1d", {"path": str(path), "centers": [3.0, 7.0]})
    png_ok(r2["plot"])
    assert abs(r2["peaks"][0]["center"] - 3.0) < 0.01


def test_cli_plot_command(tmp_path, capsys):
    from scattering_ai.cli import main

    curve = two_peak_curve()
    path = tmp_path / "c.dat"
    np.savetxt(path, np.column_stack([curve.x, curve.y]))

    out = tmp_path / "out.png"
    assert main(["plot", str(path), "--out", str(out)]) == 0
    png_ok(out)

    out_fit = tmp_path / "fit.png"
    assert main(["plot", str(path), "--fit", "3.0,7.0", "--out", str(out_fit)]) == 0
    png_ok(out_fit)
