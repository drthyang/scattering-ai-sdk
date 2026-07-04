"""Summarizing figures: packs generate plots that support the conclusion, the
agent attaches them to the report, and the renderer shows them."""

from pathlib import Path

import numpy as np
import pytest

from scattering_ai import Agent, AnalysisRequest
from scattering_ai.tools.models import Slice2D
from scattering_ai.tools.registry import _save_slice

pytest.importorskip("matplotlib")
RNG = np.random.default_rng(5)


def _agent(tmp_path):
    return Agent(workspace=tmp_path / "ws")


def _series(tmp_path, transition=50.0):
    x = np.linspace(0, 10, 1200)
    paths = []
    for temp in np.arange(5, 100, 5.0):
        center = 5.0 + (0.0 if temp < transition else 0.004 * (temp - transition))
        fwhm = 0.30 if temp < transition else 0.42
        y = 1 + 10 * np.exp(-4 * np.log(2) * (x - center) ** 2 / fwhm**2)
        y += RNG.normal(0, 0.02, x.size)
        y[:30] = -3.0
        p = tmp_path / f"scan_T_base_{temp:.1f}K.dat"
        np.savetxt(p, np.column_stack([x, y]))
        paths.append(str(p))
    return paths


def test_series_produces_transition_and_figures(tmp_path):
    report = _agent(tmp_path).analyze(
        AnalysisRequest(question="Is there a transition?", data={"files": _series(tmp_path)}))
    assert report.domain == "data"
    # a phase-transition observation with a plausible T_c
    hit = [o for o in report.observations if "phase transition" in o.lower()]
    assert hit and "K" in hit[0]
    # waterfall + tracking figures generated and on disk
    names = {Path(f).name for f in report.figures}
    assert {"series_waterfall.png", "series_tracking.png"} <= names
    assert all(Path(f).exists() for f in report.figures)


def test_pdf_produces_overview_figure(tmp_path):
    x = np.linspace(0.01, 20, 2000)
    y = -0.2 * x * np.exp(-x / 8) + 5 * np.exp(-((x - 2.6) ** 2) / 0.05)
    gr = tmp_path / "s.gr"
    gr.write_text("outputtype = gr\n#### start data\n#L r  G(r)\n"
                  + "\n".join(f"{a} {b}" for a, b in zip(x, y, strict=True)))
    report = _agent(tmp_path).analyze(AnalysisRequest(question="?", data={"files": [str(gr)]}))
    assert report.domain == "pdf"
    assert [Path(f).name for f in report.figures] == ["pdf_overview.png"]
    assert Path(report.figures[0]).exists()


def test_diffuse_slice_produces_map_figure(tmp_path):
    n = 100
    ax = np.linspace(-5, 5, n)
    xx, yy = np.meshgrid(ax, ax)
    data = 2 + np.exp(-(xx**2 + yy**2) / 30)
    for cx, cy in [(-2, -2), (0, 0), (2, 1)]:
        data += 40 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.05)
    slc = tmp_path / "s.npz"
    _save_slice(Slice2D(data=data, x_centers=ax, y_centers=ax), slc)
    report = _agent(tmp_path).analyze(AnalysisRequest(question="?", data={"files": [str(slc)]}))
    assert report.domain == "diffuse"
    assert [Path(f).name for f in report.figures] == ["diffuse_map.png"]


def test_figure_only_findings_stay_out_of_observations(tmp_path):
    x = np.linspace(0.01, 20, 2000)
    y = -0.2 * x * np.exp(-x / 8) + 5 * np.exp(-((x - 2.6) ** 2) / 0.05)
    gr = tmp_path / "s.gr"
    gr.write_text("outputtype = gr\n#### start data\n#L r  G(r)\n"
                  + "\n".join(f"{a} {b}" for a, b in zip(x, y, strict=True)))
    report = _agent(tmp_path).analyze(AnalysisRequest(question="?", data={"files": [str(gr)]}))
    assert not any("Overview plot" in o for o in report.observations)  # figure-only
    assert report.markdown.count("## Figures") == 1
    assert "pdf_overview.png" in report.markdown
