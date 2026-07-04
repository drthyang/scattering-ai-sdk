from pathlib import Path

import numpy as np
import pytest

from scattering_ai.tools.registry import default_toolkit

DATA = Path(__file__).parents[1] / "data"
SERIES = sorted((DATA / "1d" / "series").glob("GaNb4Se8_*.dat"))
XRAY_GR = DATA / "1d" / "curves" / "XRAY_FeCoSn_100K_converted.gr"

RNG = np.random.default_rng(21)


@pytest.fixture
def registry(tmp_path):
    return default_toolkit(tmp_path / "ws")


def test_skills_are_registered(registry):
    names = [s.name for s in registry.specs]
    assert "skill_characterize_slice" in names
    assert "skill_fit_pattern_peaks" in names
    assert "skill_scan_series_transitions" in names
    # skills can be disabled for tool-only registries
    bare = default_toolkit("/tmp/unused_bare_ws", skills=False)
    assert not any(n.startswith("skill_") for n in (s.name for s in bare.specs))


def test_fit_pattern_peaks_skill_synthetic(registry, tmp_path):
    pytest.importorskip("matplotlib")
    x = np.linspace(0, 10, 1500)
    y = 0.3 + RNG.normal(0, 0.02, x.size)
    for height, center in [(8.0, 3.0), (5.0, 7.0)]:
        y = y + height * np.exp(-4 * np.log(2) * (x - center) ** 2 / 0.3**2)
    path = tmp_path / "p.dat"
    np.savetxt(path, np.column_stack([x, y]))

    result = registry.execute("skill_fit_pattern_peaks", {"path": str(path)})
    assert result["step_errors"] == 0
    assert len(result["peak_table"]) == 2
    centers = sorted(p["center"] for p in result["peak_table"])
    assert abs(centers[0] - 3.0) < 0.02 and abs(centers[1] - 7.0) < 0.02
    assert result["assessment"]["fit_trustworthy"]
    assert Path(result["plots"]["fit"]).exists()
    # audited chain recorded
    assert [s["tool"] for s in result["steps"]] == ["find_peaks_1d", "plot_fit_1d"]


def test_fit_pattern_peaks_skill_range_filter(registry, tmp_path):
    x = np.linspace(0, 10, 800)
    y = 1 + 5 * np.exp(-((x - 4) ** 2) / 0.02)
    path = tmp_path / "one.dat"
    np.savetxt(path, np.column_stack([x, y]))
    result = registry.execute(
        "skill_fit_pattern_peaks", {"path": str(path), "xmin": 6.0, "xmax": 9.0}
    )
    assert "error" in result and "no peaks" in result["error"]


def test_characterize_slice_requires_cut_or_slice(registry):
    result = registry.execute("skill_characterize_slice", {"path": "whatever.nxs"})
    assert "error" in result


def test_series_paths_accept_globs_and_dirs(registry, tmp_path):
    """Regression: gemma4:26b corrupted filenames when retyping 20 long
    paths; series tools must accept globs/directories instead."""
    x = np.linspace(0, 10, 300)
    for temp in (10.0, 20.0, 30.0, 40.0, 50.0):
        y = np.exp(-((x - 5) ** 2) / 0.1)
        np.savetxt(tmp_path / f"scan_T_base_{temp:.1f}K.dat", np.column_stack([x, y]))
    (tmp_path / ".DS_Store").write_bytes(b"junk")  # macOS noise must be ignored
    (tmp_path / "notes.md").write_text("not data")

    via_glob = registry.execute("inspect_series", {"paths": [str(tmp_path / "*.dat")]})
    assert via_glob["n_curves"] == 5
    via_dir = registry.execute("inspect_series", {"paths": [str(tmp_path)]})
    assert via_dir["n_curves"] == 5

    typo = registry.execute("inspect_series", {"paths": [str(tmp_path / "nope_5.0K.dat")]})
    assert "glob pattern" in typo["error"]  # error must teach the recovery


def test_chat_files_context_compacts_to_glob(tmp_path):
    from scattering_ai.core.chat import ChatSession

    files = [f"/data/series/very_long_name_T_base_{t}.0K_suffix.dat" for t in range(20)]
    context = ChatSession._files_context(files)
    assert "/data/series/*.dat" in context
    assert context.count("very_long_name") <= 3  # not the full listing


@pytest.mark.skipif(len(SERIES) < 10, reason="GaNb4Se8 series not present")
def test_scan_series_transitions_skill_real(registry):
    pytest.importorskip("matplotlib")
    result = registry.execute(
        "skill_scan_series_transitions",
        {"paths": [str(p) for p in SERIES], "mask_value": -3.0, "n_peaks": 3,
         "fwhm_guess": 0.03},
    )
    verdict = result["verdict"]
    assert verdict["transition_detected"]
    assert 30 <= verdict["transition_estimate"] <= 55
    assert Path(result["plots"]["waterfall"]).exists()
    assert any(p.get("plot") for p in result["tracked_peaks"])
    # every plot the skill made is aggregated into a single figures list
    assert result["figures"] and all(p.endswith(".png") for p in result["figures"])
    assert result["plots"]["waterfall"] in result["figures"]
    # a human-readable summary and per-peak monitoring are always present
    assert str(verdict["transition_estimate"]).split(".")[0] in result["summary"]
    monitored = [p["monitored"] for p in result["tracked_peaks"] if "monitored" in p]
    assert any("center_shift" in m for m in monitored)


@pytest.mark.skipif(len(SERIES) < 10, reason="GaNb4Se8 series not present")
def test_scan_series_transitions_auto_masks_without_arg(registry):
    """Robustness: the skill finds the transition even when the caller forgets
    the -3.0 sentinel, by auto-detecting it."""
    pytest.importorskip("matplotlib")
    result = registry.execute(
        "skill_scan_series_transitions",
        {"paths": [str(p) for p in SERIES], "n_peaks": 3, "fwhm_guess": 0.03},
    )
    assert result["mask_value_used"] == -3.0  # detected, not passed
    assert result["verdict"]["transition_detected"]
    assert 30 <= result["verdict"]["transition_estimate"] <= 55


@pytest.mark.skipif(not XRAY_GR.exists(), reason="real data not present")
def test_fit_pattern_peaks_skill_flags_real_pdf(registry):
    pytest.importorskip("matplotlib")
    result = registry.execute(
        "skill_fit_pattern_peaks",
        {"path": str(XRAY_GR), "xmin": 2.0, "xmax": 6.0},
    )
    # Known answer: 4-peak fit of this window is honest but flagged
    # (unfitted shoulder at ~3.05 A shows in the residual)
    assert len(result["peak_table"]) >= 4
    assert not result["assessment"]["fit_trustworthy"]
