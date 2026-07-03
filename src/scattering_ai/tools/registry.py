"""Agent-facing tool registry.

Wraps the deterministic data tools as JSON-schema functions an LLM can call.
Tools exchange data through files: array-producing tools save artifacts into
a session workspace and return the path plus a JSON summary, so the LLM
chains operations without ever touching arrays (roadmap B3: every number
comes from a tool).

Safety: all tools are read-only with respect to user data; the only writes
are new artifacts inside the workspace directory.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scattering_ai.llm.base import ToolSpec
from scattering_ai.tools.models import Slice2D


@dataclass
class AgentTool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., dict[str, Any]]

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)


class ToolRegistry:
    def __init__(self, tools: list[AgentTool]):
        self._tools = {t.name: t for t in tools}

    def add(self, tool: AgentTool) -> None:
        self._tools[tool.name] = tool

    @property
    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run a tool; errors come back as structured results, never raises."""
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool '{name}'. Available: {sorted(self._tools)}"}
        try:
            return tool.fn(**arguments)
        except TypeError as exc:
            return {"error": f"Bad arguments for {name}: {exc}"}
        except Exception as exc:  # tool failures are data for the agent, not crashes
            return {"error": f"{type(exc).__name__}: {exc}"}


def _save_slice(slice2d: Slice2D, path: Path) -> None:
    np.savez(
        path,
        data=slice2d.data,
        x_centers=slice2d.x_centers,
        y_centers=slice2d.y_centers,
        xlabel=slice2d.xlabel,
        ylabel=slice2d.ylabel,
        meta=json.dumps(slice2d.meta),
    )


def load_slice(path: str | Path) -> Slice2D:
    with np.load(path, allow_pickle=False) as f:
        return Slice2D(
            data=f["data"],
            x_centers=f["x_centers"],
            y_centers=f["y_centers"],
            xlabel=str(f["xlabel"]),
            ylabel=str(f["ylabel"]),
            meta=json.loads(str(f["meta"])),
        )


def _params(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


def default_toolkit(workspace: str | Path, skills: bool = True) -> ToolRegistry:
    """The standard read-only data toolkit, saving artifacts to ``workspace``.

    ``skills=True`` also registers the built-in composite skills
    (``skill_*`` tools) on top of the individual tools.
    """
    from scattering_ai.tools import curves as c
    from scattering_ai.tools import slices as sl
    from scattering_ai.tools.io import load_curve
    from scattering_ai.tools.volumes import load_volume

    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    counter = {"n": 0}

    def artifact(stem: str, suffix: str) -> Path:
        counter["n"] += 1
        return workspace / f"{counter['n']:02d}_{stem}{suffix}"

    def _load_1d(path: str):
        if path.endswith(".cut.npy"):
            arr = np.load(path)
            from scattering_ai.tools.models import Curve1D

            return Curve1D(x=arr[0], y=arr[1])
        return load_curve(path)

    def inspect_curve(path: str) -> dict:
        curve = load_curve(path)
        out = curve.summary()
        # convention (S(Q) vs S(Q)-1) only means something for Q-space data
        if not curve.xlabel.lower().startswith("r"):
            out["detected_convention"] = c.detect_sq_convention(curve)
        return out

    def find_peaks_1d(path: str, min_prominence: float | None = None,
                      subtract_background: bool = True) -> dict:
        peaks = c.find_peaks(_load_1d(path), min_prominence=min_prominence,
                             subtract_background=subtract_background)
        return {"n_peaks": len(peaks), "peaks": peaks[:20]}

    def fit_peaks_1d(path: str, centers: list[float], fwhm_guess: float | None = None,
                     window: float | None = None) -> dict:
        return c.fit_peaks(_load_1d(path), centers=centers, fwhm_guess=fwhm_guess,
                           window=window)

    def transform_sq_to_gr(path: str, qmin: float | None = None, qmax: float | None = None,
                           rmax: float = 30.0, input_kind: str = "auto") -> dict:
        gr = c.sq_to_gr(load_curve(path), qmin=qmin, qmax=qmax, rmax=rmax,
                        input_kind=input_kind)
        out_path = artifact("gr", ".dat")
        np.savetxt(out_path, np.column_stack([gr.x, gr.y]), header="r  G(r)")
        peaks = c.find_peaks(gr, subtract_background=False)
        return {"saved": str(out_path), "summary": gr.summary(),
                "first_peaks": peaks[:8]}

    def inspect_volume(path: str) -> dict:
        return load_volume(path).summary()

    def inspect_cif(path: str, d_min: float = 1.0) -> dict:
        from scattering_ai.tools.cif import lattice_from_cif, predicted_d_spacings, read_cif

        info = read_cif(path)
        lattice = lattice_from_cif(path)
        return {
            "cell": {k: info[k] for k in ("a", "b", "c", "alpha", "beta", "gamma")},
            "space_group": info.get("space_group", ""),
            "formula": info.get("formula", ""),
            "predicted_d_spacings": predicted_d_spacings(lattice, d_min=d_min),
        }

    def slice_volume(path: str, axis: int, center: float, thickness: float) -> dict:
        s = load_volume(path).slice(axis=axis, center=center, thickness=thickness)
        out_path = artifact(f"slice_ax{axis}", ".npz")
        _save_slice(s, out_path)
        return {"saved": str(out_path), "summary": s.summary()}

    def oblique_slice_volume(path: str, origin: list[float], u_axis: list[float],
                             v_axis: list[float], u_min: float, u_max: float,
                             v_min: float, v_max: float, thickness: float = 0.0) -> dict:
        s = load_volume(path).oblique_slice(
            origin=tuple(origin), u_axis=tuple(u_axis), v_axis=tuple(v_axis),
            u_range=(u_min, u_max), v_range=(v_min, v_max), thickness=thickness,
        )
        out_path = artifact("oblique", ".npz")
        _save_slice(s, out_path)
        return {"saved": str(out_path), "summary": s.summary()}

    def find_peaks_2d(slice_path: str, min_snr: float = 10.0) -> dict:
        peaks = sl.find_peaks_2d(load_slice(slice_path), min_snr=min_snr)
        return {"n_peaks": len(peaks), "strongest": peaks[:20]}

    def detect_rings_2d(slice_path: str, x_scale: float | None = None,
                        y_scale: float | None = None) -> dict:
        s = load_slice(slice_path)
        result = sl.detect_rings(s, x_scale=x_scale, y_scale=y_scale)
        try:
            from scattering_ai.tools.plotting import plot_profile

            profile = sl.azimuthal_profile(s, x_scale, y_scale, statistic=25)
            result["plot"] = plot_profile(profile, artifact("ring_profile", ".png"),
                                          rings=result["rings"])
        except ImportError:
            pass
        return result

    def line_cut_2d(slice_path: str, x0: float, y0: float, x1: float, y1: float,
                    width: float = 0.0) -> dict:
        cut = sl.line_cut(load_slice(slice_path), start=(x0, y0), end=(x1, y1), width=width)
        out_path = artifact("cut", ".cut.npy")
        np.save(out_path, np.vstack([cut.x, cut.y]))
        peaks = c.find_peaks(cut, subtract_background=False)
        return {"saved": str(out_path), "summary": cut.summary(), "peaks": peaks[:10]}

    def _expand_paths(paths: list[str]) -> list[str]:
        """Accept globs and directories, not just exact paths: small models
        corrupt long filenames when forced to retype them."""
        import glob as globlib

        expanded: list[str] = []
        for entry in paths:
            p = Path(entry)
            if any(ch in entry for ch in "*?["):
                matches = sorted(globlib.glob(entry))
                if not matches:
                    raise FileNotFoundError(f"glob matched nothing: {entry}")
                expanded += matches
            elif p.is_dir():
                expanded += sorted(
                    str(f) for f in p.iterdir()
                    if f.is_file() and not f.name.startswith(".")
                    and f.suffix.lower() not in (".md", ".png")
                )
            else:
                if not p.exists():
                    raise FileNotFoundError(
                        f"{entry} not found — pass a glob pattern (e.g. "
                        f"'{p.parent}/*{p.suffix}') or a directory instead of "
                        "retyping long filenames"
                    )
                expanded.append(entry)
        seen: dict[str, None] = {}
        for e in expanded:
            seen.setdefault(e)
        return list(seen)

    def inspect_series(paths: list[str], mask_value: float | None = None) -> dict:
        from scattering_ai.tools.series import load_series

        series = load_series(_expand_paths(paths), mask_value=mask_value)
        out = series.summary()
        peaks = c.find_peaks(series.curves[0], subtract_background=True)
        out["peaks_in_first_curve"] = peaks[:10]
        return out

    def track_peak_series(paths: list[str], center: float,
                          fwhm_guess: float | None = None,
                          mask_value: float | None = None) -> dict:
        from scattering_ai.tools.series import detect_transition, load_series, track_peak

        series = load_series(_expand_paths(paths), mask_value=mask_value)
        tracked = track_peak(series, center=center, fwhm_guess=fwhm_guess)
        good = [r for r in tracked["rows"] if r.get("ok")]
        result: dict = {
            "n_points": tracked["n_points"],
            "n_good_fits": tracked["n_good_fits"],
            "rows": [
                {k: r.get(k) for k in
                 ("param", "center", "center_err", "fwhm", "fwhm_err", "height", "ok")}
                for r in tracked["rows"]
            ],
        }
        if len(good) >= 6:
            result["transition_on_center"] = detect_transition(
                [r["param"] for r in good], [r["center"] for r in good],
                [r["center_err"] for r in good],
            )
            result["transition_on_fwhm"] = detect_transition(
                [r["param"] for r in good], [r["fwhm"] for r in good],
                [r["fwhm_err"] for r in good],
            )
        try:
            from scattering_ai.tools.plotting import plot_tracking

            result["plot"] = plot_tracking(
                tracked, artifact("tracking", ".png"),
                transitions={
                    "center": result.get("transition_on_center"),
                    "fwhm": result.get("transition_on_fwhm"),
                },
            )
        except ImportError:
            pass
        return result

    def plot_1d(path: str, logy: bool = False, mark_peaks: bool = True) -> dict:
        from scattering_ai.tools.plotting import plot_curve

        curve = _load_1d(path)
        peaks = c.find_peaks(curve, subtract_background=True) if mark_peaks else []
        saved = plot_curve(curve, artifact("curve", ".png"), peaks=peaks, logy=logy)
        return {"saved": saved, "n_peaks_marked": len(peaks)}

    def plot_fit_1d(path: str, centers: list[float],
                    fwhm_guess: float | None = None,
                    window: float | None = None) -> dict:
        from scattering_ai.tools.plotting import plot_fit

        curve = _load_1d(path)
        fit = c.fit_peaks(curve, centers=centers, fwhm_guess=fwhm_guess, window=window)
        fit["plot"] = plot_fit(curve, fit, artifact("fit", ".png"))
        return fit

    def plot_slice_2d(slice_path: str, log: bool = False,
                      mark_peaks: bool = False) -> dict:
        from scattering_ai.tools.plotting import plot_slice

        s = load_slice(slice_path)
        peaks = sl.find_peaks_2d(s) if mark_peaks else []
        saved = plot_slice(s, artifact("slice_plot", ".png"), log=log, peaks=peaks)
        return {"saved": saved, "summary": s.summary(), "n_peaks_marked": len(peaks)}

    def plot_series_files(paths: list[str], mask_value: float | None = None,
                          xmin: float | None = None, xmax: float | None = None) -> dict:
        from scattering_ai.tools.plotting import plot_series
        from scattering_ai.tools.series import load_series

        series = load_series(_expand_paths(paths), mask_value=mask_value)
        saved = plot_series(series.curves, series.params,
                            artifact("series", ".png"),
                            param_label=series.param_label, xmin=xmin, xmax=xmax)
        return {"saved": saved, "summary": series.summary()}

    number = {"type": "number"}
    opt_number = {"type": ["number", "null"]}
    string = {"type": "string"}

    registry = ToolRegistry(
        [
            AgentTool(
                "inspect_curve",
                "Load a 1D data file (ASCII/pdfgetx) and describe it: ranges, labels, "
                "and whether it stores S(Q), S(Q)-1, or F(Q).",
                _params({"path": string}, ["path"]),
                inspect_curve,
            ),
            AgentTool(
                "find_peaks_1d",
                "Find peaks in a 1D data file or saved line cut. Returns positions, "
                "heights, prominences, FWHM estimates.",
                _params(
                    {"path": string, "min_prominence": opt_number,
                     "subtract_background": {"type": "boolean"}},
                    ["path"],
                ),
                find_peaks_1d,
            ),
            AgentTool(
                "fit_peaks_1d",
                "Fit pseudo-Voigt peaks + linear background at the given centers in a "
                "1D file or saved cut. Returns parameters with 1-sigma uncertainties "
                "and quality flags. Numbers in your answer MUST come from this tool.",
                _params(
                    {"path": string,
                     "centers": {"type": "array", "items": number},
                     "fwhm_guess": opt_number, "window": opt_number},
                    ["path", "centers"],
                ),
                fit_peaks_1d,
            ),
            AgentTool(
                "transform_sq_to_gr",
                "Fourier transform total-scattering S(Q)/F(Q) to the PDF G(r). "
                "Saves the result and returns its first peaks.",
                _params(
                    {"path": string, "qmin": opt_number, "qmax": opt_number,
                     "rmax": number, "input_kind": string},
                    ["path"],
                ),
                transform_sq_to_gr,
            ),
            AgentTool(
                "inspect_volume",
                "Describe a 3D reciprocal-space volume (.nxs): axes, ranges, unit cell.",
                _params({"path": string}, ["path"]),
                inspect_volume,
            ),
            AgentTool(
                "inspect_cif",
                "Read a CIF: unit cell, space group, formula, and geometrically "
                "allowed d-spacings/Q positions (no intensities) for matching "
                "observed peaks against a known structure.",
                _params({"path": string, "d_min": number}, ["path"]),
                inspect_cif,
            ),
            AgentTool(
                "slice_volume",
                "Cut an axis-aligned slab from a 3D volume, averaging over thickness. "
                "axis: 0/1/2 (logical). Saves a 2D slice and returns its path.",
                _params(
                    {"path": string, "axis": {"type": "integer"}, "center": number,
                     "thickness": number},
                    ["path", "axis", "center", "thickness"],
                ),
                slice_volume,
            ),
            AgentTool(
                "oblique_slice_volume",
                "Cut an arbitrary plane from a 3D volume: points = origin + s*u_axis "
                "+ t*v_axis (all in r.l.u./HKL), integrating over thickness along the "
                "plane normal. For cuts not aligned with the volume axes.",
                _params(
                    {"path": string,
                     "origin": {"type": "array", "items": number},
                     "u_axis": {"type": "array", "items": number},
                     "v_axis": {"type": "array", "items": number},
                     "u_min": number, "u_max": number,
                     "v_min": number, "v_max": number,
                     "thickness": number},
                    ["path", "origin", "u_axis", "v_axis", "u_min", "u_max",
                     "v_min", "v_max"],
                ),
                oblique_slice_volume,
            ),
            AgentTool(
                "find_peaks_2d",
                "Detect 2D peaks (e.g. Bragg) in a saved slice (.npz).",
                _params({"slice_path": string, "min_snr": number}, ["slice_path"]),
                find_peaks_2d,
            ),
            AgentTool(
                "detect_rings_2d",
                "Find powder-ring candidates in a saved slice and match them against "
                "known contaminants (Al/Cu/steel/V). Slices cut from a volume carry "
                "their lattice, so |Q| is exact (B matrix) and no scales are needed; "
                "for bare slices pass x_scale/y_scale (axis units -> 1/Angstrom). "
                "Low completeness = partial annulus = weak evidence.",
                _params(
                    {"slice_path": string, "x_scale": opt_number, "y_scale": opt_number},
                    ["slice_path"],
                ),
                detect_rings_2d,
            ),
            AgentTool(
                "line_cut_2d",
                "Extract a 1D line cut from a saved slice between (x0,y0) and (x1,y1), "
                "averaging over width. Saves the cut for use with 1D tools.",
                _params(
                    {"slice_path": string, "x0": number, "y0": number, "x1": number,
                     "y1": number, "width": number},
                    ["slice_path", "x0", "y0", "x1", "y1"],
                ),
                line_cut_2d,
            ),
            AgentTool(
                "inspect_series",
                "Load a parametric series of 1D files (temperature/field scan; the "
                "parameter is parsed from filenames like 'T_base_5.0K'). paths may "
                "be exact files, GLOB PATTERNS ('dir/*.dat' — preferred, avoids "
                "retyping long names), or a directory. Returns the parameter values "
                "and the peaks found in the first curve. mask_value: sentinel for "
                "masked points (e.g. -3.0).",
                _params(
                    {"paths": {"type": "array", "items": string}, "mask_value": opt_number},
                    ["paths"],
                ),
                inspect_series,
            ),
            AgentTool(
                "track_peak_series",
                "Fit one peak in every curve of a parametric series and track its "
                "center/FWHM/height vs the parameter, then run changepoint detection "
                "on center and FWHM trends. Use for transition hunting. paths accepts "
                "glob patterns or a directory. Saves a tracking plot when matplotlib "
                "is available.",
                _params(
                    {"paths": {"type": "array", "items": string}, "center": number,
                     "fwhm_guess": opt_number, "mask_value": opt_number},
                    ["paths", "center"],
                ),
                track_peak_series,
            ),
            AgentTool(
                "plot_1d",
                "Render a 1D data file or saved cut to a PNG, marking detected peaks. "
                "Use so the user can visually judge the data.",
                _params({"path": string, "logy": {"type": "boolean"},
                         "mark_peaks": {"type": "boolean"}}, ["path"]),
                plot_1d,
            ),
            AgentTool(
                "plot_fit_1d",
                "Fit pseudo-Voigt peaks at the given centers AND render a "
                "data+model+residual PNG with fitted values annotated. Prefer this "
                "over fit_peaks_1d when the user wants to see the fit.",
                _params(
                    {"path": string, "centers": {"type": "array", "items": number},
                     "fwhm_guess": opt_number, "window": opt_number},
                    ["path", "centers"],
                ),
                plot_fit_1d,
            ),
            AgentTool(
                "plot_slice_2d",
                "Render a saved 2D slice (.npz) to a PNG (optionally log scale, "
                "optionally marking detected peaks).",
                _params({"slice_path": string, "log": {"type": "boolean"},
                         "mark_peaks": {"type": "boolean"}}, ["slice_path"]),
                plot_slice_2d,
            ),
            AgentTool(
                "plot_series_files",
                "Render a parametric series of 1D files as a color-coded waterfall "
                "PNG (optionally restricted to an x-range). paths accepts glob "
                "patterns or a directory.",
                _params(
                    {"paths": {"type": "array", "items": string},
                     "mask_value": opt_number, "xmin": opt_number, "xmax": opt_number},
                    ["paths"],
                ),
                plot_series_files,
            ),
        ]
    )

    if skills:
        from scattering_ai.skills.builtin import register_skills

        register_skills(registry)
    return registry
