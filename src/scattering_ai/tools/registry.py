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
    category: str = ""  # skills group under this; "" for plain data tools

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

    def skills_by_category(self) -> dict[str, list[str]]:
        """Skill (``skill_*``) tool names grouped by their category, for a
        readable overview of what composite workflows are available."""
        grouped: dict[str, list[str]] = {}
        for tool in self._tools.values():
            if tool.name.startswith("skill_"):
                grouped.setdefault(tool.category or "general", []).append(tool.name)
        return {k: sorted(v) for k, v in sorted(grouped.items())}

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

    def delta_pdf(path: str, apodization: str = "hann",
                  punch_sigma: float = 6.0, fill: bool = True) -> dict:
        from scattering_ai.tools.delta_pdf import (
            central_slices,
            compute_delta_pdf,
            delta_pdf_summary,
            fill_punched,
            punch_bragg,
        )
        from scattering_ai.tools.models import Slice2D

        vol = load_volume(path)
        punched = punch_bragg(vol.load_data(), n_sigma=punch_sigma)
        if fill:  # backfill the punch holes so their lattice doesn't imprint
            punched = fill_punched(punched)
        dpdf = compute_delta_pdf(punched, apodization=apodization)
        xy = central_slices(dpdf)["xy"]
        ny, nx = xy.shape
        s = Slice2D(data=xy,
                    x_centers=np.arange(nx, dtype=float) - nx // 2,
                    y_centers=np.arange(ny, dtype=float) - ny // 2,
                    xlabel="Δ (direct-lattice steps)", ylabel="Δ (direct-lattice steps)",
                    meta={"kind": "delta_pdf_central_xy"})
        out = artifact("deltapdf_xy", ".npz")
        _save_slice(s, out)
        result = {"saved_slice": str(out), **delta_pdf_summary(dpdf)}
        try:
            from scattering_ai.tools.plotting import plot_slice

            result["plot"] = plot_slice(s, artifact("deltapdf_xy", ".png"), log=False)
        except ImportError:
            pass
        return result

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
        from scattering_ai.tools.series import (
            apply_mask,
            auto_mask_value,
            load_series,
            stack_series,
        )

        series = load_series(_expand_paths(paths))  # unmasked; detect first
        detected = auto_mask_value(series.curves)
        used = mask_value if mask_value is not None else detected
        if used is not None:
            apply_mask(series, float(used))

        out = series.summary()
        out["mask_value_detected"] = detected
        out["mask_value_used"] = used
        # Candidates come from the mean over the whole scan, so a peak present
        # in only one curve (noise) is not chosen for tracking.
        stack = stack_series(series)
        out["strongest_peaks"] = c.find_peaks(stack, subtract_background=True)[:10]
        out["peaks_in_first_curve"] = c.find_peaks(
            series.curves[0], subtract_background=True
        )[:10]
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

    def _structure(path: str):
        from scattering_ai.tools.cif import read_structure

        s = read_structure(path)
        return s["lattice"], s["positions"], s["species"]

    def find_symmetry(path: str, symprec: float = 1e-3) -> dict:
        from scattering_ai.tools import symmetry as sym

        lat, pos, sp = _structure(path)
        return sym.find_symmetry(lat, pos, sp, symprec=symprec)

    def subgroup_tree(path: str, symprec: float = 1e-3) -> dict:
        from scattering_ai.tools import symmetry as sym

        lat, pos, sp = _structure(path)
        result = sym.maximal_subgroups(lat, pos, sp, symprec=symprec)
        if "error" not in result:
            try:
                result["plot"] = sym.subgroup_tree_figure(
                    result, artifact("subgroup_tree", ".png"))
            except ImportError:
                pass
        return result

    def pseudosymmetry_scan(path: str) -> dict:
        from scattering_ai.tools import symmetry as sym

        lat, pos, sp = _structure(path)
        return sym.pseudosymmetry_scan(lat, pos, sp)

    def magnetic_symmetry(path: str, magmoms: list[float]) -> dict:
        from scattering_ai.tools import symmetry as sym

        lat, pos, sp = _structure(path)
        return sym.magnetic_symmetry(lat, pos, sp, magmoms)

    def lookup_structures(elements: list[str] | None = None, formula: str = "",
                          exclusive: bool = False, max_results: int = 10) -> dict:
        from scattering_ai.tools.optimade import query_structures

        return query_structures(elements=elements or None, formula=formula or None,
                                exclusive=exclusive, max_results=max_results)

    def simulate_gr_from_cif(path: str, rmax: float = 20.0, sigma: float = 0.1,
                             radiation: str = "neutron") -> dict:
        from scattering_ai.tools.gr_model import simulate_gr

        lat, pos, sp = _structure(path)
        sim = simulate_gr(lat, pos, sp, rmax=rmax, sigma=sigma, radiation=radiation)
        out_path = artifact("gr_model", ".dat")
        np.savetxt(out_path, np.column_stack([sim["r"], sim["g"]]),
                   header="r  G(r) model")
        from scattering_ai.tools.models import Curve1D

        peaks = c.find_peaks(Curve1D(x=sim["r"], y=sim["g"]), subtract_background=False)
        return {"saved": str(out_path), "n_atoms": sim["n_atoms"],
                "radiation": radiation, "first_peaks": peaks[:8]}

    def fit_gr_model(gr_path: str, cif_path: str, radiation: str = "neutron",
                     rmin: float = 1.0, rmax: float | None = None) -> dict:
        from scattering_ai.tools.gr_model import fit_gr

        curve = load_curve(gr_path)
        lat, pos, sp = _structure(cif_path)
        fit = fit_gr(curve.x, curve.y, lat, pos, sp, radiation=radiation,
                     rmin=rmin, rmax=rmax)
        if "error" in fit:
            return fit
        result = {k: fit[k] for k in ("rw", "scale", "sigma", "lattice_scale",
                                      "fit_range", "n_points", "radiation",
                                      "assessment", "note")}
        try:
            from scattering_ai.tools.plotting import plot_gr_fit

            result["plot"] = plot_gr_fit(fit, artifact("gr_fit", ".png"))
        except ImportError:
            pass
        return result

    def simulate_mpdf_from_mcif(path: str, rmax: float = 20.0,
                                sigma: float = 0.1) -> dict:
        from scattering_ai.tools.cif import read_structure
        from scattering_ai.tools.mpdf import simulate_mpdf

        s = read_structure(path)
        if not s.get("moments"):
            return {"error": f"{path} has no magnetic moments (_atom_site_moment loop)"}
        sim = simulate_mpdf(s["lattice"], s["positions"], s["moments"],
                            rmax=rmax, sigma=sigma)
        if "error" in sim:
            return sim
        out_path = artifact("mpdf_model", ".dat")
        np.savetxt(out_path, np.column_stack([sim["r"], sim["f"]]),
                   header="r  mPDF f(r) (ideal, arbitrary scale)")
        i_min = int(np.argmin(sim["f"]))
        i_max = int(np.argmax(sim["f"]))
        result = {"saved": str(out_path), "n_magnetic": sim["n_magnetic"],
                  "strongest_afm_distance": round(float(sim["r"][i_min]), 3),
                  "strongest_fm_distance": round(float(sim["r"][i_max]), 3),
                  "note": sim["note"]}
        try:
            from scattering_ai.tools.models import Curve1D
            from scattering_ai.tools.plotting import plot_curve

            curve = Curve1D(x=sim["r"], y=sim["f"], xlabel="r (Å)", ylabel="mPDF f(r)")
            result["plot"] = plot_curve(curve, artifact("mpdf", ".png"))
        except ImportError:
            pass
        return result

    def powder_magnetic_iq_from_mcif(path: str, qmax: float = 6.0,
                                     ion: str = "") -> dict:
        from scattering_ai.tools.cif import read_structure
        from scattering_ai.tools.mpdf import powder_magnetic_iq

        s = read_structure(path)
        if not s.get("moments"):
            return {"error": f"{path} has no magnetic moments (_atom_site_moment loop)"}
        out = powder_magnetic_iq(s["lattice"], s["positions"], s["moments"],
                                 qmax=qmax, ion=ion or None, species=s["species"])
        if "error" in out:
            return out
        saved = artifact("magnetic_iq", ".dat")
        np.savetxt(saved, np.column_stack([out["q"], out["i"]]),
                   header="Q  I(Q) magnetic (arbitrary scale)")
        i_peak = int(np.argmax(out["i"]))
        result = {"saved": str(saved), "n_magnetic": out["n_magnetic"],
                  "form_factor": out["form_factor"],
                  "strongest_peak_q": round(float(out["q"][i_peak]), 3),
                  "note": out["note"]}
        try:
            from scattering_ai.tools.models import Curve1D
            from scattering_ai.tools.plotting import plot_curve

            curve = Curve1D(x=out["q"], y=out["i"], xlabel="Q (1/Å)",
                            ylabel="I(Q) magnetic")
            result["plot"] = plot_curve(curve, artifact("magnetic_iq", ".png"))
        except ImportError:
            pass
        return result

    def spin_correlations_from_mcif(path: str, rmax: float = 12.0) -> dict:
        from scattering_ai.tools.cif import read_structure
        from scattering_ai.tools.mpdf import spin_correlations

        s = read_structure(path)
        if not s.get("moments"):
            return {"error": f"{path} has no magnetic moments (_atom_site_moment loop)"}
        return spin_correlations(s["lattice"], s["positions"], s["moments"], rmax=rmax)

    def kvector_check_from_mcif(path: str, k: list[float] | None = None,
                                rmax: float = 10.0) -> dict:
        from scattering_ai.tools.cif import read_structure
        from scattering_ai.tools.mpdf import kvector_consistency

        s = read_structure(path)
        if not s.get("moments"):
            return {"error": f"{path} has no magnetic moments (_atom_site_moment loop)"}
        return kvector_consistency(s["lattice"], s["positions"], s["moments"],
                                   k=k, rmax=rmax)

    def read_rmc6f(path: str) -> dict:
        from scattering_ai.tools.rmc_files import read_rmc6f as _read

        r = _read(path)
        return {k: r[k] for k in
                ("title", "cell", "supercell", "n_atoms", "composition")}

    def read_rmc_series(path: str) -> dict:
        from scattering_ai.tools.rmc_files import read_rmc_csv

        return read_rmc_csv(path)

    def rmc_density_map(path: str, axis: int = 2, element: str = "",
                        thickness: float = 0.25) -> dict:
        from scattering_ai.tools.models import Slice2D
        from scattering_ai.tools.rmc_files import rmc_density_slab

        r = rmc_density_slab(path, axis=axis, element=element or None, thickness=thickness)
        if "error" in r:
            return r
        s = Slice2D(data=r["density"], x_centers=r["x_centers"], y_centers=r["y_centers"],
                    xlabel=r["xlabel"], ylabel=r["ylabel"], meta={"kind": "rmc_density"})
        out = artifact("rmc_density", ".npz")
        _save_slice(s, out)
        result = {"saved": str(out), "n_points": r["n_points"],
                  "element": r["element"], "slab": r["slab"]}
        try:
            from scattering_ai.tools.plotting import plot_slice

            result["plot"] = plot_slice(s, artifact("rmc_density", ".png"), log=False)
        except ImportError:
            pass
        return result

    def systematic_absences(path: str, max_index: int = 6) -> dict:
        from scattering_ai.tools import symmetry as sym

        lat, pos, sp = _structure(path)
        return sym.systematic_absences(lat, pos, sp, max_index=max_index)

    def standardize_cell(path: str, to_primitive: bool = False) -> dict:
        from scattering_ai.tools import symmetry as sym

        lat, pos, sp = _structure(path)
        return sym.standardize_cell(lat, pos, sp, to_primitive=to_primitive)

    def plot_structure(path: str, bonds: bool = True) -> dict:
        from scattering_ai.tools.cif import read_structure
        from scattering_ai.tools.structure_viz import plot_structure as _plot

        s = read_structure(path)
        try:
            saved = _plot(s["lattice"], s["positions"], s["species"],
                          artifact("structure", ".png"), moments=s.get("moments"),
                          bonds=bonds, title=Path(path).stem)
        except ImportError:
            return {"error": "structure plotting needs matplotlib ([plots] extra)"}
        return {"saved": saved, "n_atoms": s["n_atoms"],
                "n_magnetic": s.get("n_magnetic", 0),
                "species": sorted(set(s["species"])),
                "space_group_cif": s.get("space_group_cif", "")}

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
                "delta_pdf",
                "Compute the 3D difference PDF (3D-ΔPDF) of a diffuse volume: "
                "punch out Bragg peaks, backfill the holes (fill=false to skip), "
                "apodize, and take the centred Fourier transform. Returns a "
                "central-plane slice + plot and the positive/negative correlation "
                "extremes. apodization: hann/gaussian/none.",
                _params(
                    {"path": string, "apodization": string, "punch_sigma": number,
                     "fill": {"type": "boolean"}},
                    ["path"],
                ),
                delta_pdf,
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
            AgentTool(
                "find_symmetry",
                "Detect the space group of a crystal structure (CIF) at a given "
                "tolerance: number, symbol, point group, crystal system, and "
                "Wyckoff sites (FINDSYM-like). Needs the [symmetry] extra.",
                _params({"path": string, "symprec": number}, ["path"]),
                find_symmetry,
            ),
            AgentTool(
                "subgroup_tree",
                "List the maximal subgroups of a structure's space group — the "
                "group-subgroup pathways a structural phase transition can take "
                "(type, index, number of domain variants) — and draw the tree.",
                _params({"path": string, "symprec": number}, ["path"]),
                subgroup_tree,
            ),
            AgentTool(
                "pseudosymmetry_scan",
                "Relax the position tolerance to find a higher-symmetry parent "
                "phase a slightly distorted structure sits under (the "
                "pseudosymmetry a transition breaks).",
                _params({"path": string}, ["path"]),
                pseudosymmetry_scan,
            ),
            AgentTool(
                "magnetic_symmetry",
                "Determine the magnetic (Shubnikov) space group of an ordered "
                "magnetic structure: pass the CIF and one collinear moment per "
                "atom (or an N x 3 list of moment vectors).",
                _params(
                    {"path": string, "magmoms": {"type": "array", "items": number}},
                    ["path", "magmoms"],
                ),
                magnetic_symmetry,
            ),
            AgentTool(
                "plot_structure",
                "Render a crystal structure (CIF/mCIF) to a PNG: unit cell, "
                "element-coloured atoms, bonds, and magnetic moment arrows for an "
                "mCIF. Returns the image path and a structure summary.",
                _params({"path": string, "bonds": {"type": "boolean"}}, ["path"]),
                plot_structure,
            ),
            AgentTool(
                "lookup_structures",
                "Look up candidate crystal structures in the open OPTIMADE "
                "databases (default: COD) by elements and/or reduced formula — "
                "identify known phases matching an observed cell/composition. "
                "NOTE: sends the element/formula query to an external web "
                "service (your data never leaves the machine).",
                _params(
                    {"elements": {"type": "array", "items": string},
                     "formula": string, "exclusive": {"type": "boolean"},
                     "max_results": {"type": "integer"}},
                    [],
                ),
                lookup_structures,
            ),
            AgentTool(
                "simulate_gr_from_cif",
                "Compute the model PDF G(r) of a crystal structure (CIF): pair "
                "sums with neutron scattering lengths (or radiation='xray' Z "
                "weighting) and Gaussian broadening sigma. Saves the model curve "
                "and returns its first peaks.",
                _params(
                    {"path": string, "rmax": number, "sigma": number,
                     "radiation": string},
                    ["path"],
                ),
                simulate_gr_from_cif,
            ),
            AgentTool(
                "fit_gr_model",
                "Fit a structure model (CIF) to a measured G(r): scale, Gaussian "
                "peak width, and a lattice-scale factor, with the Rw quality "
                "metric and a data/model/difference plot. THE tool for 'does "
                "this structure explain my PDF?' questions (comparison fit; full "
                "refinement is PDFgui territory).",
                _params(
                    {"gr_path": string, "cif_path": string, "radiation": string,
                     "rmin": number, "rmax": opt_number},
                    ["gr_path", "cif_path"],
                ),
                fit_gr_model,
            ),
            AgentTool(
                "simulate_mpdf_from_mcif",
                "Compute the ideal magnetic PDF (mPDF) of an ordered magnetic "
                "structure (mCIF with _atom_site_moment): negative peaks mark "
                "antiferromagnetically correlated pair distances, positive "
                "ferromagnetic. Saves the model curve; quantitative refinement "
                "is diffpy.mpdf territory.",
                _params({"path": string, "rmax": number, "sigma": number}, ["path"]),
                simulate_mpdf_from_mcif,
            ),
            AgentTool(
                "powder_magnetic_iq_from_mcif",
                "Powder-averaged magnetic diffuse scattering I(Q) from a magnetic "
                "structure (mCIF): Blech–Averbach spherical average with the <j0> "
                "magnetic form factor (ion inferred from the species, or pass ion "
                "e.g. 'Mn2'). Non-negative; ordered structures give broadened "
                "magnetic Bragg peaks. Saves the curve + plot.",
                _params({"path": string, "qmax": number, "ion": string}, ["path"]),
                powder_magnetic_iq_from_mcif,
            ),
            AgentTool(
                "kvector_check_from_mcif",
                "Test whether a magnetic structure's shell correlations match a "
                "propagation vector's ideal pattern cos(2πk·ΔR) — pass k, or omit "
                "it to scan high-symmetry candidates. ordered=true names the k; "
                "no matching k with decaying correlations flags short-range / "
                "frustrated order.",
                _params(
                    {"path": string,
                     "k": {"type": ["array", "null"], "items": number},
                     "rmax": number},
                    ["path"],
                ),
                kvector_check_from_mcif,
            ),
            AgentTool(
                "spin_correlations_from_mcif",
                "Normalized spin-pair correlations ⟨Ŝ·Ŝ⟩ per neighbour shell of a "
                "magnetic structure (mCIF): +1 = ferromagnetic shell, −1 = "
                "antiferromagnetic, 0 = uncorrelated — the real-space fingerprint "
                "spinvert-style analyses report.",
                _params({"path": string, "rmax": number}, ["path"]),
                spin_correlations_from_mcif,
            ),
            AgentTool(
                "read_rmc6f",
                "Read an RMCProfile .rmc6f configuration: the average unit cell, "
                "supercell multipliers, atom count, and composition.",
                _params({"path": string}, ["path"]),
                read_rmc6f,
            ),
            AgentTool(
                "read_rmc_series",
                "Read an RMCProfile CSV log (R-values / chi² vs step) into named "
                "columns for convergence analysis.",
                _params({"path": string}, ["path"]),
                read_rmc_series,
            ),
            AgentTool(
                "rmc_density_map",
                "Kernel-density map of an RMCProfile .rmc6f configuration folded "
                "into the average unit cell (a slab ⟂ axis 0/1/2, optionally one "
                "element) — reveals split sites and local disorder. Saves a plot.",
                _params(
                    {"path": string, "axis": {"type": "integer"},
                     "element": string, "thickness": number},
                    ["path"],
                ),
                rmc_density_map,
            ),
            AgentTool(
                "systematic_absences",
                "Symmetry-allowed reflections and systematic absences of a "
                "structure's space group (centering, screw, glide extinctions) — "
                "a symmetry-filtered Bragg peak checklist with d and |Q|.",
                _params({"path": string, "max_index": {"type": "integer"}}, ["path"]),
                systematic_absences,
            ),
            AgentTool(
                "standardize_cell",
                "Standardize a structure to its conventional (or primitive) "
                "setting; returns the cell parameters and the transformation "
                "matrix + origin shift from the input cell.",
                _params({"path": string, "to_primitive": {"type": "boolean"}}, ["path"]),
                standardize_cell,
            ),
        ]
    )

    if skills:
        from scattering_ai.skills.builtin import register_skills

        register_skills(registry)
    return registry
