"""Skills for magnetic structures (mCIF): local magnetism in one call."""

from __future__ import annotations

from typing import Any

from scattering_ai.skills.base import SkillRun
from scattering_ai.tools.registry import AgentTool, ToolRegistry, _params

CATEGORY = "magnetic"


def magnetic_diffuse(registry: ToolRegistry, path: str, rmax: float = 20.0,
                     ion: str = "") -> dict[str, Any]:
    """Full local-magnetism picture of an ordered spin structure (mCIF): the
    magnetic PDF, the spin-pair correlations per shell, and the powder magnetic
    diffuse I(Q) — with figures."""
    run = SkillRun(registry)
    mpdf = run.call("simulate_mpdf_from_mcif", path=path, rmax=rmax)
    if "error" in mpdf:
        return run.finish(error=mpdf["error"])
    corr = run.call("spin_correlations_from_mcif", path=path, rmax=min(rmax, 12.0))
    iq = run.call("powder_magnetic_iq_from_mcif", path=path, ion=ion)

    shells = corr.get("shells", []) if "error" not in corr else []
    nn = next((s for s in shells if abs(s["correlation"]) > 0.5), None)
    kind = "antiferromagnetic" if nn and nn["correlation"] < 0 else (
        "ferromagnetic" if nn else "weakly correlated")
    summary = (
        f"{mpdf['n_magnetic']} magnetic sites; nearest correlated shell is "
        f"{kind}"
        + (f" (r = {nn['r']:g} Å, ⟨Ŝ·Ŝ⟩ = {nn['correlation']:g})" if nn else "")
        + f". Strongest AFM mPDF feature at {mpdf['strongest_afm_distance']} Å; "
        f"strongest magnetic I(Q) peak at "
        f"{iq.get('strongest_peak_q', '?')} Å⁻¹.")
    return run.finish(
        summary=summary,
        n_magnetic=mpdf["n_magnetic"],
        correlation_shells=shells[:8],
        strongest_afm_distance=mpdf["strongest_afm_distance"],
        magnetic_iq_peak_q=iq.get("strongest_peak_q"),
        form_factor=iq.get("form_factor"),
        plots={"mpdf": mpdf.get("plot"), "magnetic_iq": iq.get("plot")},
        assessment={"note": "ideal forward calculation of a given moment "
                    "arrangement; refining spins/interactions to data is "
                    "spinvert/spinteract territory"},
    )


def frustration_check(registry: ToolRegistry, path: str,
                      k: list[float] | None = None) -> dict[str, Any]:
    """Is the spin arrangement a single-k ordered structure, or does it show a
    short-range / frustration signature? Compares shell correlations against
    propagation-vector patterns."""
    run = SkillRun(registry)
    check = run.call("kvector_check_from_mcif", path=path, k=k)
    if "error" in check:
        return run.finish(error=check["error"])
    corr = run.call("spin_correlations_from_mcif", path=path)

    if check["ordered"]:
        summary = (f"Ordered single-k magnetic structure: shell correlations "
                   f"match k = {check['k']} (rms {check['rms']:g}).")
    elif check["short_range"]:
        summary = ("No propagation vector reproduces the shell correlations and "
                   "|⟨Ŝ·Ŝ⟩| decays with distance — a short-range / geometric "
                   f"frustration signature (best k {check['k']}, rms {check['rms']:g}).")
    else:
        summary = (f"No single k matches (best {check['k']}, rms {check['rms']:g}) "
                   "and correlations do not simply decay — possibly multi-k, "
                   "incommensurate, or weakly correlated.")
    return run.finish(
        summary=summary,
        ordered=check["ordered"],
        short_range=check["short_range"],
        best_k=check["k"],
        rms=check["rms"],
        shells=check["shells"][:8],
        k_scan=check.get("k_scan"),
        correlation_shells=corr.get("shells", [])[:8] if "error" not in corr else [],
    )


def skills(registry: ToolRegistry) -> list[AgentTool]:
    number = {"type": "number"}
    string = {"type": "string"}
    return [
        AgentTool(
            "skill_magnetic_diffuse",
            "SKILL (composite workflow): the local-magnetism picture of a magnetic "
            "structure (mCIF) in one call — magnetic PDF, spin-pair correlations per "
            "neighbour shell, and powder magnetic diffuse I(Q), with figures. Prefer "
            "this for 'analyze the magnetism / magnetic correlations' questions.",
            _params({"path": string, "rmax": number, "ion": string}, ["path"]),
            lambda **kw: magnetic_diffuse(registry, **kw),
            category=CATEGORY,
        ),
        AgentTool(
            "skill_frustration_check",
            "SKILL (composite workflow): is this magnetic structure single-k "
            "ordered, or frustrated/short-range? Compares shell correlations to "
            "propagation-vector patterns (pass k or let it scan high-symmetry "
            "candidates) and reports the verdict with the per-shell evidence.",
            _params({"path": string,
                     "k": {"type": ["array", "null"], "items": number}}, ["path"]),
            lambda **kw: frustration_check(registry, **kw),
            category=CATEGORY,
        ),
    ]
