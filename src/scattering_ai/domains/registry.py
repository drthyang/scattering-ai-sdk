"""Domain pack registry.

Built-in packs are registered here; external packs register via the
``scattering_ai.domains`` entry point group (a callable returning a
``DomainPack``), so third parties can add techniques without touching core.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import entry_points

from scattering_ai.core.findings import Finding
from scattering_ai.core.schemas import AnalysisRequest

ENTRY_POINT_GROUP = "scattering_ai.domains"


@dataclass
class DomainPack:
    name: str
    description: str
    # (request, workspace) -> findings. Findings may carry summarizing plot
    # paths in ``evidence["figures"]`` when a workspace and matplotlib are
    # available; the agent harvests them onto the report.
    run_diagnostics: Callable[..., list[Finding]]
    knowledge_dirs: list[str] = field(default_factory=list)
    prompt_version: str = ""
    system_prompt: str = ""
    # Deterministic "what to check next" rules for this technique, keyed by
    # finding rule-key (``diagnostic`` or ``diagnostic:trend``). Domain content,
    # not core logic (decision D10) — the agent merges these as an offline floor.
    next_check_rules: dict[str, str] = field(default_factory=dict)


def _rmc_pack() -> DomainPack:
    from scattering_ai.domains.rmc import prompts
    from scattering_ai.domains.rmc.diagnostics import NEXT_CHECK_RULES, run_all
    from scattering_ai.domains.rmc.schemas import RMCRunState

    def run(request: AnalysisRequest, workspace=None) -> list[Finding]:
        return run_all(RMCRunState.from_analysis_data(request.data))

    return DomainPack(
        name="rmc",
        description="RMCProfile / reverse Monte Carlo run health analysis",
        run_diagnostics=run,
        knowledge_dirs=["rmcprofile", "scattering"],
        prompt_version=prompts.PROMPT_VERSION,
        system_prompt=prompts.SYSTEM_PROMPT,
        next_check_rules=NEXT_CHECK_RULES,
    )


def _data_pack() -> DomainPack:
    from scattering_ai.domains.data import prompts
    from scattering_ai.domains.data.diagnostics import NEXT_CHECK_RULES, run_all

    def run(request: AnalysisRequest, workspace=None) -> list[Finding]:
        return run_all(request.data.files, workspace=workspace)

    return DomainPack(
        name="data",
        description="Generic tool-driven analysis of scattering data files",
        run_diagnostics=run,
        knowledge_dirs=["scattering"],
        prompt_version=prompts.PROMPT_VERSION,
        system_prompt=prompts.SYSTEM_PROMPT,
        next_check_rules=NEXT_CHECK_RULES,
    )


def _pdf_pack() -> DomainPack:
    from scattering_ai.domains.pdf import prompts
    from scattering_ai.domains.pdf.diagnostics import NEXT_CHECK_RULES, run_all

    def run(request: AnalysisRequest, workspace=None) -> list[Finding]:
        return run_all(request.data.files, workspace=workspace)

    return DomainPack(
        name="pdf",
        description="Pair-distribution-function / total-scattering (G(r), S(Q), F(Q)) analysis",
        run_diagnostics=run,
        knowledge_dirs=["scattering"],
        prompt_version=prompts.PROMPT_VERSION,
        system_prompt=prompts.SYSTEM_PROMPT,
        next_check_rules=NEXT_CHECK_RULES,
    )


def _diffuse_pack() -> DomainPack:
    from scattering_ai.domains.diffuse import prompts
    from scattering_ai.domains.diffuse.diagnostics import NEXT_CHECK_RULES, run_all

    def run(request: AnalysisRequest, workspace=None) -> list[Finding]:
        return run_all(request.data.files, workspace=workspace)

    return DomainPack(
        name="diffuse",
        description="Single-crystal diffuse scattering / 3D-ΔPDF (volumes and slices)",
        run_diagnostics=run,
        knowledge_dirs=["scattering"],
        prompt_version=prompts.PROMPT_VERSION,
        system_prompt=prompts.SYSTEM_PROMPT,
        next_check_rules=NEXT_CHECK_RULES,
    )


_BUILTIN: dict[str, Callable[[], DomainPack]] = {
    "rmc": _rmc_pack,
    "data": _data_pack,
    "pdf": _pdf_pack,
    "diffuse": _diffuse_pack,
}


def get_domain(name: str) -> DomainPack:
    if name in _BUILTIN:
        return _BUILTIN[name]()
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == name:
            return ep.load()()
    available = sorted(set(_BUILTIN) | {ep.name for ep in entry_points(group=ENTRY_POINT_GROUP)})
    raise KeyError(f"Unknown domain '{name}'. Available domains: {available}")
