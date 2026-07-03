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
    run_diagnostics: Callable[[AnalysisRequest], list[Finding]]
    knowledge_dirs: list[str] = field(default_factory=list)
    prompt_version: str = ""
    system_prompt: str = ""


def _rmc_pack() -> DomainPack:
    from scattering_ai.domains.rmc import prompts
    from scattering_ai.domains.rmc.diagnostics import run_all
    from scattering_ai.domains.rmc.schemas import RMCRunState

    def run(request: AnalysisRequest) -> list[Finding]:
        return run_all(RMCRunState.from_analysis_data(request.data))

    return DomainPack(
        name="rmc",
        description="RMCProfile / reverse Monte Carlo run health analysis",
        run_diagnostics=run,
        knowledge_dirs=["rmcprofile", "scattering"],
        prompt_version=prompts.PROMPT_VERSION,
        system_prompt=prompts.SYSTEM_PROMPT,
    )


_BUILTIN: dict[str, Callable[[], DomainPack]] = {
    "rmc": _rmc_pack,
}


def get_domain(name: str) -> DomainPack:
    if name in _BUILTIN:
        return _BUILTIN[name]()
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == name:
            return ep.load()()
    available = sorted(set(_BUILTIN) | {ep.name for ep in entry_points(group=ENTRY_POINT_GROUP)})
    raise KeyError(f"Unknown domain '{name}'. Available domains: {available}")
