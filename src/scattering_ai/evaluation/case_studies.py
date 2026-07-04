"""Reproducible case-study runs (roadmap D5).

Each case study is a **known answer on real data**: a request the SDK should
answer in a specific way, pinned as a check on the report text (never exact
wording). They are the evidence base for the publication (D5) and double as
integration tests of the packs end-to-end.

Two RMC cases run on committed demo data (`examples/rmc_monitor_demo/`), so they
execute in CI. The rest point at real datasets under `data/` that are gitignored
(facility data): those **skip cleanly** when the data is absent and run when a
researcher has it locally. A skipped case is not a failure — it is a case whose
data has not landed yet (cf. the E8 watch queue).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

# repo root: src/scattering_ai/evaluation/case_studies.py -> parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]


class CaseStudy(BaseModel):
    name: str
    description: str
    publication_angle: str
    kind: str  # "monitor" (RMC Monitor JSON) | "files" (data files -> analyze)
    domain: str = ""
    data: list[str] = Field(default_factory=list)  # repo-relative paths/globs
    must_contain: list[str] = Field(default_factory=list)  # case-insensitive
    must_not_contain: list[str] = Field(default_factory=list)


class CaseResult(BaseModel):
    name: str
    status: str  # passed | failed | skipped
    reason: str = ""
    unmet: list[str] = Field(default_factory=list)


CASE_STUDIES: list[CaseStudy] = [
    CaseStudy(
        name="rmc_local_vs_average_conflict",
        description="Is this RMC run healthy?",
        publication_angle="Local-vs-average structure conflict detection",
        kind="monitor",
        data=["examples/rmc_monitor_demo/stalled_run.json"],
        must_contain=["conflict"],
    ),
    CaseStudy(
        name="rmc_healthy_convergence",
        description="Is this RMC run healthy?",
        publication_angle="Convergence monitoring (healthy baseline)",
        kind="monitor",
        data=["examples/rmc_monitor_demo/healthy_run.json"],
        must_contain=["decreasing"],
        must_not_contain=["conflict"],
    ),
    CaseStudy(
        name="pdf_inverted_neutron_gr",
        description="Analyse this G(r) file.",
        publication_angle="PDF quality control — inverted-.gr sign gotcha",
        kind="files",
        domain="pdf",
        data=["data/1d/curves/*GaTa4Se8*SQ.gr"],
        must_contain=["invert"],
    ),
    CaseStudy(
        name="phase_transition_gaNb4Se8",
        description="Is there a phase transition in this temperature series?",
        publication_angle="Phase-transition detection from a T-series",
        kind="files",
        domain="data",
        data=["data/1d/series/*tth.dat*.dat"],
        must_contain=["transition"],
    ),
    CaseStudy(
        name="diffuse_anisotropy_corelli",
        description="Characterise this diffuse-scattering volume.",
        publication_angle="3D diffuse feature explanation (anisotropic sampling)",
        kind="files",
        domain="diffuse",
        data=["data/3d/volumes/*.nxs"],
        must_contain=["anisotrop"],
    ),
]


def _report_text(report) -> str:
    return " ".join(
        [report.summary, *report.observations, *report.interpretation,
         *report.warnings, *report.recommended_next_checks]
    ).lower()


def _resolve(patterns: list[str], root: Path) -> list[str]:
    found: list[str] = []
    for pat in patterns:
        if any(c in pat for c in "*?["):
            found += [str(p) for p in sorted(root.glob(pat)) if p.is_file()]
        elif (root / pat).is_file():
            found.append(str(root / pat))
    return found


def run_case_study(cs: CaseStudy, agent=None, root: Path | None = None) -> CaseResult:
    root = root or REPO_ROOT
    files = _resolve(cs.data, root)
    if not files:
        return CaseResult(name=cs.name, status="skipped",
                          reason="required data not present (gitignored) — run locally with it")

    from scattering_ai import Agent, AnalysisRequest

    agent = agent or Agent()
    if cs.kind == "monitor":
        from scattering_ai.connectors.rmc_monitor import analyze_monitor

        report = analyze_monitor(files[0], agent=agent)
    else:
        report = agent.analyze(AnalysisRequest(
            domain=cs.domain or "auto", question=cs.description,
            data={"files": files}))

    text = _report_text(report)
    unmet = [f"missing '{s}'" for s in cs.must_contain if s.lower() not in text]
    unmet += [f"unexpected '{s}'" for s in cs.must_not_contain if s.lower() in text]
    return CaseResult(name=cs.name, status="failed" if unmet else "passed", unmet=unmet)


def run_all(agent=None, root: Path | None = None) -> list[CaseResult]:
    return [run_case_study(cs, agent, root) for cs in CASE_STUDIES]
