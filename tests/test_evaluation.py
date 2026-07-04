"""Domain behavior evaluation harness (roadmap D3).

Pins *required claims and grounding rules*, never exact wording. Cases came
from real failures observed during development:

- rmc_health/v1 let the model invent a config filename -> grounding rule:
  reports must not reference files absent from the input.
- data_analysis/v1 let the model claim fits it never ran -> honesty rule:
  fit language requires the fitter in used_tools (pipeline invariant +
  prompt content guard; full behavioral check needs a live model).

Live-model evals run only with SCATTERING_AI_LIVE_EVAL=1 (needs Ollama).
"""

import json
import os
import re
from pathlib import Path

import pytest

from scattering_ai import AnalysisReport, analyze

EXAMPLES = Path(__file__).parents[1] / "examples" / "rmc_monitor_demo"
LIVE = os.environ.get("SCATTERING_AI_LIVE_EVAL") == "1"


def report_text(report: AnalysisReport) -> str:
    return " ".join(
        [report.summary]
        + report.observations
        + report.interpretation
        + report.warnings
        + report.recommended_next_checks
    )


def require_claims(report: AnalysisReport, patterns: list[str]) -> list[str]:
    text = report_text(report).lower()
    return [p for p in patterns if not re.search(p, text)]


# ---------------------------------------------------------------- RMC cases


def rmc_case(name):
    return json.loads((EXAMPLES / f"{name}.json").read_text())


def test_eval_healthy_run_reports_health_without_warnings():
    case = rmc_case("healthy_run")
    report = analyze(domain="rmc", question=case["question"], data=case["data"])
    assert report.warnings == []
    missing = require_claims(report, [r"healthy", r"decreasing"])
    assert not missing, f"missing required claims: {missing}"


def test_eval_stalled_run_required_claims():
    case = rmc_case("stalled_run")
    report = analyze(domain="rmc", question=case["question"], data=case["data"])
    missing = require_claims(
        report,
        [
            r"conflict",  # bragg vs pdf disagreement must be named
            r"increasing",  # the worsening series must be identified
            r"batio3_partials\.csv",  # the missing file must be named
            r"weight",  # rebalancing weights must be recommended
        ],
    )
    assert not missing, f"missing required claims: {missing}"
    assert len(report.warnings) >= 3


def test_eval_grounding_no_invented_file_references():
    """Every file-like token in the report must exist in the input data.

    Regression for rmc_health/v1: the model recommended auditing a file
    name that was not in the input.
    """
    case = rmc_case("stalled_run")
    report = analyze(domain="rmc", question=case["question"], data=case["data"])
    rmc_meta = case["data"]["metadata"]["rmc"]
    input_files = set(rmc_meta["expected_files"]) | set(rmc_meta["present_files"])
    mentioned = set(re.findall(r"\b[\w-]+\.(?:csv|dat|log|rmc6f|out|inp)\b", report_text(report)))
    invented = mentioned - input_files
    assert not invented, f"report references files not in the input: {invented}"


def test_eval_provenance_is_complete_offline():
    case = rmc_case("stalled_run")
    report = analyze(domain="rmc", question=case["question"], data=case["data"])
    p = report.provenance
    assert p.sdk_version and p.input_hash.startswith("sha256:") and p.timestamp
    assert p.retrieved_chunks  # RAG ran and is auditable


# ------------------------------------------------- prompt regression guards


def test_prompt_guards_still_present():
    """The two documented failure modes stay addressed in the prompts."""
    from scattering_ai.domains.data import prompts as data_prompts
    from scattering_ai.domains.rmc import prompts as rmc_prompts

    assert "Never invent" in rmc_prompts.SYSTEM_PROMPT
    assert rmc_prompts.PROMPT_VERSION != "rmc_health/v1"

    assert "MUST call fit_peaks_1d" in data_prompts.SYSTEM_PROMPT
    assert data_prompts.PROMPT_VERSION != "data_analysis/v1"


def test_tool_provenance_invariant(tmp_path):
    """used_tools must exactly reflect executed tool calls (the tripwire that
    caught the unrun-fit claim)."""
    import numpy as np

    from scattering_ai import Agent, AnalysisRequest
    from scattering_ai.llm.base import LLMResponse, ModelCapabilities, ToolCall

    x = np.linspace(0, 10, 500)
    curve = tmp_path / "c.dat"
    np.savetxt(curve, np.column_stack([x, np.ones_like(x)]))

    class OneToolLLM:
        capabilities = ModelCapabilities(tool_use=True)

        def __init__(self):
            self.n = 0

        def complete(self, messages, tools=None):
            self.n += 1
            if self.n == 1:
                return LLMResponse(
                    tool_calls=[ToolCall(name="inspect_curve",
                                         arguments={"path": str(curve)}, id="a")]
                )
            return LLMResponse(content=json.dumps({"summary": "s", "confidence": "low"}))

    report = Agent(llm=OneToolLLM(), model_id="scripted", workspace=tmp_path / "ws").analyze(
        AnalysisRequest(domain="data", question="?", data={"files": [str(curve)]})
    )
    assert report.used_tools == ["inspect_curve"]
    assert [t.tool for t in report.provenance.tool_calls] == ["inspect_curve"]


# ------------------------------------------------------------- live evals


@pytest.mark.skipif(not LIVE, reason="set SCATTERING_AI_LIVE_EVAL=1 (needs Ollama)")
def test_live_fit_request_actually_fits(tmp_path):
    """Behavioral eval for data_analysis/v2: a question demanding fitted
    values must produce a fit_peaks_1d call."""
    from scattering_ai import Agent, AnalysisRequest
    from scattering_ai.core.config import SDKConfig
    from scattering_ai.llm.openai_compatible import OpenAICompatibleClient

    gr = Path(__file__).parents[1] / "data" / "1d" / "curves" / "XRAY_FeCoSn_100K_converted.gr"
    if not gr.exists():
        pytest.skip("real data not present")

    config = SDKConfig.ollama(model=os.environ.get("SCATTERING_AI_LIVE_MODEL", "qwen3:32b"))
    agent = Agent(
        llm=OpenAICompatibleClient(config),
        model_id=f"ollama:{config.model}",
        workspace=tmp_path / "ws",
    )
    report = agent.analyze(
        AnalysisRequest(
            domain="data",
            question="Fit the main peaks below r = 6 angstrom and report positions "
            "and widths with uncertainties.",
            data={"files": [str(gr)]},
        )
    )
    assert "fit_peaks_1d" in report.used_tools, (
        "model claimed analysis without fitting; prompt regression?"
    )
    assert report.summary
