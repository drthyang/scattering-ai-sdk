"""D4: provenance enforcement in output validation.

Every report the SDK returns must be fully attributable; reports with an
incomplete provenance block are rejected rather than returned.
"""

import json

import numpy as np
import pytest

from scattering_ai import (
    Agent,
    AnalysisReport,
    AnalysisRequest,
    Provenance,
    ReportValidationError,
    analyze,
)
from scattering_ai.core.schemas import REQUIRED_PROVENANCE_FIELDS
from scattering_ai.llm.base import LLMResponse, ModelCapabilities, ToolCall


def _complete_prov(**overrides):
    base = dict(
        sdk_version="0.1.0",
        input_hash="sha256:abc123",
        timestamp="2026-07-03T00:00:00+00:00",
    )
    base.update(overrides)
    return Provenance(**base)


def test_offline_analyze_report_is_valid():
    """Offline reports fill the required fields and need no model attribution."""
    report = analyze(domain="data", question="?", data={"files": []})
    assert report.provenance.missing_fields(requires_model=False) == []
    report.assert_valid()  # does not raise


def test_missing_required_field_rejected():
    for field in REQUIRED_PROVENANCE_FIELDS:
        if field == "schema_version":
            continue  # defaulted by the model
        prov = _complete_prov(**{field: ""})
        assert field in prov.missing_fields()
        with pytest.raises(ReportValidationError):
            AnalysisReport(provenance=prov).assert_valid()


def test_malformed_input_hash_rejected():
    prov = _complete_prov(input_hash="not-a-hash")
    assert "input_hash" in prov.missing_fields()
    with pytest.raises(ReportValidationError):
        AnalysisReport(provenance=prov).assert_valid()


def test_model_attribution_required_only_when_llm_ran():
    prov = _complete_prov()  # no model / prompt_version
    assert prov.missing_fields(requires_model=False) == []
    assert prov.missing_fields(requires_model=True) == ["model", "prompt_version"]


class _FakeLLM:
    capabilities = ModelCapabilities(tool_use=False)

    def __init__(self, content):
        self._content = content

    def complete(self, messages, tools=None):
        return LLMResponse(content=self._content)


def test_llm_report_records_model_and_prompt_version():
    llm = _FakeLLM(json.dumps({"summary": "ok", "confidence": "low"}))
    report = analyze(domain="rmc", question="healthy?", data={}, llm=llm,
                     model_id="ollama:qwen3:32b")
    report.assert_valid(requires_model=True)  # does not raise
    assert report.provenance.model == "ollama:qwen3:32b"
    assert report.provenance.prompt_version  # set from the pack


def test_llm_without_model_id_falls_back_to_client_identity():
    """Convenience API must stay attributable even if model_id is omitted:
    the client class stands in, never an empty or fabricated name."""
    llm = _FakeLLM(json.dumps({"summary": "ok", "confidence": "low"}))
    report = analyze(domain="rmc", question="healthy?", data={}, llm=llm)
    assert report.provenance.model == "_FakeLLM"
    report.assert_valid(requires_model=True)


def test_agent_rejects_when_llm_makes_tool_calls_but_pipeline_intact(tmp_path):
    """End-to-end: an LLM-driven run still yields a fully attributable report."""
    x = np.linspace(0, 10, 200)
    curve = tmp_path / "c.dat"
    np.savetxt(curve, np.column_stack([x, np.ones_like(x)]))

    class ToolThenAnswer:
        capabilities = ModelCapabilities(tool_use=True)

        def __init__(self):
            self.n = 0

        def complete(self, messages, tools=None):
            self.n += 1
            if self.n == 1:
                return LLMResponse(tool_calls=[ToolCall(
                    name="inspect_curve", arguments={"path": str(curve)}, id="a")])
            return LLMResponse(content=json.dumps({"summary": "s", "confidence": "low"}))

    report = Agent(llm=ToolThenAnswer(), model_id="scripted",
                   workspace=tmp_path / "ws").analyze(
        AnalysisRequest(domain="data", question="?", data={"files": [str(curve)]})
    )
    assert report.provenance.missing_fields(requires_model=True) == []
