import json

from scattering_ai import AnalysisReport, AnalysisRequest, Citation, Confidence
from scattering_ai.core.config import SDKConfig
from scattering_ai.core.schemas import SCHEMA_VERSION
from scattering_ai.llm.base import LLMClient, Message, ModelCapabilities


def test_request_roundtrip():
    request = AnalysisRequest(
        project="example_project",
        domain="rmc",
        question="Is this RMC run healthy?",
        data={"r_values": [12.1, 10.4, 9.8], "files": ["run.log"]},
    )
    restored = AnalysisRequest.model_validate_json(request.model_dump_json())
    assert restored == request
    assert restored.schema_version == SCHEMA_VERSION
    assert restored.options.use_rag is True


def test_report_roundtrip_and_json_shape():
    report = AnalysisReport(
        summary="Run appears healthy.",
        observations=["Rwp decreased from 12.1 to 9.8 over the run."],
        interpretation=["Convergence trend is consistent with a healthy refinement."],
        recommended_next_checks=["Inspect partial PDFs for unphysical features."],
        citations=[Citation(source="knowledge/rmcprofile/convergence_interpretation.md")],
        confidence=Confidence.MEDIUM,
    )
    payload = json.loads(report.model_dump_json())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["confidence"] == "medium"
    assert "provenance" in payload
    assert AnalysisReport.model_validate(payload) == report


def test_report_defaults_are_safe():
    report = AnalysisReport()
    assert report.status == "ok"
    assert report.confidence == Confidence.LOW
    assert report.provenance.schema_version == SCHEMA_VERSION


def test_config_backends():
    assert SDKConfig.lm_studio().base_url.endswith(":1234/v1")
    assert SDKConfig.ollama().base_url.endswith(":11434/v1")
    assert SDKConfig.openai(api_key="k").api_key == "k"


def test_llm_client_protocol():
    class FakeClient:
        @property
        def capabilities(self) -> ModelCapabilities:
            return ModelCapabilities(tool_use=True)

        def complete(self, messages, tools=None):
            return None

    assert isinstance(FakeClient(), LLMClient)
    message = Message(role="user", content="hello")
    assert message.model_dump() == {"role": "user", "content": "hello"}
