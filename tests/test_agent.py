import json

from scattering_ai import Confidence, analyze
from scattering_ai.llm.base import LLMResponse, ModelCapabilities


def stalled_run_data():
    return {
        "metadata": {
            "rmc": {
                "series": [
                    {"label": "bragg_neutron", "kind": "bragg", "radiation": "neutron",
                     "values": [12.0 + 0.001 * (i % 2) for i in range(30)]},
                    {"label": "pdf_neutron", "kind": "pdf", "radiation": "neutron",
                     "values": [15.0 - 0.1 * i for i in range(30)]},
                ],
                "expected_files": ["run.log", "run.rmc6f"],
                "present_files": ["run.log"],
                "log_tail": ["WARNING: constraint violated at step 12000"],
            }
        }
    }


def test_offline_analysis_end_to_end():
    report = analyze(
        domain="rmc",
        question="Why is this run not improving?",
        data=stalled_run_data(),
    )
    assert report.status == "ok"
    assert report.summary
    assert any("flat" in o for o in report.observations)
    assert report.warnings
    assert report.recommended_next_checks
    # Deterministic mode: no model, but full provenance otherwise
    assert report.provenance.model == ""
    assert report.provenance.input_hash.startswith("sha256:")
    assert report.provenance.sdk_version
    # Retrieval ran and produced citations
    assert report.citations
    assert report.provenance.retrieved_chunks


def test_offline_report_renders_markdown():
    report = analyze(domain="rmc", question="Is this run healthy?", data=stalled_run_data())
    md = report.markdown
    assert "## Conclusion" in md
    assert "## Recommended next checks" in md
    assert "deterministic-only mode" in md


class FakeLLM:
    def __init__(self, content: str):
        self._content = content
        self.last_messages = None

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(tool_use=False)

    def complete(self, messages, tools=None) -> LLMResponse:
        self.last_messages = messages
        return LLMResponse(content=self._content, model="fake")


def test_llm_interpretation_is_used():
    payload = {
        "summary": "The run is stalled due to a Bragg/PDF conflict.",
        "interpretation": ["Local and average structure disagree [K1]."],
        "recommended_next_checks": ["Rebalance dataset weights."],
        "confidence": "medium",
    }
    fake = FakeLLM(json.dumps(payload))
    report = analyze(
        domain="rmc",
        question="Why is this run not improving?",
        data=stalled_run_data(),
        llm=fake,
        model_id="fake-model",
    )
    assert report.summary == payload["summary"]
    assert report.interpretation == payload["interpretation"]
    assert "Rebalance dataset weights." in report.recommended_next_checks
    assert report.confidence == Confidence.MEDIUM
    assert report.provenance.model == "fake-model"
    assert report.provenance.prompt_version == "rmc_health/v1"
    # Diagnostics and knowledge were actually in the prompt
    prompt_text = fake.last_messages[1].content
    assert "DIAGNOSTICS" in prompt_text and "[K1]" in prompt_text


def test_invalid_llm_json_falls_back_to_deterministic():
    fake = FakeLLM("I think the run looks fine, no JSON here.")
    report = analyze(
        domain="rmc", question="Is it healthy?", data=stalled_run_data(), llm=fake
    )
    assert report.summary  # deterministic fallback summary
    assert report.confidence == Confidence.LOW
    assert any("not valid JSON" in i for i in report.interpretation)


def test_unknown_domain_raises_with_available_list():
    try:
        analyze(domain="nope", question="?", data={})
    except KeyError as exc:
        assert "rmc" in str(exc)
    else:
        raise AssertionError("expected KeyError for unknown domain")
