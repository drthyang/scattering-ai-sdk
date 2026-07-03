import json
from pathlib import Path

import pytest

from scattering_ai.connectors.rmc_monitor import analyze_monitor, from_monitor_json

EXAMPLES = Path(__file__).parents[1] / "examples" / "rmc_monitor_demo"


def decreasing(n=30, start=20.0):
    return [start - 0.3 * i for i in range(n)]


def test_full_request_shape_passthrough():
    request = from_monitor_json(EXAMPLES / "stalled_run.json")
    assert request.domain == "rmc"
    assert request.question == "Why is this run not improving?"
    # question override
    request2 = from_monitor_json(EXAMPLES / "stalled_run.json", question="Compare runs?")
    assert request2.question == "Compare runs?"


def test_rmc_state_shape():
    payload = {
        "series": [
            {"label": "bragg", "kind": "bragg", "values": decreasing()},
            {"label": "pdf", "kind": "pdf", "values": decreasing(start=15.0)},
        ],
        "expected_files": ["run.log"],
        "present_files": ["run.log"],
        "log_tail": ["step 100 ok"],
    }
    request = from_monitor_json(payload)
    assert request.domain == "rmc"
    assert request.question == "Is this RMC run healthy?"
    rmc = request.data.metadata["rmc"]
    assert len(rmc["series"]) == 2

    report = analyze_monitor(payload)
    assert report.warnings == []
    assert "healthy" in report.summary.lower()


def test_legacy_flat_shape():
    payload = {"r_values": decreasing(), "files": ["run.log"], "project": "x"}
    request = from_monitor_json(payload)
    assert request.data.r_values == decreasing()
    assert request.data.files == ["run.log"]
    report = analyze_monitor(payload)
    assert any("decreasing" in o for o in report.observations)


def test_invalid_payload_rejected():
    with pytest.raises(ValueError, match="Unrecognized RMC Monitor payload"):
        from_monitor_json({"nonsense": 1})


def test_invalid_series_rejected_with_validation_error():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        from_monitor_json({"series": [{"label": "x", "values": "not-a-list"}]})


def test_accepts_json_file_path(tmp_path):
    p = tmp_path / "monitor.json"
    p.write_text(json.dumps({"r_values": decreasing()}))
    report = analyze_monitor(p)
    assert report.provenance.input_hash.startswith("sha256:")
