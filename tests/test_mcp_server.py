import json

import numpy as np
import pytest

from scattering_ai.server.mcp import handle_tool_call, tool_definitions
from scattering_ai.tools.registry import default_toolkit


@pytest.fixture
def registry(tmp_path):
    return default_toolkit(tmp_path / "ws")


def test_tool_definitions_cover_registry_plus_analyze(registry):
    defs = tool_definitions(registry)
    names = [d["name"] for d in defs]
    assert "analyze" in names
    assert "fit_peaks_1d" in names and "slice_volume" in names
    assert len(names) == len(set(names))
    for d in defs:
        assert d["description"]
        assert d["parameters"]["type"] == "object"


def test_handle_registry_tool_call(registry, tmp_path):
    x = np.linspace(0, 10, 800)
    y = 1 + 5 * np.exp(-((x - 4) ** 2) / 0.02)
    path = tmp_path / "c.dat"
    np.savetxt(path, np.column_stack([x, y]))

    result = handle_tool_call(registry, "find_peaks_1d", {"path": str(path)},
                              tmp_path / "ws")
    assert result["n_peaks"] == 1
    assert abs(result["peaks"][0]["x"] - 4.0) < 0.05

    assert "error" in handle_tool_call(registry, "nope", {}, tmp_path / "ws")


def test_handle_analyze_tool_call(registry, tmp_path):
    args = {
        "domain": "rmc",
        "question": "Is this run healthy?",
        "data": {"r_values": [20.0 - 0.4 * i for i in range(30)]},
    }
    result = handle_tool_call(registry, "analyze", args, tmp_path / "ws")
    assert result["schema_version"] == "1"
    assert result["provenance"]["input_hash"].startswith("sha256:")
    assert any("decreasing" in o for o in result["observations"])

    bad = handle_tool_call(registry, "analyze", {"domain": "rmc"}, tmp_path / "ws")
    assert "error" in bad


def test_mcp_server_builds():
    """The stdio server wires up without errors (no client round-trip)."""
    pytest.importorskip("mcp")
    from scattering_ai.server import mcp as server_mcp

    # serve() blocks; only verify the pieces it assembles are importable and
    # the definitions serialize to JSON cleanly (what list_tools sends).
    registry = default_toolkit("/tmp/unused_ws_defs")
    payload = json.dumps(tool_definitions(registry))
    assert "analyze" in payload
    assert server_mcp.SERVER_NAME == "scattering-ai"
