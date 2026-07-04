import json

import numpy as np
import pytest

from scattering_ai.server.mcp import (
    get_prompt,
    handle_tool_call,
    prompt_definitions,
    read_resource,
    resource_definitions,
    tool_definitions,
)
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


def test_tool_definitions_include_new_tools_and_skills(registry):
    names = {d["name"] for d in tool_definitions(registry)}
    # newest tools + skills are exposed 1:1 over MCP
    assert {"delta_pdf", "find_symmetry", "read_rmc6f", "skill_delta_pdf",
            "skill_symmetry_overview"} <= names
    analyze = next(d for d in tool_definitions(registry) if d["name"] == "analyze")
    assert "symmetry" in analyze["parameters"]["properties"]["domain"]["enum"]


def test_mcp_resources_expose_knowledge_and_domains():
    resources = resource_definitions()
    uris = {r["uri"] for r in resources}
    assert "scattering-ai://domains" in uris
    assert any(u.startswith("knowledge://") for u in uris)
    # domains resource is valid JSON listing the packs
    domains = json.loads(read_resource("scattering-ai://domains"))
    assert {"rmc", "pdf", "diffuse", "symmetry", "data"} <= set(domains)
    # a knowledge resource reads back its markdown
    kn = next(u for u in uris if u.startswith("knowledge://"))
    assert read_resource(kn).strip()


def test_mcp_resource_read_rejects_traversal():
    with pytest.raises(ValueError):
        read_resource("knowledge://../../pyproject.toml")
    with pytest.raises(ValueError):
        read_resource("bogus://nope")


def test_mcp_prompts():
    names = {p["name"] for p in prompt_definitions()}
    assert names == {"domain_guidance", "analyze_files"}
    guidance = get_prompt("domain_guidance", {"domain": "pdf"})
    assert guidance["messages"][0]["role"] == "user"
    assert "S(Q)" in guidance["messages"][0]["content"]
    files = get_prompt("analyze_files", {"files": "a.gr, b.nxs", "question": "healthy?"})
    msg = files["messages"][0]["content"]
    assert "analyze" in msg and "a.gr" in msg


def test_mcp_types_wiring_constructs():
    """The serve() handlers build these mcp objects — verify the API matches."""
    types = pytest.importorskip("mcp.types")
    r = resource_definitions()[0]
    types.Resource(uri=r["uri"], name=r["name"], description=r["description"],
                   mimeType=r["mimeType"])
    types.GetPromptResult(description="d", messages=[types.PromptMessage(
        role="user", content=types.TextContent(type="text", text="hi"))])
