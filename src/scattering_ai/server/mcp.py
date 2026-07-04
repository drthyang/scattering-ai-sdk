"""MCP server (roadmap C4): the SDK's tools for any agent host.

A thin adapter — zero analysis logic lives here. Every ``AgentTool`` in the
registry is exposed 1:1 (the registry's JSON schemas are the contract), plus
one high-level ``analyze`` tool that runs the full SDK loop (diagnostics →
knowledge retrieval → optional local-LLM interpretation → provenance-carrying
report).

Run:  ``scattering-ai mcp [--workspace DIR]``
Add to Claude Code:
  ``claude mcp add scattering-ai -- scattering-ai mcp``

The host model orchestrates the individual tools itself; ``analyze`` is for
hosts that want the packaged behavior. If SCATTERING_AI_MODEL is set,
``analyze`` uses that OpenAI-compatible backend for interpretation; otherwise
it runs in deterministic-only mode (no LLM inside the LLM's tool).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from scattering_ai.tools.registry import ToolRegistry, default_toolkit

SERVER_NAME = "scattering-ai"

ANALYZE_TOOL = {
    "name": "analyze",
    "description": (
        "Run the SDK's full analysis loop on structured scientific state: "
        "deterministic diagnostics, cited knowledge retrieval, and a "
        "provenance-carrying report. domain is optional — omit it (or pass "
        "'auto') to route by the input: 'pdf' (G(r)/S(Q) curves), 'diffuse' "
        "(reciprocal-space volumes/slices), 'rmc' (RMC monitor state), or "
        "'data' (generic). The chosen pack is on report.domain. Returns JSON."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "domain": {"type": "string",
                       "enum": ["auto", "rmc", "pdf", "diffuse", "data"]},
            "question": {"type": "string"},
            "data": {
                "type": "object",
                "description": "AnalysisData payload (run_summary/r_values/files/"
                "plots/metadata); put file paths in 'files'",
            },
        },
        "required": ["question"],
    },
}


def tool_definitions(registry: ToolRegistry) -> list[dict[str, Any]]:
    """All MCP tool definitions: registry tools 1:1 plus high-level analyze."""
    defs = [
        {"name": s.name, "description": s.description, "parameters": s.parameters}
        for s in registry.specs
    ]
    return defs + [ANALYZE_TOOL]


def handle_tool_call(
    registry: ToolRegistry, name: str, arguments: dict[str, Any], workspace: Path
) -> dict[str, Any]:
    """Execute one tool call; always returns a JSON-safe dict."""
    if name == "analyze":
        return _run_analyze(arguments, workspace)
    return registry.execute(name, arguments)


def _run_analyze(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    from scattering_ai.core.agent import Agent
    from scattering_ai.core.schemas import AnalysisRequest

    try:
        request = AnalysisRequest(
            domain=arguments.get("domain") or "auto",
            question=arguments["question"],
            data=arguments.get("data") or {},
        )
    except Exception as exc:
        return {"error": f"Invalid analyze arguments: {exc}"}

    llm = None
    model_id = ""
    if os.environ.get("SCATTERING_AI_MODEL"):
        from scattering_ai.core.config import SDKConfig
        from scattering_ai.llm.openai_compatible import OpenAICompatibleClient

        config = SDKConfig.from_env()
        llm = OpenAICompatibleClient(config)
        model_id = config.model
    report = Agent(llm=llm, model_id=model_id, workspace=workspace).analyze(request)
    return json.loads(report.model_dump_json())


def serve(workspace: str | Path | None = None) -> None:
    """Run the stdio MCP server (blocking)."""
    import anyio

    try:
        import mcp.types as types
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The MCP server requires the 'mcp' package: "
            "pip install scattering-ai-sdk[mcp]"
        ) from exc

    workspace = Path(workspace) if workspace else _default_workspace()
    registry = default_toolkit(workspace)
    server: Server = Server(SERVER_NAME)

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [
            types.Tool(name=d["name"], description=d["description"],
                       inputSchema=d["parameters"])
            for d in tool_definitions(registry)
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        result = handle_tool_call(registry, name, arguments or {}, workspace)
        return [types.TextContent(type="text", text=json.dumps(result, default=str))]

    async def _run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream,
                             server.create_initialization_options())

    anyio.run(_run)


def _default_workspace() -> Path:
    import tempfile

    return Path(tempfile.mkdtemp(prefix="scattering_ai_mcp_"))
