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

The server also exposes MCP **resources** (the curated knowledge base + a domain
overview) and **prompts** (a domain-guidance template and an analyze-files
template), so a host can read the SDK's science and adopt its grounding rules.
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
                       "enum": ["auto", "rmc", "pdf", "diffuse", "symmetry", "data"]},
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
    result = registry.execute(name, arguments)
    _journal_mcp_failure(name, result)
    return result


def _journal_mcp_failure(name: str, result: dict[str, Any]) -> None:
    """Capture a redacted episode when an MCP tool call fails (self-improvement
    E6). Opt-in via ``SCATTERING_AI_JOURNAL``; best-effort — a journaling
    failure never affects the tool response. Only failures are recorded (a
    clean tool call is not a signal); the error message itself is never stored,
    only the tool name."""
    if not result.get("error"):
        return
    from scattering_ai.learning.journal import (
        Journal,
        episode_from_mcp_tool,
        resolve_journal_dir,
    )

    directory = resolve_journal_dir(None)  # env-configured for the server
    if directory is None:
        return
    try:
        Journal(directory).record(episode_from_mcp_tool(name, result))
    except Exception:
        pass


# --------------------------------------------------------------- resources

_DOMAINS_URI = "scattering-ai://domains"
_KNOWLEDGE_PREFIX = "knowledge://"


def _knowledge_root() -> Path | None:
    from scattering_ai.rag.retriever import default_knowledge_root

    root = default_knowledge_root()
    return Path(root) if root else None


def _domain_overview() -> dict[str, Any]:
    from scattering_ai.domains.registry import _BUILTIN, get_domain

    return {name: {"description": get_domain(name).description,
                   "prompt_version": get_domain(name).prompt_version}
            for name in sorted(_BUILTIN)}


def resource_definitions() -> list[dict[str, str]]:
    """The curated knowledge base + a domain overview, as MCP resources."""
    resources = [{"uri": _DOMAINS_URI, "name": "domains",
                  "description": "The SDK's domain packs and their prompt versions",
                  "mimeType": "application/json"}]
    root = _knowledge_root()
    if root is not None:
        for md in sorted(root.glob("*/*.md")):
            first = next((ln.lstrip("# ").strip()
                          for ln in md.read_text(errors="replace").splitlines()
                          if ln.strip()), md.stem)
            resources.append({
                "uri": f"{_KNOWLEDGE_PREFIX}{md.relative_to(root).as_posix()}",
                "name": md.stem, "description": first, "mimeType": "text/markdown"})
    return resources


def read_resource(uri: str) -> str:
    """Return a resource's contents; guards against path traversal."""
    if uri == _DOMAINS_URI:
        return json.dumps(_domain_overview(), indent=2)
    if uri.startswith(_KNOWLEDGE_PREFIX):
        root = _knowledge_root()
        if root is None:
            raise ValueError("no knowledge base is configured")
        target = (root / uri[len(_KNOWLEDGE_PREFIX):]).resolve()
        if root.resolve() not in target.parents or not target.is_file():
            raise ValueError(f"unknown or unsafe resource: {uri}")
        return target.read_text(errors="replace")
    raise ValueError(f"unknown resource: {uri}")


# ----------------------------------------------------------------- prompts


def prompt_definitions() -> list[dict[str, Any]]:
    return [
        {"name": "domain_guidance",
         "description": "The grounding rules / system prompt for a technique "
         "domain (rmc, pdf, diffuse, symmetry, data) — adopt the SDK's discipline.",
         "arguments": [{"name": "domain", "description": "domain name", "required": True}]},
        {"name": "analyze_files",
         "description": "Draft a request to run the SDK's `analyze` tool on data files.",
         "arguments": [
             {"name": "files", "description": "comma-separated file paths", "required": True},
             {"name": "question", "description": "the scientific question", "required": False}]},
    ]


def get_prompt(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Return {description, messages:[{role, content}]} for a named prompt."""
    if name == "domain_guidance":
        from scattering_ai.domains.registry import get_domain

        pack = get_domain(arguments["domain"])
        return {"description": f"{pack.name} domain guidance ({pack.prompt_version})",
                "messages": [{"role": "user", "content": pack.system_prompt}]}
    if name == "analyze_files":
        files = [f.strip() for f in arguments["files"].split(",") if f.strip()]
        question = arguments.get("question") or "Analyze these files and report what you find."
        content = (f"Call the `analyze` tool with data={{'files': {files}}} and "
                   f"question='{question}'. Let the SDK auto-route the domain, then "
                   "summarize the report's conclusion, figures, and provenance.")
        return {"description": "analyze files via the SDK",
                "messages": [{"role": "user", "content": content}]}
    raise ValueError(f"unknown prompt: {name}")


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

    @server.list_resources()
    async def _list_resources() -> list[types.Resource]:
        return [types.Resource(uri=r["uri"], name=r["name"],
                               description=r["description"], mimeType=r["mimeType"])
                for r in resource_definitions()]

    @server.read_resource()
    async def _read_resource(uri) -> str:
        return read_resource(str(uri))

    @server.list_prompts()
    async def _list_prompts() -> list[types.Prompt]:
        return [types.Prompt(
            name=p["name"], description=p["description"],
            arguments=[types.PromptArgument(**a) for a in p["arguments"]])
            for p in prompt_definitions()]

    @server.get_prompt()
    async def _get_prompt(name: str, arguments: dict | None) -> types.GetPromptResult:
        p = get_prompt(name, arguments or {})
        return types.GetPromptResult(
            description=p["description"],
            messages=[types.PromptMessage(
                role=m["role"], content=types.TextContent(type="text", text=m["content"]))
                for m in p["messages"]])

    async def _run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream,
                             server.create_initialization_options())

    anyio.run(_run)


def _default_workspace() -> Path:
    import tempfile

    return Path(tempfile.mkdtemp(prefix="scattering_ai_mcp_"))
