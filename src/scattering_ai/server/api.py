"""HTTP API (roadmap C3): the same core over FastAPI, for applications that
prefer a local service to a Python import.

Run: ``scattering-ai serve --port 8551``  (requires
``pip install scattering-ai-sdk[api]``).

Endpoints mirror the MCP surface — one core, many doors:

- ``GET  /health``          liveness + version
- ``GET  /tools``           JSON-schema definitions of all tools
- ``POST /tools/{name}``    execute one tool (body = arguments object)
- ``POST /analyze``         full loop on an AnalysisRequest body
"""

from __future__ import annotations

from pathlib import Path

import scattering_ai
from scattering_ai.core.schemas import AnalysisRequest
from scattering_ai.server.mcp import _run_analyze, handle_tool_call, tool_definitions
from scattering_ai.tools.registry import default_toolkit


def create_app(workspace: str | Path | None = None):
    try:
        from fastapi import FastAPI
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The HTTP API requires FastAPI: pip install scattering-ai-sdk[api]"
        ) from exc

    if workspace is None:
        import tempfile

        workspace = tempfile.mkdtemp(prefix="scattering_ai_api_")
    workspace = Path(workspace)
    registry = default_toolkit(workspace)

    app = FastAPI(
        title="scattering-ai",
        version=scattering_ai.__version__,
        description="AI SDK for scattering science: deterministic data tools "
        "and provenance-carrying analysis.",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": scattering_ai.__version__,
                "workspace": str(workspace)}

    @app.get("/tools")
    def tools() -> list[dict]:
        return tool_definitions(registry)

    @app.post("/tools/{name}")
    def call_tool(name: str, arguments: dict) -> dict:
        return handle_tool_call(registry, name, arguments, workspace)

    @app.post("/analyze")
    def analyze(request: AnalysisRequest) -> dict:
        return _run_analyze(
            {"domain": request.domain, "question": request.question,
             "data": request.data.model_dump()},
            workspace,
        )

    return app
