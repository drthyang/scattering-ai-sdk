"""Command-line interface: ``scattering-ai analyze``.

Offline (deterministic-only) by default; pass ``--backend`` to add LLM
interpretation. The input file may be a full AnalysisRequest JSON or a bare
``data`` payload combined with ``--domain``/``--question``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scattering_ai.core.agent import Agent
from scattering_ai.core.config import SDKConfig
from scattering_ai.core.schemas import AnalysisRequest


def _build_request(args: argparse.Namespace) -> AnalysisRequest:
    default_question = "Analyze this data and report what you find."
    if args.input is None:
        if not args.file:
            sys.exit("error: provide an input JSON or one or more --file arguments")
        return AnalysisRequest(
            domain=args.domain or "auto",
            question=args.question or default_question,
            data={"files": args.file},
        )
    payload = json.loads(Path(args.input).read_text())
    if "domain" in payload and "question" in payload:
        request = AnalysisRequest.model_validate(payload)
        if args.domain:
            request.domain = args.domain
        if args.question:
            request.question = args.question
        return request
    return AnalysisRequest(
        domain=args.domain or "auto",
        question=args.question or default_question,
        data=payload,
    )


def _build_agent(args: argparse.Namespace) -> Agent:
    workspace = args.workspace or None
    if args.backend == "none":
        return Agent(workspace=workspace)
    factories = {
        "lmstudio": SDKConfig.lm_studio,
        "ollama": SDKConfig.ollama,
        "openai": SDKConfig.openai,
        "env": SDKConfig.from_env,
    }
    config = factories[args.backend]()
    if args.model:
        config.model = args.model
    if args.base_url:
        config.base_url = args.base_url
    from scattering_ai.llm.openai_compatible import OpenAICompatibleClient

    return Agent(
        llm=OpenAICompatibleClient(config),
        model_id=f"{args.backend}:{config.model or 'default'}",
        workspace=workspace,
    )


def _run_chat(args: argparse.Namespace) -> int:
    from scattering_ai.core.chat import ChatSession

    agent = _build_agent(args)
    if agent.llm is None:
        sys.exit("error: chat requires an LLM backend")
    session = ChatSession(
        llm=agent.llm,
        model_id=agent.model_id,
        workspace=args.workspace or None,
        files=args.file or None,
        on_tool_call=lambda name, arguments, result: print(
            f"  ⚙ {name}({', '.join(f'{k}={v}' for k, v in list(arguments.items())[:3])})"
            + (f"  ✗ {result['error']}" if result.get("error") else "")
        ),
    )
    print(f"scattering-ai chat  |  model: {agent.model_id}  |  "
          f"workspace: {session.workspace}")
    print("Type your question (Ctrl-D or 'exit' to quit).\n")
    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in ("exit", "quit"):
            break
        try:
            print(f"\n{session.turn(user_text)}\n")
        except Exception as exc:
            print(f"error: {exc}\n")
    print(f"Transcript: {session.workspace / 'chat_transcript.md'}")
    return 0


def _run_plot(args: argparse.Namespace) -> str:
    from scattering_ai.tools import plotting

    paths = args.paths
    out = args.out or str(Path(paths[0]).with_suffix(".png"))

    if len(paths) > 1:  # series waterfall
        from scattering_ai.tools.series import load_series

        series = load_series(paths, mask_value=args.mask_value)
        return plotting.plot_series(series.curves, series.params, out,
                                    param_label=series.param_label)

    path = paths[0]
    if path.endswith(".npz"):  # saved slice
        from scattering_ai.tools.registry import load_slice

        return plotting.plot_slice(load_slice(path), out, log=args.log)

    import numpy as np

    from scattering_ai.tools.curves import find_peaks, fit_peaks
    from scattering_ai.tools.io import load_curve

    curve = load_curve(path)
    if args.mask_value is not None:
        curve.y = np.where(curve.y == args.mask_value, np.nan, curve.y)
    if args.fit:
        centers = [float(c) for c in args.fit.split(",")]
        return plotting.plot_fit(curve, fit_peaks(curve, centers=centers), out)
    peaks = find_peaks(curve, subtract_background=True)
    return plotting.plot_curve(curve, out, peaks=peaks, logy=args.log)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scattering-ai")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze_cmd = sub.add_parser("analyze", help="Analyze a scientific state JSON file")
    analyze_cmd.add_argument(
        "input", nargs="?", default=None,
        help="AnalysisRequest JSON or bare data payload (optional with --file)",
    )
    analyze_cmd.add_argument(
        "--file", action="append", default=[],
        help="Data file to analyze with tools (repeatable; implies --domain data)",
    )
    analyze_cmd.add_argument("--workspace", default="", help="Directory for tool artifacts")
    analyze_cmd.add_argument(
        "--domain", default="",
        help="Domain: rmc, pdf, diffuse, data (default: auto-detect from the input)",
    )
    analyze_cmd.add_argument("--question", default="", help="Question to answer")
    analyze_cmd.add_argument("--out", default="", help="Write Markdown report to this path")
    analyze_cmd.add_argument("--json-out", default="", help="Write JSON report to this path")
    analyze_cmd.add_argument(
        "--backend",
        choices=["none", "lmstudio", "ollama", "openai", "env"],
        default="none",
        help="LLM backend (default: none = deterministic diagnostics only)",
    )
    analyze_cmd.add_argument("--model", default="", help="Model name for the backend")
    analyze_cmd.add_argument("--base-url", default="", help="Override backend base URL")

    mcp_cmd = sub.add_parser(
        "mcp", help="Run the MCP server (stdio) exposing the SDK's tools to agent hosts"
    )
    mcp_cmd.add_argument("--workspace", default="", help="Directory for tool artifacts")

    serve_cmd = sub.add_parser("serve", help="Run the HTTP API (FastAPI/uvicorn)")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8551)
    serve_cmd.add_argument("--workspace", default="", help="Directory for tool artifacts")

    chat_cmd = sub.add_parser(
        "chat", help="Interactive analysis session (requires an LLM backend)"
    )
    chat_cmd.add_argument("--file", action="append", default=[],
                          help="Data file to work with (repeatable)")
    chat_cmd.add_argument("--workspace", default="", help="Directory for tool artifacts")
    chat_cmd.add_argument(
        "--backend", choices=["lmstudio", "ollama", "openai", "env"], default="ollama"
    )
    chat_cmd.add_argument("--model", default="", help="Model name for the backend")
    chat_cmd.add_argument("--base-url", default="", help="Override backend base URL")

    plot_cmd = sub.add_parser(
        "plot",
        help="Quick-look plots: 1D files (peaks marked), .npz slices, or a series "
        "of files (waterfall)",
    )
    plot_cmd.add_argument("paths", nargs="+", help="Data file(s); several 1D files = series")
    plot_cmd.add_argument("--out", default="", help="Output PNG path (default: alongside input)")
    plot_cmd.add_argument("--log", action="store_true", help="Log intensity scale")
    plot_cmd.add_argument("--mask-value", type=float, default=None,
                          help="Sentinel value for masked points (e.g. -3.0)")
    plot_cmd.add_argument("--fit", default="",
                          help="Comma-separated centers: fit peaks there and plot the fit")

    args = parser.parse_args(argv)
    if args.command == "mcp":
        from scattering_ai.server.mcp import serve

        serve(workspace=args.workspace or None)
        return 0
    if args.command == "serve":
        import uvicorn

        from scattering_ai.server.api import create_app

        uvicorn.run(create_app(workspace=args.workspace or None),
                    host=args.host, port=args.port)
        return 0
    if args.command == "plot":
        print(f"Plot written to: {_run_plot(args)}")
        return 0
    if args.command == "chat":
        return _run_chat(args)
    request = _build_request(args)
    report = _build_agent(args).analyze(request)

    markdown = report.markdown
    if args.out:
        Path(args.out).write_text(markdown)
    if args.json_out:
        Path(args.json_out).write_text(report.model_dump_json(indent=2))
    if not args.out and not args.json_out:
        print(markdown)
    else:
        written = [p for p in (args.out, args.json_out) if p]
        print(f"Report written to: {', '.join(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
