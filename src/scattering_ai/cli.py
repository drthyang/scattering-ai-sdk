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
    if args.input is None:
        if not args.file:
            sys.exit("error: provide an input JSON or one or more --file arguments")
        if not args.question:
            sys.exit("error: --question is required with --file")
        return AnalysisRequest(
            domain=args.domain or "data",
            question=args.question,
            data={"files": args.file},
        )
    payload = json.loads(Path(args.input).read_text())
    if "domain" in payload and "question" in payload:
        request = AnalysisRequest.model_validate(payload)
        if args.question:
            request.question = args.question
        return request
    if not args.domain or not args.question:
        sys.exit(
            "error: input file is a bare data payload; --domain and --question are required"
        )
    return AnalysisRequest(domain=args.domain, question=args.question, data=payload)


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
    analyze_cmd.add_argument("--domain", default="", help="Domain (e.g. rmc, data)")
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

    args = parser.parse_args(argv)
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
