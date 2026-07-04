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


def _expand_files(entries: list[str]) -> list[str]:
    """CLI wrapper over the shared expander: fail loudly on an empty glob."""
    from scattering_ai.core.files import expand_files

    for entry in entries:
        if any(ch in entry for ch in "*?[") and not expand_files([entry]):
            sys.exit(f"error: glob matched no files: {entry}")
    return expand_files(entries)


def _build_request(args: argparse.Namespace) -> AnalysisRequest:
    default_question = "Analyze this data and report what you find."
    if args.input is None:
        if not args.file:
            sys.exit("error: provide an input JSON or one or more --file arguments")
        return AnalysisRequest(
            domain=args.domain or "auto",
            question=args.question or default_question,
            data={"files": _expand_files(args.file)},
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
        files=_expand_files(args.file) if args.file else None,
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

    cases_cmd = sub.add_parser(
        "case-studies", help="Run reproducible known-answer case studies (D5); "
        "data-gated cases skip when their data is absent")
    cases_cmd.add_argument("--json", action="store_true", help="Emit results as JSON")

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

    learn_cmd = sub.add_parser(
        "learn", help="Self-improvement: review the local analysis journal "
        "(read-only; opt-in via SCATTERING_AI_JOURNAL)")
    learn_sub = learn_cmd.add_subparsers(dest="learn_command", required=True)
    _journal_help = "Journal directory (default: $SCATTERING_AI_JOURNAL)"
    learn_status = learn_sub.add_parser("status", help="Summarize the episode journal")
    learn_status.add_argument("--journal", default="", help=_journal_help)

    learn_signals = learn_sub.add_parser(
        "signals", help="Cluster deterministic signals (and corrections) from "
        "the journal — the evidence future proposals cite")
    learn_signals.add_argument("--journal", default="", help=_journal_help)

    learn_correct = learn_sub.add_parser(
        "correct", help="Record a HUMAN correction of a past result "
        "(seeds a regression eval); never inferred by the system")
    learn_correct.add_argument("--journal", default="", help=_journal_help)
    learn_correct.add_argument("--episode", default="",
                               help="Id of the episode being corrected (see 'learn status')")
    learn_correct.add_argument("--target", required=True,
                               help="What was wrong: domain, transition_temperature, ...")
    learn_correct.add_argument(
        "--statement", required=True,
        help="Plain-English correction, e.g. 'transitions are 50 K and 29 K, not 39 K'")
    learn_correct.add_argument("--value", default="",
                               help="Corrected value (optional, machine-usable)")
    learn_correct.add_argument("--domain", default="",
                               help="Domain the correction applies to")

    learn_review = learn_sub.add_parser(
        "review", help="Generate reviewable improvement proposals from the "
        "journal's signals (read-only; nothing is applied)")
    learn_review.add_argument("--journal", default="", help=_journal_help)
    learn_review.add_argument("--min-occurrences", type=int, default=3,
                              help="Recurrence a non-correction signal needs to propose (default: 3)")
    learn_review.add_argument("--json", action="store_true",
                              help="Emit proposals as JSON instead of markdown")
    learn_review.add_argument("--write", action="store_true",
                              help="Also write proposals.md into the journal directory")

    learn_apply = learn_sub.add_parser(
        "apply", help="Apply one proposal under its tier's guarantees "
        "(Tier-0 eval-gated diff; Tier-1 draft; Tier-2 task). Requires --approve")
    learn_apply.add_argument("--journal", default="", help=_journal_help)
    learn_apply.add_argument("--id", required=True, help="Proposal id (see 'learn review')")
    learn_apply.add_argument("--approve", action="store_true",
                             help="Explicit approval — without it, this is a dry run")
    learn_apply.add_argument("--min-occurrences", type=int, default=3,
                             help="Must match the 'learn review' threshold that produced the id")
    learn_apply.add_argument("--repo", default="",
                             help="Repo root for Tier-0 writes (default: enclosing git root)")

    learn_brief = learn_sub.add_parser(
        "brief", help="Package a Tier-1/Tier-2 proposal into an agent-ready "
        "improvement brief (evidence + intent, never a patch)")
    learn_brief.add_argument("--journal", default="", help=_journal_help)
    learn_brief.add_argument("--id", required=True, help="Proposal id (see 'learn review')")
    learn_brief.add_argument("--min-occurrences", type=int, default=3,
                             help="Must match the 'learn review' threshold that produced the id")
    learn_brief.add_argument("--write", action="store_true",
                             help="Also write the brief into <journal>/briefs/")

    learn_handoff = learn_sub.add_parser(
        "handoff", help="Prepare a one-command hand-off of a Tier-1/2 proposal to "
        "a coding agent (Codex/Claude Code) on an isolated worktree branch")
    learn_handoff.add_argument("--journal", default="", help=_journal_help)
    learn_handoff.add_argument("--id", required=True, help="Proposal id (see 'learn review')")
    learn_handoff.add_argument("--agent", choices=["codex", "claude"], default="codex")
    learn_handoff.add_argument("--min-occurrences", type=int, default=3,
                               help="Must match the 'learn review' threshold that produced the id")
    learn_handoff.add_argument("--repo", default="",
                               help="Repo root for the worktree (default: enclosing git root)")

    learn_watch = learn_sub.add_parser(
        "watch", help="Data-gated capability queue: which roadmap builds are "
        "unblocked by data now present in data/")
    learn_watch.add_argument("--data", default="data",
                             help="Data root to scan (default: ./data)")
    learn_watch.add_argument("--brief", action="store_true",
                             help="Emit an agent brief for each ready build")

    args = parser.parse_args(argv)
    if args.command == "learn":
        from scattering_ai.learning.journal import Journal, resolve_journal_dir

        if args.learn_command == "watch":  # no journal needed — scans data/
            from scattering_ai.learning.watch import watch_proposals, watch_status

            if args.brief:
                from scattering_ai.learning.briefs import brief_from_proposal, render_brief

                ready = watch_proposals(args.data)
                if not ready:
                    print("No watched builds are unblocked yet.")
                for proposal in ready:
                    print(render_brief(brief_from_proposal(proposal)))
            else:
                print(json.dumps(watch_status(args.data), indent=2))
            return 0

        directory = resolve_journal_dir(args.journal or None)
        if directory is None:
            sys.exit("error: no journal configured; set SCATTERING_AI_JOURNAL or "
                     "pass --journal DIR")
        journal = Journal(directory)
        if args.learn_command == "status":
            print(json.dumps(journal.summary(), indent=2))
        elif args.learn_command == "signals":
            from scattering_ai.learning.signals import signals_report

            print(json.dumps(signals_report(journal), indent=2))
        elif args.learn_command == "correct":
            from scattering_ai.learning.signals import make_correction

            correction = make_correction(args.episode, args.target, args.statement,
                                         args.value, args.domain)
            journal.record_correction(correction)
            print(json.dumps({"recorded": "correction", **correction.model_dump()},
                             indent=2))
        elif args.learn_command == "review":
            from scattering_ai.learning.proposals import build_proposals, render_proposals

            proposals = build_proposals(journal, min_occurrences=args.min_occurrences)
            if args.json:
                print(json.dumps([p.model_dump() for p in proposals], indent=2))
            else:
                print(render_proposals(proposals))
            if args.write:
                out = directory / "proposals.md"
                out.write_text(render_proposals(proposals), encoding="utf-8")
                print(f"\n(wrote {out})")
        elif args.learn_command == "apply":
            from scattering_ai.learning.apply import apply_proposal, find_repo_root
            from scattering_ai.learning.proposals import build_proposals

            proposals = build_proposals(journal, min_occurrences=args.min_occurrences)
            match = next((p for p in proposals if p.id == args.id), None)
            if match is None:
                sys.exit(f"error: no proposal with id {args.id!r}; run 'learn review' "
                         f"(available: {', '.join(p.id for p in proposals) or 'none'})")
            if not args.approve:
                repo = args.repo or (find_repo_root() if match.tier == 0 else directory)
                dest = "repo " + str(repo) if match.tier == 0 else "journal " + str(directory)
                print(json.dumps({
                    "dry_run": True,
                    "proposal": match.id, "tier": match.tier,
                    "change_class": match.change_class,
                    "would_write_into": dest,
                    "note": "no changes made; re-run with --approve to apply",
                }, indent=2))
                return 0
            record = apply_proposal(match, journal=journal,
                                    repo_root=args.repo or None, approve=True)
            print(json.dumps(record.model_dump(), indent=2))
        elif args.learn_command == "brief":
            from scattering_ai.learning.briefs import brief_from_proposal, render_brief
            from scattering_ai.learning.proposals import build_proposals

            proposals = build_proposals(journal, min_occurrences=args.min_occurrences)
            match = next((p for p in proposals if p.id == args.id), None)
            if match is None:
                sys.exit(f"error: no proposal with id {args.id!r}; run 'learn review' "
                         f"(available: {', '.join(p.id for p in proposals) or 'none'})")
            if match.tier == 0:
                sys.exit(f"error: {match.id!r} is Tier-0 — apply it directly with "
                         "'learn apply' (eval-gated); briefs are for Tier-1/Tier-2")
            brief = brief_from_proposal(match)
            markdown = render_brief(brief)
            print(markdown)
            if args.write:
                out = directory / "briefs" / f"{brief.proposal_id}.md"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(markdown, encoding="utf-8")
                print(f"(wrote {out})")
        elif args.learn_command == "handoff":
            from scattering_ai.learning.apply import find_repo_root
            from scattering_ai.learning.handoff import prepare_handoff, render_handoff
            from scattering_ai.learning.proposals import build_proposals

            proposals = build_proposals(journal, min_occurrences=args.min_occurrences)
            match = next((p for p in proposals if p.id == args.id), None)
            if match is None:
                sys.exit(f"error: no proposal with id {args.id!r}; run 'learn review' "
                         f"(available: {', '.join(p.id for p in proposals) or 'none'})")
            if match.tier == 0:
                sys.exit(f"error: {match.id!r} is Tier-0 — apply it directly with "
                         "'learn apply' (eval-gated); hand-off is for Tier-1/Tier-2")
            repo = args.repo or str(find_repo_root())
            plan = prepare_handoff(match, directory, repo, agent=args.agent)
            print(render_handoff(plan))
        return 0
    if args.command == "case-studies":
        from scattering_ai.evaluation.case_studies import run_all

        results = run_all()
        if args.json:
            print(json.dumps([r.model_dump() for r in results], indent=2))
        else:
            for r in results:
                extra = f" — {r.reason}" if r.reason else (
                    f" — {', '.join(r.unmet)}" if r.unmet else "")
                print(f"{r.status.upper():8} {r.name}{extra}")
        return 1 if any(r.status == "failed" for r in results) else 0
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
