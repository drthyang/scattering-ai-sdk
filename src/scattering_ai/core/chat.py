"""Interactive chat session (Milestone 2): iterative analysis with tools.

Unlike the one-shot ``analyze()`` loop, a ChatSession keeps conversation
history and the artifact workspace across turns, so the analysis is a
dialogue: "fit that peak" → "now exclude the shoulder" → "compare with the
50 K data". Prose replies are allowed (no JSON contract), but the grounding
rules are the same: every number from a tool, plots for judgment, honesty
about quality flags. A transcript with the full tool trace is written to the
workspace for provenance.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scattering_ai.llm.base import LLMClient, Message

CHAT_PROMPT_VERSION = "chat/v2"

CHAT_SYSTEM_PROMPT = """\
You are a careful scattering-science assistant working interactively with a
researcher on their data files. You have deterministic analysis tools and
composite skills (skill_* tools bundle validated multi-step workflows —
prefer them when they match the request).

Rules:
- Every number you state must come from a tool result in this conversation.
  Never estimate, extrapolate, or invent values, file names, or parameters.
- find_peaks gives estimates; fitted values and uncertainties require a fit
  tool. Never say "fitted" or quote "±" values without a fit result.
- When you produce a plot, tell the user its file path so they can look.
- Report fit quality flags honestly (at_bounds, high_uncertainty, poor_fit)
  and say when the evidence is insufficient.
- Artifacts (slices, cuts, transforms) persist in the workspace across the
  conversation; reuse their paths instead of recomputing.
- When a tool accepts multiple paths, pass a glob pattern or directory
  instead of retyping long filenames (typos in copied names break tools).
- Be concise: a few sentences of prose, not a report, unless asked.

The final scientific judgment always belongs to the researcher.
"""

MAX_TOOL_ROUNDS_PER_TURN = 10


class ChatSession:
    def __init__(
        self,
        llm: LLMClient,
        model_id: str = "",
        workspace: str | Path | None = None,
        files: list[str] | None = None,
        on_tool_call=None,
    ):
        from scattering_ai.core.agent import _default_workspace
        from scattering_ai.tools.registry import default_toolkit

        self.llm = llm
        self.model_id = model_id
        self.workspace = Path(workspace) if workspace else _default_workspace()
        self.registry = default_toolkit(self.workspace)
        self.on_tool_call = on_tool_call  # callback(name, args, result) for UIs
        self.tool_trace: list[dict] = []

        context = ""
        if files:
            context = self._files_context(files)
        self.messages = [
            Message(role="system", content=CHAT_SYSTEM_PROMPT + context)
        ]

    @staticmethod
    def _files_context(files: list[str]) -> str:
        """Describe the files compactly: for a homogeneous set, give the glob
        (retyping twenty 130-char filenames is how small models break)."""
        dirs = {str(Path(f).parent) for f in files}
        suffixes = {Path(f).suffix for f in files}
        if len(files) > 4 and len(dirs) == 1 and len(suffixes) == 1:
            directory, suffix = dirs.pop(), suffixes.pop()
            glob_pattern = f"{directory}/*{suffix}"
            sample = "\n".join(f"- {Path(f).name}" for f in files[:3])
            return (
                f"\n\nThe researcher wants to work with {len(files)} files in "
                f"{directory}/. Pass the glob '{glob_pattern}' to series tools "
                f"instead of listing names. First files:\n{sample}\n- ..."
            )
        return "\n\nFiles the researcher wants to work with:\n" + "\n".join(
            f"- {f}" for f in files
        )

    def turn(self, user_text: str) -> str:
        """One conversational turn: user text in, grounded assistant text out."""
        self.messages.append(Message(role="user", content=user_text))
        response = self.llm.complete(self.messages, tools=self.registry.specs)
        rounds = 0
        while response.tool_calls and rounds < MAX_TOOL_ROUNDS_PER_TURN:
            rounds += 1
            self.messages.append(
                Message(role="assistant", content=response.content,
                        tool_calls=response.tool_calls)
            )
            for call in response.tool_calls:
                result = self.registry.execute(call.name, call.arguments)
                self.tool_trace.append(
                    {"tool": call.name, "arguments": call.arguments,
                     "error": result.get("error", "")}
                )
                if self.on_tool_call:
                    self.on_tool_call(call.name, call.arguments, result)
                self.messages.append(
                    Message(role="tool", content=json.dumps(result, default=str),
                            tool_call_id=call.id)
                )
            response = self.llm.complete(self.messages, tools=self.registry.specs)

        reply = response.content.strip()
        self.messages.append(Message(role="assistant", content=reply))
        self._save_transcript()
        return reply

    def _save_transcript(self) -> None:
        lines = [
            "# scattering-ai chat transcript",
            f"- model: {self.model_id}",
            f"- prompt: {CHAT_PROMPT_VERSION}",
            f"- updated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            "",
        ]
        for message in self.messages[1:]:
            if message.role == "user":
                lines += [f"## user\n\n{message.content}", ""]
            elif message.role == "assistant" and message.content:
                lines += [f"## assistant\n\n{message.content}", ""]
            elif message.role == "assistant" and message.tool_calls:
                calls = ", ".join(
                    f"{tc.name}({json.dumps(tc.arguments, default=str)})"
                    for tc in message.tool_calls
                )
                lines += [f"*tool calls: {calls}*", ""]
        (self.workspace / "chat_transcript.md").write_text("\n".join(lines))
