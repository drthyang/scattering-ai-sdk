"""Agent skills: procedural knowledge as deterministic composite workflows.

A **tool** is one operation; a **skill** encodes *which tools to chain, in
what order, with what checks* — the analysis judgment a scientist would
apply. Skills run deterministically over the tool registry (the LLM invokes
one skill instead of improvising a chain), return every intermediate result
as evidence, and record the chain in ``steps`` so reports stay auditable.

Skills are exposed to agents exactly like tools, with a ``skill_`` name
prefix. Domain packs and external plugins can ship their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from scattering_ai.tools.registry import ToolRegistry


def _figure_paths(obj: Any) -> list[str]:
    """Every ``.png`` path anywhere in a (possibly nested) tool result."""
    if isinstance(obj, str):
        return [obj] if obj.endswith(".png") else []
    if isinstance(obj, dict):
        return [p for v in obj.values() for p in _figure_paths(v)]
    if isinstance(obj, list):
        return [p for v in obj for p in _figure_paths(v)]
    return []


@dataclass
class SkillRun:
    """Collects the audited chain of tool calls a skill makes."""

    registry: ToolRegistry
    steps: list[dict[str, Any]] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)

    def call(self, tool: str, **arguments) -> dict[str, Any]:
        result = self.registry.execute(tool, arguments)
        self.steps.append({"tool": tool, "arguments": arguments,
                           "error": result.get("error", "")})
        for path in _figure_paths(result):  # aggregate every plot the skill made
            if path not in self.figures:
                self.figures.append(path)
        return result

    def finish(self, **payload) -> dict[str, Any]:
        errors = [s for s in self.steps if s["error"]]
        return {
            **payload,
            "figures": self.figures,
            "steps": self.steps,
            "n_steps": len(self.steps),
            "step_errors": len(errors),
        }
