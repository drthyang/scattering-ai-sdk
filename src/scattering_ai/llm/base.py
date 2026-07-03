"""Provider-agnostic LLM client protocol.

All model access in the SDK goes through :class:`LLMClient`. Skills query
``capabilities`` and degrade gracefully (e.g. skip plot-image analysis on a
text-only local model) instead of failing.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str


class ToolSpec(BaseModel):
    """JSON-schema tool definition in the OpenAI function-calling shape."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    model: str = ""
    finish_reason: str = ""


class ModelCapabilities(BaseModel):
    tool_use: bool = False
    vision: bool = False
    structured_output: bool = False
    context_window: int = 0


@runtime_checkable
class LLMClient(Protocol):
    @property
    def capabilities(self) -> ModelCapabilities: ...

    def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> LLMResponse: ...
