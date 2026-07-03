"""LLM client for any OpenAI-compatible chat completions server.

Covers LM Studio, Ollama, vLLM, and OpenAI itself — the backend is chosen
entirely by ``SDKConfig.base_url``. Requires the ``openai`` package
(``pip install scattering-ai-sdk[llm]``); imported lazily so the base
install stays dependency-light.
"""

from __future__ import annotations

import json

from scattering_ai.core.config import SDKConfig
from scattering_ai.llm.base import LLMResponse, Message, ModelCapabilities, ToolCall, ToolSpec


class OpenAICompatibleClient:
    def __init__(self, config: SDKConfig, capabilities: ModelCapabilities | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The OpenAI-compatible client requires the 'openai' package. "
                "Install it with: pip install scattering-ai-sdk[llm]"
            ) from exc

        self.config = config
        self._capabilities = capabilities or ModelCapabilities(tool_use=True)
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    @staticmethod
    def _serialize_message(message: Message) -> dict:
        payload: dict = {"role": message.role, "content": message.content}
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in message.tool_calls
            ]
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        return payload

    def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        kwargs: dict = {
            "model": self.config.model,
            "messages": [self._serialize_message(m) for m in messages],
        }
        if tools and self._capabilities.tool_use:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        completion = self._client.chat.completions.create(**kwargs)
        choice = completion.choices[0]

        tool_calls = [
            ToolCall(
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments or "{}"),
                id=tc.id or f"call_{i}",
            )
            for i, tc in enumerate(choice.message.tool_calls or [])
        ]
        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            model=completion.model,
            finish_reason=choice.finish_reason or "",
        )
