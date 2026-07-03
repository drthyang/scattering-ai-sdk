"""SDK configuration for LLM backend selection.

Any OpenAI-compatible server works by pointing ``base_url`` at it; the
constructors below cover the common local-first setups. Switching backends
is a config change, never a code change.
"""

from __future__ import annotations

import os

from pydantic import BaseModel

LM_STUDIO_URL = "http://localhost:1234/v1"
OLLAMA_URL = "http://localhost:11434/v1"
OPENAI_URL = "https://api.openai.com/v1"


class SDKConfig(BaseModel):
    base_url: str = LM_STUDIO_URL
    model: str = ""
    api_key: str = "not-needed"  # local servers ignore it; cloud requires a real key
    # Local models with a large tool-schema prompt can take minutes per turn.
    timeout: float = 600.0

    @classmethod
    def lm_studio(cls, model: str = "") -> SDKConfig:
        return cls(base_url=LM_STUDIO_URL, model=model)

    @classmethod
    def ollama(cls, model: str = "") -> SDKConfig:
        return cls(base_url=OLLAMA_URL, model=model)

    @classmethod
    def openai(cls, model: str = "gpt-4o-mini", api_key: str | None = None) -> SDKConfig:
        return cls(
            base_url=OPENAI_URL,
            model=model,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
        )

    @classmethod
    def from_env(cls) -> SDKConfig:
        """Build config from SCATTERING_AI_* environment variables."""
        return cls(
            base_url=os.environ.get("SCATTERING_AI_BASE_URL", LM_STUDIO_URL),
            model=os.environ.get("SCATTERING_AI_MODEL", ""),
            api_key=os.environ.get("SCATTERING_AI_API_KEY", "not-needed"),
            timeout=float(os.environ.get("SCATTERING_AI_TIMEOUT", "120")),
        )
