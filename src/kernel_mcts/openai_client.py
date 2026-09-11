from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .llm import LLMCompletion


@dataclass(frozen=True, slots=True)
class OpenAIResponsesConfig:
    model: str
    reasoning_effort: str = "medium"
    max_output_tokens: int = 16_384
    timeout_seconds: float = 180.0
    api_key_env: str = "OPENAI_API_KEY"
    store: bool = False

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("OpenAI model must not be empty")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


class OpenAIResponsesClient:
    """Make independent OpenAI Responses API calls for candidate generation."""

    def __init__(
        self,
        config: OpenAIResponsesConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        if client is None:
            api_key = os.environ.get(config.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"environment variable {config.api_key_env!r} is not set"
                )
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError(
                    "OpenAI generation requires the 'llm' optional dependency"
                ) from error
            client = OpenAI(api_key=api_key, timeout=config.timeout_seconds)
        self._client = client

    def complete(self, prompt: str) -> LLMCompletion:
        started = perf_counter()
        response = self._client.responses.create(
            model=self.config.model,
            instructions=(
                "Return only the complete replacement kernel source. "
                "Do not use Markdown fences or explanatory prose."
            ),
            input=prompt,
            max_output_tokens=self.config.max_output_tokens,
            reasoning={"effort": self.config.reasoning_effort},
            store=self.config.store,
        )
        latency = perf_counter() - started
        usage = getattr(response, "usage", None)
        return LLMCompletion(
            response_id=str(response.id),
            output_text=str(response.output_text or ""),
            model=str(response.model),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            latency_seconds=latency,
            metadata={
                "provider": "openai",
                "status": str(getattr(response, "status", "unknown")),
                "reasoning_effort": self.config.reasoning_effort,
                "max_output_tokens": self.config.max_output_tokens,
                "store": self.config.store,
            },
        )
