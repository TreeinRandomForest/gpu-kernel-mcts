from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit

from .llm import DEFAULT_GENERATION_INSTRUCTIONS, LLMCompletion


OPENAI_GENERATION_INSTRUCTIONS = DEFAULT_GENERATION_INSTRUCTIONS


@dataclass(frozen=True, slots=True)
class OpenAIResponsesConfig:
    model: str
    reasoning_effort: str = "medium"
    max_output_tokens: int = 16_384
    timeout_seconds: float = 180.0
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    store: bool = False
    instructions: str = OPENAI_GENERATION_INSTRUCTIONS
    history_turns: int = 0

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("OpenAI model must not be empty")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.history_turns < 0:
            raise ValueError("history_turns cannot be negative")
        if self.base_url is not None:
            parsed = urlsplit(self.base_url)
            if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
                raise ValueError("Responses base URL must be HTTPS without embedded credentials")


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
            options: dict[str, Any] = {
                "api_key": api_key,
                "timeout": config.timeout_seconds,
            }
            if config.base_url is not None:
                options["base_url"] = config.base_url
            client = OpenAI(**options)
        self._client = client
        self._history: list[tuple[str, str]] = []

    def complete(self, prompt: str) -> LLMCompletion:
        started = perf_counter()
        history = self._history[-self.config.history_turns:] if self.config.history_turns else []
        messages = [message for user, assistant in history for message in (
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        )]
        messages.append({"role": "user", "content": prompt})
        response = self._client.responses.create(
            model=self.config.model,
            instructions=self.config.instructions,
            input=messages if self.config.history_turns else prompt,
            max_output_tokens=self.config.max_output_tokens,
            reasoning={"effort": self.config.reasoning_effort},
            store=self.config.store,
        )
        latency = perf_counter() - started
        if self.config.history_turns:
            self._history.append((prompt, str(response.output_text or "")))
            self._history = self._history[-self.config.history_turns:]
        usage = getattr(response, "usage", None)
        input_details = getattr(usage, "input_tokens_details", None)
        return LLMCompletion(
            response_id=str(response.id),
            output_text=str(response.output_text or ""),
            model=str(response.model),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            latency_seconds=latency,
            metadata={
                "provider": "openai_compatible" if self.config.base_url else "openai",
                "status": str(getattr(response, "status", "unknown")),
                "reasoning_effort": self.config.reasoning_effort,
                "max_output_tokens": self.config.max_output_tokens,
                "store": self.config.store,
                "history_turns": len(history),
                "cached_input_tokens": getattr(input_details, "cached_tokens", None),
                "cache_write_input_tokens": getattr(input_details, "cache_write_tokens", None),
                **({"conversation_input": messages} if self.config.history_turns else {}),
            },
            instructions_text=self.config.instructions,
        )
