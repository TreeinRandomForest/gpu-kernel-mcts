"""Client for gateways implementing the Anthropic Messages HTTP format."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

from .llm import DEFAULT_GENERATION_INSTRUCTIONS, LLMCompletion


SendRequest = Callable[[Request, float], bytes]


@dataclass(frozen=True, slots=True)
class MessagesConfig:
    model: str
    url: str
    max_output_tokens: int = 16_384
    timeout_seconds: float = 180.0
    api_key_env: str = "ENMAAS_KEY"
    api_version: str = "2023-06-01"
    instructions: str = DEFAULT_GENERATION_INSTRUCTIONS
    history_turns: int = 0
    thinking_mode: str = "default"

    def __post_init__(self) -> None:
        parsed = urlsplit(self.url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("Messages URL must be HTTPS without embedded credentials")
        if not self.model.strip():
            raise ValueError("Messages model must not be empty")
        if self.max_output_tokens <= 0 or self.timeout_seconds <= 0:
            raise ValueError("Messages token limit and timeout must be positive")
        if self.history_turns < 0:
            raise ValueError("history_turns cannot be negative")
        if self.thinking_mode not in {"default", "disabled"}:
            raise ValueError("thinking_mode must be default or disabled")
        if not self.api_key_env or not self.api_version:
            raise ValueError("Messages key environment variable and API version are required")


class MessagesClient:
    """Send one request per generation, optionally replaying bounded text turns."""

    def __init__(self, config: MessagesConfig, *, send: SendRequest | None = None) -> None:
        self.config = config
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(f"environment variable {config.api_key_env!r} is not set")
        self._api_key = api_key
        self._send = send or _send_request
        self._history: list[tuple[str, str]] = []

    def complete(self, prompt: str) -> LLMCompletion:
        history = self._history[-self.config.history_turns:] if self.config.history_turns else []
        messages = [message for user, assistant in history for message in (
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        )]
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_output_tokens,
            "system": self.config.instructions,
            "messages": messages,
        }
        if self.config.thinking_mode == "disabled":
            payload["thinking"] = {"type": "disabled"}
        request = Request(
            self.config.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": self.config.api_version,
                "content-type": "application/json",
            },
            method="POST",
        )
        started = perf_counter()
        try:
            raw = self._send(request, self.config.timeout_seconds)
        except HTTPError as error:
            raise RuntimeError(f"Messages API returned HTTP {error.code}") from None
        except (URLError, TimeoutError, OSError):
            raise RuntimeError("Messages API transport failed") from None
        latency = perf_counter() - started
        try:
            response = json.loads(raw)
            if not isinstance(response, dict) or not isinstance(response.get("content"), list):
                raise ValueError
            output = "\n".join(
                block["text"] for block in response["content"]
                if isinstance(block, dict) and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            )
        except (TypeError, ValueError, KeyError, UnicodeDecodeError):
            raise RuntimeError("Messages API returned an invalid response") from None
        if self.config.history_turns:
            self._history.append((prompt, output))
            self._history = self._history[-self.config.history_turns:]
        usage = response.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        input_parts = (
            usage.get("input_tokens"), usage.get("cache_creation_input_tokens"),
            usage.get("cache_read_input_tokens"),
        )
        input_tokens = sum(value for value in input_parts if type(value) is int) \
            if any(type(value) is int for value in input_parts) else None
        response_id = response.get("id")
        return LLMCompletion(
            response_id=response_id if isinstance(response_id, str) and response_id else f"messages-{uuid4()}",
            output_text=output,
            model=str(response.get("model") or self.config.model),
            input_tokens=input_tokens,
            output_tokens=usage.get("output_tokens") if type(usage.get("output_tokens")) is int else None,
            latency_seconds=latency,
            metadata={
                "provider": "messages",
                "stop_reason": response.get("stop_reason"),
                "max_output_tokens": self.config.max_output_tokens,
                "thinking_mode": self.config.thinking_mode,
                "history_turns": len(history),
                "cached_input_tokens": usage.get("cache_read_input_tokens"),
                "cache_write_input_tokens": usage.get("cache_creation_input_tokens"),
                **({"conversation_input": messages} if self.config.history_turns else {}),
            },
            instructions_text=self.config.instructions,
        )


def _send_request(request: Request, timeout: float) -> bytes:
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise RuntimeError("Messages API response exceeds size limit")
    return raw
