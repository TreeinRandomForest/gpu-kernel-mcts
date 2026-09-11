from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol


@dataclass(frozen=True, slots=True)
class LLMCompletion:
    response_id: str
    output_text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_seconds: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


class LLMClient(Protocol):
    def complete(self, prompt: str) -> LLMCompletion: ...
