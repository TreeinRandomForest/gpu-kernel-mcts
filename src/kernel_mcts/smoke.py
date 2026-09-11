from __future__ import annotations

import hashlib

from .benchmarks import load_bf16_gemm_smoke_candidate
from .generation import GenerationRequest, GenerationResult


class SmokeKernelGenerator:
    """One-shot deterministic generator for remote orchestration validation."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("smoke generator supports exactly one generation")
        program = load_bf16_gemm_smoke_candidate()
        prompt = (
            f"deterministic-smoke:{request.workload.benchmark_id}:"
            f"{request.strategy.id}:{program.backend}"
        )
        return GenerationResult(
            generation_id="smoke-generation-1",
            raw_output=program.source,
            program=program,
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
            metadata={"generator": "deterministic-smoke", "llm_call": False},
        )
