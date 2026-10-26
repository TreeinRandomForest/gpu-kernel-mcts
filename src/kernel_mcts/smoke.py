from __future__ import annotations

import hashlib

from .benchmarks import load_bf16_gemm_smoke_candidate, load_triton_bf16_gemm_root
from .domain import KernelProgram
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


class TritonSmokeKernelGenerator:
    """One distinct, deterministic Triton proposal for remote integration checks."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("smoke generator supports exactly one generation")
        source = load_triton_bf16_gemm_root().source
        marker = "bm, bn, bk = 32, 64, 32"
        if source.count(marker) != 1:
            raise RuntimeError("Triton smoke root tile marker changed")
        program = KernelProgram(source.replace(marker, "bm, bn, bk = 64, 64, 32"),
                                backend="triton_h100")
        prompt = f"deterministic-smoke:{request.workload.benchmark_id}:{request.strategy.id}:triton_h100"
        return GenerationResult(
            generation_id="triton-smoke-generation-1", raw_output=program.source,
            program=program, prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
            metadata={"generator": "deterministic-smoke", "llm_call": False},
        )
