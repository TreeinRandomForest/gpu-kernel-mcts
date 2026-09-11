from __future__ import annotations

import hashlib
import json

from kernel_mcts.domain import (
    CompilationEvidence,
    CompileStatus,
    EvaluationResult,
    InvalidReason,
    KernelProgram,
    ProposalStatus,
    ShapeCase,
    Strategy,
    WorkloadContract,
)
from kernel_mcts.generation import GenerationRequest
from kernel_mcts.llm import LLMCompletion
from kernel_mcts.llm_generation import LLMKernelGenerator, build_generation_prompt


WORKLOAD = WorkloadContract(
    "gemm",
    "gemm",
    "bfloat16",
    (ShapeCase({"M": 4096, "N": 4096, "K": 4096}, 1.0),),
    0.02,
    0.02,
    metadata={"entry_point": "bf16_gemm_root"},
)
STRATEGY = Strategy(
    "shared_memory",
    "Improve on-chip reuse",
    {"cuda_cpp": "Stage reusable values in shared memory."},
)


class FakeLLMClient:
    def __init__(self, output: str) -> None:
        self.output = output
        self.prompts = []

    def complete(self, prompt: str) -> LLMCompletion:
        self.prompts.append(prompt)
        return LLMCompletion(
            "response-1",
            self.output,
            "test-model",
            input_tokens=100,
            output_tokens=200,
            latency_seconds=1.5,
            metadata={"provider": "fake"},
        )


def request(**values) -> GenerationRequest:
    defaults = {
        "parent": KernelProgram('extern "C" __global__ void bf16_gemm_root() {}'),
        "strategy": STRATEGY,
        "workload": WORKLOAD,
        "hardware": {"gpu_model": "H100", "form_factor": "SXM"},
        "profile": {"occupancy": 0.5},
    }
    defaults.update(values)
    return GenerationRequest(**defaults)


def test_llm_generator_builds_complete_fresh_prompt_and_records_metadata() -> None:
    client = FakeLLMClient("```cuda\noptimized source\n```")
    subject = LLMKernelGenerator(client)

    result = subject.generate(request())

    assert len(client.prompts) == 1
    payload = json.loads(client.prompts[0])
    assert payload["backend"] == "cuda_cpp"
    assert payload["target_hardware"]["gpu_model"] == "H100"
    assert payload["workload"]["benchmark_id"] == "gemm"
    assert payload["strategy"]["id"] == "shared_memory"
    assert payload["strategy"]["backend_prompt"].startswith("Stage reusable")
    assert payload["parent_profile"] == {"occupancy": 0.5}
    assert "bf16_gemm_root" in payload["parent_kernel"]
    assert result.generation_id == "response-1"
    assert result.program == KernelProgram("optimized source")
    assert result.prompt_text == client.prompts[0]
    assert result.prompt_hash == hashlib.sha256(client.prompts[0].encode()).hexdigest()
    assert result.input_tokens == 100
    assert result.output_tokens == 200
    assert result.latency_seconds == 1.5
    assert result.metadata == {
        "provider": "fake",
        "generator": "llm",
        "llm_call": True,
        "model": "test-model",
        "response_id": "response-1",
    }


def test_repair_prompt_contains_failed_candidate_and_bounded_diagnostics() -> None:
    stderr = "x" * 9_000 + "useful compiler error"
    failed_program = KernelProgram("failed candidate")
    failed_result = EvaluationResult(
        ProposalStatus.INVALID,
        program=failed_program,
        invalid_reason=InvalidReason.COMPILE_FAILURE,
        compile_status=CompileStatus.FAIL,
        compilation=CompilationEvidence(None, "", stderr),
    )

    prompt = build_generation_prompt(
        request(
            attempt=1,
            previous_program=failed_program,
            previous_result=failed_result,
        )
    )
    repair = json.loads(prompt)["repair"]

    assert repair["previous_candidate"] == "failed candidate"
    assert repair["previous_evaluation"]["invalid_reason"] == "COMPILE_FAILURE"
    recorded_stderr = repair["previous_evaluation"]["compilation"]["stderr"]
    assert len(recorded_stderr) == 8_000
    assert recorded_stderr.endswith("useful compiler error")


def test_empty_model_output_produces_no_program() -> None:
    result = LLMKernelGenerator(FakeLLMClient("  ")).generate(request())

    assert result.program is None
