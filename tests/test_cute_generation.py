from __future__ import annotations

import json

from kernel_mcts.cute_generation import CuteTypedLLMGenerator
from kernel_mcts.cute_mutations import CUTE_MUTATION_STRATEGIES
from kernel_mcts.cute_program import (
    PinnedCuteGemmRenderer,
    REFERENCE_CUTE_GEMM,
)
from kernel_mcts.domain import KernelProgram, ShapeCase, WorkloadContract
from kernel_mcts.generation import GenerationRequest, GenerationResult


WORKLOAD = WorkloadContract(
    "bf16_gemm_4096_h100",
    "gemm",
    "bfloat16",
    (ShapeCase({"M": 4096, "N": 4096, "K": 4096}, 1.0),),
    0.02,
    0.02,
)


class FixedGenerator:
    def __init__(self, output: str) -> None:
        self.output = output

    def generate(self, _request):
        return GenerationResult(
            "response-1",
            self.output,
            KernelProgram(self.output, "cute_dsl"),
            "prompt",
            metadata={"generator": "llm", "llm_call": True},
        )


def request() -> GenerationRequest:
    return GenerationRequest(
        PinnedCuteGemmRenderer().render(REFERENCE_CUTE_GEMM),
        CUTE_MUTATION_STRATEGIES[1],
        WORKLOAD,
        {"gpu_model": "H100"},
        None,
    )


def test_typed_llm_json_is_validated_and_rendered_canonically() -> None:
    representation = {
        **REFERENCE_CUTE_GEMM.as_dict(),
        "cluster_m": 2,
        "cluster_n": 1,
    }

    result = CuteTypedLLMGenerator(
        FixedGenerator(json.dumps(representation))
    ).generate(request())

    assert result.program == PinnedCuteGemmRenderer().render(
        type(REFERENCE_CUTE_GEMM)(**representation)
    )
    assert result.metadata["proposal_mechanism"] == "llm_typed_representation"
    assert result.metadata["typed_output_format"] == "json"
    assert result.metadata["static_validation"]["valid"] is True
    assert result.metadata["transformation"]["changed_fields"] == {
        "cluster_m": {"before": 1, "after": 2}
    }


def test_full_source_with_incorrect_embedded_hash_is_canonicalized() -> None:
    candidate = type(REFERENCE_CUTE_GEMM)(
        **{
            **REFERENCE_CUTE_GEMM.as_dict(),
            "cluster_m": 2,
            "cluster_n": 1,
        }
    )
    canonical = PinnedCuteGemmRenderer().render(candidate)
    noncanonical = canonical.source.replace(candidate.configuration_hash, "wrong-hash")

    result = CuteTypedLLMGenerator(FixedGenerator(noncanonical)).generate(request())

    assert result.program == canonical
    assert result.raw_output == noncanonical
    assert result.metadata["typed_output_format"] == "rendered_source"


def test_statically_unsupported_typed_output_produces_no_program() -> None:
    representation = {**REFERENCE_CUTE_GEMM.as_dict(), "pipeline_stages": 5}

    result = CuteTypedLLMGenerator(
        FixedGenerator(json.dumps(representation))
    ).generate(request())

    assert result.program is None
    assert result.metadata["static_validation"]["valid"] is False
    assert result.metadata["configuration_hash"]
