"""Pure-Triton sequential choices keep catalog action fidelity."""

import json
from pathlib import Path

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD
from kernel_mcts.budget import GenerationBudget
from kernel_mcts.config import load_data, parse_strategies
from kernel_mcts.domain import (
    BenchmarkResult, EvaluationResult, InvalidReason, KernelProgram,
    ProposalStatus, Strategy,
)
from kernel_mcts.generation import GenerationRequest, GenerationResult
from kernel_mcts.llm import LLMCompletion
from kernel_mcts.proposals import run_proposal
from kernel_mcts.sequential_generation import SequentialKernelGenerator, build_sequential_prompt


ROOT = Path("src/kernel_mcts/benchmarks/kernels/bf16_gemm_triton.py")
NARROW = Path("docs/experiments/triton-h100-winner-crossover/ungrouped_bn128_w8.py")
CATALOG = Path("configs/triton_h100_pure_local_v20.json")
FREEFORM = Strategy("freeform", "", {"triton_h100": ""})


class Evaluator:
    calls = 0

    def evaluate(self, program, workload):
        self.calls += 1
        return EvaluationResult(
            ProposalStatus.VALID, program, "valid-state", 1.0,
            BenchmarkResult((250.0,), 250.0),
        )


def _request(source: str) -> GenerationRequest:
    return GenerationRequest(
        KernelProgram(source, "triton_h100"), FREEFORM,
        BF16_GEMM_WORKLOAD, {"gpu_model": "H100"}, None,
    )


def test_triton_sequential_prompt_uses_launch_abi_and_calibrated_reference() -> None:
    request = GenerationRequest(
        KernelProgram(ROOT.read_text(), "triton_h100"), FREEFORM,
        BF16_GEMM_WORKLOAD, {"gpu_model": "H100"}, None,
        reference_baselines={"cublas": {"median_us": 178.0}},
    )
    payload = json.loads(build_sequential_prompt(request))
    assert payload["workload"]["metadata"]["entry_point"] == "launch"
    assert payload["workload"]["metadata"]["abi"]["parameters"] == [
        "A", "B", "C", "M", "N", "K",
    ]
    assert payload["performance_references"] == {"cublas": {"median_us": 178.0}}


def test_triton_sequential_menu_filters_parent_and_checks_selected_guard() -> None:
    catalog = parse_strategies(load_data(CATALOG))

    class Client:
        prompt = None

        def complete(self, prompt):
            self.prompt = json.loads(prompt)
            return LLMCompletion("response", json.dumps({
                "selected_strategy_id": "triton_pointer_advance_tile_128",
                "hypothesis": "advance pointers", "source": NARROW.read_text(),
            }), "fake")

    client = Client()
    generator = SequentialKernelGenerator(
        client, strategy_catalog=catalog,
        strategy_menu=tuple(f"{item.id}: {item.description}" for item in catalog),
        strategy_ids=tuple(item.id for item in catalog),
    )
    evaluator = Evaluator()
    budget = GenerationBudget(1)
    outcome = run_proposal(
        generator=generator, evaluator=evaluator, budget=budget,
        request=_request(ROOT.read_text()), max_repairs=0,
        max_infrastructure_retries=0,
        selected_strategies={item.id: item for item in catalog},
        allowed_candidate_stages=frozenset({"pointer_triton", "descriptor_triton"}),
    )
    assert outcome.result.status == ProposalStatus.VALID
    assert evaluator.calls == 1 and budget.snapshot().used == 1
    assert len(client.prompt["strategy_menu"]) == 1
    assert "triton_pointer_advance_tile_128" in client.prompt["strategy_menu"][0]
    assert "verified Triton kernel" in client.prompt["task"]


def test_triton_sequential_rejects_wrong_action_before_gpu() -> None:
    catalog = parse_strategies(load_data(CATALOG))

    class Generator:
        def generate(self, request):
            source = NARROW.read_text()
            return GenerationResult("response", source,
                                    KernelProgram(source, "triton_h100"), "hash",
                                    metadata={"selected_strategy_id": "triton_widen_bn_256_eight_warps",
                                              "strategy_choice_status": "valid"})

    evaluator = Evaluator()
    outcome = run_proposal(
        generator=Generator(), evaluator=evaluator, budget=GenerationBudget(1),
        request=_request(ROOT.read_text()), max_repairs=0,
        max_infrastructure_retries=0,
        selected_strategies={item.id: item for item in catalog},
        allowed_candidate_stages=frozenset({"pointer_triton", "descriptor_triton"}),
    )
    assert outcome.result.invalid_reason == InvalidReason.STRATEGY_MISMATCH
    assert evaluator.calls == 0


def test_triton_sequential_applies_selected_action_source_guard() -> None:
    catalog = parse_strategies(load_data(CATALOG))

    class Generator:
        def generate(self, request):
            source = NARROW.read_text().replace("num_warps=8", "num_warps=4")
            return GenerationResult("response", source,
                                    KernelProgram(source, "triton_h100"), "hash",
                                    metadata={"selected_strategy_id": "triton_pointer_advance_tile_128",
                                              "strategy_choice_status": "valid"})

    evaluator = Evaluator()
    outcome = run_proposal(
        generator=Generator(), evaluator=evaluator, budget=GenerationBudget(1),
        request=_request(ROOT.read_text()), max_repairs=0,
        max_infrastructure_retries=0,
        selected_strategies={item.id: item for item in catalog},
        allowed_candidate_stages=frozenset({"pointer_triton", "descriptor_triton"}),
    )
    assert outcome.result.invalid_reason == InvalidReason.STRATEGY_MISMATCH
    assert outcome.result.metadata["strategy_edit_guard"] == "triton_pointer_advance_tile_128"
    assert evaluator.calls == 0


def test_triton_sequential_freeform_rejects_gluon_stage_before_gpu() -> None:
    class Generator:
        def generate(self, request):
            source = "from triton.experimental import gluon\n@gluon.jit\ndef gemm(): pass\ndef launch(A,B,C,M,N,K):\n    return gemm[(1,)]()\n"
            return GenerationResult("response", source,
                                    KernelProgram(source, "triton_h100"), "hash")

    evaluator = Evaluator()
    outcome = run_proposal(
        generator=Generator(), evaluator=evaluator, budget=GenerationBudget(1),
        request=_request(ROOT.read_text()), max_repairs=0,
        max_infrastructure_retries=0,
        allowed_candidate_stages=frozenset({"pointer_triton", "descriptor_triton"}),
    )
    assert outcome.result.invalid_reason == InvalidReason.STRATEGY_MISMATCH
    assert evaluator.calls == 0
