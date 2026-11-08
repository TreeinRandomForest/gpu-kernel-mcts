from __future__ import annotations

import json

import pytest

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD, load_triton_bf16_gemm_root
from kernel_mcts.budget import GenerationBudget
from kernel_mcts.domain import (
    BenchmarkResult, EvaluationResult, InvalidReason, KernelProgram, ProposalStatus,
)
from kernel_mcts.generation import GenerationRequest, GenerationResult
from kernel_mcts.llm_generation import build_generation_prompt
from kernel_mcts.priors import UniformStrategyPrior
from kernel_mcts.search import MCTS, MCTSConfig
from kernel_mcts.search_cli import _search_components, build_parser
from kernel_mcts.strategy_applicability import applicable_strategies
from kernel_mcts.triton_freeform import (
    TRITON_FREEFORM_ACTION, TRITON_FREEFORM_PROMPT_VERSION,
    pure_triton_freeform_strategy,
)


PURE_CHILD = """import triton

@triton.jit
def gemm(A, B, C, M, N, K):
    return

def launch(A, B, C, M, N, K):
    return gemm[(1,)](A, B, C, M, N, K)
"""

GLUON_CHILD = PURE_CHILD.replace("@triton.jit", "@gluon.jit")


def test_freeform_action_is_only_eligible_for_h100_gemm_pure_triton_parents() -> None:
    action = pure_triton_freeform_strategy()
    root = load_triton_bf16_gemm_root()
    hardware = {"gpu_model": "H100"}
    assert applicable_strategies((action,), root, BF16_GEMM_WORKLOAD, hardware) == (action,)
    assert applicable_strategies(
        (action,), KernelProgram(GLUON_CHILD, "triton_h100"), BF16_GEMM_WORKLOAD, hardware,
    ) == ()
    assert applicable_strategies((action,), root, BF16_GEMM_WORKLOAD, {"gpu_model": "MI300X"}) == ()


def test_freeform_prompt_is_versioned_and_requires_pure_triton() -> None:
    prompt = json.loads(build_generation_prompt(GenerationRequest(
        parent=load_triton_bf16_gemm_root(),
        strategy=pure_triton_freeform_strategy(),
        workload=BF16_GEMM_WORKLOAD,
        hardware={"gpu_model": "H100"},
        profile=None,
    )))
    assert prompt["strategy"]["id"] == TRITON_FREEFORM_ACTION
    assert prompt["strategy"]["prompt_version"] == TRITON_FREEFORM_PROMPT_VERSION
    assert "freely" in prompt["task"]
    assert "do not use Gluon" in " ".join(prompt["constraints"])
    assert "@triton.jit" in " ".join(prompt["constraints"])


@pytest.mark.parametrize("invalid_program", [
    KernelProgram(GLUON_CHILD, "triton_h100"),
    KernelProgram(PURE_CHILD, "cuda_cpp"),
])
def test_freeform_mcts_charges_rejected_repair_and_admits_valid_local_slowdown(
    invalid_program: KernelProgram,
) -> None:
    class Generator:
        def __init__(self) -> None:
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            program = invalid_program if request.attempt == 0 else KernelProgram(PURE_CHILD, "triton_h100")
            return GenerationResult(
                f"generation:{len(self.requests)}", program.source, program, "prompt",
            )

    class Evaluator:
        def __init__(self) -> None:
            self.sources = []

        def evaluate(self, program, workload):
            self.sources.append(program.source)
            return EvaluationResult(
                ProposalStatus.VALID, program, "child", -0.1,
                BenchmarkResult((110.0,), 110.0, {"shape": 110.0}),
            )

    class Events:
        def __init__(self) -> None:
            self.records = []

        def emit(self, event, payload):
            self.records.append((event, payload))

    generator, evaluator, events = Generator(), Evaluator(), Events()
    root_program = load_triton_bf16_gemm_root()
    root = EvaluationResult(
        ProposalStatus.VALID, root_program, "root", 0.0,
        BenchmarkResult((100.0,), 100.0, {"shape": 100.0}),
    )
    result = MCTS(
        strategies=(pure_triton_freeform_strategy(),),
        workload=BF16_GEMM_WORKLOAD,
        generator=generator,
        evaluator=evaluator,
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(2),
        hardware={"gpu_model": "H100"},
        config=MCTSConfig(max_repairs=1, max_depth=2),
        events=events,
    ).run(root)

    assert result.generations == 2
    assert result.best is result.root
    assert len(result.nodes) == 2
    assert evaluator.sources == [PURE_CHILD]
    assert [request.attempt for request in generator.requests] == [0, 1]
    generations = [payload for event, payload in events.records if event == "generation"]
    assert [event["strategy_id"] for event in generations] == [TRITON_FREEFORM_ACTION] * 2
    assert generations[0]["evaluation"]["invalid_reason"] == InvalidReason.STRATEGY_MISMATCH
    assert generations[1]["evaluation"]["status"] == ProposalStatus.VALID


def test_cli_selects_freeform_only_without_strategy_file(monkeypatch) -> None:
    parser = build_parser()
    arguments = parser.parse_args([
        "--image", "worker:triton", "--trace", "trace.sqlite",
        "--backend", "triton_h100", "--generator", "openai", "--model", "test",
        "--triton-freeform-mcts",
    ])
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    strategies, _, _, _ = _search_components(parser, arguments)
    assert [strategy.id for strategy in strategies] == [TRITON_FREEFORM_ACTION]

    arguments.strategies = "configs/triton_h100_strategies.json"
    with pytest.raises(SystemExit):
        _search_components(parser, arguments)
    arguments.strategies = None
    arguments.mutation_budget = 1
    with pytest.raises(SystemExit):
        _search_components(parser, arguments)
