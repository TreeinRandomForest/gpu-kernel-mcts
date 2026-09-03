from __future__ import annotations

from kernel_mcts.baselines import independent_best_of_n
from kernel_mcts.budget import GenerationBudget
from kernel_mcts.domain import (
    EvaluationResult,
    InvalidReason,
    KernelProgram,
    ProposalStatus,
    ShapeCase,
    Strategy,
    WorkloadContract,
)
from kernel_mcts.generation import GenerationRequest, GenerationResult
from kernel_mcts.proposals import run_proposal


WORKLOAD = WorkloadContract("toy", "toy", "fp32", (ShapeCase({"n": 1}, 1.0),), 0.0, 0.0)
STRATEGY = Strategy("repair", "repair", {"cuda_cpp": "repair"})


class RepairGenerator:
    def __init__(self) -> None:
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        source = "bad" if request.attempt == 0 else "good"
        return GenerationResult(
            f"generation:{len(self.requests)}",
            source,
            KernelProgram(source),
            "prompt",
        )


class RepairEvaluator:
    def evaluate(self, program, workload):
        if program.source == "bad":
            return EvaluationResult(
                ProposalStatus.INVALID,
                program=program,
                invalid_reason=InvalidReason.COMPILE_FAILURE,
                metadata={"compiler_stderr": "compile failed"},
            )
        return EvaluationResult(ProposalStatus.VALID, program, "good-state", 1.0)


def request() -> GenerationRequest:
    return GenerationRequest(KernelProgram("root"), STRATEGY, WORKLOAD, {}, None)


def test_repair_generation_is_charged_and_receives_failure_context() -> None:
    generator = RepairGenerator()
    budget = GenerationBudget(3)
    outcome = run_proposal(
        generator=generator,
        evaluator=RepairEvaluator(),
        budget=budget,
        request=request(),
        max_repairs=2,
        max_infrastructure_retries=0,
    )

    assert outcome.result.status == ProposalStatus.VALID
    assert budget.snapshot().used == 2
    assert [attempt.budget_index for attempt in outcome.attempts] == [1, 2]
    assert generator.requests[1].attempt == 1
    assert generator.requests[1].previous_program == KernelProgram("bad")
    assert generator.requests[1].previous_result is outcome.attempts[0].evaluation


def test_global_budget_prevents_repair_call() -> None:
    generator = RepairGenerator()
    budget = GenerationBudget(1)
    outcome = run_proposal(
        generator=generator,
        evaluator=RepairEvaluator(),
        budget=budget,
        request=request(),
        max_repairs=2,
        max_infrastructure_retries=0,
    )

    assert outcome.result.status == ProposalStatus.INVALID
    assert len(generator.requests) == 1
    assert budget.snapshot().used == 1


def test_infrastructure_retry_does_not_generate_again() -> None:
    generator = RepairGenerator()

    class FlakyEvaluator:
        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, program, workload):
            self.calls += 1
            if self.calls == 1:
                return EvaluationResult(ProposalStatus.INFRASTRUCTURE_FAILURE)
            return EvaluationResult(ProposalStatus.VALID, program, "state", 1.0)

    evaluator = FlakyEvaluator()
    budget = GenerationBudget(3)
    outcome = run_proposal(
        generator=generator,
        evaluator=evaluator,
        budget=budget,
        request=request(),
        max_repairs=2,
        max_infrastructure_retries=1,
    )

    assert outcome.result.status == ProposalStatus.VALID
    assert len(generator.requests) == 1
    assert evaluator.calls == 2
    assert budget.snapshot().used == 1


def test_baseline_uses_same_repair_accounting() -> None:
    generator = RepairGenerator()
    result = independent_best_of_n(
        KernelProgram("root"),
        (STRATEGY,),
        WORKLOAD,
        generator,
        RepairEvaluator(),
        GenerationBudget(2),
        max_repairs=2,
    )

    assert result.generations == 2
    assert result.best_program == KernelProgram("good")
