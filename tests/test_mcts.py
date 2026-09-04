from __future__ import annotations

from kernel_mcts.budget import GenerationBudget
import pytest

from kernel_mcts.domain import (
    BenchmarkResult,
    EvaluationResult,
    InvalidReason,
    KernelProgram,
    ProposalStatus,
    ShapeCase,
    Strategy,
    WorkloadContract,
)
from kernel_mcts.generation import GenerationResult
from kernel_mcts.priors import UniformStrategyPrior
from kernel_mcts.search import MCTS, MCTSConfig
from kernel_mcts.search.model import RealizationEdge, SearchNode, StrategyEdge


WORKLOAD = WorkloadContract("toy", "toy", "fp32", (ShapeCase({"n": 1}, 1.0),), 0.0, 0.0)
STRATEGIES = (
    Strategy("a", "increment one", {"cuda_cpp": "a"}),
    Strategy("b", "increment two", {"cuda_cpp": "b"}),
)


def valid_evaluation(source: str, state_key: str, reward: float) -> EvaluationResult:
    return EvaluationResult(
        ProposalStatus.VALID,
        KernelProgram(source),
        state_key,
        reward,
        BenchmarkResult((1.0,), 1.0, {"n=1": 1.0}),
        metadata={"artifact_id": f"artifact:{state_key}"},
    )


class ToyGenerator:
    def __init__(self) -> None:
        self.counter = 0

    def generate(self, request):
        self.counter += 1
        parent_value = int(request.parent.source)
        value = parent_value + (1 if request.strategy.id == "a" else 2)
        return GenerationResult(f"generation:{self.counter}", str(value), KernelProgram(str(value)), "prompt")


class ToyEvaluator:
    def evaluate(self, program, workload):
        value = int(program.source)
        return valid_evaluation(program.source, f"state:{value}", float(value))


def test_mcts_obeys_budget_and_finds_improvement() -> None:
    result = MCTS(
        strategies=STRATEGIES,
        workload=WORKLOAD,
        generator=ToyGenerator(),
        evaluator=ToyEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(12),
        config=MCTSConfig(max_depth=5, k_max=2),
    ).run(valid_evaluation("0", "state:0", 0.0))
    assert result.generations == 12
    assert result.best.reward > 0
    assert len(result.nodes) > 1
    assert all(edge.visits >= 0 for node in result.nodes for edge in node.actions.values())


class DuplicateGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        return GenerationResult(
            f"generation:{self.calls}",
            "same",
            KernelProgram("same"),
            "prompt",
        )


class DuplicateEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, program, workload):
        self.calls += 1
        return EvaluationResult(
            ProposalStatus.VALID,
            program,
            "same-state",
            1.0,
            BenchmarkResult((float(self.calls),), float(self.calls), {"n=1": float(self.calls)}),
            metadata={"artifact_id": f"artifact:call:{self.calls}"},
        )


def test_transpositions_reuse_state() -> None:
    generator = DuplicateGenerator()
    evaluator = DuplicateEvaluator()
    result = MCTS(
        strategies=STRATEGIES,
        workload=WORKLOAD,
        generator=generator,
        evaluator=evaluator,
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(6),
        config=MCTSConfig(max_depth=3, k_max=2),
    ).run(valid_evaluation("root", "root", 0.0))
    assert len(result.nodes) == 2
    cached = next(node for node in result.nodes if node.state_key == "same-state")
    assert cached.evaluation.benchmark == BenchmarkResult((1.0,), 1.0, {"n=1": 1.0})
    assert cached.evaluation.metadata["artifact_id"] == "artifact:call:1"
    assert generator.calls == result.generations
    assert evaluator.calls == result.generations


def test_search_node_rejects_invalid_evaluation() -> None:
    invalid = EvaluationResult(
        ProposalStatus.INVALID,
        invalid_reason=InvalidReason.COMPILE_FAILURE,
    )
    with pytest.raises(ValueError, match="valid evaluations"):
        SearchNode("invalid", invalid)


def test_root_and_candidate_cache_complete_evaluations() -> None:
    root_evaluation = valid_evaluation("0", "state:0", 0.0)
    result = MCTS(
        strategies=STRATEGIES,
        workload=WORKLOAD,
        generator=ToyGenerator(),
        evaluator=ToyEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(1),
    ).run(root_evaluation)

    assert result.root.evaluation is root_evaluation
    candidate = next(node for node in result.nodes if node is not result.root)
    assert candidate.evaluation.benchmark is not None
    assert candidate.program is candidate.evaluation.program


def test_profiler_receives_cached_evaluation() -> None:
    class RecordingProfiler:
        def __init__(self) -> None:
            self.evaluations = []

        def lightweight_profile(self, evaluation, workload):
            self.evaluations.append(evaluation)
            return {"profiled": True}

    root_evaluation = valid_evaluation("0", "state:0", 0.0)
    profiler = RecordingProfiler()
    MCTS(
        strategies=STRATEGIES,
        workload=WORKLOAD,
        generator=ToyGenerator(),
        evaluator=ToyEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(1),
        profiler=profiler,
    ).run(root_evaluation)

    assert profiler.evaluations == [root_evaluation]


def test_mcts_charges_and_logs_repair_generation() -> None:
    class RepairGenerator:
        def __init__(self) -> None:
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            source = "bad" if request.attempt == 0 else "1"
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
                )
            return valid_evaluation("1", "state:1", 1.0)

    class RecordingEvents:
        def __init__(self) -> None:
            self.events = []

        def emit(self, event_type, payload):
            self.events.append((event_type, payload))

    generator = RepairGenerator()
    events = RecordingEvents()
    result = MCTS(
        strategies=(STRATEGIES[0],),
        workload=WORKLOAD,
        generator=generator,
        evaluator=RepairEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(2),
        config=MCTSConfig(max_repairs=1),
        events=events,
    ).run(valid_evaluation("0", "state:0", 0.0))

    action = result.root.actions["a"]
    generation_events = [payload for event, payload in events.events if event == "generation"]
    assert result.generations == 2
    assert len(result.nodes) == 2
    assert action.proposal_count == 1
    assert action.generation_attempt_count == 2
    assert action.repair_generation_count == 1
    assert action.visits == 1
    assert action.value_sum == 1.0
    assert [payload["b_gen"] for payload in generation_events] == [1, 2]
    assert [payload["repair_attempt"] for payload in generation_events] == [0, 1]


def test_zero_visit_puct_uses_seeded_tie_breaking() -> None:
    node = SearchNode("root", valid_evaluation("0", "state:0", 0.0))
    node.actions = {
        "high-prior": StrategyEdge("high-prior", 0.99),
        "low-prior": StrategyEdge("low-prior", 0.01),
    }
    selections = []
    for _ in range(2):
        mcts = MCTS(
            strategies=STRATEGIES,
            workload=WORKLOAD,
            generator=ToyGenerator(),
            evaluator=ToyEvaluator(),
            prior_provider=UniformStrategyPrior(),
            budget=GenerationBudget(1),
            seed=0,
        )
        selections.append(mcts._select_action(node).strategy_id)

    assert selections == ["low-prior", "low-prior"]


def test_puct_uses_priors_after_an_action_visit() -> None:
    node = SearchNode("root", valid_evaluation("0", "state:0", 0.0))
    node.actions = {
        "high-prior": StrategyEdge("high-prior", 0.9, visits=1),
        "low-prior": StrategyEdge("low-prior", 0.1),
    }
    mcts = MCTS(
        strategies=STRATEGIES,
        workload=WORKLOAD,
        generator=ToyGenerator(),
        evaluator=ToyEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(1),
        seed=0,
    )

    assert mcts._select_action(node).strategy_id == "high-prior"


class FixedGenerator:
    def __init__(self, sources):
        self.sources = iter(sources)
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        source = next(self.sources)
        return GenerationResult(
            f"generation:{self.calls}",
            source,
            KernelProgram(source),
            "prompt",
        )


def test_invalid_proposal_does_not_update_search_statistics() -> None:
    class InvalidEvaluator:
        def evaluate(self, program, workload):
            return EvaluationResult(
                ProposalStatus.INVALID,
                program=program,
                invalid_reason=InvalidReason.CORRECTNESS_FAILURE,
            )

    result = MCTS(
        strategies=(STRATEGIES[0],),
        workload=WORKLOAD,
        generator=FixedGenerator(("invalid",)),
        evaluator=InvalidEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(1),
        config=MCTSConfig(max_repairs=0),
    ).run(valid_evaluation("0", "state:0", 0.0))

    action = result.root.actions["a"]
    assert action.visits == 0
    assert action.value_sum == 0.0
    assert action.q_max == float("-inf")
    assert action.proposal_count == 1
    assert action.invalid_proposal_count == 1
    assert action.generation_attempt_count == 1


def test_infrastructure_failure_does_not_update_search_statistics() -> None:
    class InfrastructureEvaluator:
        def evaluate(self, program, workload):
            return EvaluationResult(ProposalStatus.INFRASTRUCTURE_FAILURE)

    result = MCTS(
        strategies=(STRATEGIES[0],),
        workload=WORKLOAD,
        generator=FixedGenerator(("candidate",)),
        evaluator=InfrastructureEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(1),
        config=MCTSConfig(max_repairs=0, max_infrastructure_retries=1),
    ).run(valid_evaluation("0", "state:0", 0.0))

    action = result.root.actions["a"]
    assert action.visits == 0
    assert action.value_sum == 0.0
    assert action.q_max == float("-inf")
    assert action.proposal_count == 1
    assert action.invalid_proposal_count == 0


def test_failed_descendant_does_not_partially_back_up_valid_ancestor_path() -> None:
    class ValidThenInvalidEvaluator:
        def evaluate(self, program, workload):
            if program.source == "1":
                return valid_evaluation("1", "state:1", 1.0)
            return EvaluationResult(
                ProposalStatus.INVALID,
                program=program,
                invalid_reason=InvalidReason.COMPILE_FAILURE,
            )

    result = MCTS(
        strategies=(STRATEGIES[0],),
        workload=WORKLOAD,
        generator=FixedGenerator(("1", "invalid")),
        evaluator=ValidThenInvalidEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(2),
        config=MCTSConfig(k_max=1, max_repairs=0),
    ).run(valid_evaluation("0", "state:0", 0.0))

    root_action = result.root.actions["a"]
    realization = next(iter(root_action.realizations.values()))
    child = next(node for node in result.nodes if node.state_key == "state:1")
    child_action = child.actions["a"]
    assert root_action.visits == 1
    assert root_action.value_sum == 1.0
    assert root_action.q_max == 1.0
    assert realization.descents == 1
    assert realization.value_sum == 1.0
    assert child_action.visits == 0
    assert child_action.value_sum == 0.0


def test_transposed_node_uses_current_path_depth_for_expansion() -> None:
    mcts = MCTS(
        strategies=(STRATEGIES[0],),
        workload=WORKLOAD,
        generator=ToyGenerator(),
        evaluator=ToyEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(1),
        config=MCTSConfig(k_max=1, max_depth=2),
    )
    detour = SearchNode("detour", valid_evaluation("10", "state:10", 10.0))
    middle = SearchNode("middle", valid_evaluation("11", "state:11", 11.0))
    shared = SearchNode("shared", valid_evaluation("1", "state:1", 1.0))
    detour.actions = {
        "a": StrategyEdge(
            "a",
            1.0,
            visits=1,
            realizations={middle.id: RealizationEdge(middle.id, descents=1, value_sum=1.0)},
        )
    }
    middle.actions = {
        "a": StrategyEdge(
            "a",
            1.0,
            visits=1,
            realizations={shared.id: RealizationEdge(shared.id, descents=1, value_sum=1.0)},
        )
    }
    root = SearchNode("root", valid_evaluation("0", "state:0", 0.0))
    root.actions = {
        "a": StrategyEdge(
            "a",
            1.0,
            visits=1,
            realizations={shared.id: RealizationEdge(shared.id, descents=1, value_sum=1.0)},
        )
    }
    for node in (detour, middle, shared, root):
        mcts.nodes.add(node)

    leaf = mcts._iterate(root)

    assert leaf is not None
    assert leaf.state_key == "state:2"
    assert mcts.nodes.get("state:1") is shared
    assert not hasattr(shared, "depth")
