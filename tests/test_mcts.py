from __future__ import annotations

import json

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
from kernel_mcts.trace_records import IterationStatus, SelectionMode


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
    assert result.prior_calls == 0
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
    events = RecordingEvents()
    result = MCTS(
        strategies=STRATEGIES,
        workload=WORKLOAD,
        generator=generator,
        evaluator=evaluator,
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(6),
        config=MCTSConfig(max_depth=3, k_max=2),
        events=events,
    ).run(valid_evaluation("root", "root", 0.0))
    assert len(result.nodes) == 2
    cached = next(node for node in result.nodes if node.state_key == "same-state")
    assert cached.evaluation.benchmark == BenchmarkResult((1.0,), 1.0, {"n=1": 1.0})
    assert cached.evaluation.metadata["artifact_id"] == "artifact:call:1"
    assert generator.calls == result.generations
    assert evaluator.calls == result.generations
    reused = [payload for event, payload in events.events if event == "node_reused"]
    assert reused
    assert all(payload["node_id"] == cached.id for payload in reused)


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
    backup_events = [payload for event, payload in events.events if event == "backup"]
    iteration_events = [payload for event, payload in events.events if event == "iteration_completed"]
    assert result.generations == 2
    assert len(result.nodes) == 2
    assert action.proposal_count == 1
    assert action.generation_attempt_count == 2
    assert action.repair_generation_count == 1
    assert action.visits == 1
    assert action.value_sum == 1.0
    assert [payload["b_gen"] for payload in generation_events] == [1, 2]
    assert [payload["repair_attempt"] for payload in generation_events] == [0, 1]
    assert [payload["iteration"] for payload in generation_events] == [1, 1]
    assert generation_events[-1]["created_node_id"] is not None
    assert generation_events[-1]["compile_status"] == "NOT_ATTEMPTED"
    assert len(backup_events) == 1
    assert backup_events[0]["strategy"]["q_mean"] == 1.0
    assert backup_events[0]["strategy"]["q_max"] == 1.0
    assert iteration_events[0]["steps"][0]["selection_mode"] == "EXPAND"


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

    outcome = mcts._iterate(root)

    assert outcome.status == IterationStatus.VALID
    assert outcome.leaf is not None
    assert outcome.leaf.state_key == "state:2"
    assert [step.selection_mode for step in outcome.steps] == [
        SelectionMode.UCB,
        SelectionMode.EXPAND,
    ]
    assert [step.node_id for step in outcome.steps] == [root.id, shared.id]
    assert outcome.expanded_parent_node_id == shared.id
    assert outcome.selected_strategy_id == "a"
    assert outcome.backed_up_reward == 2.0
    assert mcts.nodes.get("state:1") is shared
    assert not hasattr(shared, "depth")


def test_invalid_iteration_outcome_has_no_leaf_or_backup_reward() -> None:
    class InvalidEvaluator:
        def evaluate(self, program, workload):
            return EvaluationResult(
                ProposalStatus.INVALID,
                program=program,
                invalid_reason=InvalidReason.COMPILE_FAILURE,
            )

    mcts = MCTS(
        strategies=(STRATEGIES[0],),
        workload=WORKLOAD,
        generator=FixedGenerator(("invalid",)),
        evaluator=InvalidEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(1),
        config=MCTSConfig(max_repairs=0),
    )
    root = SearchNode("root", valid_evaluation("0", "state:0", 0.0))
    mcts.nodes.add(root)

    outcome = mcts._iterate(root)

    assert outcome.status == IterationStatus.INVALID
    assert outcome.leaf is None
    assert outcome.backed_up_reward is None
    assert len(outcome.steps) == 1
    assert outcome.steps[0].selection_mode == SelectionMode.EXPAND
    assert outcome.steps[0].child_node_id is None


def test_valid_root_expansion_records_expand_step_and_reward() -> None:
    mcts = MCTS(
        strategies=(STRATEGIES[0],),
        workload=WORKLOAD,
        generator=ToyGenerator(),
        evaluator=ToyEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(1),
    )
    root = SearchNode("root", valid_evaluation("0", "state:0", 0.0))
    mcts.nodes.add(root)

    outcome = mcts._iterate(root)

    assert outcome.status == IterationStatus.VALID
    assert outcome.leaf is not None
    assert outcome.backed_up_reward == outcome.leaf.reward
    assert outcome.expanded_parent_node_id == root.id
    assert len(outcome.steps) == 1
    assert outcome.steps[0].selection_mode == SelectionMode.EXPAND
    assert outcome.steps[0].child_node_id == outcome.leaf.id


def test_infrastructure_iteration_outcome_has_no_leaf_or_backup_reward() -> None:
    class InfrastructureEvaluator:
        def evaluate(self, program, workload):
            return EvaluationResult(ProposalStatus.INFRASTRUCTURE_FAILURE)

    mcts = MCTS(
        strategies=(STRATEGIES[0],),
        workload=WORKLOAD,
        generator=FixedGenerator(("candidate",)),
        evaluator=InfrastructureEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(1),
        config=MCTSConfig(max_repairs=0, max_infrastructure_retries=0),
    )
    root = SearchNode("root", valid_evaluation("0", "state:0", 0.0))
    mcts.nodes.add(root)

    outcome = mcts._iterate(root)

    assert outcome.status == IterationStatus.INFRASTRUCTURE_FAILURE
    assert outcome.leaf is None
    assert outcome.backed_up_reward is None


def test_cycle_iteration_outcome_records_ucb_step() -> None:
    mcts = MCTS(
        strategies=(STRATEGIES[0],),
        workload=WORKLOAD,
        generator=ToyGenerator(),
        evaluator=ToyEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(0),
        config=MCTSConfig(k_max=1),
    )
    root = SearchNode("root", valid_evaluation("0", "state:0", 0.0))
    root.actions = {
        "a": StrategyEdge(
            "a",
            1.0,
            visits=1,
            realizations={root.id: RealizationEdge(root.id)},
        )
    }
    mcts.nodes.add(root)

    outcome = mcts._iterate(root)

    assert outcome.status == IterationStatus.CYCLE
    assert outcome.leaf is root
    assert outcome.backed_up_reward == root.reward
    assert len(outcome.steps) == 1
    assert outcome.steps[0].selection_mode == SelectionMode.UCB
    assert outcome.steps[0].child_node_id == root.id


def test_depth_limit_iteration_outcome_records_traversed_path() -> None:
    mcts = MCTS(
        strategies=(STRATEGIES[0],),
        workload=WORKLOAD,
        generator=ToyGenerator(),
        evaluator=ToyEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(0),
        config=MCTSConfig(k_max=1, max_depth=1),
    )
    root = SearchNode("root", valid_evaluation("0", "state:0", 0.0))
    child = SearchNode("child", valid_evaluation("1", "state:1", 1.0))
    root.actions = {
        "a": StrategyEdge(
            "a",
            1.0,
            visits=1,
            realizations={child.id: RealizationEdge(child.id)},
        )
    }
    mcts.nodes.add(root)
    mcts.nodes.add(child)

    outcome = mcts._iterate(root)

    assert outcome.status == IterationStatus.DEPTH_LIMIT
    assert outcome.leaf is child
    assert outcome.backed_up_reward == child.reward
    assert [step.child_node_id for step in outcome.steps] == [child.id]


class CountingLLMPrior:
    name = "test-llm"
    counts_toward_b_prior = True

    def __init__(self) -> None:
        self.calls = 0

    def get_priors(self, program, workload, strategies, profile):
        self.calls += 1
        return {strategy.id: index + 1.0 for index, strategy in enumerate(strategies)}


class RecordingEvents:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event_type, payload):
        self.events.append((event_type, payload))


def test_llm_prior_is_counted_reported_and_logged() -> None:
    provider = CountingLLMPrior()
    events = RecordingEvents()
    result = MCTS(
        strategies=STRATEGIES,
        workload=WORKLOAD,
        generator=ToyGenerator(),
        evaluator=ToyEvaluator(),
        prior_provider=provider,
        budget=GenerationBudget(1),
        events=events,
    ).run(valid_evaluation("0", "state:0", 0.0))

    prior_events = [payload for event, payload in events.events if event == "strategy_priors"]
    assert provider.calls == 1
    assert result.prior_calls == 1
    assert prior_events == [
        {
            "iteration": 1,
            "node_id": result.root.id,
            "provider": "test-llm",
            "counts_toward_b_prior": True,
            "b_prior": 1,
            "priors": {"a": 1.0 / 3.0, "b": 2.0 / 3.0},
        }
    ]


def test_prior_is_cached_per_node_and_counted_on_new_node() -> None:
    provider = CountingLLMPrior()
    mcts = MCTS(
        strategies=STRATEGIES,
        workload=WORKLOAD,
        generator=ToyGenerator(),
        evaluator=ToyEvaluator(),
        prior_provider=provider,
        budget=GenerationBudget(0),
    )
    first = SearchNode("first", valid_evaluation("0", "state:0", 0.0))
    second = SearchNode("second", valid_evaluation("1", "state:1", 1.0))

    mcts._ensure_actions(first)
    mcts._ensure_actions(first)
    mcts._ensure_actions(second)

    assert provider.calls == 2
    assert mcts.prior_calls == 2


def test_invalid_llm_priors_are_counted_but_not_cached() -> None:
    class InvalidLLMPrior(CountingLLMPrior):
        def get_priors(self, program, workload, strategies, profile):
            self.calls += 1
            return {"unknown": 1.0}

    provider = InvalidLLMPrior()
    mcts = MCTS(
        strategies=STRATEGIES,
        workload=WORKLOAD,
        generator=ToyGenerator(),
        evaluator=ToyEvaluator(),
        prior_provider=provider,
        budget=GenerationBudget(0),
    )
    node = SearchNode("root", valid_evaluation("0", "state:0", 0.0))

    with pytest.raises(ValueError, match="prior keys"):
        mcts._ensure_actions(node)

    assert provider.calls == 1
    assert mcts.prior_calls == 1
    assert node.actions == {}


def test_run_emits_lifecycle_node_iteration_and_backup_events() -> None:
    events = RecordingEvents()
    result = MCTS(
        strategies=(STRATEGIES[0],),
        workload=WORKLOAD,
        generator=ToyGenerator(),
        evaluator=ToyEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(1),
        events=events,
    ).run(valid_evaluation("0", "state:0", 0.0))

    event_types = [event for event, _ in events.events]
    assert event_types[:2] == ["run_started", "node_created"]
    assert event_types[-1] == "run_completed"
    assert event_types.count("node_created") == 2
    assert event_types.count("generation") == 1
    assert event_types.count("backup") == 1
    assert event_types.count("iteration_completed") == 1
    assert event_types.count("node_snapshot") == len(result.nodes)
    assert event_types.count("strategy_edge_snapshot") == 1
    assert event_types.count("realization_edge_snapshot") == 1
    completed = events.events[-1][1]
    assert completed["iterations"] == result.iterations
    assert completed["b_gen"] == result.generations
    assert completed["best_node_id"] == result.best.id
    for _, payload in events.events:
        json.dumps(payload)


def test_invalid_run_emits_iteration_without_backup_or_node() -> None:
    class InvalidEvaluator:
        def evaluate(self, program, workload):
            return EvaluationResult(
                ProposalStatus.INVALID,
                program=program,
                invalid_reason=InvalidReason.COMPILE_FAILURE,
            )

    events = RecordingEvents()
    MCTS(
        strategies=(STRATEGIES[0],),
        workload=WORKLOAD,
        generator=FixedGenerator(("invalid",)),
        evaluator=InvalidEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(1),
        config=MCTSConfig(max_repairs=0),
        events=events,
    ).run(valid_evaluation("0", "state:0", 0.0))

    event_types = [event for event, _ in events.events]
    assert event_types.count("node_created") == 1
    assert "backup" not in event_types
    iteration = next(payload for event, payload in events.events if event == "iteration_completed")
    assert iteration["status"] == "INVALID"
    assert iteration["backed_up_reward"] is None
    snapshot = next(
        payload for event, payload in events.events if event == "strategy_edge_snapshot"
    )
    assert snapshot["strategy"]["visits"] == 0
    assert snapshot["strategy"]["proposal_count"] == 1
    assert snapshot["strategy"]["invalid_proposal_count"] == 1
    assert snapshot["strategy"]["q_max"] is None


def test_run_failure_is_logged_and_reraised() -> None:
    class FailingGenerator:
        def generate(self, request):
            raise RuntimeError("generation failed")

    events = RecordingEvents()
    mcts = MCTS(
        strategies=(STRATEGIES[0],),
        workload=WORKLOAD,
        generator=FailingGenerator(),
        evaluator=ToyEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(1),
        events=events,
    )

    with pytest.raises(RuntimeError, match="generation failed"):
        mcts.run(valid_evaluation("0", "state:0", 0.0))

    failed = [payload for event, payload in events.events if event == "run_failed"]
    assert failed == [
        {
            "iterations": 1,
                "b_gen": 1,
                "b_prior": 0,
                "profile_calls": 0,
                "error_type": "RuntimeError",
            "message": "generation failed",
        }
    ]
    assert all(event != "run_completed" for event, _ in events.events)
