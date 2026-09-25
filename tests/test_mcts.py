from __future__ import annotations

import json
import math

from kernel_mcts.budget import GenerationBudget, MutationBudget
from kernel_mcts.cute_mutations import (
    CUTE_MUTATION_STRATEGIES,
    CHANGE_EPILOGUE_STAGES,
    CHANGE_PIPELINE_STAGES,
    CHANGE_SHARED_MEMORY_SWIZZLE,
    CuteMutationGenerator,
)
from kernel_mcts.cute_program import (
    CuteGemmProgram,
    PinnedCuteGemmRenderer,
    REFERENCE_CUTE_GEMM,
    cute_gemm_program_from_source,
)
from kernel_mcts.cute_independent import make_independent_cute_gemm
from kernel_mcts.cute_independent_program import (
    IndependentCuteGemmRenderer,
    independent_cute_gemm_from_source,
)
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
from kernel_mcts.generation import GenerationResult, MutationFirstGenerator
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


def test_mcts_searches_typed_cute_mutations_under_separate_budget() -> None:
    class CuteEvaluator:
        def evaluate(self, program, workload):
            representation = cute_gemm_program_from_source(program.source)
            reward = (256 - representation.tile_n) / 128 + (
                representation.cluster_m - 1
            )
            return EvaluationResult(
                ProposalStatus.VALID,
                program,
                representation.configuration_hash,
                float(reward),
                BenchmarkResult((1.0,), 1.0, {"n=1": 1.0}),
                metadata={
                    "representation": representation.as_dict(),
                    "representation_schema_version": representation.schema_version,
                    "configuration_hash": representation.configuration_hash,
                },
            )

    root_program = PinnedCuteGemmRenderer().render(REFERENCE_CUTE_GEMM)
    root = CuteEvaluator().evaluate(root_program, WORKLOAD)
    events = RecordingEvents()
    result = MCTS(
        strategies=CUTE_MUTATION_STRATEGIES,
        workload=WORKLOAD,
        generator=CuteMutationGenerator(),
        evaluator=CuteEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(0),
        mutation_budget=MutationBudget(4),
        config=MCTSConfig(max_depth=3, k_max=2),
        events=events,
    ).run(root)

    assert result.generations == 0
    assert result.mutations == 4
    assert len(result.nodes) >= 3
    generations = [payload for event, payload in events.events if event == "generation"]
    assert [payload["b_mut"] for payload in generations] == [1, 2, 3, 4]
    assert all(payload["b_gen"] == 0 for payload in generations)
    assert all(
        payload["proposal_mechanism"] == "typed_mutation"
        for payload in generations
    )


def test_mcts_creates_schema_v3_epilogue_node_under_mutation_budget() -> None:
    class CuteEvaluator:
        def evaluate(self, program, workload):
            representation = cute_gemm_program_from_source(program.source)
            reward = float(4 - (representation.epilogue_stages or 4))
            return EvaluationResult(
                ProposalStatus.VALID,
                program,
                representation.configuration_hash,
                reward,
                BenchmarkResult((1.0,), 1.0, {"n=1": 1.0}),
                metadata={
                    "representation": representation.as_dict(),
                    "representation_schema_version": representation.schema_version,
                    "configuration_hash": representation.configuration_hash,
                },
            )

    parent = CuteGemmProgram(128, 256, 2, 1)
    root = CuteEvaluator().evaluate(PinnedCuteGemmRenderer().render(parent), WORKLOAD)
    strategy = next(
        item
        for item in CUTE_MUTATION_STRATEGIES
        if item.id == CHANGE_EPILOGUE_STAGES
    )
    result = MCTS(
        strategies=(strategy,),
        workload=WORKLOAD,
        generator=CuteMutationGenerator(),
        evaluator=CuteEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(0),
        mutation_budget=MutationBudget(1),
        config=MCTSConfig(max_depth=1, k_max=2),
    ).run(root)

    assert result.generations == 0
    assert result.mutations == 1
    assert len(result.nodes) == 2
    candidate = next(node for node in result.nodes if node is not result.root)
    representation = cute_gemm_program_from_source(candidate.program.source)
    assert representation.schema_version == 3
    assert representation.epilogue_stages == 2


def test_mcts_creates_sw64_node_under_mutation_budget() -> None:
    class CuteEvaluator:
        def evaluate(self, program, workload):
            representation = cute_gemm_program_from_source(program.source)
            reward = float(representation.shared_memory_swizzle == "sw64")
            return EvaluationResult(
                ProposalStatus.VALID,
                program,
                representation.configuration_hash,
                reward,
                BenchmarkResult((1.0,), 1.0, {"n=1": 1.0}),
                metadata={
                    "representation": representation.as_dict(),
                    "representation_schema_version": representation.schema_version,
                    "configuration_hash": representation.configuration_hash,
                },
            )

    parent = CuteGemmProgram(128, 256, 2, 1)
    root = CuteEvaluator().evaluate(PinnedCuteGemmRenderer().render(parent), WORKLOAD)
    strategy = next(
        item
        for item in CUTE_MUTATION_STRATEGIES
        if item.id == CHANGE_SHARED_MEMORY_SWIZZLE
    )
    result = MCTS(
        strategies=(strategy,),
        workload=WORKLOAD,
        generator=CuteMutationGenerator(),
        evaluator=CuteEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(0),
        mutation_budget=MutationBudget(1),
        config=MCTSConfig(max_depth=1, k_max=1),
    ).run(root)

    assert result.generations == 0
    assert result.mutations == 1
    assert len(result.nodes) == 2
    candidate = next(node for node in result.nodes if node is not result.root)
    representation = cute_gemm_program_from_source(candidate.program.source)
    assert representation.schema_version == 3
    assert representation.shared_memory_swizzle == "sw64"


def test_mcts_creates_independent_sw64_node_under_mutation_budget() -> None:
    class IndependentEvaluator:
        def evaluate(self, program, workload):
            representation = independent_cute_gemm_from_source(program.source)
            swizzle = representation.mainloop.a_copy.swizzle_bytes
            return EvaluationResult(
                ProposalStatus.VALID,
                program,
                representation.configuration_hash,
                float(swizzle == 64),
                BenchmarkResult((1.0,), 1.0, {"n=1": 1.0}),
                metadata={"representation": representation.as_dict()},
            )

    renderer = IndependentCuteGemmRenderer()
    root = IndependentEvaluator().evaluate(
        renderer.render(make_independent_cute_gemm()), WORKLOAD
    )
    strategy = next(
        item
        for item in CUTE_MUTATION_STRATEGIES
        if item.id == CHANGE_SHARED_MEMORY_SWIZZLE
    )

    result = MCTS(
        strategies=(strategy,),
        workload=WORKLOAD,
        generator=CuteMutationGenerator(),
        evaluator=IndependentEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(0),
        mutation_budget=MutationBudget(1),
        config=MCTSConfig(max_depth=1, k_max=1),
    ).run(root)

    assert result.generations == 0
    assert result.mutations == 1
    assert len(result.nodes) == 2
    assert result.best.reward == 1.0
    candidate = next(node for node in result.nodes if node is not result.root)
    representation = independent_cute_gemm_from_source(candidate.program.source)
    assert representation.mainloop.a_copy.swizzle_bytes == 64


def test_mcts_creates_independent_two_stage_node_under_mutation_budget() -> None:
    class IndependentEvaluator:
        def evaluate(self, program, workload):
            representation = independent_cute_gemm_from_source(program.source)
            stages = representation.mainloop.pipeline_stages
            return EvaluationResult(
                ProposalStatus.VALID,
                program,
                representation.configuration_hash,
                -0.01 if stages == 2 else 0.0,
                BenchmarkResult((1.0,), 1.0, {"n=1": 1.0}),
                metadata={"representation": representation.as_dict()},
            )

    root = IndependentEvaluator().evaluate(
        IndependentCuteGemmRenderer().render(make_independent_cute_gemm()),
        WORKLOAD,
    )
    strategy = next(
        item
        for item in CUTE_MUTATION_STRATEGIES
        if item.id == CHANGE_PIPELINE_STAGES
    )

    result = MCTS(
        strategies=(strategy,),
        workload=WORKLOAD,
        generator=CuteMutationGenerator(),
        evaluator=IndependentEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(0),
        mutation_budget=MutationBudget(1),
        config=MCTSConfig(max_depth=1, k_max=1),
    ).run(root)

    assert result.generations == 0
    assert result.mutations == 1
    assert len(result.nodes) == 2
    assert result.best is result.root
    candidate = next(node for node in result.nodes if node is not result.root)
    assert candidate.reward == -0.01
    representation = independent_cute_gemm_from_source(candidate.program.source)
    assert representation.mainloop.pipeline_stages == 2


def test_mcts_routes_mutation_before_generation_with_separate_budgets() -> None:
    class OneMutation:
        def __init__(self):
            self.issued = False

        def can_generate(self, _request):
            return not self.issued

        def generate(self, _request):
            self.issued = True
            return GenerationResult(
                "mutation:1", "1", KernelProgram("1"), "mutation"
            )

    class OneGeneration:
        def __init__(self):
            self.issued = False

        def can_generate(self, _request):
            return not self.issued

        def generate(self, _request):
            self.issued = True
            return GenerationResult(
                "generation:1", "2", KernelProgram("2"), "generation"
            )

    events = RecordingEvents()
    result = MCTS(
        strategies=(STRATEGIES[0],),
        workload=WORKLOAD,
        generator=MutationFirstGenerator(OneMutation(), OneGeneration()),
        evaluator=ToyEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(1),
        mutation_budget=MutationBudget(1),
        config=MCTSConfig(max_depth=1, k_max=2),
        events=events,
    ).run(valid_evaluation("0", "state:0", 0.0))

    assert result.mutations == 1
    assert result.generations == 1
    generations = [payload for event, payload in events.events if event == "generation"]
    assert [payload["generation_id"] for payload in generations] == [
        "mutation:1",
        "generation:1",
    ]
    assert generations[0]["proposal_mechanism_candidates"] == [
        "mutation",
        "generation",
    ]
    assert generations[1]["proposal_mechanism_candidates"] == ["generation"]
    assert generations[1]["proposal_mechanism"] == "generation"


def test_exhausted_leaf_does_not_hide_available_root_strategy() -> None:
    class FiniteGenerator:
        def __init__(self):
            self.issued = set()

        def can_generate(self, request):
            return request.parent.source == "0" and request.strategy.id not in self.issued

        def generate(self, request):
            self.issued.add(request.strategy.id)
            source = "1" if request.strategy.id == "a" else "2"
            return GenerationResult(
                f"generation:{request.strategy.id}",
                source,
                KernelProgram(source),
                "prompt",
            )

    result = MCTS(
        strategies=STRATEGIES,
        workload=WORKLOAD,
        generator=FiniteGenerator(),
        evaluator=ToyEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(2),
        config=MCTSConfig(max_depth=3, k_max=2),
        seed=1,
    ).run(valid_evaluation("0", "state:0", 0.0))

    assert result.generations == 2
    assert {node.program.source for node in result.nodes} == {"0", "1", "2"}
    assert result.iterations == 3


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
    result = MCTS(
        strategies=STRATEGIES,
        workload=WORKLOAD,
        generator=ToyGenerator(),
        evaluator=ToyEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(1),
        profiler=profiler,
    ).run(root_evaluation)

    assert profiler.evaluations == [root_evaluation, result.best.evaluation]
    assert result.profile_calls == 2
    assert result.best.profile == {"profiled": True}


def test_final_best_profile_is_persisted_with_distinct_trigger() -> None:
    class Profiler:
        def lightweight_profile(self, evaluation, workload):
            return {"state_key": evaluation.state_key}

    events = RecordingEvents()
    result = MCTS(
        strategies=STRATEGIES,
        workload=WORKLOAD,
        generator=ToyGenerator(),
        evaluator=ToyEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(1),
        profiler=Profiler(),
        events=events,
    ).run(valid_evaluation("0", "state:0", 0.0))

    profile_events = [
        payload for event, payload in events.events if event == "node_profiled"
    ]
    assert [payload["trigger"] for payload in profile_events] == [
        "expansion",
        "final_best",
    ]
    assert all(
        payload["metric_set"] == "lightweight_v1" for payload in profile_events
    )
    assert profile_events[-1]["node_id"] == result.best.id
    assert profile_events[-1]["profile"] == {"state_key": result.best.state_key}


def test_final_best_is_not_reprofiled_when_expansion_already_profiled_it() -> None:
    class Evaluator:
        def evaluate(self, program, workload):
            reward = 2.0 if program.source == "1" else 1.0
            return valid_evaluation(program.source, f"state:{program.source}", reward)

    class Profiler:
        def lightweight_profile(self, evaluation, workload):
            return {"state_key": evaluation.state_key}

    events = RecordingEvents()
    result = MCTS(
        strategies=(STRATEGIES[0],),
        workload=WORKLOAD,
        generator=ToyGenerator(),
        evaluator=Evaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(2),
        config=MCTSConfig(k_max=1),
        profiler=Profiler(),
        events=events,
    ).run(valid_evaluation("0", "state:0", 0.0))

    profile_events = [
        payload for event, payload in events.events if event == "node_profiled"
    ]
    assert result.best.program.source == "1"
    assert result.profile_calls == 2
    assert [payload["trigger"] for payload in profile_events] == [
        "expansion",
        "expansion",
    ]


def test_opt_in_profile_delta_uses_active_incoming_edge() -> None:
    class RecordingGenerator(ToyGenerator):
        def __init__(self) -> None:
            super().__init__()
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            return super().generate(request)

    class Profiler:
        def lightweight_profile(self, evaluation, workload):
            value = int(evaluation.program.source)
            return {
                "summary": {
                    "occupancy": 90.0 - value * 10.0,
                    "registers": 32 + value * 8,
                }
            }

    generator = RecordingGenerator()
    MCTS(
        strategies=(STRATEGIES[0],),
        workload=WORKLOAD,
        generator=generator,
        evaluator=ToyEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(2),
        config=MCTSConfig(
            k_max=1,
            max_repairs=0,
            include_incoming_profile_delta=True,
        ),
        profiler=Profiler(),
    ).run(valid_evaluation("0", "state:0", 0.0))

    assert generator.requests[0].incoming_profile_delta is None
    delta = generator.requests[1].incoming_profile_delta
    assert delta is not None
    assert delta["basis"] == "selected_incoming_edge"
    assert delta["strategy_id"] == "a"
    assert delta["reward_delta"] == 1.0
    assert delta["speedup_ratio"] == pytest.approx(math.e)
    assert delta["summary_delta"] == {
        "occupancy": -10.0,
        "registers": 8.0,
    }


def test_profile_delta_is_disabled_by_default() -> None:
    class RecordingGenerator(ToyGenerator):
        def __init__(self) -> None:
            super().__init__()
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            return super().generate(request)

    class Profiler:
        def lightweight_profile(self, evaluation, workload):
            return {"summary": {"occupancy": 50.0}}

    generator = RecordingGenerator()
    MCTS(
        strategies=(STRATEGIES[0],),
        workload=WORKLOAD,
        generator=generator,
        evaluator=ToyEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(2),
        config=MCTSConfig(k_max=1, max_repairs=0),
        profiler=Profiler(),
    ).run(valid_evaluation("0", "state:0", 0.0))

    assert all(request.incoming_profile_delta is None for request in generator.requests)


def test_mcts_rejects_unknown_profile_metric_set() -> None:
    with pytest.raises(ValueError, match="unknown profile metric set"):
        MCTSConfig(profile_metric_set="unknown")


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


def test_selection_score_snapshots_use_exact_puct_and_ucb_terms() -> None:
    node = SearchNode("root", valid_evaluation("0", "state:0", 0.0))
    first = StrategyEdge("a", 0.75, visits=3, value_sum=6.0, q_max=3.0)
    second = StrategyEdge("b", 0.25, visits=1, value_sum=1.0, q_max=1.5)
    node.actions = {"a": first, "b": second}
    first.realizations = {
        "child-a": RealizationEdge("child-a", descents=2, value_sum=3.0),
        "child-b": RealizationEdge("child-b", descents=1, value_sum=2.0),
    }
    mcts = MCTS(
        strategies=STRATEGIES,
        workload=WORKLOAD,
        generator=ToyGenerator(),
        evaluator=ToyEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(1),
        config=MCTSConfig(c_puct=2.0, c_ucb=1.25),
        seed=0,
    )

    selected, puct, total = mcts._select_action_with_scores(node)
    assert total == 4
    assert selected.strategy_id == "a"
    first_score = next(item for item in puct if item["strategy_id"] == "a")
    assert first_score["exploit_term"] == 2.0
    assert first_score["explore_term"] == pytest.approx(2.0 * 0.75 * 2.0 / 4.0)
    assert first_score["total_score"] == pytest.approx(
        first_score["exploit_term"] + first_score["explore_term"]
    )

    realization, ucb = mcts._select_realization_with_scores(first)
    expected = {
        item["child_node_id"]: item["q_mean"]
        + 1.25 * math.sqrt(math.log1p(3) / (1 + item["descents"]))
        for item in ucb
    }
    assert realization.child_id == max(expected, key=expected.get)
    assert all(
        item["total_score"] == pytest.approx(expected[item["child_node_id"]])
        for item in ucb
    )


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


def test_explicit_drift_policy_remeasures_without_changing_cached_rewards() -> None:
    class DriftMonitor:
        def __init__(self):
            self.calls = []

        def remeasure(self, evaluation, workload):
            self.calls.append(evaluation.state_key)
            assert evaluation.benchmark is not None
            return EvaluationResult(
                ProposalStatus.VALID,
                evaluation.program,
                evaluation.state_key,
                evaluation.reward,
                BenchmarkResult((1.1,), 1.1),
            )

    monitor = DriftMonitor()
    events = RecordingEvents()
    result = MCTS(
        strategies=(STRATEGIES[0],),
        workload=WORKLOAD,
        generator=ToyGenerator(),
        evaluator=ToyEvaluator(),
        prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(2),
        config=MCTSConfig(
            k_max=1,
            measurement_drift_interval=1,
            measurement_drift_threshold=0.05,
        ),
        events=events,
        drift_monitor=monitor,
    ).run(valid_evaluation("0", "state:0", 0.0))

    probes = [payload for event, payload in events.events if event == "measurement_drift_probe"]
    assert result.drift_probe_calls == 4
    assert len(probes) == 4
    assert all(payload["suspect"] for payload in probes)
    assert result.root.reward == 0.0
    assert result.best.reward == 2.0


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
                    "b_mut": 0,
                    "b_prior": 0,
                "profile_calls": 0,
                "drift_probe_calls": 0,
                "error_type": "RuntimeError",
            "message": "generation failed",
        }
    ]
    assert all(event != "run_completed" for event, _ in events.events)
