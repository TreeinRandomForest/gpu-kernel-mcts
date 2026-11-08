from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from kernel_mcts.benchmarks import (
    BF16_GEMM_WORKLOAD,
    load_triton_bf16_gemm_gluon_reference,
    load_triton_bf16_gemm_root,
)
from kernel_mcts.budget import GenerationBudget
from kernel_mcts.config import load_data, parse_strategies
from kernel_mcts.domain import (
    BenchmarkResult, CompileStatus, EvaluationResult, InvalidReason, KernelProgram,
    ProposalStatus,
)
from kernel_mcts.generation import GenerationRequest, GenerationResult
from kernel_mcts.llm_generation import build_generation_prompt
from kernel_mcts.priors import UniformStrategyPrior
from kernel_mcts.proposals import run_proposal
from kernel_mcts.search import MCTS, MCTSConfig
from kernel_mcts.search.model import SearchNode
from kernel_mcts.strategy_applicability import applicable_strategies, triton_parent_stage


CATALOG = Path("configs/triton_h100_strategies_v2.json")
STAGED_CATALOG = Path("configs/triton_h100_strategies_v3_staged.json")
BRIDGE_CATALOG = Path("configs/triton_h100_gluon_bridge_v4.json")
CHAIN_CATALOG = Path("configs/triton_h100_gluon_bridge_local_v5.json")
REVISED_CHAIN_CATALOG = Path("configs/triton_h100_gluon_bridge_local_v7.json")
DESCRIPTOR_CHAIN_CATALOG = Path("configs/triton_h100_gluon_bridge_local_v8.json")
WGMMA_CHAIN_CATALOG = Path("configs/triton_h100_gluon_bridge_local_v9.json")
WGMMA_FIXED_CHAIN_CATALOG = Path("configs/triton_h100_gluon_bridge_local_v10.json")
WGMMA_VERIFIED_API_CATALOG = Path("configs/triton_h100_gluon_bridge_local_v11.json")
WGMMA_GUARDED_CATALOG = Path("configs/triton_h100_gluon_bridge_local_v12.json")
WGMMA_ACCUMULATOR_CATALOG = Path("configs/triton_h100_gluon_bridge_local_v14.json")
WGMMA_INVALIDATE_CATALOG = Path("configs/triton_h100_gluon_bridge_local_v15.json")
WGMMA_BYTES_CATALOG = Path("configs/triton_h100_gluon_bridge_local_v16.json")
HARDWARE = {"gpu_model": "H100"}


def catalog():
    return parse_strategies(load_data(CATALOG))


def staged_catalog():
    return parse_strategies(load_data(STAGED_CATALOG))


def test_v4_bridge_is_opt_in_and_applies_only_to_descriptor_triton() -> None:
    strategies = parse_strategies(load_data(BRIDGE_CATALOG))
    assert len(strategies) == 1
    strategy = strategies[0]
    assert strategy.id == "gluon_mma_v2_pointer_port"
    assert strategy.candidate_stage == "gluon_mma_v2"
    assert strategy.scaffold_resource is None
    descriptor = KernelProgram(DESCRIPTOR_TRITON_SOURCE, backend="triton_h100")
    assert applicable_strategies(strategies, descriptor, BF16_GEMM_WORKLOAD, HARDWARE) == (
        strategy,)
    assert applicable_strategies(strategies, load_triton_bf16_gemm_root(),
                                 BF16_GEMM_WORKLOAD, HARDWARE) == ()


def test_v5_catalog_routes_pointer_descriptor_and_actionable_gluon_stages() -> None:
    strategies = parse_strategies(load_data(CHAIN_CATALOG))
    root = load_triton_bf16_gemm_root()
    descriptor = KernelProgram(DESCRIPTOR_TRITON_SOURCE, "triton_h100")
    path = Path("docs/experiments/triton-h100-gluon-search-adapter")
    misordered = KernelProgram((path / "v4-mma-v2-thread-layout-repair/candidate.py").read_text(),
                               "triton_h100")
    corrected = KernelProgram((path / "v4_mma_v2_coalesced_layout_fixture.py").read_text(),
                              "triton_h100")
    names = lambda program: {item.id for item in applicable_strategies(
        strategies, program, BF16_GEMM_WORKLOAD, HARDWARE)}
    assert names(root) == {"descriptor_tma_staging"}
    assert names(descriptor) == {"gluon_mma_v2_pointer_port"}
    assert names(misordered) == {"gluon_mma_v2_coalesced_layout"}
    assert names(corrected) == set()


def test_v7_catalog_preserves_stage_routing_with_revised_bridge_prompt() -> None:
    strategies = parse_strategies(load_data(REVISED_CHAIN_CATALOG))
    assert [item.id for item in strategies] == [
        "descriptor_tma_staging", "gluon_mma_v2_pointer_port",
        "gluon_mma_v2_coalesced_layout",
    ]
    bridge = strategies[1]
    assert bridge.scaffold_resource is None
    assert bridge.candidate_stage == "gluon_mma_v2"
    assert "gl.arange" in bridge.prompt_for("triton_h100")
    descriptor = KernelProgram(DESCRIPTOR_TRITON_SOURCE, "triton_h100")
    assert applicable_strategies(strategies, descriptor,
                                 BF16_GEMM_WORKLOAD, HARDWARE) == (bridge,)


def test_descriptor_stage_follows_constructed_arguments_into_kernel() -> None:
    source = """
@triton.jit
def gemm(A, B, C):
    a = A.load([0, 0])
    b = B.load([0, 0])
def launch(A, B, C, M, N, K):
    a_desc = TensorDescriptor(A, shape=[M,K], strides=[K,1], block_shape=[64,64])
    b_desc = TensorDescriptor(B, shape=[N,K], strides=[K,1], block_shape=[128,64])
    return gemm[(1,)](a_desc, b_desc, C)
"""
    assert triton_parent_stage(source) == "descriptor_triton"
    pointer_source = source.replace("(a_desc, b_desc, C)", "(A, B, C)")
    assert triton_parent_stage(pointer_source) == "pointer_triton"


def test_v8_catalog_revises_descriptor_prompt_without_changing_stage_route() -> None:
    strategies = parse_strategies(load_data(DESCRIPTOR_CHAIN_CATALOG))
    assert [strategy.id for strategy in strategies] == [
        "descriptor_tma_staging", "gluon_mma_v2_pointer_port",
        "gluon_mma_v2_coalesced_layout",
    ]
    assert "TensorDescriptor(A, shape=[M,K]" in strategies[0].prompt_for("triton_h100")
    assert applicable_strategies(strategies, load_triton_bf16_gemm_root(),
                                 BF16_GEMM_WORKLOAD, HARDWARE) == (strategies[0],)


def test_v9_serial_wgmma_action_routes_from_valid_mma_v2_parent() -> None:
    strategies = parse_strategies(load_data(WGMMA_CHAIN_CATALOG))
    action = strategies[-1]
    assert action.id == "single_cta_gluon_tma_wgmma_serial"
    assert action.candidate_stage == "single_cta_gluon"
    assert action.scaffold_resource is None
    parent = KernelProgram(Path(
        "docs/experiments/triton-h100-gluon-search-adapter/"
        "v8-packaged-root-mcts-bgen4-complete-gluon.py"
    ).read_text(), "triton_h100")
    assert triton_parent_stage(parent.source) == "gluon_mma_v2"
    assert action in applicable_strategies(strategies, parent,
                                           BF16_GEMM_WORKLOAD, HARDWARE)


def test_v10_wgmma_prompt_preserves_loop_carried_accumulator_type() -> None:
    strategy = parse_strategies(load_data(WGMMA_FIXED_CHAIN_CATALOG))[-1]
    assert "WITHOUT assigning its return inside the loop" in strategy.prompt_for(
        "triton_h100")


def test_v11_wgmma_prompt_names_the_validated_host_layout_api() -> None:
    strategy = parse_strategies(load_data(WGMMA_VERIFIED_API_CATALOG))[-1]
    prompt = strategy.prompt_for("triton_h100")
    assert "gl.NVMMASharedLayout.get_default_for([BM, BK], gl.bfloat16)" in prompt
    assert "gl.NVMMASharedLayout.get_default_for([BN, BK], gl.bfloat16)" in prompt
    assert "gl.NVMMASharedLayout.get_default_for([BM, BN], gl.bfloat16)" in prompt


def test_v12_guarded_serial_action_is_opt_in_and_stage_scoped() -> None:
    strategies = parse_strategies(load_data(WGMMA_GUARDED_CATALOG))
    action = strategies[-1]
    assert action.id == "single_cta_gluon_tma_wgmma_serial_guarded"
    assert action.candidate_stage == "single_cta_gluon"
    parent = KernelProgram(Path(
        "docs/experiments/triton-h100-gluon-search-adapter/"
        "v8-packaged-root-mcts-bgen4-complete-gluon.py"
    ).read_text(), "triton_h100")
    assert action in applicable_strategies(strategies, parent,
                                           BF16_GEMM_WORKLOAD, HARDWARE)


def test_v14_prompt_states_wgmma_accumulator_and_output_contract() -> None:
    strategy = parse_strategies(load_data(WGMMA_ACCUMULATOR_CATALOG))[-1]
    prompt = strategy.prompt_for("triton_h100")
    assert "acc = warpgroup_mma(a_smem" in prompt
    assert "c_smem.store(acc.to(gl.bfloat16))" in prompt


def test_v15_prompt_names_mbarrier_invalidation_method() -> None:
    strategy = parse_strategies(load_data(WGMMA_INVALIDATE_CATALOG))[-1]
    assert "mbarrier.invalidate(bar)" in strategy.prompt_for("triton_h100")


def test_v16_prompt_names_descriptor_transfer_byte_count() -> None:
    strategy = parse_strategies(load_data(WGMMA_BYTES_CATALOG))[-1]
    assert "a_desc.nbytes_per_cta + b_desc.nbytes_per_cta" in strategy.prompt_for(
        "triton_h100")


DESCRIPTOR_TRITON_SOURCE = """
@triton.jit
def gemm(a_desc):
    a = a_desc.load([0, 0])
def launch(A, B, C, M, N, K):
    a_desc = TensorDescriptor.from_tensor(A, [128, 64])
    return gemm[(1,)](a_desc)
"""

SINGLE_CTA_GLUON_SOURCE = """
@gluon.jit
def gemm(a_desc):
    bar = mbarrier.allocate_mbarrier(batch=1)
    acc = warpgroup_mma(a_desc, a_desc, 0)
def launch(A, B, C, M, N, K):
    a_desc = TensorDescriptor.from_tensor(A, [128, 64])
    return gemm[(1,)](a_desc, num_ctas=1)
"""

GLUON_MMA_V2_SOURCE = """
@gluon.jit
def gemm(A, B, C):
    acc = mma_v2(A, B, 0)
def launch(A, B, C, M, N, K):
    return gemm[(1,)](A, B, C, num_ctas=1)
"""


def test_v2_catalog_is_separate_and_stage_classification_is_structural() -> None:
    old = parse_strategies(load_data("configs/triton_h100_strategies.json"))
    new = catalog()
    assert len(old) == 5
    assert len(new) == len({item.id for item in new}) == 6
    assert load_data(CATALOG)["version"] == "triton_h100_stage_v2"
    assert triton_parent_stage(load_triton_bf16_gemm_root().source) == "pointer_triton"
    assert triton_parent_stage(load_triton_bf16_gemm_gluon_reference().source) == "clustered_gluon"
    assert triton_parent_stage("# @gluon.jit\nnot Python") == "unknown"
    assert triton_parent_stage("""
@gluon.jit
def dormant():
    pass
@triton.jit
def active():
    pass
def launch(A, B, C, M, N, K):
    return active[(1,)]()
""") == "pointer_triton"


def test_stage_and_hardware_filter_before_prior() -> None:
    root = load_triton_bf16_gemm_root()
    gluon = load_triton_bf16_gemm_gluon_reference()
    root_actions = {item.id for item in applicable_strategies(
        catalog(), root, BF16_GEMM_WORKLOAD, HARDWARE)}
    gluon_actions = {item.id for item in applicable_strategies(
        catalog(), gluon, BF16_GEMM_WORKLOAD, HARDWARE)}
    assert root_actions == {
        "tile_warp_co_design", "descriptor_tma_staging", "cross_cta_tile_traversal",
        "clustered_gluon_wgmma_tma",
    }
    assert gluon_actions == {
        "tile_warp_co_design", "cross_cta_tile_traversal",
        "three_slot_wait_one_pipeline", "persistent_cluster_assignment",
    }
    assert applicable_strategies(catalog(), root, BF16_GEMM_WORKLOAD,
                                 {"gpu_model": "MI300X"}) == ()
    assert applicable_strategies(catalog(), root,
                                 replace(BF16_GEMM_WORKLOAD, operation="attention"),
                                 HARDWARE) == ()


def test_staged_v3_catalog_offers_one_structural_transition_at_each_stage() -> None:
    assert triton_parent_stage(DESCRIPTOR_TRITON_SOURCE) == "descriptor_triton"
    assert triton_parent_stage(GLUON_MMA_V2_SOURCE) == "gluon_mma_v2"
    assert triton_parent_stage(SINGLE_CTA_GLUON_SOURCE) == "single_cta_gluon"
    assert load_data(STAGED_CATALOG)["version"] == "triton_h100_staged_v3"
    strategies = staged_catalog()
    stages = (
        (load_triton_bf16_gemm_root(),
         {"descriptor_tma_staging", "gluon_mma_v2_port"}),
        (KernelProgram(DESCRIPTOR_TRITON_SOURCE, "triton_h100"),
         {"gluon_mma_v2_port", "single_cta_gluon_wgmma_tma"}),
        (KernelProgram(GLUON_MMA_V2_SOURCE, "triton_h100"),
         {"single_cta_gluon_wgmma_tma"}),
        (KernelProgram(SINGLE_CTA_GLUON_SOURCE, "triton_h100"),
         {"two_cta_cluster_partition", "single_cta_tile_shape"}),
        (load_triton_bf16_gemm_gluon_reference(), {"three_slot_wait_one_pipeline"}),
    )
    structural = {"descriptor_tma_staging", "single_cta_gluon_wgmma_tma",
                  "gluon_mma_v2_port", "two_cta_cluster_partition",
                  "three_slot_wait_one_pipeline", "single_cta_tile_shape"}
    for program, expected in stages:
        offered = {item.id for item in applicable_strategies(
            strategies, program, BF16_GEMM_WORKLOAD, HARDWARE)}
        assert (offered & structural) == expected
    assert {item.id: item.candidate_stage for item in strategies if item.id in structural} == {
        "descriptor_tma_staging": "descriptor_triton",
        "gluon_mma_v2_port": "gluon_mma_v2",
        "single_cta_gluon_wgmma_tma": "single_cta_gluon",
        "two_cta_cluster_partition": "clustered_gluon",
        "three_slot_wait_one_pipeline": "clustered_gluon",
        "single_cta_tile_shape": "single_cta_gluon",
    }
    prompt_by_id = {item.id: item.prompts["triton_h100"] for item in strategies}
    assert "unicast TMA load" in prompt_by_id["single_cta_gluon_wgmma_tma"]
    assert "unicast per-CTA TMA loads" in prompt_by_id["two_cta_cluster_partition"]


def test_mcts_prior_receives_only_parent_eligible_actions() -> None:
    root = load_triton_bf16_gemm_root()
    evaluation = EvaluationResult(
        ProposalStatus.VALID, root, "root", 0.0, BenchmarkResult((1.0,), 1.0),
    )

    class RecordingPrior(UniformStrategyPrior):
        def __init__(self):
            self.seen = ()

        def get_priors(self, program, workload, strategies, profile):
            self.seen = tuple(item.id for item in strategies)
            return super().get_priors(program, workload, strategies, profile)

    prior = RecordingPrior()
    search = MCTS(
        strategies=catalog(), workload=BF16_GEMM_WORKLOAD,
        generator=object(), evaluator=object(), prior_provider=prior,
        budget=GenerationBudget(1), hardware=HARDWARE,
    )
    node = SearchNode("root-node", evaluation)
    search._ensure_actions(node)
    assert set(prior.seen) == set(node.actions)
    assert "persistent_cluster_assignment" not in node.actions
    assert sum(edge.prior for edge in node.actions.values()) == 1.0


def test_v2_catalog_rejects_malformed_applicability() -> None:
    data = load_data(CATALOG)
    data["strategies"][0]["applicability"]["parent_stages"] = "pointer_triton"
    with pytest.raises(ValueError, match="parent_stages"):
        parse_strategies(data)
    data = load_data(CATALOG)
    data["strategies"][0]["scaffold_resource"] = "../other.py"
    with pytest.raises(ValueError, match="scaffold_resource"):
        parse_strategies(data)
    data = load_data(CATALOG)
    data["strategies"][0]["candidate_stage"] = "not_a_stage"
    with pytest.raises(ValueError, match="candidate_stage"):
        parse_strategies(data)


def test_structural_action_prompt_keeps_abi_and_ownership_guards() -> None:
    strategy = next(item for item in catalog() if item.id == "clustered_gluon_wgmma_tma")
    prompt = json.loads(build_generation_prompt(GenerationRequest(
        parent=load_triton_bf16_gemm_root(), strategy=strategy,
        workload=BF16_GEMM_WORKLOAD, hardware=HARDWARE, profile=None,
    )))
    text = prompt["strategy"]["backend_prompt"]
    assert "B.T" in text and "two-CTA" in text
    assert "one writer" in text and "ordinary B loads" in text
    assert hashlib.sha256(text.encode()).hexdigest() == (
        "d0af9043b309370b220498a438be9e080bab64d379390c1b85723fc436e4480b"
    )
    assert prompt["parent_kernel"] == load_triton_bf16_gemm_root().source
    assert "source_scaffold" not in prompt["strategy"]
    scaffold_prompt = json.loads(build_generation_prompt(GenerationRequest(
        parent=load_triton_bf16_gemm_root(),
        strategy=replace(strategy, scaffold_resource="bf16_gemm_gluon_wait1.py"),
        workload=BF16_GEMM_WORKLOAD, hardware=HARDWARE, profile=None,
    )))
    scaffold = scaffold_prompt["strategy"]["source_scaffold"]
    assert scaffold["source"] == load_triton_bf16_gemm_gluon_reference().source
    assert scaffold["sha256"] == hashlib.sha256(scaffold["source"].encode()).hexdigest()


def test_mcts_routes_one_structural_realization_without_changing_budget() -> None:
    root = load_triton_bf16_gemm_root()
    reference = load_triton_bf16_gemm_gluon_reference()
    strategy = next(item for item in catalog() if item.id == "clustered_gluon_wgmma_tma")

    class FixtureGenerator:
        def generate(self, request):
            assert request.strategy.id == strategy.id
            assert request.parent.source == root.source
            return GenerationResult("gluon-fixture", reference.source, reference, "fixture")

    class FixtureEvaluator:
        def evaluate(self, program, workload):
            assert triton_parent_stage(program.source) == "clustered_gluon"
            return EvaluationResult(
                ProposalStatus.VALID, program, "gluon-fixture", 1.0,
                BenchmarkResult((1.0,), 1.0),
            )

    root_evaluation = EvaluationResult(
        ProposalStatus.VALID, root, "triton-root", 0.0,
        BenchmarkResult((2.0,), 2.0),
    )
    result = MCTS(
        strategies=(strategy,), workload=BF16_GEMM_WORKLOAD,
        generator=FixtureGenerator(), evaluator=FixtureEvaluator(),
        prior_provider=UniformStrategyPrior(), budget=GenerationBudget(1),
        hardware=HARDWARE, config=MCTSConfig(max_repairs=0),
    ).run(root_evaluation)
    assert result.generations == 1
    assert result.best.program.source == reference.source
    assert tuple(result.root.actions) == (strategy.id,)


def test_inapplicable_catalog_ends_without_generation() -> None:
    root = load_triton_bf16_gemm_root()
    strategy = next(item for item in catalog() if item.id == "persistent_cluster_assignment")
    result = MCTS(
        strategies=(strategy,), workload=BF16_GEMM_WORKLOAD,
        generator=object(), evaluator=object(), prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(1), hardware=HARDWARE,
    ).run(EvaluationResult(
        ProposalStatus.VALID, root, "root", 0.0, BenchmarkResult((1.0,), 1.0),
    ))
    assert result.generations == 0
    assert not result.root.actions


def test_gluon_action_rejects_triton_realization_before_gpu_and_repairs() -> None:
    root = load_triton_bf16_gemm_root()
    gluon = load_triton_bf16_gemm_gluon_reference()
    strategy = next(item for item in catalog() if item.id == "clustered_gluon_wgmma_tma")

    class TwoStageGenerator:
        def __init__(self):
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            program = root if request.attempt == 0 else gluon
            return GenerationResult(f"generation-{len(self.requests)}", program.source,
                                    program, "prompt")

    class CountingEvaluator:
        def __init__(self):
            self.calls = 0

        def evaluate(self, program, workload):
            self.calls += 1
            assert program.source == gluon.source
            return EvaluationResult(ProposalStatus.VALID, program, "gluon", 1.0)

    generator = TwoStageGenerator()
    evaluator = CountingEvaluator()
    budget = GenerationBudget(2)
    outcome = run_proposal(
        generator=generator, evaluator=evaluator, budget=budget,
        request=GenerationRequest(root, strategy, BF16_GEMM_WORKLOAD, HARDWARE, None),
        max_repairs=1, max_infrastructure_retries=0,
    )
    rejected = outcome.attempts[0].evaluation
    assert budget.snapshot().used == 2
    assert evaluator.calls == 1
    assert rejected.status == ProposalStatus.INVALID
    assert rejected.invalid_reason == InvalidReason.STRATEGY_MISMATCH
    assert rejected.compile_status == CompileStatus.NOT_ATTEMPTED
    assert rejected.metadata["expected_candidate_stage"] == "clustered_gluon"
    assert rejected.metadata["observed_candidate_stage"] == "pointer_triton"
    assert generator.requests[1].previous_result is rejected
    repair = json.loads(build_generation_prompt(generator.requests[1]))["repair"]
    assert repair["previous_evaluation"]["expected_candidate_stage"] == "clustered_gluon"
    assert outcome.result.status == ProposalStatus.VALID


def test_baseline_strategy_without_stage_gate_still_reaches_evaluator() -> None:
    root = load_triton_bf16_gemm_root()
    strategy = next(item for item in catalog() if item.id == "tile_warp_co_design")

    class Generator:
        def generate(self, request):
            return GenerationResult("generation", root.source, root, "prompt")

    class Evaluator:
        def __init__(self):
            self.calls = 0

        def evaluate(self, program, workload):
            self.calls += 1
            return EvaluationResult(ProposalStatus.VALID, program, "root", 0.0)

    evaluator = Evaluator()
    outcome = run_proposal(
        generator=Generator(), evaluator=evaluator, budget=GenerationBudget(1),
        request=GenerationRequest(root, strategy, BF16_GEMM_WORKLOAD, HARDWARE, None),
        max_repairs=0, max_infrastructure_retries=0,
    )
    assert evaluator.calls == 1
    assert outcome.result.status == ProposalStatus.VALID


def test_mcts_does_not_admit_mislabeled_gluon_child() -> None:
    root = load_triton_bf16_gemm_root()
    strategy = next(item for item in catalog() if item.id == "clustered_gluon_wgmma_tma")

    class Generator:
        def generate(self, request):
            return GenerationResult("wrong-stage", root.source, root, "prompt")

    class Evaluator:
        def evaluate(self, program, workload):
            raise AssertionError("mismatched source must not reach the GPU evaluator")

    result = MCTS(
        strategies=(strategy,), workload=BF16_GEMM_WORKLOAD,
        generator=Generator(), evaluator=Evaluator(), prior_provider=UniformStrategyPrior(),
        budget=GenerationBudget(1), hardware=HARDWARE,
        config=MCTSConfig(max_repairs=0),
    ).run(EvaluationResult(
        ProposalStatus.VALID, root, "root", 0.0,
        BenchmarkResult((1.0,), 1.0),
    ))
    assert result.generations == 1
    assert len(result.nodes) == 1
    assert result.root.actions[strategy.id].invalid_proposal_count == 1
