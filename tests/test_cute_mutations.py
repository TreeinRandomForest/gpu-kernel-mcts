from __future__ import annotations

import pytest

from kernel_mcts.cute_mutations import (
    CUTE_MUTATION_STRATEGIES,
    CHANGE_CLUSTER_SHAPE,
    CHANGE_CTA_TILE,
    CHANGE_EPILOGUE_STAGES,
    CHANGE_PIPELINE_STAGES,
    CuteMutationGenerator,
    enumerate_cute_mutations,
    mutate_cute_program,
)
from kernel_mcts.cute_program import (
    REFERENCE_CUTE_GEMM,
    CuteGemmProgram,
    PinnedCuteGemmRenderer,
)
from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD
from kernel_mcts.generation import GenerationRequest, ProposalBudgetKind


def test_enumerates_stable_valid_one_hop_neighborhood() -> None:
    proposals = enumerate_cute_mutations(REFERENCE_CUTE_GEMM)

    assert [proposal.strategy_id for proposal in proposals] == [
        CHANGE_CTA_TILE,
        CHANGE_CTA_TILE,
        CHANGE_CLUSTER_SHAPE,
        CHANGE_CLUSTER_SHAPE,
        CHANGE_PIPELINE_STAGES,
        CHANGE_PIPELINE_STAGES,
    ]
    assert all(proposal.validation.valid for proposal in proposals)
    assert len({proposal.candidate.configuration_hash for proposal in proposals}) == 6
    assert all(
        PinnedCuteGemmRenderer().render(proposal.candidate).backend == "cute_dsl"
        for proposal in proposals
    )


def test_mutation_records_complete_transformation_evidence() -> None:
    proposal = mutate_cute_program(
        REFERENCE_CUTE_GEMM,
        CHANGE_CTA_TILE,
        {"tile_m": 128, "tile_n": 128},
    )

    assert proposal.changed_fields == {
        "tile_n": {"before": 256, "after": 128}
    }
    serialized = proposal.as_dict()
    assert serialized["proposal_mechanism"] == "typed_mutation"
    assert serialized["static_validation"] == {"valid": True, "violations": []}
    assert serialized["transformation"]["parent_configuration_hash"] == (
        REFERENCE_CUTE_GEMM.configuration_hash
    )
    assert serialized["configuration_hash"] == proposal.candidate.configuration_hash


def test_different_mutation_paths_reach_same_canonical_state() -> None:
    tile_first = mutate_cute_program(
        REFERENCE_CUTE_GEMM,
        CHANGE_CTA_TILE,
        {"tile_m": 128, "tile_n": 128},
    ).candidate
    tile_then_cluster = mutate_cute_program(
        tile_first,
        CHANGE_CLUSTER_SHAPE,
        {"cluster_m": 2, "cluster_n": 1},
    ).candidate
    cluster_first = mutate_cute_program(
        REFERENCE_CUTE_GEMM,
        CHANGE_CLUSTER_SHAPE,
        {"cluster_m": 2, "cluster_n": 1},
    ).candidate
    cluster_then_tile = mutate_cute_program(
        cluster_first,
        CHANGE_CTA_TILE,
        {"tile_m": 128, "tile_n": 128},
    ).candidate

    assert tile_then_cluster == cluster_then_tile
    assert tile_then_cluster.configuration_hash == cluster_then_tile.configuration_hash


def test_pipeline_and_tile_mutation_paths_transpose() -> None:
    pipeline_first = mutate_cute_program(
        REFERENCE_CUTE_GEMM,
        CHANGE_PIPELINE_STAGES,
        {"pipeline_stages": 3},
    ).candidate
    pipeline_then_tile = mutate_cute_program(
        pipeline_first,
        CHANGE_CTA_TILE,
        {"tile_m": 128, "tile_n": 128},
    ).candidate
    tile_first = mutate_cute_program(
        REFERENCE_CUTE_GEMM,
        CHANGE_CTA_TILE,
        {"tile_m": 128, "tile_n": 128},
    ).candidate
    tile_then_pipeline = mutate_cute_program(
        tile_first,
        CHANGE_PIPELINE_STAGES,
        {"pipeline_stages": 3},
    ).candidate

    assert pipeline_then_tile == tile_then_pipeline
    assert pipeline_then_tile.configuration_hash == (
        tile_then_pipeline.configuration_hash
    )


def test_pipeline_neighborhood_excludes_explicit_stage_four_alias() -> None:
    pipeline_proposals = [
        proposal
        for proposal in enumerate_cute_mutations(REFERENCE_CUTE_GEMM)
        if proposal.strategy_id == CHANGE_PIPELINE_STAGES
    ]

    assert [proposal.candidate.pipeline_stages for proposal in pipeline_proposals] == [
        2,
        3,
    ]


def test_epilogue_neighborhood_is_limited_to_validated_schedule() -> None:
    assert not any(
        proposal.strategy_id == CHANGE_EPILOGUE_STAGES
        for proposal in enumerate_cute_mutations(REFERENCE_CUTE_GEMM)
    )

    validated_parent = CuteGemmProgram(128, 256, 2, 1)
    epilogue_proposals = [
        proposal
        for proposal in enumerate_cute_mutations(validated_parent)
        if proposal.strategy_id == CHANGE_EPILOGUE_STAGES
    ]

    assert [
        proposal.candidate.epilogue_stages for proposal in epilogue_proposals
    ] == [2, 3]
    assert all(proposal.validation.valid for proposal in epilogue_proposals)


def test_explicit_epilogue_stage_four_alias_is_not_a_mutation() -> None:
    proposal = mutate_cute_program(
        CuteGemmProgram(128, 256, 2, 1),
        CHANGE_EPILOGUE_STAGES,
        {"epilogue_stages": 4},
    )

    assert proposal.validation.valid is False
    assert proposal.validation.violations[0].code == "unsupported_structural_value"

def test_rejects_noop_unknown_and_malformed_mutations() -> None:
    with pytest.raises(ValueError, match="must change"):
        mutate_cute_program(
            REFERENCE_CUTE_GEMM,
            CHANGE_CTA_TILE,
            {"tile_m": 128, "tile_n": 256},
        )
    with pytest.raises(ValueError, match="unknown"):
        mutate_cute_program(REFERENCE_CUTE_GEMM, "unknown", {})
    with pytest.raises(ValueError, match="exactly"):
        mutate_cute_program(
            REFERENCE_CUTE_GEMM,
            CHANGE_CLUSTER_SHAPE,
            {"cluster_m": 2},
        )


def test_generator_emits_distinct_deterministic_candidates_without_llm_calls() -> None:
    generator = CuteMutationGenerator()
    request = GenerationRequest(
        PinnedCuteGemmRenderer().render(REFERENCE_CUTE_GEMM),
        CUTE_MUTATION_STRATEGIES[0],
        BF16_GEMM_WORKLOAD,
        {},
        None,
    )

    first = generator.generate(request)
    second = generator.generate(request)

    assert generator.proposal_budget_kind(request) == ProposalBudgetKind.MUTATION
    assert first.program != second.program
    assert first.metadata["llm_call"] is False
    assert first.metadata["proposal_mechanism"] == "typed_mutation"
    assert generator.can_generate(request) is False


def test_generator_exposes_epilogue_mutations_only_at_validated_schedule() -> None:
    generator = CuteMutationGenerator()
    strategy = next(
        item for item in CUTE_MUTATION_STRATEGIES
        if item.id == CHANGE_EPILOGUE_STAGES
    )
    root_request = GenerationRequest(
        PinnedCuteGemmRenderer().render(REFERENCE_CUTE_GEMM),
        strategy,
        BF16_GEMM_WORKLOAD,
        {},
        None,
    )
    validated_parent = CuteGemmProgram(128, 256, 2, 1)
    validated_request = GenerationRequest(
        PinnedCuteGemmRenderer().render(validated_parent),
        strategy,
        BF16_GEMM_WORKLOAD,
        {},
        None,
    )

    assert generator.can_generate(root_request) is False
    first = generator.generate(validated_request)
    second = generator.generate(validated_request)

    assert first.program is not None
    assert second.program is not None
    assert first.program != second.program
    assert generator.can_generate(validated_request) is False
