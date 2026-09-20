from __future__ import annotations

import pytest

from kernel_mcts.cute_mutations import (
    CHANGE_CLUSTER_SHAPE,
    CHANGE_CTA_TILE,
    enumerate_cute_mutations,
    mutate_cute_program,
)
from kernel_mcts.cute_program import REFERENCE_CUTE_GEMM, PinnedCuteGemmRenderer


def test_enumerates_stable_valid_one_hop_neighborhood() -> None:
    proposals = enumerate_cute_mutations(REFERENCE_CUTE_GEMM)

    assert [proposal.strategy_id for proposal in proposals] == [
        CHANGE_CTA_TILE,
        CHANGE_CTA_TILE,
        CHANGE_CLUSTER_SHAPE,
        CHANGE_CLUSTER_SHAPE,
    ]
    assert all(proposal.validation.valid for proposal in proposals)
    assert len({proposal.candidate.configuration_hash for proposal in proposals}) == 4
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
