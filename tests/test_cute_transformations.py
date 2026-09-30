from __future__ import annotations

import pytest

from kernel_mcts.cute_independent import make_independent_cute_gemm
from kernel_mcts.cute_mutations import (
    _legacy_enumerate_independent_cute_mutations,
    enumerate_independent_cute_mutations,
)
from kernel_mcts.cute_transformations import INDEPENDENT_STRUCTURAL_TRANSFORMATIONS


@pytest.mark.parametrize(
    "parent",
    (
        make_independent_cute_gemm(),
        make_independent_cute_gemm(pipeline_stages=2),
        make_independent_cute_gemm(swizzle_bytes=64),
        make_independent_cute_gemm(cluster_m=2),
        make_independent_cute_gemm(tile_m=128, tile_n=256),
        make_independent_cute_gemm(tile_m=64, tile_n=128),
        make_independent_cute_gemm(tile_m=128, tile_n=128),
        make_independent_cute_gemm(mainloop_schedule="prefetch"),
        make_independent_cute_gemm(
            mainloop_schedule="prefetch",
            producer_consumer_mode="warp_specialized",
        ),
        make_independent_cute_gemm(
            tile_m=128,
            mainloop_schedule="prefetch",
            producer_consumer_mode="warp_specialized",
        ),
    ),
)
def test_descriptor_graph_preserves_validated_mutation_order(parent) -> None:
    current = enumerate_independent_cute_mutations(parent)
    legacy = _legacy_enumerate_independent_cute_mutations(parent)

    def identity(proposal):
        return (
            proposal.strategy_id,
            dict(proposal.parameters),
            proposal.candidate.configuration_hash,
        )

    assert [identity(item) for item in current] == [identity(item) for item in legacy]
    assert all(item.transformation_id is not None for item in current)


def test_structural_descriptors_declare_generalization_contract() -> None:
    identifiers = [
        transformation.transformation_id
        for transformation in INDEPENDENT_STRUCTURAL_TRANSFORMATIONS
    ]

    assert len(identifiers) == len(set(identifiers))
    assert all(
        transformation.operation_families == ("gemm",)
        and transformation.dtypes == ("bfloat16",)
        and transformation.required_capabilities == ("sm90a", "tma", "wgmma")
        and transformation.parameter_schema
        and transformation.coupled_fields
        for transformation in INDEPENDENT_STRUCTURAL_TRANSFORMATIONS
    )
