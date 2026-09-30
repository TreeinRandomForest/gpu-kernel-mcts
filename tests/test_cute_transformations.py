from __future__ import annotations

from kernel_mcts.cute_independent import make_independent_cute_gemm
from kernel_mcts.cute_mutations import enumerate_independent_cute_mutations
from kernel_mcts.cute_transformations import INDEPENDENT_STRUCTURAL_TRANSFORMATIONS


def test_descriptor_graph_preserves_validated_root_proposal_order() -> None:
    proposals = enumerate_independent_cute_mutations(make_independent_cute_gemm())

    assert [proposal.strategy_id for proposal in proposals] == [
        "change_cta_tile",
        "change_cta_tile",
        "change_cta_tile",
        "change_cluster_shape",
        "change_shared_memory_swizzle",
        "change_pipeline_stages",
        "change_pipeline_stages",
        "change_mainloop_schedule",
    ]
    assert all(proposal.transformation_id is not None for proposal in proposals)
    assert all(proposal.validation.valid for proposal in proposals)


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
