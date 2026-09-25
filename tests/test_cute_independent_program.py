from __future__ import annotations

from kernel_mcts.cute_independent import make_independent_cute_gemm
from kernel_mcts.cute_independent_program import (
    IndependentCuteGemmRenderer,
    independent_cute_gemm_from_source,
)


def test_independent_renderer_is_deterministic_and_round_trips_typed_state() -> None:
    representation = make_independent_cute_gemm()
    renderer = IndependentCuteGemmRenderer()

    first = renderer.render(representation)
    second = renderer.render(representation)

    assert first == second
    assert first.backend == "cute_dsl"
    assert independent_cute_gemm_from_source(first.source) == representation
    assert "REPOSITORY_CONTRACT = True" in first.source
    assert "DEBUG_STAGE = 'wgmma_full_workload'" in first.source


def test_independent_repository_renderer_uses_canonical_inputs_and_reference() -> None:
    source = IndependentCuteGemmRenderer().render(
        make_independent_cute_gemm()
    ).source

    assert '"/usr/local/bin/kernel-mcts-bf16-inputs"' in source
    assert '"/usr/local/lib/kernel-mcts-bf16-reference.so"' in source
    assert '"comparable_to_repository_baselines": True' in source
    assert '"implementation": "independent_cute_gemm_v1"' in source
