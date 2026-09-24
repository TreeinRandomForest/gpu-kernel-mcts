from __future__ import annotations

import json

import pytest

from kernel_mcts.cute_program import (
    CUTE_GEMM_SCHEMA_VERSION,
    REFERENCE_CUTE_GEMM,
    CuteGemmProgram,
    PinnedCuteGemmRenderer,
    validate_cute_gemm_program,
)


def test_reference_representation_is_canonical_and_workload_independent() -> None:
    encoded = REFERENCE_CUTE_GEMM.canonical_json()
    value = json.loads(encoded)

    assert value["schema_version"] == CUTE_GEMM_SCHEMA_VERSION
    assert value["tile_m"] == 128
    assert value["schema_version"] == 3
    assert value["shared_memory_swizzle"] == "heuristic"
    assert "wgmma_inflight_groups" not in value
    assert "dtype" not in value
    assert "warmup_count" not in value
    assert encoded == REFERENCE_CUTE_GEMM.canonical_json()
    assert len(REFERENCE_CUTE_GEMM.configuration_hash) == 64


def test_configuration_identity_changes_with_structure() -> None:
    clustered = CuteGemmProgram(128, 256, 2, 1)

    assert clustered.configuration_hash != REFERENCE_CUTE_GEMM.configuration_hash
    assert clustered.configuration_hash == CuteGemmProgram(128, 256, 2, 1).configuration_hash


def test_legality_returns_structured_violations() -> None:
    program = CuteGemmProgram(
        96,
        128,
        1,
        1,
        pipeline_stages=5,
    )

    result = validate_cute_gemm_program(program)

    assert result.valid is False
    assert {item.code for item in result.violations} == {
        "invalid_schedule",
        "unsupported_structural_value",
    }
    assert result.as_dict()["valid"] is False


def test_renderer_is_deterministic_and_backend_typed() -> None:
    renderer = PinnedCuteGemmRenderer()
    first = renderer.render(REFERENCE_CUTE_GEMM)
    second = renderer.render(REFERENCE_CUTE_GEMM)

    assert first == second
    assert first.backend == "cute_dsl"
    assert "tile_shape_mn=(128, 256)" in first.source
    assert "cluster_shape_mn=(1, 1)" in first.source
    assert REFERENCE_CUTE_GEMM.configuration_hash in first.source


def test_renderer_rejects_unimplemented_structural_controls() -> None:
    with pytest.raises(ValueError, match="pipeline_stages must be one of"):
        PinnedCuteGemmRenderer().render(
            CuteGemmProgram(128, 256, 1, 1, pipeline_stages=5)
        )


@pytest.mark.parametrize("pipeline_stages", (2, 3, 4))
def test_renderer_supports_bounded_mainloop_pipeline_override(
    pipeline_stages: int,
) -> None:
    program = CuteGemmProgram(
        128,
        256,
        1,
        1,
        pipeline_stages=pipeline_stages,
    )

    legality = validate_cute_gemm_program(program)
    rendered = PinnedCuteGemmRenderer().render(program)

    assert legality.valid is True
    assert f"pipeline_stages = {pipeline_stages}" in rendered.source
    assert "_, epi_stage = pinned_mainloop_compute_stages(" in rendered.source
    assert "return pipeline_stages, epi_stage" in rendered.source


def test_default_renderer_preserves_pinned_pipeline_heuristic() -> None:
    rendered = PinnedCuteGemmRenderer().render(REFERENCE_CUTE_GEMM)

    assert "pipeline_stages = None" in rendered.source
    assert "if pipeline_stages is not None:" in rendered.source
    assert "epilogue_stages = None" in rendered.source


@pytest.mark.parametrize("epilogue_stages", (2, 3))
def test_renderer_supports_validated_epilogue_pipeline_depth(
    epilogue_stages: int,
) -> None:
    program = CuteGemmProgram(
        128,
        256,
        2,
        1,
        epilogue_stages=epilogue_stages,
    )

    legality = validate_cute_gemm_program(program)
    rendered = PinnedCuteGemmRenderer().render(program)

    assert legality.valid is True
    assert f"epilogue_stages = {epilogue_stages}" in rendered.source
    assert "mainloop_stages, _ = pinned_epilogue_compute_stages(" in rendered.source
    assert "return mainloop_stages, epilogue_stages" in rendered.source


def test_epilogue_depth_is_limited_to_validated_schedule() -> None:
    program = CuteGemmProgram(128, 256, 1, 1, epilogue_stages=2)

    result = validate_cute_gemm_program(program)

    assert result.valid is False
    assert [item.code for item in result.violations] == [
        "incompatible_structural_values"
    ]


def test_explicit_epilogue_stage_four_alias_is_rejected() -> None:
    program = CuteGemmProgram(128, 256, 2, 1, epilogue_stages=4)

    with pytest.raises(ValueError, match="epilogue_stages must be one of"):
        PinnedCuteGemmRenderer().render(program)


def test_mainloop_and_epilogue_renderer_overrides_use_distinct_closures() -> None:
    program = CuteGemmProgram(
        128,
        256,
        2,
        1,
        pipeline_stages=3,
        epilogue_stages=2,
    )

    rendered = PinnedCuteGemmRenderer().render(program)

    assert "pinned_mainloop_compute_stages" in rendered.source
    assert "pinned_epilogue_compute_stages" in rendered.source
    compile(rendered.source, "rendered_cute_program.py", "exec")


def test_renderer_supports_single_warp_group_configuration() -> None:
    program = CuteGemmProgram(
        128,
        256,
        1,
        1,
        wgmma_configuration="single_warp_group",
    )

    legality = validate_cute_gemm_program(program)
    rendered = PinnedCuteGemmRenderer().render(program)

    assert legality.valid is True
    assert "wgmma_configuration = 'single_warp_group'" in rendered.source
    assert "self.atom_layout_mnk = (1, 1, 1)" in rendered.source
    assert "self.mma_warp_groups = 1" in rendered.source


def test_renderer_rejects_unknown_wgmma_configuration() -> None:
    program = CuteGemmProgram(
        128,
        256,
        1,
        1,
        wgmma_configuration="unknown",
    )

    with pytest.raises(ValueError, match="wgmma_configuration must be one of"):
        PinnedCuteGemmRenderer().render(program)


def test_renderer_supports_validated_sw64_shared_memory_layout() -> None:
    program = CuteGemmProgram(
        128,
        256,
        2,
        1,
        shared_memory_swizzle="sw64",
    )

    legality = validate_cute_gemm_program(program)
    rendered = PinnedCuteGemmRenderer().render(program)

    assert legality.valid is True
    assert "shared_memory_swizzle = 'sw64'" in rendered.source
    assert "select_sw64_layout_atom" in rendered.source
    assert '"MN_SW64" if selected_name.startswith("MN_") else "K_SW64"' in (
        rendered.source
    )
    compile(rendered.source, "rendered_cute_program.py", "exec")


def test_sw64_layout_is_limited_to_validated_schedule() -> None:
    program = CuteGemmProgram(
        128,
        256,
        1,
        1,
        shared_memory_swizzle="sw64",
    )

    result = validate_cute_gemm_program(program)

    assert result.valid is False
    assert [item.code for item in result.violations] == [
        "incompatible_structural_values"
    ]


def test_renderer_rejects_unknown_shared_memory_swizzle() -> None:
    program = CuteGemmProgram(
        128,
        256,
        2,
        1,
        shared_memory_swizzle="unknown",
    )

    with pytest.raises(ValueError, match="shared_memory_swizzle must be one of"):
        PinnedCuteGemmRenderer().render(program)
