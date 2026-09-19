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
        pipeline_stages=3,
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
    with pytest.raises(ValueError, match="explicit pipeline stages"):
        PinnedCuteGemmRenderer().render(
            CuteGemmProgram(128, 256, 1, 1, pipeline_stages=3)
        )
