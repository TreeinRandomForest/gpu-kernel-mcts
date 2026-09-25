from __future__ import annotations

from dataclasses import replace

import pytest

from kernel_mcts.cute_independent import (
    make_independent_cute_gemm,
    make_independent_tma_smem_mainloop,
    render_independent_cute_gemm_contract,
    render_independent_tma_smem_contract,
    validate_independent_cute_gemm,
    validate_independent_tma_smem_mainloop,
)


def test_initial_tma_smem_variants_are_valid_distinct_and_deterministic() -> None:
    sw128 = make_independent_tma_smem_mainloop(swizzle_bytes=128)
    sw64 = make_independent_tma_smem_mainloop(swizzle_bytes=64)

    assert validate_independent_tma_smem_mainloop(sw128).valid is True
    assert validate_independent_tma_smem_mainloop(sw64).valid is True
    assert sw128.configuration_hash != sw64.configuration_hash
    assert sw128.shared_memory_bytes == 147_456
    assert sw128.canonical_json() == make_independent_tma_smem_mainloop(
        swizzle_bytes=128
    ).canonical_json()


def test_contract_coordinates_operand_tiles_multicast_and_storage() -> None:
    plan = make_independent_tma_smem_mainloop()

    assert (plan.a_copy.tile_rows, plan.a_copy.tile_columns) == (128, 64)
    assert (plan.b_copy.tile_rows, plan.b_copy.tile_columns) == (64, 256)
    assert plan.a_copy.multicast_axis == "none"
    assert plan.b_copy.multicast_axis == "cluster_m"
    assert plan.a_copy.storage_offset_bytes == 0
    assert plan.b_copy.storage_offset_bytes == 49_152
    assert plan.barrier_slots == plan.pipeline_stages


def test_partial_swizzle_transition_is_statically_rejected() -> None:
    plan = make_independent_tma_smem_mainloop()
    partial = replace(
        plan,
        b_copy=replace(plan.b_copy, swizzle_bytes=64),
    )

    result = validate_independent_tma_smem_mainloop(partial)

    assert result.valid is False
    assert "partial_layout_transition" in {
        violation.code for violation in result.violations
    }


def test_incompatible_multicast_is_statically_rejected() -> None:
    plan = make_independent_tma_smem_mainloop()
    invalid = replace(
        plan,
        b_copy=replace(plan.b_copy, multicast_axis="none"),
    )

    result = validate_independent_tma_smem_mainloop(invalid)

    assert result.valid is False
    assert "incompatible_multicast_partition" in {
        violation.code for violation in result.violations
    }


def test_overlapping_operand_storage_is_statically_rejected() -> None:
    plan = make_independent_tma_smem_mainloop()
    overlapping = replace(
        plan,
        b_copy=replace(plan.b_copy, storage_offset_bytes=16_384),
    )

    result = validate_independent_tma_smem_mainloop(overlapping)

    assert result.valid is False
    assert "overlapping_operand_storage" in {
        violation.code for violation in result.violations
    }


def test_renderer_is_deterministic_and_rejects_illegal_contracts() -> None:
    plan = make_independent_tma_smem_mainloop()
    first = render_independent_tma_smem_contract(plan)
    second = render_independent_tma_smem_contract(plan)

    assert first == second
    assert plan.configuration_hash in first
    assert "not an executable kernel" in first
    compile(first, "independent_tma_smem_contract.py", "exec")

    invalid = replace(plan, barrier_slots=2)
    with pytest.raises(ValueError, match="arrival-barrier"):
        render_independent_tma_smem_contract(invalid)


def test_complete_kernel_contract_coordinates_wgmma_and_epilogue() -> None:
    kernel = make_independent_cute_gemm()

    assert validate_independent_cute_gemm(kernel).valid is True
    assert kernel.consumer.instruction_m * kernel.consumer.warp_groups_m == 128
    assert kernel.consumer.instruction_n * kernel.consumer.warp_groups_n == 256
    assert kernel.mainloop.tile_k % kernel.consumer.instruction_k == 0
    assert kernel.epilogue.accumulator_source == "registers"
    assert kernel.epilogue.store_kind == "tma"
    assert kernel.shared_memory_bytes == 180_224


def test_complete_kernel_identity_includes_memory_layout() -> None:
    sw128 = make_independent_cute_gemm(swizzle_bytes=128)
    sw64 = make_independent_cute_gemm(swizzle_bytes=64)

    assert sw128.configuration_hash != sw64.configuration_hash
    assert validate_independent_cute_gemm(sw128).valid is True
    assert validate_independent_cute_gemm(sw64).valid is True


def test_wgmma_must_cover_the_complete_cta_tile() -> None:
    kernel = make_independent_cute_gemm()
    invalid = replace(
        kernel,
        consumer=replace(kernel.consumer, warp_groups_m=1),
    )

    result = validate_independent_cute_gemm(invalid)

    assert result.valid is False
    assert "incomplete_wgmma_tile_coverage" in {
        violation.code for violation in result.violations
    }


def test_epilogue_storage_must_not_overlap_mainloop() -> None:
    kernel = make_independent_cute_gemm()
    invalid = replace(
        kernel,
        epilogue=replace(kernel.epilogue, storage_offset_bytes=0),
    )

    result = validate_independent_cute_gemm(invalid)

    assert result.valid is False
    assert "overlapping_mainloop_epilogue_storage" in {
        violation.code for violation in result.violations
    }


def test_complete_contract_renderer_is_deterministic_but_not_executable() -> None:
    kernel = make_independent_cute_gemm()
    rendered = render_independent_cute_gemm_contract(kernel)

    assert rendered == render_independent_cute_gemm_contract(kernel)
    assert kernel.configuration_hash in rendered
    assert "future lowering" in rendered
    compile(rendered, "independent_cute_gemm_contract.py", "exec")


def test_execution_schedule_makes_agents_buffers_and_dependencies_explicit() -> None:
    kernel = make_independent_cute_gemm()

    assert [agent.agent_id for agent in kernel.execution.agents] == [
        "tma_load_agent",
        "wgmma_agents",
        "epilogue_agents",
        "tma_store_agent",
    ]
    assert [buffer.stages for buffer in kernel.execution.buffers] == [3, 3, 4]
    assert [phase.phase_id for phase in kernel.execution.phases] == [
        "tma_load",
        "wgmma",
        "epilogue_stage",
        "tma_store",
    ]
    assert kernel.execution.phases[1].waits_for == ("mainloop_full",)


def test_execution_schedule_rejects_unknown_shared_buffer() -> None:
    kernel = make_independent_cute_gemm()
    phases = list(kernel.execution.phases)
    phases[1] = replace(phases[1], reads=("shared_missing",))
    invalid = replace(
        kernel,
        execution=replace(kernel.execution, phases=tuple(phases)),
    )

    result = validate_independent_cute_gemm(invalid)

    assert result.valid is False
    assert "unknown_phase_buffer" in {
        violation.code for violation in result.violations
    }


def test_execution_schedule_rejects_inconsistent_buffer_stage_count() -> None:
    kernel = make_independent_cute_gemm()
    buffers = list(kernel.execution.buffers)
    buffers[0] = replace(buffers[0], stages=2)
    invalid = replace(
        kernel,
        execution=replace(kernel.execution, buffers=tuple(buffers)),
    )

    result = validate_independent_cute_gemm(invalid)

    assert result.valid is False
    assert "inconsistent_buffer_rings" in {
        violation.code for violation in result.violations
    }


def test_execution_schedule_rejects_dependency_cycle() -> None:
    kernel = make_independent_cute_gemm()
    phases = list(kernel.execution.phases)
    phases[0] = replace(phases[0], waits_for=("epilogue_empty",))
    invalid = replace(
        kernel,
        execution=replace(kernel.execution, phases=tuple(phases)),
    )

    result = validate_independent_cute_gemm(invalid)

    assert result.valid is False
    assert "cyclic_execution_dependencies" in {
        violation.code for violation in result.violations
    }
