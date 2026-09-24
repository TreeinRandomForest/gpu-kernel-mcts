from __future__ import annotations

from types import SimpleNamespace

import pytest

from kernel_mcts.cute_baseline import (
    _collect_timing_samples,
    _install_mainloop_pipeline_override,
    _install_epilogue_pipeline_override,
    _install_tma_load_policy_override,
    _install_wgmma_configuration_override,
    _launch_once,
    _tensor_helpers,
    run_same_worker_comparison,
    run_hopper_bf16_feasibility,
)
from kernel_mcts.cute_schedule import CuteSchedule
from kernel_mcts.domain import BenchmarkResult


class FakeKernel:
    @staticmethod
    def is_valid_dtypes(*_arguments):
        return False


class FakePipelineKernel:
    def __init__(self):
        self.atom_layout_mnk = (2, 1, 1)
        self.mma_warp_groups = 2
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = 256

    @staticmethod
    def _compute_stages(*_arguments):
        return 4, 4

    @staticmethod
    def _make_tma_atoms_and_tensors(
        _tensor, _smem_layout_staged, _smem_tile, mcast_dim
    ):
        return mcast_dim


def test_mainloop_override_preserves_epilogue_and_restores_descriptor() -> None:
    original = vars(FakePipelineKernel)["_compute_stages"]

    saved = _install_mainloop_pipeline_override(FakePipelineKernel, 2)

    assert FakePipelineKernel._compute_stages(None, None, None, 232448, 1) == (2, 4)
    setattr(FakePipelineKernel, "_compute_stages", saved)
    assert vars(FakePipelineKernel)["_compute_stages"] is original


def test_epilogue_override_preserves_mainloop_and_restores_descriptor() -> None:
    original = vars(FakePipelineKernel)["_compute_stages"]

    saved = _install_epilogue_pipeline_override(FakePipelineKernel, 2)

    assert FakePipelineKernel._compute_stages(None, None, None, 232448, 1) == (4, 2)
    setattr(FakePipelineKernel, "_compute_stages", saved)
    assert vars(FakePipelineKernel)["_compute_stages"] is original


def test_pipeline_overrides_compose_without_cross_talk() -> None:
    original = vars(FakePipelineKernel)["_compute_stages"]

    _install_mainloop_pipeline_override(FakePipelineKernel, 2)
    _install_epilogue_pipeline_override(FakePipelineKernel, 3)

    assert FakePipelineKernel._compute_stages(None, None, None, 232448, 1) == (2, 3)
    setattr(FakePipelineKernel, "_compute_stages", original)


def test_epilogue_override_rejects_unsupported_depth() -> None:
    with pytest.raises(ValueError, match="epilogue stages must be one of"):
        _install_epilogue_pipeline_override(FakePipelineKernel, 1)


def test_single_warp_group_override_updates_dependent_thread_counts() -> None:
    original = vars(FakePipelineKernel)["__init__"]

    saved = _install_wgmma_configuration_override(
        FakePipelineKernel, "single_warp_group"
    )
    kernel = FakePipelineKernel()

    assert kernel.atom_layout_mnk == (1, 1, 1)
    assert kernel.mma_warp_groups == 1
    assert kernel.threads_per_cta == 128
    setattr(FakePipelineKernel, "__init__", saved)
    assert vars(FakePipelineKernel)["__init__"] is original


def test_non_multicast_tma_override_forces_single_cta_copy() -> None:
    original = vars(FakePipelineKernel)["_make_tma_atoms_and_tensors"]

    saved = _install_tma_load_policy_override(
        FakePipelineKernel, "non_multicast"
    )

    assert FakePipelineKernel._make_tma_atoms_and_tensors(None, None, None, 2) == 1
    setattr(FakePipelineKernel, "_make_tma_atoms_and_tensors", saved)
    assert vars(FakePipelineKernel)["_make_tma_atoms_and_tensors"] is original


def test_tma_override_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="unsupported TMA load policy"):
        _install_tma_load_policy_override(FakePipelineKernel, "unknown")


def test_feasibility_runner_uses_fixed_bf16_contract_and_marks_limitations(tmp_path) -> None:
    calls = []
    original_validator = FakeKernel.is_valid_dtypes

    def run(**arguments):
        calls.append(arguments)
        assert FakeKernel.is_valid_dtypes(
            arguments["a_dtype"],
            arguments["b_dtype"],
            arguments["acc_dtype"],
            arguments["c_dtype"],
            arguments["a_major"],
            arguments["b_major"],
        )
        return 123.5

    example = SimpleNamespace(HopperWgmmaGemmKernel=FakeKernel, run=run)
    cutlass = SimpleNamespace(BFloat16="bf16", Float32="fp32")

    result = run_hopper_bf16_feasibility(
        tmp_path / "not-read-for-injected-module.py",
        example_module=example,
        cutlass_module=cutlass,
    )

    assert calls == [
        {
            "mnkl": (4096, 4096, 4096, 1),
            "a_dtype": "bf16",
            "b_dtype": "bf16",
            "c_dtype": "bf16",
            "acc_dtype": "fp32",
            "a_major": "k",
            "b_major": "k",
            "c_major": "n",
            "tile_shape_mn": (128, 256),
            "cluster_shape_mn": (1, 1),
            "tolerance": 0.02,
            "warmup_iterations": 10,
            "iterations": 30,
            "skip_ref_check": False,
            "use_cold_l2": False,
        }
    ]
    assert result["correctness"]["success"] is True
    assert result["benchmark"]["aggregate_mean_us"] == 123.5
    assert result["comparable_to_repository_baselines"] is False
    assert len(result["comparability_blockers"]) == 3
    assert FakeKernel.is_valid_dtypes is original_validator


def test_feasibility_runner_passes_typed_schedule_to_pinned_example(tmp_path) -> None:
    calls = []

    def run(**arguments):
        calls.append(arguments)
        return 100.0

    example = SimpleNamespace(HopperWgmmaGemmKernel=FakeKernel, run=run)
    cutlass = SimpleNamespace(BFloat16="bf16", Float32="fp32")
    schedule = CuteSchedule(64, 128, 2, 1)

    result = run_hopper_bf16_feasibility(
        tmp_path / "unused.py",
        schedule=schedule,
        example_module=example,
        cutlass_module=cutlass,
    )

    assert calls[0]["tile_shape_mn"] == (64, 128)
    assert calls[0]["cluster_shape_mn"] == (2, 1)
    assert result["contract"]["schedule_id"] == schedule.configuration_id


def test_invalid_schedule_is_rejected_before_example_execution(tmp_path) -> None:
    calls = []
    example = SimpleNamespace(
        HopperWgmmaGemmKernel=FakeKernel,
        run=lambda **arguments: calls.append(arguments),
    )
    cutlass = SimpleNamespace(BFloat16="bf16", Float32="fp32")

    with pytest.raises(ValueError, match="invalid CuTe schedule"):
        run_hopper_bf16_feasibility(
            tmp_path / "unused.py",
            schedule=CuteSchedule(96, 128, 1, 1),
            example_module=example,
            cutlass_module=cutlass,
        )

    assert calls == []


def test_collects_individual_samples_after_one_warmup_sequence() -> None:
    clock = [0.0]
    launches = []

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing is True
            self.time = None

        def record(self):
            self.time = clock[0]

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return other.time - self.time

    synchronizations = []
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            Event=Event,
            synchronize=lambda: synchronizations.append(True),
        )
    )

    def launch(value):
        launches.append(value)
        clock[0] += 0.25

    samples = _collect_timing_samples(
        launch,
        {
            "iterations": 3,
            "warmup_iterations": 2,
            "workspace_generator": lambda: SimpleNamespace(
                args=("workspace",), kwargs={}
            ),
            "workspace_count": 1,
        },
        torch,
    )

    assert samples == [250.0, 250.0, 250.0]
    assert synchronizations == [True]
    assert launches == ["workspace"] * 5


def test_profile_launch_executes_exactly_once_without_timing() -> None:
    launches = []
    synchronizations = []
    torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: synchronizations.append(True))
    )

    _launch_once(
        lambda value: launches.append(value),
        {
            "workspace_generator": lambda: SimpleNamespace(
                args=("workspace",), kwargs={}
            )
        },
        torch,
    )

    assert launches == ["workspace"]
    assert synchronizations == [True]


def test_pinned_example_tensor_helpers_are_reached_through_cutlass_module() -> None:
    helpers = SimpleNamespace(create_and_permute_torch_tensor=object())
    imports = []

    def import_module(name):
        imports.append(name)
        return helpers

    assert _tensor_helpers(import_module) is helpers
    assert imports == ["cutlass.torch"]


def test_runtime_hooks_do_not_require_example_module_torch_attribute() -> None:
    example = SimpleNamespace()

    assert not hasattr(example, "torch")


def test_same_worker_comparison_records_order_manifest_and_ratios() -> None:
    def vendor_runner():
        return {
            "cublas": {
                "correctness": {"success": True},
                "benchmark": BenchmarkResult((2.0, 2.0), 2.0),
            },
            "cutlass": {
                "correctness": {"success": True},
                "benchmark": BenchmarkResult((4.0, 4.0), 4.0),
            },
        }

    def cute_runner():
        return {
            "correctness": {"success": True},
            "benchmark": {"timings_us": [3.0, 3.0], "median_us": 3.0},
            "comparable_to_repository_baselines": True,
        }

    result = run_same_worker_comparison(
        vendor_runner, cute_runner, {"manifest_id": "manifest-1"}
    )

    assert result["execution_order"] == ["cublas", "cutlass", "cute_dsl"]
    assert result["environment_manifest"]["manifest_id"] == "manifest-1"
    assert result["comparison"]["latency_ratio_vs_cublas"] == {
        "cublas": 1.0,
        "cutlass": 2.0,
        "cute_dsl": 1.5,
    }
    assert result["comparison"]["speedup_vs_cutlass"] == 4.0 / 3.0
