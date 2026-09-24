from __future__ import annotations

import hashlib
import importlib
import ctypes
import statistics
import subprocess
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping

from .domain import BenchmarkResult
from .cute_schedule import DEFAULT_CUTE_SCHEDULE, CuteSchedule, validate_cute_schedule
from .cute_diagnostics import describe_kernel_callable
from .cute_source_transform import (
    load_pinned_cute_gemm,
    validate_pinned_cute_gemm_source,
)
from .serialization import serialize_benchmark


DEFAULT_EXAMPLE = Path(
    "/opt/cutlass/examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py"
)
DEFAULT_INPUT_GENERATOR = Path("/usr/local/bin/kernel-mcts-bf16-inputs")
DEFAULT_REFERENCE_LIBRARY = Path("/usr/local/lib/kernel-mcts-bf16-reference.so")


def run_same_worker_comparison(
    vendor_runner: Callable[[], Mapping[str, object]],
    cute_runner: Callable[[], Mapping[str, Any]],
    environment_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Run all fixed baselines sequentially and summarize median latency ratios."""
    vendor = vendor_runner()
    cute = dict(cute_runner())
    implementations: dict[str, Any] = {}
    for name in ("cublas", "cutlass"):
        result = vendor[name]
        correctness = result["correctness"]
        if not isinstance(correctness, Mapping) or not correctness.get("success"):
            raise RuntimeError(f"{name} baseline failed correctness")
        benchmark = result["benchmark"]
        if not isinstance(benchmark, BenchmarkResult):
            raise TypeError(f"{name} baseline returned an invalid benchmark")
        implementations[name] = {
            "correctness": correctness,
            "benchmark": serialize_benchmark(benchmark),
        }
    if not cute.get("comparable_to_repository_baselines"):
        raise RuntimeError("CuTe DSL result did not complete the comparable contract")
    cute_correctness = cute.get("correctness")
    if not isinstance(cute_correctness, Mapping) or not cute_correctness.get("success"):
        raise RuntimeError("CuTe DSL baseline failed correctness")
    implementations["cute_dsl"] = cute

    medians = {
        name: float(result["benchmark"]["median_us"])
        for name, result in implementations.items()
    }
    cublas_median = medians["cublas"]
    return {
        "status": "ok",
        "benchmark_id": "bf16_gemm_4096_h100",
        "execution_order": ["cublas", "cutlass", "cute_dsl"],
        "environment_manifest": dict(environment_manifest),
        "implementations": implementations,
        "comparison": {
            "median_us": medians,
            "latency_ratio_vs_cublas": {
                name: median / cublas_median for name, median in medians.items()
            },
            "speedup_vs_cutlass": medians["cutlass"] / medians["cute_dsl"],
        },
    }


def run_hopper_bf16_feasibility(
    example_path: Path = DEFAULT_EXAMPLE,
    *,
    schedule: CuteSchedule = DEFAULT_CUTE_SCHEDULE,
    example_module: ModuleType | None = None,
    cutlass_module: ModuleType | None = None,
) -> Mapping[str, Any]:
    """Run the pinned Hopper example with its dtype guard extended to BF16.

    This is deliberately labelled a feasibility result: NVIDIA's example owns
    input generation, reference checking, and aggregated timing, so its result
    is not yet comparable with the repository's CUDA-event sample protocol.
    """
    _require_legal_schedule(schedule)
    example = example_module or _load_example(example_path)
    if cutlass_module is None:
        import cutlass as imported_cutlass

        cutlass_module = imported_cutlass

    kernel_type = example.HopperWgmmaGemmKernel
    original_validator = kernel_type.is_valid_dtypes

    def bf16_validator(a_dtype, b_dtype, acc_dtype, c_dtype, a_major, b_major):
        exact_bf16_contract = (
            a_dtype == cutlass_module.BFloat16
            and b_dtype == cutlass_module.BFloat16
            and c_dtype == cutlass_module.BFloat16
            and acc_dtype == cutlass_module.Float32
            and a_major == "k"
            and b_major == "k"
        )
        return exact_bf16_contract or original_validator(
            a_dtype, b_dtype, acc_dtype, c_dtype, a_major, b_major
        )

    kernel_type.is_valid_dtypes = staticmethod(bf16_validator)
    try:
        mean_us = example.run(
            mnkl=(4096, 4096, 4096, 1),
            a_dtype=cutlass_module.BFloat16,
            b_dtype=cutlass_module.BFloat16,
            c_dtype=cutlass_module.BFloat16,
            acc_dtype=cutlass_module.Float32,
            a_major="k",
            b_major="k",
            c_major="n",
            tile_shape_mn=(schedule.tile_m, schedule.tile_n),
            cluster_shape_mn=(schedule.cluster_m, schedule.cluster_n),
            tolerance=2.0e-2,
            warmup_iterations=10,
            iterations=30,
            skip_ref_check=False,
            use_cold_l2=False,
        )
    finally:
        kernel_type.is_valid_dtypes = original_validator

    return {
        "status": "ok",
        "implementation": "cute_dsl_hopper_dense_gemm_v4.5.1_adapted_bf16",
        "contract": {
            "mnkl": [4096, 4096, 4096, 1],
            "input_dtype": "bfloat16",
            "accumulator_dtype": "float32",
            "output_dtype": "bfloat16",
            "a_major": "k",
            "b_major": "k",
            "c_major": "n",
            "tile_shape_mn": [schedule.tile_m, schedule.tile_n],
            "cluster_shape_mn": [schedule.cluster_m, schedule.cluster_n],
            "schedule_id": schedule.configuration_id,
            "warmup_count": 10,
            "measurement_count": 30,
        },
        "correctness": {"success": True, "example_tolerance": 2.0e-2},
        "benchmark": {"aggregate_mean_us": float(mean_us)},
        "comparable_to_repository_baselines": False,
        "comparability_blockers": [
            "the NVIDIA example generates different deterministic inputs",
            "the NVIDIA example uses atol=0.02 and rtol=0.001 rather than the workload's separate tolerances",
            "the NVIDIA example returns an aggregate time rather than all CUDA-event samples",
        ],
        "example_sha256": _sha256(example_path) if example_path.is_file() else None,
    }


def run_hopper_bf16_comparable(
    example_path: Path = DEFAULT_EXAMPLE,
    input_generator: Path = DEFAULT_INPUT_GENERATOR,
    reference_library: Path = DEFAULT_REFERENCE_LIBRARY,
    *,
    schedule: CuteSchedule = DEFAULT_CUTE_SCHEDULE,
    pipeline_stages: int | None = None,
    wgmma_configuration: str = "pinned_default",
    wgmma_inflight_groups: int = 1,
    tma_load_policy: str = "auto_multicast",
    raise_on_correctness_failure: bool = True,
    capture_jit_diagnostics: bool = False,
    profile_single_launch: bool = False,
) -> Mapping[str, Any]:
    """Evaluate the pinned Hopper kernel under the repository benchmark contract."""
    import cutlass
    import torch

    _require_legal_schedule(schedule)
    if tma_load_policy != "auto_multicast":
        validate_pinned_cute_gemm_source(example_path)
    example = _load_example(
        example_path, wgmma_inflight_groups=wgmma_inflight_groups
    )
    if not input_generator.is_file():
        raise RuntimeError(f"BF16 input generator is unavailable: {input_generator}")
    reference = _CublasReference(reference_library)

    with tempfile.TemporaryDirectory(prefix="kernel-mcts-cute-inputs-") as directory:
        root = Path(directory)
        a_path = root / "a.bf16"
        b_path = root / "b.bf16"
        subprocess.run(
            [
                str(input_generator),
                "4096",
                "4096",
                "4096",
                "0",
                str(a_path),
                str(b_path),
            ],
            check=True,
            timeout=120,
        )
        a_cpu = torch.from_file(
            str(a_path), shared=False, size=4096 * 4096, dtype=torch.bfloat16
        ).reshape(4096, 4096, 1)
        b_cpu = torch.from_file(
            str(b_path), shared=False, size=4096 * 4096, dtype=torch.bfloat16
        ).reshape(4096, 4096, 1)
        c_cpu = torch.zeros((4096, 4096, 1), dtype=torch.bfloat16)
        result = _run_with_repository_hooks(
            example,
            cutlass,
            torch,
            (a_cpu, b_cpu, c_cpu),
            reference,
            schedule,
            raise_on_correctness_failure,
            pipeline_stages=pipeline_stages,
            wgmma_configuration=wgmma_configuration,
            wgmma_inflight_groups=wgmma_inflight_groups,
            tma_load_policy=tma_load_policy,
            capture_jit_diagnostics=capture_jit_diagnostics,
            profile_single_launch=profile_single_launch,
        )

    result["example_sha256"] = _sha256(example_path)
    return result


def _run_with_repository_hooks(
    example,
    cutlass,
    torch,
    inputs,
    reference,
    schedule: CuteSchedule,
    raise_on_correctness_failure: bool = True,
    pipeline_stages: int | None = None,
    wgmma_configuration: str = "pinned_default",
    wgmma_inflight_groups: int = 1,
    tma_load_policy: str = "auto_multicast",
    capture_jit_diagnostics: bool = False,
    profile_single_launch: bool = False,
) -> dict[str, Any]:
    kernel_type = example.HopperWgmmaGemmKernel
    tensor_helpers = _tensor_helpers()
    original_validator = kernel_type.is_valid_dtypes
    original_tensor_factory = tensor_helpers.create_and_permute_torch_tensor
    original_einsum = torch.einsum
    original_assert_close = torch.testing.assert_close
    original_benchmark = example.testing.benchmark
    original_compute_stages = _install_mainloop_pipeline_override(
        kernel_type, pipeline_stages
    )
    original_init = _install_wgmma_configuration_override(
        kernel_type, wgmma_configuration
    )
    original_tma_loader = _install_tma_load_policy_override(
        kernel_type, tma_load_policy
    )
    tensor_index = 0
    timings: list[float] = []
    correctness: dict[str, Any] = {}
    jit_diagnostics: dict[str, Any] = {}

    def bf16_validator(a_dtype, b_dtype, acc_dtype, c_dtype, a_major, b_major):
        return (
            a_dtype == cutlass.BFloat16
            and b_dtype == cutlass.BFloat16
            and c_dtype == cutlass.BFloat16
            and acc_dtype == cutlass.Float32
            and a_major == "k"
            and b_major == "k"
        ) or original_validator(
            a_dtype, b_dtype, acc_dtype, c_dtype, a_major, b_major
        )

    def tensor_factory(*_arguments, **_keywords):
        nonlocal tensor_index
        tensor = inputs[tensor_index % 3]
        tensor_index += 1
        return tensor.clone()

    def cublas_reference(equation, a, b):
        if equation != "mkl,nkl->mnl":
            return original_einsum(equation, a, b)
        a_gpu = a[..., 0].to(device="cuda", dtype=torch.bfloat16)
        b_gpu = b[..., 0].to(device="cuda", dtype=torch.bfloat16)
        c_gpu = torch.empty((4096, 4096), device="cuda", dtype=torch.bfloat16)
        reference(a_gpu.data_ptr(), b_gpu.data_ptr(), c_gpu.data_ptr(), 4096, 4096, 4096)
        torch.cuda.synchronize()
        return c_gpu.to(torch.float32).cpu().unsqueeze(-1)

    def repository_assert_close(actual, expected, **_ignored):
        actual_f32 = actual.to(torch.float32)
        expected_f32 = expected.to(torch.float32)
        error = (actual_f32 - expected_f32).abs()
        allowed = 2.0e-2 + 2.0e-2 * expected_f32.abs()
        finite = torch.isfinite(actual_f32)
        success = bool(torch.all(finite & (error <= allowed)).item())
        correctness.update(
            {
                "success": success,
                "maximum_error": float(error.max().item()),
                "mean_error": float(error.mean().item()),
                "failed_test_id": None if success else "fixed_shape",
                "reference_metadata": {
                    "implementation": "cuBLAS",
                    "compute_type": "CUBLAS_COMPUTE_32F",
                    "seed": 0,
                },
            }
        )
        if not success and raise_on_correctness_failure:
            raise AssertionError("CuTe DSL result failed repository correctness tolerances")

    def sample_benchmark(callable, **keywords):
        if capture_jit_diagnostics and not jit_diagnostics:
            jit_diagnostics.update(describe_kernel_callable(callable, keywords))
        if profile_single_launch:
            _launch_once(callable, keywords, torch)
            return 0.0
        timings.extend(_collect_timing_samples(callable, keywords, torch))
        return statistics.fmean(timings)

    kernel_type.is_valid_dtypes = staticmethod(bf16_validator)
    tensor_helpers.create_and_permute_torch_tensor = tensor_factory
    torch.einsum = cublas_reference
    torch.testing.assert_close = repository_assert_close
    example.testing.benchmark = sample_benchmark
    try:
        example.run(
            mnkl=(4096, 4096, 4096, 1),
            a_dtype=cutlass.BFloat16,
            b_dtype=cutlass.BFloat16,
            c_dtype=cutlass.BFloat16,
            acc_dtype=cutlass.Float32,
            a_major="k",
            b_major="k",
            c_major="n",
            tile_shape_mn=(schedule.tile_m, schedule.tile_n),
            cluster_shape_mn=(schedule.cluster_m, schedule.cluster_n),
            tolerance=2.0e-2,
            warmup_iterations=0 if profile_single_launch else 10,
            iterations=1 if profile_single_launch else 30,
            skip_ref_check=profile_single_launch,
            use_cold_l2=False,
        )
    finally:
        kernel_type.is_valid_dtypes = original_validator
        tensor_helpers.create_and_permute_torch_tensor = original_tensor_factory
        torch.einsum = original_einsum
        torch.testing.assert_close = original_assert_close
        example.testing.benchmark = original_benchmark
        kernel_type._compute_stages = original_compute_stages
        kernel_type.__init__ = original_init
        kernel_type._make_tma_atoms_and_tensors = original_tma_loader

    if profile_single_launch:
        return {
            "status": "ok",
            "profile_launch": True,
            "jit_diagnostics": jit_diagnostics,
        }
    if not correctness or len(timings) != 30:
        raise RuntimeError("CuTe DSL adapter did not complete the evaluation contract")
    result = {
        "status": "ok",
        "implementation": "cute_dsl_hopper_dense_gemm_v4.5.1_repository_contract",
        "contract": {
            "mnkl": [4096, 4096, 4096, 1],
            "input_dtype": "bfloat16",
            "accumulator_dtype": "float32",
            "output_dtype": "bfloat16",
            "a_major": "k",
            "b_major": "k",
            "c_major": "n",
            "tile_shape_mn": [schedule.tile_m, schedule.tile_n],
            "cluster_shape_mn": [schedule.cluster_m, schedule.cluster_n],
            "schedule_id": schedule.configuration_id,
            "pipeline_stages": pipeline_stages,
            "wgmma_configuration": wgmma_configuration,
            "wgmma_inflight_groups": wgmma_inflight_groups,
            "tma_load_policy": tma_load_policy,
            "rtol": 2.0e-2,
            "atol": 2.0e-2,
            "seed": 0,
            "warmup_count": 10,
            "measurement_count": 30,
        },
        "correctness": correctness,
        "benchmark": {
            "timings_us": timings,
            "median_us": statistics.median(timings),
            "mean_us": statistics.fmean(timings),
            "stddev_us": statistics.pstdev(timings),
            "min_us": min(timings),
            "max_us": max(timings),
        },
        "comparable_to_repository_baselines": True,
        "comparability_blockers": [],
    }
    if capture_jit_diagnostics:
        result["jit_diagnostics"] = jit_diagnostics
    return result


def _install_mainloop_pipeline_override(kernel_type, pipeline_stages: int | None):
    original_descriptor = vars(kernel_type)["_compute_stages"]
    if pipeline_stages is None:
        return original_descriptor
    pinned_compute_stages = kernel_type._compute_stages

    def compute_stages_with_mainloop_override(
        tile_shape_mnk, a_dtype, b_dtype, smem_capacity, occupancy
    ):
        _, epi_stage = pinned_compute_stages(
            tile_shape_mnk, a_dtype, b_dtype, smem_capacity, occupancy
        )
        return pipeline_stages, epi_stage

    kernel_type._compute_stages = staticmethod(compute_stages_with_mainloop_override)
    return original_descriptor


def _install_wgmma_configuration_override(kernel_type, configuration: str):
    original_descriptor = vars(kernel_type)["__init__"]
    if configuration == "pinned_default":
        return original_descriptor
    if configuration != "single_warp_group":
        raise ValueError(f"unsupported WGMMA configuration {configuration!r}")
    pinned_init = kernel_type.__init__

    def init_with_single_warp_group(self, *args, **kwargs):
        pinned_init(self, *args, **kwargs)
        self.atom_layout_mnk = (1, 1, 1)
        self.mma_warp_groups = 1
        self.threads_per_cta = self.num_threads_per_warp_group

    kernel_type.__init__ = init_with_single_warp_group
    return original_descriptor


def _install_tma_load_policy_override(kernel_type, policy: str):
    original_descriptor = vars(kernel_type)["_make_tma_atoms_and_tensors"]
    if policy == "auto_multicast":
        return original_descriptor
    if policy != "non_multicast":
        raise ValueError(f"unsupported TMA load policy {policy!r}")
    pinned_make_tma_atoms = kernel_type._make_tma_atoms_and_tensors

    def make_non_multicast_tma_atoms(
        tensor, smem_layout_staged, smem_tile, _mcast_dim
    ):
        return pinned_make_tma_atoms(
            tensor,
            smem_layout_staged,
            smem_tile,
            1,
        )

    kernel_type._make_tma_atoms_and_tensors = staticmethod(
        make_non_multicast_tma_atoms
    )
    return original_descriptor


def _collect_timing_samples(
    kernel_callable: Callable[..., Any], keywords: Mapping[str, Any], torch
) -> list[float]:
    if keywords.get("use_cuda_graphs", False) or keywords.get("use_cupti", False):
        raise ValueError("repository timing requires direct CUDA-event measurement")
    if int(keywords.get("workspace_count", 1)) != 1:
        raise ValueError("repository timing requires one reusable workspace")
    workspace = keywords.get("kernel_arguments")
    if workspace is None:
        generator = keywords.get("workspace_generator")
        if not callable(generator):
            raise ValueError("repository timing requires kernel arguments")
        workspace = generator()
    arguments = workspace.args
    named_arguments = workspace.kwargs
    warmups = int(keywords["warmup_iterations"])
    measurements = int(keywords["iterations"])

    for _ in range(warmups):
        kernel_callable(*arguments, **named_arguments)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(measurements):
        start.record()
        kernel_callable(*arguments, **named_arguments)
        stop.record()
        stop.synchronize()
        timings.append(float(start.elapsed_time(stop)) * 1_000.0)
    return timings


def _launch_once(
    kernel_callable: Callable[..., Any], keywords: Mapping[str, Any], torch
) -> None:
    workspace = keywords.get("kernel_arguments")
    if workspace is None:
        generator = keywords.get("workspace_generator")
        if not callable(generator):
            raise ValueError("CuTe profile launch requires kernel arguments")
        workspace = generator()
    kernel_callable(*workspace.args, **workspace.kwargs)
    torch.cuda.synchronize()


def _require_legal_schedule(schedule: CuteSchedule) -> None:
    reasons = validate_cute_schedule(schedule)
    if reasons:
        raise ValueError("invalid CuTe schedule: " + "; ".join(reasons))


class _CublasReference:
    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise RuntimeError(f"cuBLAS reference library is unavailable: {path}")
        library = ctypes.CDLL(str(path))
        function = library.kernel_mcts_bf16_reference
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_int
        self._library = library
        self._function = function

    def __call__(self, a: int, b: int, c: int, m: int, n: int, k: int) -> None:
        status = self._function(a, b, c, m, n, k)
        if status != 0:
            raise RuntimeError(f"cuBLAS reference failed with status {status}")


def _tensor_helpers(import_module=importlib.import_module):
    """Import the helper explicitly before the pinned example's local import runs."""
    return import_module("cutlass.torch")


def _load_example(
    path: Path, *, wgmma_inflight_groups: int = 1
) -> ModuleType:
    if not path.is_file():
        raise RuntimeError(f"pinned CuTe DSL example is unavailable: {path}")
    return load_pinned_cute_gemm(
        path, wgmma_inflight_groups=wgmma_inflight_groups
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
