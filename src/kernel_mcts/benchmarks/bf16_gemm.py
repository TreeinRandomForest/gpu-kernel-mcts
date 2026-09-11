from __future__ import annotations

from importlib.resources import files

from ..domain import KernelProgram, ShapeCase, WorkloadContract


BF16_GEMM_WORKLOAD = WorkloadContract(
    benchmark_id="bf16_gemm_4096_h100",
    operation="gemm",
    dtype="bfloat16",
    shapes=(ShapeCase({"M": 4096, "N": 4096, "K": 4096}, 1.0),),
    rtol=2.0e-2,
    atol=2.0e-2,
    metadata={
        "target": {
            "gpu_model": "H100",
            "form_factor": "SXM",
            "minimum_compute_capability": "9.0",
        },
        "entry_point": "bf16_gemm_root",
        "accumulation_dtype": "float32",
        "output_dtype": "bfloat16",
        "tensors": {
            "A": {"shape": ["M", "K"], "layout": "row_major"},
            "B": {"shape": ["K", "N"], "layout": "column_major"},
            "C": {"shape": ["M", "N"], "layout": "row_major"},
        },
        "abi": {
            "parameters": [
                "const __nv_bfloat16* A",
                "const __nv_bfloat16* B",
                "__nv_bfloat16* C",
                "int M",
                "int N",
                "int K",
            ],
            "launch": {
                "block": [16, 16, 1],
                "grid": ["ceil_div(N, 16)", "ceil_div(M, 16)", 1],
                "dynamic_shared_memory_bytes": 0,
            },
        },
    },
)


def load_bf16_gemm_root() -> KernelProgram:
    """Load the fixed correctness-first root program from package resources."""
    source = (
        files("kernel_mcts.benchmarks")
        .joinpath("kernels", "bf16_gemm_root.cu")
        .read_text(encoding="utf-8")
    )
    return KernelProgram(source=source, backend="cuda_cpp")


def load_bf16_gemm_smoke_candidate() -> KernelProgram:
    """Load a distinct correctness-first candidate for remote smoke tests."""
    source = (
        files("kernel_mcts.benchmarks")
        .joinpath("kernels", "bf16_gemm_smoke_candidate.cu")
        .read_text(encoding="utf-8")
    )
    return KernelProgram(source=source, backend="cuda_cpp")
