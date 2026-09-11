from __future__ import annotations

from importlib.resources import files

from kernel_mcts.benchmarks import (
    BF16_GEMM_WORKLOAD,
    load_bf16_gemm_root,
    load_bf16_gemm_smoke_candidate,
)


def test_bf16_gemm_workload_contract_is_fixed_and_explicit() -> None:
    workload = BF16_GEMM_WORKLOAD

    assert workload.benchmark_id == "bf16_gemm_4096_h100"
    assert workload.operation == "gemm"
    assert workload.dtype == "bfloat16"
    assert len(workload.shapes) == 1
    assert workload.shapes[0].dimensions == {"M": 4096, "N": 4096, "K": 4096}
    assert workload.shapes[0].weight == 1.0
    assert workload.rtol == 2.0e-2
    assert workload.atol == 2.0e-2


def test_bf16_gemm_metadata_defines_target_layout_and_abi() -> None:
    metadata = BF16_GEMM_WORKLOAD.metadata

    assert metadata["target"] == {
        "gpu_model": "H100",
        "form_factor": "SXM",
        "minimum_compute_capability": "9.0",
    }
    assert metadata["entry_point"] == "bf16_gemm_root"
    assert metadata["accumulation_dtype"] == "float32"
    assert metadata["output_dtype"] == "bfloat16"
    assert metadata["tensors"] == {
        "A": {"shape": ["M", "K"], "layout": "row_major"},
        "B": {"shape": ["K", "N"], "layout": "column_major"},
        "C": {"shape": ["M", "N"], "layout": "row_major"},
    }
    assert metadata["abi"]["launch"] == {
        "block": [16, 16, 1],
        "grid": ["ceil_div(N, 16)", "ceil_div(M, 16)", 1],
        "dynamic_shared_memory_bytes": 0,
    }


def test_bf16_gemm_root_loads_deterministically_from_package_resources() -> None:
    first = load_bf16_gemm_root()
    second = load_bf16_gemm_root()

    assert first == second
    assert first is not second
    assert first.backend == "cuda_cpp"
    assert first.source.strip()
    assert 'extern "C" __global__ void bf16_gemm_root' in first.source
    assert "const __nv_bfloat16* A" in first.source
    assert "const __nv_bfloat16* B" in first.source
    assert "__nv_bfloat16* C" in first.source
    assert "int M" in first.source
    assert "int N" in first.source
    assert "int K" in first.source


def test_bf16_gemm_root_is_available_as_a_package_resource() -> None:
    resource = files("kernel_mcts.benchmarks").joinpath(
        "kernels", "bf16_gemm_root.cu"
    )

    assert resource.is_file()
    assert resource.read_text(encoding="utf-8") == load_bf16_gemm_root().source


def test_smoke_candidate_is_distinct_and_preserves_fixed_abi_and_layout() -> None:
    root = load_bf16_gemm_root()
    candidate = load_bf16_gemm_smoke_candidate()

    assert candidate != root
    assert 'extern "C" __global__ void bf16_gemm_root' in candidate.source
    assert "A[row * K + k]" in candidate.source
    assert "B[col * K + k]" in candidate.source
    assert "C[row * N + col]" in candidate.source
