"""Run the packaged Triton root and Gluon adapter through the search backend."""

from __future__ import annotations

import hashlib
import argparse
import json
import statistics
from pathlib import Path

import torch

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD, load_triton_bf16_gemm_root
from kernel_mcts.benchmarks.kernels.bf16_gemm_gluon_wait1 import launch as gluon_launch
from kernel_mcts.benchmarks.triton_bf16_gemm import load_triton_bf16_gemm_gluon_reference
from kernel_mcts.domain import KernelProgram
from kernel_mcts.triton_backend import TritonBackendConfig, TritonH100Backend


ARTIFACT_ROOT = Path("/tmp/gluon-search-adapter-artifacts")


def cublas_timing() -> dict:
    torch.manual_seed(0)
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16).T
    c = torch.empty((4096, 4096), device="cuda", dtype=torch.bfloat16)
    for _ in range(10):
        torch.mm(a, b, out=c)
    torch.cuda.synchronize()
    timings = []
    for _ in range(30):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.mm(a, b, out=c)
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000)
    return {"median_us": statistics.median(timings), "timings_us": timings}


def exact_gluon_check() -> bool:
    torch.manual_seed(0)
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16).T
    c = torch.empty((4096, 4096), device="cuda", dtype=torch.bfloat16)
    gluon_launch(a, b, c, 4096, 4096, 4096)
    reference = torch.mm(a, b)
    return bool(torch.equal(c, reference))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    config = TritonBackendConfig(artifact_root=ARTIFACT_ROOT)
    backend = TritonH100Backend(config)
    result = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "candidates": {},
        "complete": False,
    }
    import triton
    result["triton_version"] = triton.__version__
    candidates = [("triton_root", load_triton_bf16_gemm_root())]
    if args.candidate is None:
        candidates.append(("gluon_wait1", load_triton_bf16_gemm_gluon_reference()))
    else:
        candidates.append(("generated_candidate", KernelProgram(
            args.candidate.read_text(), backend="triton_h100")))
    try:
        for label, program in candidates:
            print(f"candidate {label}: compiling", flush=True)
            row = {"source_sha256": hashlib.sha256(program.source.encode()).hexdigest()}
            result["candidates"][label] = row
            compiled = backend.compile(program, BF16_GEMM_WORKLOAD)
            row["compile_success"] = compiled.success
            if not compiled.success or compiled.artifact is None:
                row["compile_stdout"] = compiled.stdout
                row["compile_stderr"] = compiled.stderr
                raise RuntimeError(
                    f"{label} compile failed: {(compiled.stdout or compiled.stderr)[-800:]}"
                )
            artifact = compiled.artifact
            row["artifact_id"] = artifact.artifact_id
            row["kernel_name"] = artifact.kernel_name
            row["compiler_context"] = compiled.compiler_context
            print(f"candidate {label}: correctness", flush=True)
            correctness = backend.check_correctness(artifact, BF16_GEMM_WORKLOAD)
            row["correctness_success"] = correctness.success
            row["maximum_error"] = correctness.maximum_error
            if not correctness.success:
                raise RuntimeError(f"{label} correctness failed")
            print(f"candidate {label}: benchmarking", flush=True)
            timing = backend.benchmark(artifact, BF16_GEMM_WORKLOAD)
            row["median_us"] = timing.median_us
            row["timings_us"] = timing.timings_us
            print(f"candidate {label}: {timing.median_us:.3f} us, kernel {artifact.kernel_name}",
                  flush=True)
            if label in {"gluon_wait1", "generated_candidate"}:
                print(f"candidate {label}: NCU", flush=True)
                if label == "gluon_wait1":
                    row["exact_vs_torch_mm"] = exact_gluon_check()
                row["lightweight_profile"] = backend.lightweight_profile(
                    artifact, BF16_GEMM_WORKLOAD, "lightweight_v1")
                if row["lightweight_profile"]["kernel_name"] != artifact.kernel_name:
                    raise RuntimeError("NCU selected a different kernel")
        result["cublas"] = cublas_timing()
        print(f"cuBLAS: {result['cublas']['median_us']:.3f} us", flush=True)
        result["complete"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"[-2000:]
        raise
    finally:
        print("RESULT_JSON=" + json.dumps(result, default=str), flush=True)


if __name__ == "__main__":
    main()
