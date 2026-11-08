"""Isolated GPU process for the fixed Triton/Gluon H100 GEMM contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


GRAPH_TIMING_CONTRACT = "triton_h100_cuda_graph_v1"


def _candidate(path: Path):
    spec = importlib.util.spec_from_file_location("candidate_triton", path)
    if spec is None or spec.loader is None:
        raise ValueError("candidate module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    launch = getattr(module, "launch", None)
    if not callable(launch):
        raise ValueError("candidate must define launch(A, B, C, M, N, K)")
    return launch


def _inputs(m: int, n: int, k: int, seed: int):
    import torch

    torch.manual_seed(seed)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    # A transpose of [N,K] has shape [K,N] and column-major strides.
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).T
    c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    return a, b, c


def _event_timings(torch, call, count: int) -> list[float]:
    timings = []
    for _ in range(count):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0)
    return timings


def _compiled_cubin_hash(compiled) -> str:
    asm = getattr(compiled, "asm", None)
    if not isinstance(asm, dict) or not isinstance(asm.get("cubin"), bytes):
        raise ValueError("launch must return a Triton CompiledKernel with cubin")
    return hashlib.sha256(asm["cubin"]).hexdigest()


def run(arguments: argparse.Namespace) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable")
    a, b, c = _inputs(arguments.M, arguments.N, arguments.K, arguments.seed)
    launch = _candidate(arguments.candidate)
    compiled = launch(a, b, c, arguments.M, arguments.N, arguments.K)
    torch.cuda.synchronize()
    if arguments.mode == "compile":
        _compiled_cubin_hash(compiled)
        asm = compiled.asm
        name = getattr(compiled, "name", None)
        if not isinstance(name, str) or not name or len(name) > 256:
            raise ValueError("launch must return a named Triton CompiledKernel")
        arguments.artifact_dir.mkdir(parents=True, exist_ok=True)
        (arguments.artifact_dir / "kernel.cubin").write_bytes(asm["cubin"])
        ptx = asm.get("ptx")
        if isinstance(ptx, str):
            (arguments.artifact_dir / "kernel.ptx").write_text(ptx, encoding="utf-8")
        metadata = getattr(compiled, "metadata", None)
        return {
            "status": "ok",
            "kernel_name": name,
            "registers_per_thread": getattr(compiled, "n_regs", None),
            "shared_memory_bytes": getattr(metadata, "shared", None),
            "spills": getattr(compiled, "n_spills", None),
        }
    if arguments.mode == "correctness":
        reference = a @ b
        delta = (c.float() - reference.float()).abs()
        maximum = float(delta.max().item())
        mean = float(delta.mean().item())
        success = bool(torch.allclose(c, reference, rtol=arguments.rtol,
                                      atol=arguments.atol))
        return {
            "status": "ok", "success": success,
            "maximum_error": maximum, "mean_error": mean,
            "failed_test_id": None if success else "bf16_gemm_4096_h100",
            "reference_metadata": {"implementation": "torch.matmul/cuBLAS", "seed": arguments.seed},
        }
    if arguments.mode in {"benchmark", "profile"}:
        for _ in range(arguments.warmups):
            launch(a, b, c, arguments.M, arguments.N, arguments.K)
        torch.cuda.synchronize()
        if arguments.mode == "profile":
            torch.cuda.nvtx.range_push("KERNEL_MCTS_TARGET")
            try:
                launch(a, b, c, arguments.M, arguments.N, arguments.K)
            finally:
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            return {"status": "ok"}
        expected = (arguments.artifact_dir / "kernel.cubin").read_bytes()
        if _compiled_cubin_hash(compiled) != hashlib.sha256(expected).hexdigest():
            raise ValueError("benchmark cubin differs from compiled artifact")
        call = lambda: launch(a, b, c, arguments.M, arguments.N, arguments.K)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = call()
        if _compiled_cubin_hash(captured) != _compiled_cubin_hash(compiled):
            raise ValueError("captured graph uses a different compiled cubin")
        reference = a @ b
        c.zero_()
        graph.replay()
        torch.cuda.synchronize()
        if not bool(torch.allclose(c, reference, rtol=arguments.rtol,
                                   atol=arguments.atol)):
            raise ValueError("captured graph failed correctness")
        for _ in range(arguments.warmups):
            graph.replay()
        torch.cuda.synchronize()
        timings = _event_timings(torch, graph.replay, arguments.measurements)
        try:
            ordinary_diagnostics = {
                "ordinary_launch_timings_us": _event_timings(
                    torch, call, arguments.measurements),
            }
        except Exception as error:
            ordinary_diagnostics = {"ordinary_launch_error_type": type(error).__name__}
        return {"status": "ok", "timing_contract": GRAPH_TIMING_CONTRACT,
                "timings_us": timings,
                "diagnostics": ordinary_diagnostics}
    raise ValueError(f"unsupported mode {arguments.mode!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("compile", "correctness", "benchmark", "profile"), required=True)
    for name in ("M", "N", "K"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--measurements", type=int, default=30)
    arguments = parser.parse_args(argv)
    try:
        payload = run(arguments)
    except Exception as error:
        # Candidate errors are data; traceback and source stay out of the wire record.
        payload = {"status": "candidate_failure", "error_type": type(error).__name__,
                   "message": str(error)[-2000:]}
    print(json.dumps(payload, default=str))
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
