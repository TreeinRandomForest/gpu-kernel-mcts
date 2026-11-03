"""Time Triton B and torch/cuBLAS on the same BF16 tensors and event harness."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
from pathlib import Path


M = N = K = 4096
SOURCE_SHA256 = "b9b4574de16a817477990842f3e1e4d4dbd95b662e17f375f5c82981a463fba4"


def load_launch(source: Path):
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Triton source is not the corrected seed-0 B kernel")
    spec = importlib.util.spec_from_file_location("paired_triton_b", source)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load Triton source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.launch


def run(source: Path, mode: str, target: str | None) -> dict:
    import torch

    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16).T
    triton_out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    cublas_out = torch.empty_like(triton_out)
    launch = load_launch(source)

    def call(name: str) -> None:
        if name == "triton":
            launch(a, b, triton_out, M, N, K)
        elif name == "cublas":
            torch.mm(a, b, out=cublas_out)
        else:
            raise ValueError(name)

    # JIT and library dispatch are resolved before timing or profiling.
    call("triton")
    call("cublas")
    torch.cuda.synchronize()
    maximum_error = float((triton_out.float() - cublas_out.float()).abs().max().item())
    if not torch.allclose(triton_out, cublas_out, rtol=0.02, atol=0.02):
        raise RuntimeError("Triton and cuBLAS outputs differ")
    for name in ("triton", "cublas"):
        for _ in range(10):
            call(name)
    torch.cuda.synchronize()

    if mode == "profile":
        if target not in ("triton", "cublas"):
            raise ValueError("profile target must be triton or cublas")
        range_name = f"PAIRED_{target.upper()}"
        torch.cuda.nvtx.range_push(range_name)
        call(target)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        return {"status": "ok", "mode": mode, "target": target,
                "source_sha256": SOURCE_SHA256, "maximum_error": maximum_error,
                "nvtx_range": range_name}

    if mode != "benchmark":
        raise ValueError(mode)
    observations = []
    for round_index in range(3):
        order = ["triton", "cublas"]
        random.Random(300 + round_index).shuffle(order)
        for name in order:
            samples = []
            for _ in range(30):
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                call(name)
                stop.record()
                stop.synchronize()
                samples.append(start.elapsed_time(stop) * 1000.0)
            observations.append({"round": round_index, "implementation": name,
                                 "order": order.index(name), "timings_us": samples,
                                 "median_us": statistics.median(samples)})
    return {"status": "ok", "mode": mode, "source_sha256": SOURCE_SHA256,
            "maximum_error": maximum_error, "observations": observations}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mode", choices=("benchmark", "profile"), required=True)
    parser.add_argument("--target", choices=("triton", "cublas"))
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.source, arguments.mode, arguments.target), sort_keys=True))


if __name__ == "__main__":
    main()
