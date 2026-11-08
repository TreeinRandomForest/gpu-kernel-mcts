"""Compare fast B, its TMA-load variant, and cuBLAS on identical tensors."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
from pathlib import Path


M = N = K = 4096
NAMES = ("baseline", "tma", "cublas")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(baseline_source: Path, tma_source: Path, mode: str, target: str | None) -> dict:
    import torch

    source_hashes = {"baseline": hashlib.sha256(baseline_source.read_bytes()).hexdigest(),
                     "tma": hashlib.sha256(tma_source.read_bytes()).hexdigest()}
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16).T
    outputs = {name: torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
               for name in NAMES}
    baseline = load_module("fast_b", baseline_source)
    tma = load_module("tma_variant", tma_source)
    tma_launch = tma.prepare(a, b, outputs["tma"], M, N, K)

    def call(name: str) -> None:
        if name == "baseline":
            baseline.launch(a, b, outputs[name], M, N, K)
        elif name == "tma":
            tma_launch()
        elif name == "cublas":
            torch.mm(a, b, out=outputs[name])
        else:
            raise ValueError(name)

    for name in NAMES:
        call(name)
    torch.cuda.synchronize()
    correctness = {}
    for name in ("baseline", "tma"):
        actual = outputs[name]
        reference = outputs["cublas"]
        maximum_error = float((actual.float() - reference.float()).abs().max().item())
        if not torch.allclose(actual, reference, rtol=0.02, atol=0.02):
            raise RuntimeError(f"{name} differs from cuBLAS")
        correctness[name] = {"maximum_error_vs_cublas": maximum_error,
                             "exact_mismatches_vs_cublas": int((actual != reference).sum().item())}
    correctness["tma"]["exact_mismatches_vs_baseline"] = int(
        (outputs["tma"] != outputs["baseline"]).sum().item())
    for name in NAMES:
        for _ in range(10):
            call(name)
    torch.cuda.synchronize()

    common = {"status": "ok", "mode": mode, "source_sha256": source_hashes,
              "correctness": correctness}
    if mode == "profile":
        if target not in ("baseline", "tma"):
            raise ValueError("profile target must be baseline or tma")
        range_name = f"TMA_TEST_{target.upper()}"
        torch.cuda.nvtx.range_push(range_name)
        call(target)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        return {**common, "target": target, "nvtx_range": range_name}
    if mode != "benchmark":
        raise ValueError(mode)
    observations = []
    for round_index in range(3):
        order = list(NAMES)
        random.Random(700 + round_index).shuffle(order)
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
    return {**common, "observations": observations}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--tma", required=True, type=Path)
    parser.add_argument("--mode", choices=("benchmark", "profile"), required=True)
    parser.add_argument("--target", choices=("baseline", "tma"))
    args = parser.parse_args()
    print(json.dumps(run(args.baseline, args.tma, args.mode, args.target), sort_keys=True))


if __name__ == "__main__":
    main()
