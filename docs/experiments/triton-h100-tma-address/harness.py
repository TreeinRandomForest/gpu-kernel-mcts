"""Same-input correctness, CUDA-event timing, and selected NCU launch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
from pathlib import Path


M = N = K = 4096
NAMES = ("baseline", "unmasked_store", "constant_group", "both")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"tma_address_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(variants_dir: Path, mode: str, target: str | None) -> dict:
    import torch

    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16).T
    outputs = {name: torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
               for name in (*NAMES, "cublas")}
    paths = {name: variants_dir / f"{name}.py" for name in NAMES}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in paths.items()}
    launchers = {name: load_module(path).prepare(a, b, outputs[name], M, N, K)
                 for name, path in paths.items()}

    def call(name: str) -> None:
        if name == "cublas":
            torch.mm(a, b, out=outputs[name])
        else:
            launchers[name]()

    for name in (*NAMES, "cublas"):
        call(name)
    torch.cuda.synchronize()
    correctness = {}
    for name in NAMES:
        actual, reference = outputs[name], outputs["cublas"]
        if not torch.allclose(actual, reference, rtol=0.02, atol=0.02):
            raise RuntimeError(f"{name} differs from cuBLAS")
        correctness[name] = {
            "maximum_error_vs_cublas": float((actual.float() - reference.float()).abs().max().item()),
            "exact_mismatches_vs_cublas": int((actual != reference).sum().item()),
        }
    common = {"status": "ok", "mode": mode, "source_sha256": hashes,
              "correctness": correctness}
    if mode == "profile":
        if target not in NAMES:
            raise ValueError("unknown profile target")
        for _ in range(10):
            call(target)
        torch.cuda.synchronize()
        range_name = f"TMA_ADDRESS_{target.upper()}"
        torch.cuda.nvtx.range_push(range_name)
        call(target)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        return {**common, "target": target, "nvtx_range": range_name}
    if mode != "benchmark":
        raise ValueError(mode)
    for name in (*NAMES, "cublas"):
        for _ in range(10):
            call(name)
    torch.cuda.synchronize()
    observations = []
    for round_index in range(3):
        order = list((*NAMES, "cublas"))
        random.Random(1811 + round_index).shuffle(order)
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
    parser.add_argument("--variants-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("benchmark", "profile"), required=True)
    parser.add_argument("--target", choices=NAMES)
    args = parser.parse_args()
    print(json.dumps(run(args.variants_dir, args.mode, args.target), sort_keys=True))


if __name__ == "__main__":
    main()
