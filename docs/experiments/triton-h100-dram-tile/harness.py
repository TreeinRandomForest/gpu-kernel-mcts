"""Compare same-area TMA tile orientations and CTA orders on one input set."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
from pathlib import Path


M = N = K = 4096
ARMS = {
    "tile_128x256_g1": (128, 256, 1),
    "tile_128x256_g8": (128, 256, 8),
    "tile_256x128_g1": (256, 128, 1),
    "tile_256x128_g8": (256, 128, 8),
    "tile_256x128_g16": (256, 128, 16),
}
NAMES = (*ARMS, "cublas")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("fixed_tma_kernel", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(tma_source: Path, mode: str, target: str | None) -> dict:
    import torch
    import triton

    source_hash = hashlib.sha256(tma_source.read_bytes()).hexdigest()
    tma = load_module(tma_source)
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16).T
    outputs = {name: torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
               for name in NAMES}
    descriptors = {}
    for bm, bn in {(bm, bn) for bm, bn, _ in ARMS.values()}:
        descriptors[(bm, bn)] = (
            tma.create_2d_tma_descriptor(a.data_ptr(), M, K, bm, 64, a.element_size()),
            tma.create_2d_tma_descriptor(b.data_ptr(), N, K, bn, 64, b.element_size()),
        )

    def call(name: str) -> None:
        if name == "cublas":
            torch.mm(a, b, out=outputs[name])
        elif name in ARMS:
            bm, bn, group = ARMS[name]
            a_desc, b_desc = descriptors[(bm, bn)]
            grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
            tma.bf16_gemm_tma[grid](
                a_desc, b_desc, outputs[name], M, N, K, bm, bn, 64, group,
                num_warps=8, num_stages=3,
            )
        else:
            raise ValueError(name)

    for name in NAMES:
        call(name)
    torch.cuda.synchronize()
    correctness = {}
    for name in ARMS:
        actual, reference = outputs[name], outputs["cublas"]
        if not torch.allclose(actual, reference, rtol=0.02, atol=0.02):
            raise RuntimeError(f"{name} differs from cuBLAS")
        correctness[name] = {
            "maximum_error_vs_cublas": float((actual.float() - reference.float()).abs().max().item()),
            "exact_mismatches_vs_cublas": int((actual != reference).sum().item()),
        }
    common = {"status": "ok", "mode": mode, "source_sha256": source_hash,
              "correctness": correctness}
    if mode == "profile":
        if target not in NAMES:
            raise ValueError("unknown profile target")
        for _ in range(10):
            call(target)
        torch.cuda.synchronize()
        range_name = f"DRAM_TILE_{target.upper()}"
        torch.cuda.nvtx.range_push(range_name)
        call(target)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        return {**common, "target": target, "nvtx_range": range_name}
    if mode != "benchmark":
        raise ValueError(mode)
    for name in NAMES:
        for _ in range(10):
            call(name)
    torch.cuda.synchronize()
    observations = []
    for round_index in range(3):
        order = list(NAMES)
        random.Random(1270 + round_index).shuffle(order)
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
    parser.add_argument("--tma", type=Path, required=True)
    parser.add_argument("--mode", choices=("benchmark", "profile"), required=True)
    parser.add_argument("--target", choices=NAMES)
    args = parser.parse_args()
    print(json.dumps(run(args.tma, args.mode, args.target), sort_keys=True))


if __name__ == "__main__":
    main()
