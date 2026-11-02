"""In-kernel repeated-read probe for L2 and HBM source-path scaling."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl

from bench_inside import gpu_state, quantile


BLOCK_ELEMENTS = 1024
CONFIGS = ((8 << 20, 1), (8 << 20, 4), (8 << 20, 16), (8 << 20, 64),
           (256 << 20, 1), (256 << 20, 4))


@triton.jit
def repeated_read_probe(source, output, blocks: tl.constexpr,
                        repeats: tl.constexpr, block: tl.constexpr):
    lane = tl.arange(0, block)
    accumulated = tl.full((block,), 0, tl.int32)
    for iteration in range(repeats):
        block_id = (tl.program_id(0) + iteration) & (blocks - 1)
        values = tl.load(source + block_id * block + lane, cache_modifier=".cg")
        accumulated += values
    tl.store(output + tl.program_id(0), tl.sum(accumulated, 0))


def allocate(payload_bytes: int):
    elements = payload_bytes // 4
    source = torch.ones(elements, device="cuda", dtype=torch.int32)
    output = torch.empty(elements // BLOCK_ELEMENTS, device="cuda", dtype=torch.int32)
    return source, output


def launch(source, output, repeats: int):
    return repeated_read_probe[(output.numel(),)](
        source, output, output.numel(), repeats, BLOCK_ELEMENTS, num_warps=4,
    )


def check(output, repeats: int) -> None:
    if not torch.all(output == BLOCK_ELEMENTS * repeats).item():
        raise RuntimeError("repeated-read probe failed correctness")


def benchmark_case(payload_bytes: int, repeats: int, *, rounds: int,
                   launches_per_replay: int) -> dict:
    source, output = allocate(payload_bytes)
    compiled = launch(source, output, repeats)
    ptx_load_lines = [line.strip() for line in compiled.asm["ptx"].splitlines()
                      if "ld.global" in line]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(launches_per_replay):
            launch(source, output, repeats)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / launches_per_replay)
    check(output, repeats)
    median = statistics.median(samples)
    return {
        "probe": "in_kernel_repeated_read",
        "measurement_scope": "read_path_plus_address_arithmetic_reduction_and_output",
        "payload_bytes": payload_bytes,
        "repeats": repeats,
        "requested_read_bytes_per_launch": payload_bytes * repeats,
        "working_set_bytes": payload_bytes,
        "working_set_policy": "cyclic_block_shift_within_kernel",
        "cache_modifier": ".cg",
        "ptx_global_load_lines": ptx_load_lines[:8],
        "rounds": rounds,
        "launches_per_graph_replay": launches_per_replay,
        "raw_us_per_launch": samples,
        "median_us_per_launch": median,
        "p10_us_per_launch": quantile(samples, 0.1),
        "p90_us_per_launch": quantile(samples, 0.9),
        "effective_requested_read_gb_per_s": payload_bytes * repeats / (median * 1000),
        "physical_link_attribution": "unverified",
    }


def profile_case(payload_bytes: int, repeats: int) -> None:
    source, output = allocate(payload_bytes)
    for _ in range(11):
        launch(source, output, repeats)
    torch.cuda.synchronize()
    check(output, repeats)
    print("PROFILE_PROBE_OK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bench", "profile"), default="bench")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--size", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--launches-per-replay", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("probe requires one H100-class CUDA 9.0 GPU")
    if args.mode == "profile":
        if (args.size, args.repeats) not in CONFIGS:
            parser.error("unsupported profile case")
        profile_case(args.size, args.repeats)
        return
    if args.output is None or args.rounds < 3 or args.launches_per_replay < 2:
        parser.error("bench mode requires --output, >=3 rounds, and >=2 graph launches")
    before = gpu_state()
    configs = list(CONFIGS)
    random.Random(2).shuffle(configs)
    observations = [benchmark_case(size, repeats, rounds=args.rounds,
                                   launches_per_replay=args.launches_per_replay)
                    for size, repeats in configs]
    observations.sort(key=lambda item: (item["payload_bytes"], item["repeats"]))
    result = {
        "schema_version": 1,
        "device_name": torch.cuda.get_device_name(),
        "compute_capability": "9.0",
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "gpu_state_before": before,
        "gpu_state_after": gpu_state(),
        "timing_method": "CUDA events around CUDA Graph replay; no NCU during timing",
        "observations": observations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cases": len(observations), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
