"""Generic H100 read-path sensitivity to access order and grid size."""

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
SIZES_BYTES = (8 << 20, 256 << 20)
GRID_SIZES = (132, 528, 2112)
PATTERNS = ("contiguous", "stride_32_words")
CONFIGS = tuple((size, grid, pattern) for size in SIZES_BYTES
                for grid in GRID_SIZES for pattern in PATTERNS)


@triton.jit
def sensitivity_read_probe(source, output, elements: tl.constexpr,
                           programs: tl.constexpr, iterations: tl.constexpr,
                           stride_pattern: tl.constexpr, block: tl.constexpr):
    lane = tl.arange(0, block)
    accumulated = tl.full((block,), 0, tl.int32)
    for iteration in range(iterations):
        logical = (iteration * programs + tl.program_id(0)) * block + lane
        if stride_pattern:
            columns = elements // 32
            physical = (logical % columns) * 32 + logical // columns
        else:
            physical = logical
        values = tl.load(source + physical, mask=logical < elements,
                         other=0, cache_modifier=".cg")
        accumulated += values
    tl.store(output + tl.program_id(0), tl.sum(accumulated, 0))


def launch(source, output, grid: int, pattern: str):
    elements = source.numel()
    iterations = triton.cdiv(elements, grid * BLOCK_ELEMENTS)
    return sensitivity_read_probe[(grid,)](
        source, output, elements, grid, iterations, pattern == "stride_32_words",
        BLOCK_ELEMENTS, num_warps=4,
    )


def check(output, elements: int) -> None:
    if output.to(torch.int64).sum().item() != elements:
        raise RuntimeError("sensitivity probe failed correctness")


def benchmark_case(size: int, grid: int, pattern: str, *, rounds: int,
                   launches_per_replay: int) -> dict:
    elements = size // 4
    source = torch.ones(elements, device="cuda", dtype=torch.int32)
    output = torch.empty(grid, device="cuda", dtype=torch.int32)
    compiled = launch(source, output, grid, pattern)
    ptx_load_lines = [line.strip() for line in compiled.asm["ptx"].splitlines()
                      if "ld.global" in line]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(launches_per_replay):
            launch(source, output, grid, pattern)
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
    check(output, elements)
    median = statistics.median(samples)
    return {
        "probe": "global_read_reduce_sensitivity",
        "measurement_scope": "read_path_plus_reduction_and_output",
        "payload_bytes": size,
        "working_set_bytes": size,
        "grid_programs": grid,
        "block_elements": BLOCK_ELEMENTS,
        "iterations_per_program": triton.cdiv(elements, grid * BLOCK_ELEMENTS),
        "access_pattern": pattern,
        "cache_modifier": ".cg",
        "ptx_global_load_lines": ptx_load_lines[:8],
        "requested_read_bytes_per_launch": size,
        "rounds": rounds,
        "launches_per_graph_replay": launches_per_replay,
        "raw_us_per_launch": samples,
        "median_us_per_launch": median,
        "p10_us_per_launch": quantile(samples, 0.1),
        "p90_us_per_launch": quantile(samples, 0.9),
        "effective_requested_read_gb_per_s": size / (median * 1000),
        "physical_link_attribution": "unverified",
    }


def profile_case(size: int, grid: int, pattern: str) -> None:
    elements = size // 4
    source = torch.ones(elements, device="cuda", dtype=torch.int32)
    output = torch.empty(grid, device="cuda", dtype=torch.int32)
    for _ in range(11):
        launch(source, output, grid, pattern)
    torch.cuda.synchronize()
    check(output, elements)
    print("PROFILE_PROBE_OK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bench", "profile"), default="bench")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--size", type=int)
    parser.add_argument("--grid", type=int)
    parser.add_argument("--pattern", choices=PATTERNS)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--launches-per-replay", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("probe requires one H100-class CUDA 9.0 GPU")
    if args.mode == "profile":
        if (args.size, args.grid, args.pattern) not in CONFIGS:
            parser.error("unsupported profile case")
        profile_case(args.size, args.grid, args.pattern)
        return
    if args.output is None or args.rounds < 3 or args.launches_per_replay < 2:
        parser.error("bench mode requires --output, >=3 rounds, and >=2 graph launches")
    sweeps = []
    for sweep in range(3):
        before = gpu_state()
        configs = list(CONFIGS)
        random.Random(10 + sweep).shuffle(configs)
        cases = [benchmark_case(size, grid, pattern, rounds=args.rounds,
                                launches_per_replay=args.launches_per_replay)
                 for size, grid, pattern in configs]
        cases.sort(key=lambda item: (item["payload_bytes"], item["grid_programs"],
                                     item["access_pattern"]))
        sweeps.append({"index": sweep, "gpu_state_before": before,
                       "gpu_state_after": gpu_state(), "cases": cases})
    result = {
        "schema_version": 1,
        "device_name": torch.cuda.get_device_name(),
        "compute_capability": "9.0",
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "timing_method": "CUDA events around CUDA Graph replay; no NCU during timing",
        "sweeps": sweeps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"sweeps": len(sweeps), "cases_per_sweep": len(CONFIGS)}), flush=True)


if __name__ == "__main__":
    main()
