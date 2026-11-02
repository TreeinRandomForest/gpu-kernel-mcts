"""Read-only cache-path probes for the pinned H100 worker image."""

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
SIZES_BYTES = (1 << 20, 8 << 20, 32 << 20, 64 << 20, 256 << 20)
CACHE_MODIFIERS = (".cg", ".ca")


@triton.jit
def read_reduce_probe(source, output, cache_modifier: tl.constexpr,
                      block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    values = tl.load(source + offsets, cache_modifier=cache_modifier)
    checksum = tl.sum(values.to(tl.int64), 0)
    tl.store(output + tl.program_id(0), checksum)


def cases() -> list[tuple[int, str]]:
    result = [(size, ".cg") for size in SIZES_BYTES]
    result += [(8 << 20, ".ca"), (256 << 20, ".ca")]
    random.Random(1).shuffle(result)
    return result


def allocate(payload_bytes: int):
    elements = payload_bytes // 4
    source = torch.arange(elements, device="cuda", dtype=torch.int32)
    output = torch.empty(elements // BLOCK_ELEMENTS, device="cuda", dtype=torch.int64)
    return source, output


def launch(source, output, cache_modifier: str):
    return read_reduce_probe[(output.numel(),)](
        source, output, cache_modifier, BLOCK_ELEMENTS, num_warps=4,
    )


def check(source, output) -> None:
    expected = source.to(torch.int64).reshape(-1, BLOCK_ELEMENTS).sum(dim=1)
    if not torch.equal(output, expected):
        raise RuntimeError("read-reduction probe failed correctness")


def benchmark_case(payload_bytes: int, cache_modifier: str, *, rounds: int,
                   launches_per_replay: int) -> dict:
    source, output = allocate(payload_bytes)
    compiled = launch(source, output, cache_modifier)
    ptx_load_lines = [line.strip() for line in compiled.asm["ptx"].splitlines()
                      if "ld.global" in line]
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(launches_per_replay):
            launch(source, output, cache_modifier)
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
    check(source, output)
    median = statistics.median(samples)
    return {
        "probe": "global_read_reduce",
        "measurement_scope": "read_path_plus_reduction_and_output",
        "payload_bytes": payload_bytes,
        "working_set_bytes": payload_bytes,
        "working_set_policy": "repeat_same_buffer",
        "access_pattern": "contiguous_aligned_int32_read",
        "cache_modifier": cache_modifier,
        "ptx_global_load_lines": ptx_load_lines[:8],
        "requested_read_bytes_per_launch": payload_bytes,
        "output_bytes_per_launch": 8 * payload_bytes // (4 * BLOCK_ELEMENTS),
        "block_elements": BLOCK_ELEMENTS,
        "warps_per_cta": 4,
        "rounds": rounds,
        "launches_per_graph_replay": launches_per_replay,
        "raw_us_per_launch": samples,
        "median_us_per_launch": median,
        "p10_us_per_launch": quantile(samples, 0.1),
        "p90_us_per_launch": quantile(samples, 0.9),
        "effective_requested_read_gb_per_s": payload_bytes / (median * 1000),
        "physical_link_attribution": "unverified",
    }


def profile_case(payload_bytes: int, cache_modifier: str) -> None:
    source, output = allocate(payload_bytes)
    for _ in range(11):
        launch(source, output, cache_modifier)
    torch.cuda.synchronize()
    check(source, output)
    print("PROFILE_PROBE_OK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bench", "profile"), default="bench")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--size", type=int)
    parser.add_argument("--cache-modifier", choices=CACHE_MODIFIERS, default=".cg")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--launches-per-replay", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("probe requires one H100-class CUDA 9.0 GPU")
    if args.mode == "profile":
        if args.size not in SIZES_BYTES:
            parser.error("profile mode requires a supported --size")
        profile_case(args.size, args.cache_modifier)
        return
    if args.output is None or args.rounds < 3 or args.launches_per_replay < 2:
        parser.error("bench mode requires --output, >=3 rounds, and >=2 graph launches")
    before = gpu_state()
    observations = [benchmark_case(size, modifier, rounds=args.rounds,
                                   launches_per_replay=args.launches_per_replay)
                    for size, modifier in cases()]
    observations.sort(key=lambda item: (item["payload_bytes"], item["cache_modifier"]))
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
