"""Standalone H100 global-copy path probe; runs inside the pinned GPU image."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
from pathlib import Path

import torch
import triton
import triton.language as tl


SCHEMA_VERSION = 1
BLOCK_ELEMENTS = 1024
SIZES_BYTES = (4 << 10, 64 << 10, 1 << 20, 8 << 20,
               32 << 20, 128 << 20, 256 << 20)


@triton.jit
def copy_path_probe(source, target, count: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    values = tl.load(source + offsets, offsets < count, other=0)
    tl.store(target + offsets, values, offsets < count)


@triton.jit
def minimal_kernel_probe(target):
    tl.store(target, 1)


def ring_count(payload_bytes: int, policy: str) -> int:
    if policy == "repeat_same_buffer":
        return 1
    if policy != "rotate_buffers":
        raise ValueError("unknown working-set policy")
    return min(16, max(4, (256 << 20) // (2 * payload_bytes)))


def quantile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def gpu_state() -> dict[str, object]:
    command = ["nvidia-smi", "--query-gpu=name,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu",
               "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    return {"query": command[1], "raw": result.stdout.strip() if result.returncode == 0 else None,
            "status": "available" if result.returncode == 0 else "unavailable"}


def allocate_pairs(payload_bytes: int, count: int):
    elements = payload_bytes // 4
    return [(torch.randint(0, 2**31 - 1, (elements,), device="cuda", dtype=torch.int32),
             torch.empty(elements, device="cuda", dtype=torch.int32))
            for _ in range(count)]


def launch_copy(pair) -> None:
    source, target = pair
    copy_path_probe[(triton.cdiv(source.numel(), BLOCK_ELEMENTS),)](
        source, target, source.numel(), BLOCK_ELEMENTS, num_warps=4,
    )


def benchmark_case(payload_bytes: int, policy: str, *, rounds: int,
                   launches_per_replay: int) -> dict[str, object]:
    count = ring_count(payload_bytes, policy)
    pairs = allocate_pairs(payload_bytes, count)
    launch_copy(pairs[0])  # compile before graph capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for launch in range(launches_per_replay):
            launch_copy(pairs[launch % count])
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    samples_us = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000 / launches_per_replay)
    for source, target in pairs:
        if not torch.equal(source, target):
            raise RuntimeError("global-copy probe failed correctness")
    median_us = statistics.median(samples_us)
    return {
        "probe": "global_copy_path",
        "measurement_scope": "end_to_end_path",
        "candidate_links": ["hbm_l2", "l2_l1", "l1_registers"],
        "payload_bytes": payload_bytes,
        "bytes_requested_per_launch": 2 * payload_bytes,
        "working_set_policy": policy,
        "working_set_bytes": 2 * payload_bytes * count,
        "ring_count": count,
        "access_pattern": "contiguous_aligned_int32_load_store",
        "block_elements": BLOCK_ELEMENTS,
        "warps_per_cta": 4,
        "rounds": rounds,
        "launches_per_graph_replay": launches_per_replay,
        "raw_us_per_launch": samples_us,
        "median_us_per_launch": median_us,
        "p10_us_per_launch": quantile(samples_us, 0.1),
        "p90_us_per_launch": quantile(samples_us, 0.9),
        "effective_gb_per_s": 2 * payload_bytes / (median_us * 1000),
        "cache_level_attribution": "unverified",
    }


def launch_floor(*, rounds: int, launches_per_replay: int) -> dict[str, object]:
    target = torch.empty(1, device="cuda", dtype=torch.int32)
    minimal_kernel_probe[(1,)](target, num_warps=4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(launches_per_replay):
            minimal_kernel_probe[(1,)](target, num_warps=4)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    samples_us = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000 / launches_per_replay)
    if target.item() != 1:
        raise RuntimeError("launch-floor probe failed correctness")
    return {"probe": "minimal_kernel_graph_replay", "raw_us_per_launch": samples_us,
            "median_us_per_launch": statistics.median(samples_us),
            "p10_us_per_launch": quantile(samples_us, 0.1),
            "p90_us_per_launch": quantile(samples_us, 0.9),
            "rounds": rounds, "launches_per_graph_replay": launches_per_replay}


def run_probe_once(payload_bytes: int, count: int) -> None:
    pairs = allocate_pairs(payload_bytes, count)
    for launch in range(11):
        launch_copy(pairs[launch % count])
    torch.cuda.synchronize()
    if not torch.equal(*pairs[10 % count]):
        raise RuntimeError("profile probe failed correctness")
    print("PROFILE_PROBE_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bench", "profile"), default="bench")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--size", type=int)
    parser.add_argument("--ring-count", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--launches-per-replay", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("probe requires one H100-class CUDA 9.0 GPU")
    if args.mode == "profile":
        if not args.size or args.size % 4 or args.ring_count < 1:
            parser.error("profile mode requires aligned --size and positive --ring-count")
        run_probe_once(args.size, args.ring_count)
        return
    if args.output is None or args.rounds < 3 or args.launches_per_replay < 2:
        parser.error("bench mode requires --output, >=3 rounds, and >=2 graph launches")
    before = gpu_state()
    cases = [(size, "repeat_same_buffer") for size in SIZES_BYTES]
    cases += [(size, "rotate_buffers") for size in SIZES_BYTES if size >= 8 << 20]
    random.Random(0).shuffle(cases)
    observations = [benchmark_case(size, policy, rounds=args.rounds,
                                   launches_per_replay=args.launches_per_replay)
                    for size, policy in cases]
    observations.sort(key=lambda item: (item["payload_bytes"], item["working_set_policy"]))
    result = {
        "schema_version": SCHEMA_VERSION,
        "device_name": torch.cuda.get_device_name(),
        "compute_capability": "9.0",
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "gpu_state_before": before,
        "gpu_state_after": gpu_state(),
        "timing_method": "CUDA events around CUDA Graph replay; no NCU during timing",
        "launch_floor": launch_floor(rounds=args.rounds,
                                     launches_per_replay=args.launches_per_replay),
        "observations": observations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cases": len(observations), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
