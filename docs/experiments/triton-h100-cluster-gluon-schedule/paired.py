"""Paired correctness, compiler, and timing checks for B-load request order."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import traceback
from pathlib import Path


ARMS = {
    "ordinary_a_first": (True, False, True, False),
    "ordinary_b_first": (True, False, True, True),
    "multicast_a_first": (True, True, True, False),
}


def load_kernel(path: Path):
    spec = importlib.util.spec_from_file_location("paired_gluon_kernel", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def measure(launch, torch) -> list[float]:
    for _ in range(10):
        launch()
    torch.cuda.synchronize()
    samples = []
    for _ in range(30):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import torch
    import triton

    source = load_kernel(args.source)
    m = n = k = 4096
    torch.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    reference = torch.mm(a, b.T)
    c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    report = {"schema_version": 1, "complete": False,
              "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "triton_version": triton.__version__, "torch_version": torch.__version__,
              "gpu_name": torch.cuda.get_device_name(), "observations": [], "orders": {}}
    save(args.output, report)
    for round_index in range(3):
        order = list(ARMS)
        random.Random(4127 + round_index).shuffle(order)
        report["orders"][str(round_index)] = order
        for name in order:
            c.zero_()
            clustered, multicast, pipelined, b_first = ARMS[name]
            try:
                launch = lambda: source.launch(a, b, c, clustered=clustered,
                                               multicast=multicast, pipelined=pipelined,
                                               b_first=b_first)
                compiled = launch()
                torch.cuda.synchronize()
                error = (c.float() - reference.float()).abs()
                allowed = 0.02 + 0.02 * reference.float().abs()
                success = bool(torch.all(torch.isfinite(c) & (error <= allowed)).item())
                ptx = compiled.asm["ptx"]
                cubin = compiled.asm["cubin"]
                prefix = args.output.parent / f"{name}-r{round_index}"
                prefix.with_suffix(".ptx").write_text(ptx)
                prefix.with_suffix(".cubin").write_bytes(cubin)
                result = {"round": round_index, "name": name,
                          "clustered": clustered, "multicast": multicast,
                          "pipelined": pipelined, "b_first": b_first,
                          "status": "ok" if success else "incorrect",
                          "correctness": {"success": success,
                                          "maximum_error": float(error.max().item()),
                                          "mean_error": float(error.mean().item())},
                          "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
                          "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
                          "ptx_multicast_sites": ptx.count("multicast::cluster"),
                          "registers_per_thread": compiled.n_regs,
                          "spills": compiled.n_spills,
                          "shared_memory_bytes": compiled.metadata.shared}
                if success:
                    samples = measure(launch, torch)
                    result["benchmark"] = {
                        "timings_us": samples, "median_us": statistics.median(samples),
                        "mean_us": statistics.fmean(samples),
                        "stddev_us": statistics.pstdev(samples),
                    }
            except Exception as error:
                result = {"round": round_index, "name": name, "status": "failed",
                          "error_type": type(error).__name__, "error": str(error)[-1500:],
                          "traceback_tail": traceback.format_exc()[-2500:]}
            report["observations"].append(result)
            save(args.output, report)
            print(f"round {round_index} {name}: {result['status']} "
                  f"{result.get('benchmark', {}).get('median_us')}", flush=True)
            if result["status"] != "ok":
                return
    report["complete"] = True
    save(args.output, report)


if __name__ == "__main__":
    main()
