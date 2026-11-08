"""Same-cubin ordinary versus multicast B comparison for wait-one Gluon GEMM."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import traceback
from pathlib import Path


ARMS = ("ordinary", "multicast", "triton_b", "cublas")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path: Path, report: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def tensors(torch, seed: int):
    torch.manual_seed(seed)
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    return a, b, torch.mm(a, b.T)


def run(args, report):
    import torch

    source = load(args.source, "wait1_multicast_source")
    fast_b = load(args.fast_b, "wait1_multicast_fast_b")
    helper = load(args.helper, "wait1_multicast_helper")
    a, b, reference = tensors(torch, 0)
    outputs = {name: torch.empty_like(reference) for name in ARMS}
    desc = {}
    compiled = {}
    grid = None
    for name, mode in (("ordinary", 1), ("multicast", 0)):
        desc[name], grid = helper.descriptors(source, a, b, outputs[name])
        compiled[name] = source.gemm[grid](*desc[name], True, True, mode,
                                           num_warps=source.WARPS, num_ctas=2)
    report["compiler"] = {name: helper.write_compiler_artifacts(
        args.output.parent, name, compiled[name]) for name in ("ordinary", "multicast")}
    report["same_cubin"] = (report["compiler"]["ordinary"]["cubin_sha256"] ==
                            report["compiler"]["multicast"]["cubin_sha256"])
    save(args.output, report)
    if not report["same_cubin"]:
        raise ValueError("ordinary and multicast compiled to different cubins")
    calls = {
        "ordinary": lambda: compiled["ordinary"][grid](*desc["ordinary"], True, True, 1),
        "multicast": lambda: compiled["multicast"][grid](*desc["multicast"], True, True, 0),
        "triton_b": lambda: fast_b.launch(a, b.T, outputs["triton_b"], 4096, 4096, 4096),
        "cublas": lambda: torch.mm(a, b.T, out=outputs["cublas"]),
    }
    for name in ARMS:
        calls[name]()
        torch.cuda.synchronize()
        exact = bool(torch.equal(outputs[name], reference))
        report.setdefault("correctness", {})[name] = exact
        save(args.output, report)
        if not exact:
            raise ValueError(f"incorrect seed-0 output: {name}")
    graphs = {name: helper.capture_graph(calls[name], torch) for name in ARMS}
    for round_index in range(3):
        order = list(ARMS)
        random.Random(9931 + round_index).shuffle(order)
        report["orders"][str(round_index)] = order
        for name in order:
            benchmark = helper.measure_graph(graphs[name], torch)
            report["observations"].append({"round": round_index, "name": name,
                                            "benchmark": benchmark})
            save(args.output, report)
            print(round_index, name, benchmark["median_us"], flush=True)
        for name in ARMS:
            exact = bool(torch.equal(outputs[name], reference))
            report.setdefault("post_round_correctness", {}).setdefault(name, []).append(exact)
            save(args.output, report)
            if not exact:
                raise ValueError(f"incorrect output after round {round_index}: {name}")
    # Additional input seeds exercise the shared-memory ring and multicast
    # ownership without contaminating the fixed seed-0 latency comparison.
    for seed in (1, 2):
        extra_a, extra_b, extra_reference = tensors(torch, seed)
        report.setdefault("extra_seed_correctness", {})[str(seed)] = {}
        for name, mode in (("ordinary", 1), ("multicast", 0)):
            extra_c = torch.empty_like(extra_reference)
            extra_desc, extra_grid = helper.descriptors(source, extra_a, extra_b, extra_c)
            for _ in range(5):
                compiled[name][extra_grid](*extra_desc, True, True, mode)
                torch.cuda.synchronize()
                if not torch.equal(extra_c, extra_reference):
                    raise ValueError(f"incorrect {name} output at seed {seed}")
            report["extra_seed_correctness"][str(seed)][name] = True
            save(args.output, report)
    report["complete"] = True
    save(args.output, report)


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "fast-b", "helper", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "complete": False, "orders": {}, "observations": [],
              "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "fast_b_sha256": hashlib.sha256(args.fast_b.read_bytes()).hexdigest(),
              "helper_sha256": hashlib.sha256(args.helper.read_bytes()).hexdigest()}
    save(args.output, report)
    try:
        run(args, report)
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)[-1600:],
                           "traceback_tail": traceback.format_exc()[-2500:]}
        save(args.output, report)
        raise


if __name__ == "__main__":
    main()
