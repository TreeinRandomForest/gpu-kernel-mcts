"""Matched wait-one Gluon versus cuBLAS CUDA Graph timing on one H100."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import traceback
from pathlib import Path


ORDERS = (("gluon", "cublas"), ("cublas", "gluon")) * 4


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(args, report):
    import torch

    source = load(args.source, "wait1_cublas_gluon")
    helper = load(args.helper, "wait1_cublas_helper")
    torch.manual_seed(0)
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    reference = torch.mm(a, b.T)
    outputs = {name: torch.empty_like(reference) for name in ("gluon", "cublas")}
    desc, grid = helper.descriptors(source, a, b, outputs["gluon"])
    compiled = source.gemm[grid](*desc, True, True, 1,
                                  num_warps=source.WARPS, num_ctas=2)
    report["compiler"] = helper.write_compiler_artifacts(args.output.parent, "gluon", compiled)
    save(args.output, report)
    calls = {
        "gluon": lambda: compiled[grid](*desc, True, True, 1),
        "cublas": lambda: torch.mm(a, b.T, out=outputs["cublas"]),
    }
    for name, call in calls.items():
        call()
        torch.cuda.synchronize()
        exact = bool(torch.equal(outputs[name], reference))
        report.setdefault("correctness", {})[name] = exact
        save(args.output, report)
        if not exact:
            raise ValueError(f"incorrect output: {name}")
    graphs = {name: helper.capture_graph(call, torch) for name, call in calls.items()}
    for round_index, order in enumerate(ORDERS):
        for name in order:
            measurement = helper.measure_graph(graphs[name], torch)
            report["observations"].append({"round": round_index, "position": order.index(name),
                                            "name": name, "benchmark": measurement})
            save(args.output, report)
            print(round_index, name, measurement["median_us"], flush=True)
        for name in order:
            if not torch.equal(outputs[name], reference):
                raise ValueError(f"incorrect output after round {round_index}: {name}")
    report["complete"] = True
    save(args.output, report)


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "helper", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "complete": False, "observations": [],
              "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
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
