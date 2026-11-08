"""Balanced-order timing follow-up for validated wait-two and wait-one cubins."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import traceback
from pathlib import Path


ARMS = ("candidate", "control", "triton_b", "cublas")
ORDERS = (
    ("candidate", "control", "triton_b", "cublas"),
    ("control", "cublas", "candidate", "triton_b"),
    ("triton_b", "candidate", "cublas", "control"),
    ("cublas", "triton_b", "control", "candidate"),
)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    for name in ("candidate", "control", "fast-b", "helper", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "complete": False, "observations": [],
              "source_sha256": {name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
                                for name in ("candidate", "control", "fast_b", "helper")}}
    save(args.output, report)
    try:
        import torch
        candidate = load(args.candidate, "balanced_wait2")
        control = load(args.control, "balanced_wait1")
        fast_b = load(args.fast_b, "balanced_triton_b")
        helper = load(args.helper, "balanced_helper")
        torch.manual_seed(0)
        a = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
        reference = torch.mm(a, b.T)
        outputs = {name: torch.empty_like(reference) for name in ARMS}
        calls = {}
        report["cubin_sha256"] = {}
        for name, module in (("candidate", candidate), ("control", control)):
            desc, grid = helper.descriptors(module, a, b, outputs[name])
            compiled = module.gemm[grid](*desc, True, True, 1,
                                          num_warps=module.WARPS, num_ctas=2)
            report["cubin_sha256"][name] = hashlib.sha256(compiled.asm["cubin"]).hexdigest()
            calls[name] = lambda compiled=compiled, desc=desc, grid=grid: compiled[grid](
                *desc, True, True, 1)
        calls["triton_b"] = lambda: fast_b.launch(a, b.T, outputs["triton_b"], 4096, 4096, 4096)
        calls["cublas"] = lambda: torch.mm(a, b.T, out=outputs["cublas"])
        for name in ARMS:
            calls[name]()
            torch.cuda.synchronize()
            if not torch.equal(outputs[name], reference):
                raise ValueError(f"incorrect initial output: {name}")
        graphs = {name: helper.capture_graph(calls[name], torch) for name in ARMS}
        for round_index, order in enumerate(ORDERS + ORDERS):
            for name in order:
                benchmark = helper.measure_graph(graphs[name], torch)
                report["observations"].append({"round": round_index,
                                                "position": order.index(name),
                                                "name": name, "benchmark": benchmark})
                save(args.output, report)
                print(round_index, name, benchmark["median_us"], flush=True)
            if not all(torch.equal(outputs[name], reference) for name in ARMS):
                raise ValueError(f"incorrect post-round output: {round_index}")
        report["complete"] = True
        save(args.output, report)
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)[-1600:],
                           "traceback_tail": traceback.format_exc()[-2500:]}
        save(args.output, report)
        raise


if __name__ == "__main__":
    main()
