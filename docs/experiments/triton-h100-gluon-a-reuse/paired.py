"""Exactness and paired CUDA Graph timing for two 128-wide output tiles."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import traceback
from pathlib import Path


NAMES = ("original_persistent", "split_reload", "split_reuse", "cublas")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path: Path, report: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare(name: str, kernel, baseline, helper, a, b, c):
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

    if name not in NAMES[:-1]:
        raise ValueError(f"unknown kernel arm: {name}")
    bn = 256 if name == "original_persistent" else 128
    block_m, block_k, workers = 256, 64, 66
    a_layout = gl.NVMMASharedLayout.get_default_for(
        [block_m, block_k], gl.bfloat16, cga_layout=[[1, 0]])
    b_layout = gl.NVMMASharedLayout.get_default_for(
        [bn, block_k], gl.bfloat16, cga_layout=[[0, 0]])
    c_layout = gl.NVMMASharedLayout.get_default_for(
        [block_m, bn], gl.bfloat16, cga_layout=[[1, 0]])
    desc = (TensorDescriptor.from_tensor(a, [block_m, block_k], a_layout),
            TensorDescriptor.from_tensor(b, [bn, block_k], b_layout),
            TensorDescriptor.from_tensor(c, [block_m, bn], c_layout))
    grid = (workers, 1, 1)
    m_tiles = a.shape[0] // block_m
    n_pairs = b.shape[0] // 256
    if name == "original_persistent":
        launch_args = (*desc, m_tiles, n_pairs, workers, 0)
        target = baseline.gemm
    else:
        launch_args = (*desc, m_tiles, n_pairs, workers, name == "split_reuse")
        target = kernel.gemm
    compiled = target[grid](*launch_args, num_warps=8, num_ctas=2)
    call = lambda: compiled[grid](*launch_args)
    return compiled, call, {"grid": grid, "tile_m": block_m, "tile_n": bn,
                            "outputs_per_cluster": 1 if bn == 256 else 2,
                            "workers": workers, "reuse_a": name == "split_reuse"}


def run(args, report):
    import torch
    import triton

    kernel = load(args.kernel, "a_reuse_kernel")
    baseline = load(args.baseline, "a_reuse_baseline")
    helper = load(args.helper, "a_reuse_helper")
    torch.manual_seed(0)
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    reference = torch.mm(a, b.T)
    outputs = {name: torch.empty_like(reference) for name in NAMES}
    report["versions"] = {"torch": torch.__version__, "triton": triton.__version__}
    calls = {"cublas": lambda: torch.mm(a, b.T, out=outputs["cublas"])}
    for name in NAMES[:-1]:
        try:
            compiled, call, factors = prepare(name, kernel, baseline, helper,
                                                a, b, outputs[name])
            report["variants"][name] = {
                "status": "compiled", "factors": factors,
                "compiler": helper.write_compiler_artifacts(args.output.parent, name, compiled),
            }
            calls[name] = call
        except Exception as error:
            report["variants"][name] = {
                "status": "compile_failed", "error": str(error)[-1600:],
                "traceback_tail": traceback.format_exc()[-3500:],
            }
        save(args.output, report)
        print("compile", name, report["variants"][name]["status"], flush=True)
    if len(calls) != len(NAMES):
        raise ValueError("at least one variant did not compile")
    for name, call in calls.items():
        call()
        torch.cuda.synchronize()
        exact = bool(torch.equal(outputs[name], reference))
        report["correctness"][name] = exact
        save(args.output, report)
        print("exact", name, exact, flush=True)
        if not exact:
            raise ValueError(f"incorrect output: {name}")
    graphs = {name: helper.capture_graph(call, torch) for name, call in calls.items()}
    for round_index in range(10):
        order = NAMES[round_index % len(NAMES):] + NAMES[:round_index % len(NAMES)]
        for position, name in enumerate(order):
            measurement = helper.measure_graph(graphs[name], torch)
            report["observations"].append({"round": round_index,
                                           "position": position, "name": name,
                                           "benchmark": measurement})
            save(args.output, report)
            print("timing", round_index, name, measurement["median_us"], flush=True)
        for name in NAMES:
            if not torch.equal(outputs[name], reference):
                raise ValueError(f"incorrect output after timing round: {name}")
    report["complete"] = True
    save(args.output, report)


def main():
    parser = argparse.ArgumentParser()
    for name in ("kernel", "baseline", "helper", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "complete": False,
              "sha256": {name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
                         for name in ("kernel", "baseline", "helper")},
              "variants": {}, "correctness": {}, "observations": [], "names": NAMES}
    save(args.output, report)
    try:
        run(args, report)
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)[-1600:],
                           "traceback_tail": traceback.format_exc()[-3500:]}
        save(args.output, report)
        raise


if __name__ == "__main__":
    main()
