"""Paired exactness and CUDA Graph timing for wait-one Gluon factors."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import traceback
from pathlib import Path


VARIANTS = tuple((bn, order, persistent)
                 for bn, order, persistent in itertools.product(
                     (256, 128), (0, 1), (False, True)))
NAMES = ("original", "cublas") + tuple(
    f"n{bn}_{'mfast' if order == 0 else 'nfast'}_{'persistent' if persistent else 'regular'}"
    for bn, order, persistent in VARIANTS)


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


def variant(name: str):
    for configuration, candidate in zip(VARIANTS, NAMES[2:], strict=True):
        if name == candidate:
            return configuration
    raise ValueError(f"unknown variant {name}")


def descriptors(a, b, c, bn: int):
    import triton
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

    block_m = 256
    block_k = 64
    a_layout = gl.NVMMASharedLayout.get_default_for(
        [block_m, block_k], gl.bfloat16, cga_layout=[[1, 0]])
    b_layout = gl.NVMMASharedLayout.get_default_for(
        [bn, block_k], gl.bfloat16, cga_layout=[[0, 0]])
    c_layout = gl.NVMMASharedLayout.get_default_for(
        [block_m, bn], gl.bfloat16, cga_layout=[[1, 0]])
    args = (TensorDescriptor.from_tensor(a, [block_m, block_k], a_layout),
            TensorDescriptor.from_tensor(b, [bn, block_k], b_layout),
            TensorDescriptor.from_tensor(c, [block_m, bn], c_layout))
    return args, (triton.cdiv(a.shape[0], block_m), triton.cdiv(b.shape[0], bn))


def prepare(name: str, kernel, original, helper, a, b, c):
    if name == "original":
        args, grid = helper.descriptors(original, a, b, c)
        compiled = original.gemm[grid](*args, True, True, 1,
                                       num_warps=original.WARPS, num_ctas=2)
        call = lambda: compiled[grid](*args, True, True, 1)
        return compiled, call, {"grid": grid, "tile_n": 256, "order": "mfast",
                                "persistent": False, "workers": 256}
    bn, order, persistent = variant(name)
    args, (m_tiles, n_tiles) = descriptors(a, b, c, bn)
    workers = 66 if persistent else m_tiles * n_tiles
    grid = (workers, 1, 1)
    compiled = kernel.gemm[grid](*args, m_tiles, n_tiles, workers, order,
                                 num_warps=kernel.WARPS, num_ctas=2)
    call = lambda: compiled[grid](*args, m_tiles, n_tiles, workers, order)
    return compiled, call, {"grid": grid, "tile_n": bn,
                            "order": "mfast" if order == 0 else "nfast",
                            "persistent": persistent, "workers": workers}


def run(args, report):
    import torch
    import triton

    kernel = load(args.kernel, "factor_kernel")
    original = load(args.original, "factor_original")
    helper = load(args.helper, "factor_helper")
    torch.manual_seed(0)
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    reference = torch.mm(a, b.T)
    outputs = {name: torch.empty_like(reference) for name in NAMES}
    report["versions"] = {"torch": torch.__version__, "triton": triton.__version__}
    calls = {"cublas": lambda: torch.mm(a, b.T, out=outputs["cublas"])}
    for name in NAMES:
        if name == "cublas":
            continue
        try:
            compiled, call, factors = prepare(
                name, kernel, original, helper, a, b, outputs[name])
            report["variants"][name] = {
                "status": "compiled", "factors": factors,
                "compiler": helper.write_compiler_artifacts(args.output.parent, name, compiled),
            }
            calls[name] = call
        except Exception as error:
            report["variants"][name] = {
                "status": "compile_failed", "error": str(error)[-1600:],
                "traceback_tail": traceback.format_exc()[-2500:],
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
    for round_index in range(len(NAMES)):
        order = NAMES[round_index:] + NAMES[:round_index]
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
    for name in ("kernel", "original", "helper", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "complete": False,
              "sha256": {name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
                         for name in ("kernel", "original", "helper")},
              "variants": {}, "correctness": {}, "observations": [], "names": NAMES}
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
