"""Check source-level B TMA scheduling beside fast Triton B and cuBLAS."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import subprocess
import traceback
from pathlib import Path


ARMS = ("source_ordinary", "source_multicast", "static_ordinary",
        "static_multicast", "triton_b", "cublas")
FAST_B_SHA256 = "b9b4574de16a817477990842f3e1e4d4dbd95b662e17f375f5c82981a463fba4"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def descriptors(source, a, b, c):
    import triton
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

    m, k = a.shape
    n = b.shape[0]
    block_m = source.CTA_M * 2
    a_layout = gl.NVMMASharedLayout.get_default_for(
        [block_m, source.BLOCK_K], gl.bfloat16, cga_layout=[[1, 0]])
    b_layout = gl.NVMMASharedLayout.get_default_for(
        [source.CTA_N, source.BLOCK_K], gl.bfloat16, cga_layout=[[0, 0]])
    c_layout = gl.NVMMASharedLayout.get_default_for(
        [block_m, source.CTA_N], gl.bfloat16, cga_layout=[[1, 0]])
    args = (TensorDescriptor.from_tensor(a, [block_m, source.BLOCK_K], a_layout),
            TensorDescriptor.from_tensor(b, [source.CTA_N, source.BLOCK_K], b_layout),
            TensorDescriptor.from_tensor(c, [block_m, source.CTA_N], c_layout))
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, source.CTA_N), 1)
    return args, grid


def capture_graph(call, torch):
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    return graph


def measure_graph(graph, torch) -> dict:
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(30):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return {"timings_us": samples, "median_us": statistics.median(samples)}


def write_compiler_artifacts(output: Path, name: str, compiled) -> dict:
    ptx = compiled.asm["ptx"]
    cubin = compiled.asm["cubin"]
    (output / f"{name}.ptx").write_text(ptx)
    cubin_path = output / f"{name}.cubin"
    cubin_path.write_bytes(cubin)
    sass = subprocess.run(["nvdisasm", "-c", str(cubin_path)], capture_output=True,
                          text=True, timeout=60, check=True).stdout
    (output / f"{name}.sass").write_text(sass)
    return {"ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
            "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
            "ptx_multicast_sites": ptx.count("multicast::cluster"),
            "sass_multicast_sites": sass.count("UTMALDG.2D.MULTICAST"),
            "sass_warpgroup_wait_sites": sass.count("WARPGROUP.DEPBAR.LE"),
            "registers_per_thread": compiled.n_regs,
            "shared_memory_bytes": compiled.metadata.shared,
            "spills": compiled.n_spills}


def run(args, report):
    import torch
    import triton

    if hashlib.sha256(args.fast_b.read_bytes()).hexdigest() != FAST_B_SHA256:
        raise ValueError("fast Triton B source changed")
    dynamic = load_module(args.source, "source_schedule_dynamic")
    static = load_module(args.static, "source_schedule_static")
    fast_b = load_module(args.fast_b, "source_schedule_fast_b")
    m = n = k = 4096
    torch.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    reference = torch.mm(a, b.T)
    outputs = {name: torch.empty_like(reference) for name in ARMS}
    dynamic_args = {}
    static_args = {}
    for name in ("source_ordinary", "source_multicast"):
        dynamic_args[name], grid = descriptors(dynamic, a, b, outputs[name])
    for name in ("static_ordinary", "static_multicast"):
        static_args[name], static_grid = descriptors(static, a, b, outputs[name])
    compiled_source_ordinary = dynamic.gemm[grid](
        *dynamic_args["source_ordinary"], True, True, 1,
        num_warps=dynamic.WARPS, num_ctas=2)
    compiled_source_multicast = dynamic.gemm[grid](
        *dynamic_args["source_multicast"], True, True, 0,
        num_warps=dynamic.WARPS, num_ctas=2)
    compiled_static_ordinary = static.gemm[static_grid](
        *static_args["static_ordinary"], True, False, True,
        num_warps=static.WARPS, num_ctas=2)
    compiled_static_multicast = static.gemm[static_grid](
        *static_args["static_multicast"], True, True, True,
        num_warps=static.WARPS, num_ctas=2)
    compiled = {"source_ordinary": compiled_source_ordinary,
                "source_multicast": compiled_source_multicast,
                "static_ordinary": compiled_static_ordinary,
                "static_multicast": compiled_static_multicast}
    report["compiler"] = {name: write_compiler_artifacts(args.output.parent, name, value)
                          for name, value in compiled.items()}
    report["dynamic_same_cubin"] = (report["compiler"]["source_ordinary"]["cubin_sha256"] ==
                                    report["compiler"]["source_multicast"]["cubin_sha256"])
    save(args.output, report)
    calls = {
        "source_ordinary": lambda: compiled_source_ordinary[grid](
            *dynamic_args["source_ordinary"], True, True, 1),
        "source_multicast": lambda: compiled_source_multicast[grid](
            *dynamic_args["source_multicast"], True, True, 0),
        "static_ordinary": lambda: compiled_static_ordinary[static_grid](
            *static_args["static_ordinary"], True, False, True),
        "static_multicast": lambda: compiled_static_multicast[static_grid](
            *static_args["static_multicast"], True, True, True),
        "triton_b": lambda: fast_b.launch(a, b.T, outputs["triton_b"], m, n, k),
        "cublas": lambda: torch.mm(a, b.T, out=outputs["cublas"]),
    }
    report["correctness"] = {}
    for name in ARMS:
        calls[name]()
        torch.cuda.synchronize()
        correct = bool(torch.equal(outputs[name], reference))
        report["correctness"][name] = {"exact": correct,
                                       "maximum_error": float((outputs[name].float() -
                                                               reference.float()).abs().max().item())}
        save(args.output, report)
        if not correct:
            raise ValueError(f"incorrect output: {name}")
    graphs = {name: capture_graph(call, torch) for name, call in calls.items()}
    for round_index in range(3):
        order = list(ARMS)
        random.Random(9127 + round_index).shuffle(order)
        report["orders"][str(round_index)] = order
        for name in order:
            result = {"round": round_index, "name": name,
                      "benchmark": measure_graph(graphs[name], torch)}
            report["observations"].append(result)
            save(args.output, report)
            print(f"round {round_index} {name}: {result['benchmark']['median_us']}", flush=True)
    report["complete"] = True
    save(args.output, report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--fast-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "complete": False,
              "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "static_sha256": hashlib.sha256(args.static.read_bytes()).hexdigest(),
              "fast_b_sha256": hashlib.sha256(args.fast_b.read_bytes()).hexdigest(),
              "orders": {}, "observations": []}
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
