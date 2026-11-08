"""Same-worker timing modes for identical single- and two-CTA Gluon cubins."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import time
import traceback
from pathlib import Path


KERNELS = ("single_cta", "persistent", "cublas")
MODES = ("search_event", "prepared_event", "batched_event", "graph_event")
ROUNDS = 8
SAMPLES = 30


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


def prepared_call(module, compiled, a, b, c, persistent: bool):
    import triton
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

    m = n = k = 4096
    block_m = 256 if persistent else 128
    block_n = 256
    block_k = 64
    if persistent:
        a_layout = gl.NVMMASharedLayout.get_default_for(
            [block_m, block_k], gl.bfloat16, cga_layout=[[1, 0]])
        b_layout = gl.NVMMASharedLayout.get_default_for(
            [block_n, block_k], gl.bfloat16, cga_layout=[[0, 0]])
        c_layout = gl.NVMMASharedLayout.get_default_for(
            [block_m, block_n], gl.bfloat16, cga_layout=[[1, 0]])
    else:
        a_layout = gl.NVMMASharedLayout.get_default_for(
            [block_m, block_k], gl.bfloat16)
        b_layout = gl.NVMMASharedLayout.get_default_for(
            [block_n, block_k], gl.bfloat16)
        c_layout = gl.NVMMASharedLayout.get_default_for(
            [block_m, block_n], gl.bfloat16)
    desc = (TensorDescriptor.from_tensor(a, [block_m, block_k], a_layout),
            TensorDescriptor.from_tensor(b.T, [block_n, block_k], b_layout),
            TensorDescriptor.from_tensor(c, [block_m, block_n], c_layout))
    if persistent:
        grid = (66, 1, 1)
        launch_args = (*desc, m // block_m, n // block_n, 66, 0)
    else:
        grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n), 1)
        launch_args = (*desc, False, True, 1)
    call = lambda: compiled[grid](*launch_args)
    # A prepared call must use the same compiled device program as launch().
    checked = module.gemm[grid](*launch_args, num_warps=8,
                                num_ctas=2 if persistent else 1)
    if hashlib.sha256(checked.asm["cubin"]).digest() != hashlib.sha256(compiled.asm["cubin"]).digest():
        raise ValueError("prepared call compiled a different cubin")
    return call


def capture(call, torch):
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    torch.cuda.synchronize()
    return graph


def event_samples(call, torch):
    timings = []
    for _ in range(SAMPLES):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0)
    return timings


def batched_sample(call, torch):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(SAMPLES):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / SAMPLES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", type=Path, required=True)
    parser.add_argument("--persistent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "complete": False, "rounds": ROUNDS,
              "samples_per_mode": SAMPLES, "candidates": {}, "observations": [],
              "source_sha256": {name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
                                for name in ("single", "persistent")}}
    save(args.output, report)
    try:
        import torch
        import triton

        report["environment"] = {"gpu": torch.cuda.get_device_name(),
                                 "torch": torch.__version__, "triton": triton.__version__,
                                 "cuda": torch.version.cuda}
        torch.manual_seed(0)
        a = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16).T
        reference = torch.mm(a, b)
        modules = {"single_cta": load(args.single, "timing_single_cta"),
                   "persistent": load(args.persistent, "timing_persistent")}
        outputs = {name: torch.empty_like(reference) for name in KERNELS}
        calls = {}
        for name, module in modules.items():
            c = outputs[name]
            launch = lambda m=module, out=c: m.launch(a, b, out, 4096, 4096, 4096)
            compiled = launch()
            torch.cuda.synchronize()
            if not torch.equal(c, reference):
                raise ValueError(f"{name} output differs from torch.mm")
            prepared = prepared_call(module, compiled, a, b, c,
                                     persistent=name == "persistent")
            prepared()
            torch.cuda.synchronize()
            if not torch.equal(c, reference):
                raise ValueError(f"{name} prepared output differs from torch.mm")
            report["candidates"][name] = {
                "cubin_sha256": hashlib.sha256(compiled.asm["cubin"]).hexdigest(),
                "grid": [66, 1, 1] if name == "persistent" else [32, 16, 1],
                "registers_per_thread": compiled.n_regs,
                "shared_memory_bytes": compiled.metadata.shared,
                "correctness_exact": True,
            }
            calls[name] = {"search_event": launch, "prepared_event": prepared,
                           "batched_event": launch,
                           "graph_event": capture(launch, torch).replay}
            save(args.output, report)
        c = outputs["cublas"]
        cublas = lambda: torch.mm(a, b, out=c)
        cublas()
        torch.cuda.synchronize()
        if not torch.equal(c, reference):
            raise ValueError("cuBLAS output differs from reference")
        calls["cublas"] = {"search_event": cublas, "prepared_event": cublas,
                           "batched_event": cublas,
                           "graph_event": capture(cublas, torch).replay}
        report["candidates"]["cublas"] = {"correctness_exact": True}
        save(args.output, report)
        tasks = [(name, mode) for name in KERNELS for mode in MODES]
        for round_index in range(ROUNDS):
            order = tasks.copy()
            random.Random(4100 + round_index).shuffle(order)
            for position, (name, mode) in enumerate(order):
                call = calls[name][mode]
                for _ in range(10):
                    call()
                torch.cuda.synchronize()
                if mode == "batched_event":
                    timings = [batched_sample(call, torch)]
                else:
                    timings = event_samples(call, torch)
                row = {"round": round_index, "position": position,
                       "candidate": name, "mode": mode,
                       "median_us": statistics.median(timings),
                       "samples_us": timings}
                report["observations"].append(row)
                save(args.output, report)
                print("timing", round_index, name, mode, row["median_us"], flush=True)
                if not torch.equal(outputs[name], reference):
                    raise ValueError(f"{name} output changed during {mode}")
        # This is a host-side measure of launch(), not a GPU latency measure.
        for name in modules:
            wall = []
            for _ in range(60):
                start = time.perf_counter_ns()
                calls[name]["search_event"]()
                wall.append((time.perf_counter_ns() - start) / 1000.0)
            torch.cuda.synchronize()
            report["candidates"][name]["host_launch_median_us"] = statistics.median(wall)
        report["complete"] = True
        save(args.output, report)
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)[-1600:],
                           "traceback_tail": traceback.format_exc()[-3000:]}
        save(args.output, report)
        raise


if __name__ == "__main__":
    main()
