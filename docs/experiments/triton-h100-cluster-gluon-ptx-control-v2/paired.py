"""Validate and time one-instruction B-TMA PTX controls on H100."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import random
import statistics
import traceback
from pathlib import Path


ARMS = ("generated_ordinary", "generated_multicast", "reassembled_multicast",
        "patched_ordinary")
EXPECTED_PTX_SHA256 = "981b8c868288cb7962c2a9f81766aab420a735a6a6d77f1ba5a0a2120a66054a"


def load_source(path: Path):
    spec = importlib.util.spec_from_file_location("gluon_ptx_control_source", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def clone_with_cubin(compiled, cubin: bytes):
    """Give the original Triton ABI and launch metadata a distinct CUDA module."""
    cloned = copy.copy(compiled)
    cloned.kernel = cubin
    cloned.module = None
    cloned._module_pid = None
    cloned.function = None
    cloned._run = None
    return cloned


def descriptors(source, a, b, c):
    import triton
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

    m, k = a.shape
    n = b.shape[0]
    block_m = source.CTA_M * 2
    cga_m = [[1, 0]]
    cga_b = [[0, 0]]
    a_layout = gl.NVMMASharedLayout.get_default_for(
        [block_m, source.BLOCK_K], gl.bfloat16, cga_layout=cga_m)
    b_layout = gl.NVMMASharedLayout.get_default_for(
        [source.CTA_N, source.BLOCK_K], gl.bfloat16, cga_layout=cga_b)
    c_layout = gl.NVMMASharedLayout.get_default_for(
        [block_m, source.CTA_N], gl.bfloat16, cga_layout=cga_m)
    args = (TensorDescriptor.from_tensor(a, [block_m, source.BLOCK_K], a_layout),
            TensorDescriptor.from_tensor(b, [source.CTA_N, source.BLOCK_K], b_layout),
            TensorDescriptor.from_tensor(c, [block_m, source.CTA_N], c_layout))
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, source.CTA_N), 1)
    return args, grid


def measure(launch, torch) -> dict:
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
    return {"timings_us": samples, "median_us": statistics.median(samples)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--patched", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    import torch
    import triton

    source = load_source(args.source)
    m = n = k = 4096
    torch.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    reference = torch.mm(a, b.T)
    tensor_args, grid = descriptors(source, a, b, c)
    ordinary = source.gemm[grid](*tensor_args, True, False, True,
                                 num_warps=source.WARPS, num_ctas=2)
    multicast = source.gemm[grid](*tensor_args, True, True, True,
                                  num_warps=source.WARPS, num_ctas=2)
    actual_ptx = hashlib.sha256(multicast.asm["ptx"].encode()).hexdigest()
    if actual_ptx != EXPECTED_PTX_SHA256:
        raise ValueError(f"compiled PTX differs from patched input: {actual_ptx}")
    reassembled = clone_with_cubin(multicast, args.original.read_bytes())
    patched = clone_with_cubin(multicast, args.patched.read_bytes())
    launches = {
        "generated_ordinary": lambda: ordinary[grid](*tensor_args, True, False, True),
        "generated_multicast": lambda: multicast[grid](*tensor_args, True, True, True),
        "reassembled_multicast": lambda: reassembled[grid](*tensor_args, True, True, True),
        "patched_ordinary": lambda: patched[grid](*tensor_args, True, True, True),
    }
    report = {"schema_version": 1, "complete": False,
              "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "ptx_sha256": actual_ptx,
              "original_cubin_sha256": hashlib.sha256(args.original.read_bytes()).hexdigest(),
              "patched_cubin_sha256": hashlib.sha256(args.patched.read_bytes()).hexdigest(),
              "triton_version": triton.__version__, "torch_version": torch.__version__,
              "gpu_name": torch.cuda.get_device_name(), "orders": {}, "observations": []}
    save(args.output, report)
    for round_index in range(3):
        order = list(ARMS)
        random.Random(8127 + round_index).shuffle(order)
        report["orders"][str(round_index)] = order
        for name in order:
            c.zero_()
            try:
                launches[name]()
                torch.cuda.synchronize()
                error = (c.float() - reference.float()).abs()
                exact = bool(torch.equal(c, reference))
                result = {"round": round_index, "name": name,
                          "status": "ok" if exact else "incorrect",
                          "correctness": {"exact": exact,
                                          "maximum_error": float(error.max().item())}}
                if exact:
                    result["benchmark"] = measure(launches[name], torch)
            except Exception as exc:
                result = {"round": round_index, "name": name, "status": "failed",
                          "error_type": type(exc).__name__, "error": str(exc)[-1500:],
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
