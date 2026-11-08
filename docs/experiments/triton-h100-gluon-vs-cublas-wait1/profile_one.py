"""NCU-select one matched wait-one Gluon or cuBLAS launch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--arm", choices=("gluon", "cublas"), required=True)
    args = parser.parse_args()
    import torch
    import triton

    torch.manual_seed(0)
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    reference = torch.mm(a, b.T)
    c = torch.empty_like(reference)
    compiled = None
    if args.arm == "gluon":
        source = load(args.source, "profile_wait1_gluon")
        helper = load(args.helper, "profile_wait1_helper")
        desc, grid = helper.descriptors(source, a, b, c)
        compiled = source.gemm[grid](*desc, True, True, 1,
                                     num_warps=source.WARPS, num_ctas=2)
        launch = lambda: compiled[grid](*desc, True, True, 1)
    else:
        launch = lambda: torch.mm(a, b.T, out=c)
    launch()
    torch.cuda.synchronize()
    exact = bool(torch.equal(c, reference))
    if not exact:
        raise ValueError(f"incorrect output: {args.arm}")
    for _ in range(10):
        launch()
    torch.cuda.synchronize()
    range_name = f"WAIT1_VS_CUBLAS_{args.arm.upper()}"
    torch.cuda.nvtx.range_push(range_name)
    launch()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print(json.dumps({"arm": args.arm, "correctness_exact": exact,
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(args.helper.read_bytes()).hexdigest(),
        "cubin_sha256": (hashlib.sha256(compiled.asm["cubin"]).hexdigest()
                         if compiled is not None else None),
        "triton_version": triton.__version__, "torch_version": torch.__version__,
        "nvtx_range": range_name}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
