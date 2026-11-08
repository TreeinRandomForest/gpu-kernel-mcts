"""Profile one wait-two or wait-one Gluon GEMM under NCU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ARMS = ("candidate", "control")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()

    import torch
    import triton

    m = n = k = 4096
    torch.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    reference = torch.mm(a, b.T)
    c = torch.empty_like(reference)
    compiled = None
    source_path = getattr(args, args.arm)
    source = load_module(source_path, "profile_gluon_" + args.arm)
    paired = load_module(args.paired, "profile_gluon_paired")
    desc, grid = paired.descriptors(source, a, b, c)
    compiled = source.gemm[grid](*desc, True, True, 1,
                                 num_warps=source.WARPS, num_ctas=2)
    launch = lambda: compiled[grid](*desc, True, True, 1)
    launch()
    torch.cuda.synchronize()
    exact = bool(torch.equal(c, reference))
    if not exact:
        raise ValueError(f"incorrect output: {args.arm}")
    for _ in range(10):
        launch()
    torch.cuda.synchronize()
    range_name = f"GLUON_WAIT2_{args.arm.upper()}"
    torch.cuda.nvtx.range_push(range_name)
    launch()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print(json.dumps({
        "arm": args.arm, "correctness": {"exact": exact, "maximum_error": 0.0},
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "paired_sha256": hashlib.sha256(args.paired.read_bytes()).hexdigest(),
        "cubin_sha256": (hashlib.sha256(compiled.asm["cubin"]).hexdigest()
                         if compiled is not None else None),
        "triton_version": triton.__version__, "torch_version": torch.__version__,
        "nvtx_range": range_name,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
