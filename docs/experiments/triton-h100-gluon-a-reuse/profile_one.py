"""Profile one exactness-passing A-reuse arm with NCU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    for name in ("kernel", "baseline", "helper", "paired"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    import torch

    spec = importlib.util.spec_from_file_location("a_reuse_paired_profile", args.paired)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load paired harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if args.arm not in module.NAMES:
        raise ValueError(f"unknown arm: {args.arm}")
    torch.manual_seed(0)
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    reference = torch.mm(a, b.T)
    c = torch.empty_like(reference)
    compiled = None
    if args.arm == "cublas":
        call = lambda: torch.mm(a, b.T, out=c)
    else:
        kernel = module.load(args.kernel, "profile_a_reuse_kernel")
        baseline = module.load(args.baseline, "profile_a_reuse_baseline")
        helper = module.load(args.helper, "profile_a_reuse_helper")
        compiled, call, _ = module.prepare(args.arm, kernel, baseline, helper, a, b, c)
    call()
    torch.cuda.synchronize()
    exact = bool(torch.equal(c, reference))
    if not exact:
        raise ValueError(f"incorrect output: {args.arm}")
    for _ in range(10):
        call()
    torch.cuda.synchronize()
    range_name = "AREUSE_" + args.arm.upper()
    torch.cuda.nvtx.range_push(range_name)
    call()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print(json.dumps({"arm": args.arm, "correctness_exact": exact,
                      "sha256": {name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
                                 for name in ("kernel", "baseline", "helper", "paired")},
                      "cubin_sha256": (hashlib.sha256(compiled.asm["cubin"]).hexdigest()
                                       if compiled is not None else None),
                      "nvtx_range": range_name}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
