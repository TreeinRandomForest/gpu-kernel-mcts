"""Launch one verified Gluon GEMM inside a targeted NCU NVTX range."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ARMS = {
    "single_pipeline": (False, False, True),
    "cluster_pipeline_ordinary": (True, False, True),
    "cluster_pipeline_multicast": (True, True, True),
}


def load_source(path: Path):
    spec = importlib.util.spec_from_file_location("gluon_cluster_ncu_source", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    args = parser.parse_args()
    import torch
    import triton

    source = load_source(args.source)
    clustered, multicast, pipelined = ARMS[args.arm]
    torch.manual_seed(0)
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    c = torch.empty_like(a)
    reference = torch.mm(a, b.T)
    launch = lambda: source.launch(a, b, c, clustered=clustered,
                                   multicast=multicast, pipelined=pipelined)
    compiled = launch()
    torch.cuda.synchronize()
    error = (c.float() - reference.float()).abs()
    if not bool(torch.all(c == reference).item()):
        raise AssertionError(f"{args.arm} does not exactly match cuBLAS")
    for _ in range(10):
        launch()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"GLUON_PROFILE_{args.arm.upper()}")
    try:
        launch()
    finally:
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print(json.dumps({
        "arm": args.arm, "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "triton_version": triton.__version__, "torch_version": torch.__version__,
        "correctness": {"exact": True, "maximum_error": float(error.max().item())},
        "cubin_sha256": hashlib.sha256(compiled.asm["cubin"]).hexdigest(),
        "registers_per_thread": compiled.n_regs,
        "spills": compiled.n_spills,
        "shared_memory_bytes": compiled.metadata.shared,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
