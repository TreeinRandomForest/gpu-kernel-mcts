"""Check whether pinned Triton lowers two-CTA TMA GEMM to multicast PTX."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import traceback
from pathlib import Path


M = N = K = 4096
CONFIGS = {
    "baseline": (128, 256, 1),
    "cluster_same_tile": (128, 256, 2),
    "cluster_m": (256, 256, 2),
    "cluster_n": (128, 512, 2),
}


def classify_ptx(ptx: str) -> dict[str, object]:
    return {
        "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
        "ptx_size_bytes": len(ptx.encode()),
        "cluster_mentions": ptx.count("shared::cluster"),
        "multicast_mentions": ptx.count("multicast::cluster"),
        "tma_load_mentions": ptx.count("cp.async.bulk.tensor.2d"),
        "cluster_rank_mentions": ptx.count("cluster_ctarank"),
    }


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("tma_cluster_probe_kernel", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(source: Path, name: str, output_dir: Path) -> dict:
    import torch
    import triton

    bm, bn, num_ctas = CONFIGS[name]
    tma = load_module(source)
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16).T
    c = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    reference = torch.mm(a, b)
    a_desc = tma.create_2d_tma_descriptor(a.data_ptr(), M, K, bm, 64, a.element_size())
    b_desc = tma.create_2d_tma_descriptor(b.data_ptr(), N, K, bn, 64, b.element_size())
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    kernel = tma.bf16_gemm_tma[grid](
        a_desc, b_desc, c, M, N, K, bm, bn, 64, 8,
        num_warps=8, num_stages=3, num_ctas=num_ctas,
    )
    torch.cuda.synchronize()
    if not torch.allclose(c, reference, rtol=0.02, atol=0.02):
        raise RuntimeError("result differs from cuBLAS")
    ptx = kernel.asm["ptx"]
    output_dir.mkdir(exist_ok=True)
    (output_dir / f"{name}.ptx").write_text(ptx)
    return {
        "status": "ok", "variant": name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "tile": [bm, bn, 64], "num_ctas": num_ctas, "grid_programs": grid[0],
        "exact_mismatches_vs_cublas": int((c != reference).sum().item()),
        "maximum_error_vs_cublas": float((c.float() - reference.float()).abs().max().item()),
        "registers_per_thread": kernel.n_regs, "spills": kernel.n_spills,
        "shared_memory_bytes": kernel.metadata.shared,
        **classify_ptx(ptx),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.source, args.variant, args.output_dir)
    except Exception as error:
        result = {"status": "failed", "variant": args.variant,
                  "error_type": type(error).__name__, "error": str(error)[-800:],
                  "traceback_tail": traceback.format_exc()[-1200:]}
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
