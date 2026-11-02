"""Small real causal attention kernel for operation-independent overlay validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import triton
import triton.language as tl


SEQ, DIM, BLOCK_M, BLOCK_N = 256, 64, 32, 64


@triton.jit
def causal_attention_probe(Q, K, V, O, N: tl.constexpr, D: tl.constexpr,
                           BM: tl.constexpr, BN: tl.constexpr,
                           SCALE: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.arange(0, BN)
    d = tl.arange(0, D)
    q = tl.load(Q + m[:, None] * D + d[None, :], m[:, None] < N, other=0)
    running_max = tl.full((BM,), float("-inf"), tl.float32)
    running_sum = tl.full((BM,), 0, tl.float32)
    acc = tl.full((BM, D), 0, tl.float32)
    for start_n in range(0, N, BN):
        cols = start_n + n
        k = tl.load(K + cols[None, :] * D + d[:, None],
                    cols[None, :] < N, other=0)
        scores = tl.dot(q, k) * SCALE
        scores = tl.where((cols[None, :] <= m[:, None]) &
                          (cols[None, :] < N), scores, -1.0e6)
        next_max = tl.maximum(running_max, tl.max(scores, 1))
        rescale = tl.exp(running_max - next_max)
        probabilities = tl.exp(scores - next_max[:, None])
        v = tl.load(V + cols[:, None] * D + d[None, :],
                    cols[:, None] < N, other=0)
        acc = acc * rescale[:, None]
        acc = tl.dot(probabilities.to(tl.bfloat16), v, acc)
        running_sum = running_sum * rescale + tl.sum(probabilities, 1)
        running_max = next_max
    output = acc / running_sum[:, None]
    tl.store(O + m[:, None] * D + d[None, :], output, m[:, None] < N)


def launch(q, k, v, out):
    return causal_attention_probe[(triton.cdiv(SEQ, BLOCK_M),)](
        q, k, v, out, SEQ, DIM, BLOCK_M, BLOCK_N, DIM ** -0.5,
        num_warps=4, num_stages=2,
    )


def correctness(q, k, v, out):
    score = q.float() @ k.float().T * (DIM ** -0.5)
    rows = torch.arange(SEQ, device="cuda")
    score.masked_fill_(rows[:, None] < rows[None, :], float("-inf"))
    reference = torch.softmax(score, dim=-1) @ v.float()
    actual = out.float()
    if not torch.isfinite(actual).all().item():
        raise RuntimeError("attention output contains nonfinite values")
    absolute = (actual - reference).abs()
    torch.testing.assert_close(actual, reference, atol=0.025, rtol=0.03)
    return {"max_absolute_error": absolute.max().item(),
            "mean_absolute_error": absolute.mean().item(),
            "tolerance_atol": 0.025, "tolerance_rtol": 0.03}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("compile", "profile"), default="compile")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("attention probe requires an H100-class CUDA 9.0 GPU")
    torch.manual_seed(23)
    q = torch.randn((SEQ, DIM), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((SEQ, DIM), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((SEQ, DIM), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q)
    compiled = launch(q, k, v, out)
    check = correctness(q, k, v, out)
    if args.mode == "profile":
        for _ in range(10):
            launch(q, k, v, out)
        torch.cuda.synchronize()
        print("ATTENTION_PROFILE_OK", flush=True)
        return
    if args.output is None:
        parser.error("--output required in compile mode")
    args.output.mkdir(parents=True, exist_ok=True)
    ptx = compiled.asm["ptx"]
    (args.output / "attention.ptx").write_text(ptx)
    result = {
        "schema_version": 1, "kernel": "causal_attention_probe",
        "shape": {"sequence": SEQ, "head_dimension": DIM,
                  "query_block": BLOCK_M, "key_block": BLOCK_N},
        "dtype": "bfloat16", "causal": True,
        "correctness": check,
        "compiler_resources": {
            "registers_per_thread": compiled.n_regs,
            "spills": compiled.n_spills,
            "static_shared_memory_bytes": compiled.metadata.shared,
        },
        "ptx_bytes": len(ptx.encode()),
        "device_name": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
