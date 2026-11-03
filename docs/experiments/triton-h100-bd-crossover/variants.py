"""Deterministic 2×2 source crossover around exact traced B; D is a control."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent


def once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"expected exactly one source site: {old[:50]!r}")
    return source.replace(old, new, 1)


def hybrid(source: str, *, offset_loop: bool, hoist_masks: bool) -> str:
    if offset_loop:
        source = once(source,
                      "    for k in range(0, tl.cdiv(K, BK)):\n",
                      "    for k0 in range(0, K, BK):\n        k = k0 + kk\n")
        source = once(source, "((kk[None, :] + k * BK) < K)", "(k[None, :] < K)")
        source = once(source, "((kk[:, None] + k * BK) < K)", "(k[:, None] < K)")
    if hoist_masks:
        source = once(source,
                      "    acc = tl.zeros((BM, BN), dtype=tl.float32)\n",
                      "    acc = tl.zeros((BM, BN), dtype=tl.float32)\n"
                      "    a_mask_row = row[:, None] < M\n"
                      "    b_mask_col = col[None, :] < N\n")
        source = once(source,
                      "a = tl.load(a_ptrs,\n                    mask=(row[:, None] < M) &",
                      "a = tl.load(a_ptrs,\n                    mask=a_mask_row &")
        source = once(source, "& (col[None, :] < N),", "& b_mask_col,")
        source = once(source,
                      "mask=(row[:, None] < M) & (col[None, :] < N))",
                      "mask=a_mask_row & b_mask_col)")
    ast.parse(source)
    return source


def sources() -> dict[str, str]:
    original_b = (HERE / "trace_B.py").read_text()
    original_d = (HERE / "trace_D.py").read_text()
    return {
        "B_exact": original_b,
        "B_loop_only": hybrid(original_b, offset_loop=True, hoist_masks=False),
        "B_mask_only": hybrid(original_b, offset_loop=False, hoist_masks=True),
        "B_both": hybrid(original_b, offset_loop=True, hoist_masks=True),
        "D_exact": original_d,
    }


def write_sources() -> dict[str, dict[str, str]]:
    result = {}
    for name, source in sources().items():
        path = HERE / f"{name}.py"
        if path.exists() and path.read_text() != source:
            raise FileExistsError(f"refusing to replace different source: {path}")
        path.write_text(source)
        result[name] = {"path": path.name,
                        "sha256": hashlib.sha256(source.encode()).hexdigest()}
    return result


if __name__ == "__main__":
    for name, record in write_sources().items():
        print(name, record["sha256"])
