"""Construct isolated TMA output-mask and grouped-CTA arithmetic variants."""

from __future__ import annotations

import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASELINE = ROOT / "docs/experiments/triton-h100-tma-load/tma_kernel.py"
OUTPUT = HERE / "output"
NAMES = ("baseline", "unmasked_store", "constant_group", "both")

STORE = "tl.store(c_ptrs, c, mask=(row[:, None] < M) & (col[None, :] < N))"
GROUP = "group_size_m = min(num_pid_m - first_pid_m, GROUP_M)"
GUARD = "if M % 128 or N % 256 or K % 64:"


def sources(baseline: str) -> dict[str, str]:
    if baseline.count(STORE) != 1 or baseline.count(GROUP) != 1 or baseline.count(GUARD) != 1:
        raise ValueError("pinned TMA source markers changed")
    variants = {"baseline": baseline}
    variants["unmasked_store"] = baseline.replace(STORE, "tl.store(c_ptrs, c)")
    group = baseline.replace(GROUP, "group_size_m = GROUP_M").replace(
        GUARD, "if M % 128 or N % 256 or K % 64 or M % 1024:"
    )
    variants["constant_group"] = group
    variants["both"] = group.replace(STORE, "tl.store(c_ptrs, c)")
    if set(variants) != set(NAMES) or len(set(variants.values())) != len(NAMES):
        raise ValueError("invalid TMA variant construction")
    return variants


def write_variants() -> dict[str, dict[str, str]]:
    OUTPUT.mkdir(exist_ok=True)
    variants = sources(BASELINE.read_text())
    manifest = {}
    for name, source in variants.items():
        path = OUTPUT / f"{name}.py"
        if path.exists() and path.read_text() != source:
            raise FileExistsError(f"refusing to overwrite changed variant: {path}")
        path.write_text(source)
        manifest[name] = {"path": str(path),
                          "sha256": hashlib.sha256(source.encode()).hexdigest()}
    return manifest


if __name__ == "__main__":
    for name, record in write_variants().items():
        print(name, record["sha256"])
