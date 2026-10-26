from __future__ import annotations

from importlib.resources import files

from ..domain import KernelProgram


def load_triton_bf16_gemm_root() -> KernelProgram:
    source = (
        files("kernel_mcts.benchmarks")
        .joinpath("kernels", "bf16_gemm_triton.py")
        .read_text(encoding="utf-8")
    )
    return KernelProgram(source=source, backend="triton_h100")
