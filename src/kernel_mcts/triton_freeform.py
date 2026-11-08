"""Versioned, opt-in freeform action for pure-Triton MCTS."""

from .domain import Strategy


TRITON_FREEFORM_ACTION = "freeform_triton"
TRITON_FREEFORM_PROMPT_VERSION = 1
TRITON_FREEFORM_PROMPT = (
    "Optimize the current verified Triton kernel freely for this workload. "
    "You may make coupled source changes and need not choose a catalog strategy. "
    "Return one complete Python source file using @triton.jit and triton.language; "
    "do not use Gluon. Preserve launch(A,B,C,M,N,K), the BF16 tensor layouts, "
    "FP32 accumulation, correctness tolerance, and output ownership. "
    "Use the measured parent timing and any supplied context to guide the edit. "
    "A local slowdown is permitted for exploration."
)


def pure_triton_freeform_strategy() -> Strategy:
    return Strategy(
        TRITON_FREEFORM_ACTION,
        "Open-ended pure-Triton source optimization",
        {"triton_h100": TRITON_FREEFORM_PROMPT},
        applicability={
            "backend": "triton_h100",
            "operation": "gemm",
            "gpu_model": "H100",
            "parent_stages": ["pointer_triton", "descriptor_triton"],
        },
    )
