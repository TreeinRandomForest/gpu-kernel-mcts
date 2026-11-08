"""One v5 bridge repair with explicit static Gluon API feedback; no GPU claim."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD
from kernel_mcts.config import load_data, parse_strategies
from kernel_mcts.domain import EvaluationResult, InvalidReason, KernelProgram, ProposalStatus
from kernel_mcts.generation import GenerationRequest
from kernel_mcts.llm_generation import LLMKernelGenerator
from kernel_mcts.messages_client import MessagesClient, MessagesConfig
from kernel_mcts.strategy_applicability import triton_parent_stage


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = HERE / "v5-mma-v2-bridge-retry"
OUTPUT = HERE / "v5-mma-v2-bridge-static-repair"
FINDINGS = (
    "Static review before GPU compilation: `import gluon` and "
    "`import gluon.language as gl` are invalid; use `from triton.experimental "
    "import gluon` and `from triton.experimental.gluon import language as gl`. "
    "Inside @gluon.jit use only gl.program_id, gl.arange, gl.reshape, "
    "gl.convert_layout, gl.float32 and gl.bfloat16, never tl.*. "
    "Use `gl.NVMMADistributedLayout(version=[2,0], warps_per_cta=[4,1], "
    "instr_shape=[16,8])`; supply dtype then layout to "
    "`gl.zeros((BM,BN), gl.float32, layout=mma_layout)`. "
    "Keep the A/B/output BlockedLayout orders and 32-thread warp products. "
    "Return complete Python code only."
)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base = os.environ.get("ENMAAS_BASE_URL", "").rstrip("/")
    if not base.startswith("https://") or not os.environ.get("ENMAAS_KEY"):
        raise RuntimeError("ENMAAS_BASE_URL and ENMAAS_KEY must be set")
    parent = KernelProgram((HERE / "v3-descriptor-repair/attempt-1-candidate.py").read_text(),
                           backend="triton_h100")
    previous = KernelProgram((PRIOR / "attempt-0-candidate.py").read_text(),
                             backend="triton_h100")
    strategy = next(item for item in parse_strategies(load_data(
        ROOT / "configs/triton_h100_gluon_bridge_local_v5.json"))
        if item.id == "gluon_mma_v2_pointer_port")
    strategy = replace(strategy, prompts={"triton_h100": strategy.prompt_for("triton_h100")
                                           + " " + FINDINGS})
    request = GenerationRequest(
        parent=parent, strategy=strategy, workload=BF16_GEMM_WORKLOAD,
        hardware={"gpu_model": "H100"}, profile=None,
        current_benchmark={"median_us": 300.448, "measurement_count": 30},
        reference_baselines={"cublas_median_us": 181.536}, attempt=1,
        previous_program=previous,
        previous_result=EvaluationResult(ProposalStatus.INVALID, program=previous,
                                         invalid_reason=InvalidReason.OTHER),
    )
    client = MessagesClient(MessagesConfig(
        model="rits/zai-org/glm-5-3", url=base + "/messages",
        max_output_tokens=8192, timeout_seconds=180, thinking_mode="disabled"))
    OUTPUT.mkdir()
    result = LLMKernelGenerator(client).generate(request)
    (OUTPUT / "prompt.json").write_text(result.prompt_text or "")
    (OUTPUT / "raw.txt").write_text(result.raw_output)
    source = result.program.source if result.program else None
    if source is not None:
        (OUTPUT / "candidate.py").write_text(source)
    report = {
        "schema_version": 1, "b_gen_for_this_probe": 1,
        "cumulative_bridge_b_gen": 2, "gpu_compile_attempted": False,
        "model": (result.metadata or {}).get("model"),
        "generation_id": result.generation_id, "prompt_sha256": result.prompt_hash,
        "feedback_sha256": hashlib.sha256(FINDINGS.encode()).hexdigest(),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest() if source else None,
        "candidate_stage": triton_parent_stage(source) if source else "missing",
        "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
