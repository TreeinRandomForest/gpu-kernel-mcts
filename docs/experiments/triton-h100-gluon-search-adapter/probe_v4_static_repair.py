"""One extra B_gen call with explicit source-level feedback; no GPU claim."""

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
PRIOR = HERE / "v4-mma-v2-pointer-repair"
OUTPUT = HERE / "v4-mma-v2-static-repair"
FINDINGS = (
    "Static source review of the previous candidate, before any GPU compile: "
    "replace both tl.program_id calls with gl.program_id; gl.BlockedLayout "
    "takes four arguments (size_per_thread, threads_per_warp, warps_per_cta, order), "
    "not five; gl.load(pointer, mask=..., other=...) and gl.store(pointer, value, "
    "mask=...) have no layout positional argument. For a 2D [M,K] or [K,N] "
    "BlockedLayout, the row-axis offsets use gl.SliceLayout(1, layout) and the "
    "column-axis offsets use gl.SliceLayout(0, layout). Make output row/column "
    "offsets with slices of the output layout after converting the accumulator. "
    "Use range(0,K,BK) if uncertain about gl.range. Return only the complete "
    "replacement source, with no explanations or Markdown."
)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base = os.environ.get("ENMAAS_BASE_URL", "").rstrip("/")
    if not base.startswith("https://") or not os.environ.get("ENMAAS_KEY"):
        raise RuntimeError("ENMAAS_BASE_URL and ENMAAS_KEY must be set")
    parent = KernelProgram((HERE / "v3-descriptor-repair/attempt-1-candidate.py").read_text(),
                           backend="triton_h100")
    previous = KernelProgram((PRIOR / "attempt-1-candidate.py").read_text(),
                             backend="triton_h100")
    strategy = parse_strategies(load_data(ROOT / "configs/triton_h100_gluon_bridge_v4.json"))[0]
    strategy = replace(strategy, prompts={"triton_h100": strategy.prompt_for("triton_h100")
                                           + " " + FINDINGS})
    request = GenerationRequest(
        parent=parent, strategy=strategy, workload=BF16_GEMM_WORKLOAD,
        hardware={"gpu_model": "H100"}, profile=None,
        current_benchmark={"median_us": 300.448, "measurement_count": 30},
        reference_baselines={"cublas_median_us": 181.536}, attempt=2,
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
        "cumulative_bridge_b_gen": 3, "gpu_compile_attempted": False,
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
