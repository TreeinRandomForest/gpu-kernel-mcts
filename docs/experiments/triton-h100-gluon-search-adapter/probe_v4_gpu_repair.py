"""One B_gen repair using an H100 compiler error from the previous candidate."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD
from kernel_mcts.config import load_data, parse_strategies
from kernel_mcts.domain import (
    CompilationEvidence, CompileStatus, EvaluationResult, InvalidReason, KernelProgram,
    ProposalStatus,
)
from kernel_mcts.generation import GenerationRequest
from kernel_mcts.llm_generation import LLMKernelGenerator
from kernel_mcts.messages_client import MessagesClient, MessagesConfig
from kernel_mcts.strategy_applicability import triton_parent_stage


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = HERE / "v4-mma-v2-gpu-repair"
PRIOR_RESULTS = HERE / "v4-mma-v2-bridge-diagnostic-results.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base = os.environ.get("ENMAAS_BASE_URL", "").rstrip("/")
    if not base.startswith("https://") or not os.environ.get("ENMAAS_KEY"):
        raise RuntimeError("ENMAAS_BASE_URL and ENMAAS_KEY must be set")
    record = json.loads(PRIOR_RESULTS.read_text())
    failure = record["measurement"]["candidates"]["generated_candidate"]
    if failure["compile_success"]:
        raise ValueError("previous candidate unexpectedly compiled")
    parent = KernelProgram((HERE / "v3-descriptor-repair/attempt-1-candidate.py").read_text(),
                           backend="triton_h100")
    previous = KernelProgram((HERE / "v4-mma-v2-static-repair/candidate.py").read_text(),
                             backend="triton_h100")
    strategy = parse_strategies(load_data(ROOT / "configs/triton_h100_gluon_bridge_v4.json"))[0]
    strategy = replace(strategy, prompts={"triton_h100": strategy.prompt_for("triton_h100")
        + " Repair the H100 compilation failure below. Annotate every layout object "
          "assignment with ': gl.constexpr' (a_layout, b_layout, mma_layout, "
          "a_op, b_op, out_layout). Keep the rest of the source as close as "
          "possible to the previous candidate. Return only complete Python code."})
    evaluation = EvaluationResult(
        ProposalStatus.INVALID, program=previous,
        invalid_reason=InvalidReason.COMPILE_FAILURE, compile_status=CompileStatus.FAIL,
        compilation=CompilationEvidence(None, failure["compile_stdout"],
                                        failure["compile_stderr"]),
    )
    request = GenerationRequest(
        parent=parent, strategy=strategy, workload=BF16_GEMM_WORKLOAD,
        hardware={"gpu_model": "H100"}, profile=None,
        current_benchmark={"median_us": 300.448, "measurement_count": 30},
        reference_baselines={"cublas_median_us": 181.536}, attempt=3,
        previous_program=previous, previous_result=evaluation,
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
        "cumulative_bridge_b_gen": 4,
        "previous_h100_compile_success": False,
        "model": (result.metadata or {}).get("model"),
        "generation_id": result.generation_id, "prompt_sha256": result.prompt_hash,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest() if source else None,
        "candidate_stage": triton_parent_stage(source) if source else "missing",
        "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
