"""Bounded GLM probe of a local grouped traversal edit from valid Gluon."""

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
from kernel_mcts.triton_action_fidelity import traversal_edit_mismatch
from kernel_mcts.triton_compiler_summary import prompt_triton_compiler_summary


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CATALOG = ROOT / "configs/triton_h100_gluon_local_v1.json"
PARENT = HERE / "single_cta_gluon_fixture.py"
EVIDENCE = HERE / "v3-single-cta-fixture-results.json"
OUTPUT = HERE / "local-traversal-probe-v1"


def save(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base = os.environ.get("ENMAAS_BASE_URL", "").rstrip("/")
    if not base.startswith("https://") or not os.environ.get("ENMAAS_KEY"):
        raise RuntimeError("ENMAAS_BASE_URL and ENMAAS_KEY are required")
    url = base if base.endswith("/messages") else base + "/messages"
    strategy, = parse_strategies(load_data(CATALOG))
    parent = KernelProgram(PARENT.read_text(), backend="triton_h100")
    recorded = json.loads(EVIDENCE.read_text())
    if (not recorded["complete"] or recorded["candidate_sha256"] != hashlib.sha256(
        parent.source.encode()).hexdigest()):
        raise ValueError("parent H100 evidence does not match source")
    row = recorded["measurement"]["candidates"]["generated_candidate"]
    if not row["compile_success"] or not row["correctness_success"]:
        raise ValueError("parent was not validated")
    request = GenerationRequest(
        parent=parent, strategy=strategy, workload=BF16_GEMM_WORKLOAD,
        hardware={"gpu_model": "H100"}, profile=row["lightweight_profile"],
        compiler_summary=prompt_triton_compiler_summary(row["compiler_context"]),
        current_benchmark={"median_us": row["median_us"], "measurement_count": 30},
        reference_baselines={"cublas_median_us": recorded["measurement"]["cublas"]["median_us"]},
    )
    generator = LLMKernelGenerator(MessagesClient(MessagesConfig(
        model="rits/zai-org/glm-5-3", url=url,
        max_output_tokens=8192, timeout_seconds=180, thinking_mode="disabled",
    )))
    OUTPUT.mkdir()
    report = {
        "schema_version": 1, "model": "rits/zai-org/glm-5-3",
        "catalog_sha256": hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
        "parent_source_sha256": hashlib.sha256(parent.source.encode()).hexdigest(),
        "parent_evidence_sha256": hashlib.sha256(EVIDENCE.read_bytes()).hexdigest(),
        "strategy_id": strategy.id, "max_b_gen": 2, "max_repairs": 1,
        "attempts": [], "complete": False,
    }
    save(OUTPUT / "summary.json", report)
    for index in range(2):
        result = generator.generate(request)
        prefix = f"attempt-{index}"
        (OUTPUT / f"{prefix}-prompt.json").write_text(result.prompt_text or "")
        (OUTPUT / f"{prefix}-raw.txt").write_text(result.raw_output)
        source = result.program.source if result.program is not None else None
        if source is not None:
            (OUTPUT / f"{prefix}-candidate.py").write_text(source)
        stage = triton_parent_stage(source) if source else "missing"
        mismatch = (traversal_edit_mismatch(parent.source, source)
                    if source and stage == "single_cta_gluon" else
                    f"expected single_cta_gluon, observed {stage}")
        attempt = {
            "attempt": index, "b_gen": index + 1,
            "generation_id": result.generation_id,
            "prompt_sha256": result.prompt_hash,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest() if source else None,
            "stage": stage, "edit_mismatch": mismatch,
            "accepted_for_gpu": mismatch is None,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_seconds": result.latency_seconds,
        }
        report["attempts"].append(attempt)
        save(OUTPUT / "summary.json", report)
        print(json.dumps(attempt, sort_keys=True), flush=True)
        if mismatch is None:
            break
        request = replace(
            request, attempt=index + 1, previous_program=result.program,
            previous_result=EvaluationResult(
                ProposalStatus.INVALID,
                program=result.program,
                invalid_reason=InvalidReason.STRATEGY_MISMATCH,
                metadata={"detail": mismatch,
                          "expected_candidate_stage": "single_cta_gluon",
                          "observed_candidate_stage": stage,
                          "strategy_edit_guard": strategy.id},
            ),
        )
    report["b_gen_used"] = len(report["attempts"])
    report["accepted_for_gpu"] = bool(report["attempts"][-1]["accepted_for_gpu"])
    report["complete"] = True
    save(OUTPUT / "summary.json", report)


if __name__ == "__main__":
    main()
