"""Bounded scaffold-free staged generation with one stage-mismatch repair."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD, load_triton_bf16_gemm_root
from kernel_mcts.config import load_data, parse_strategies
from kernel_mcts.domain import EvaluationResult, InvalidReason, KernelProgram, ProposalStatus
from kernel_mcts.generation import GenerationRequest
from kernel_mcts.llm_generation import LLMKernelGenerator
from kernel_mcts.messages_client import MessagesClient, MessagesConfig
from kernel_mcts.strategy_applicability import applicable_strategies, triton_parent_stage
from kernel_mcts.triton_action_fidelity import (
    coalesced_layout_edit_mismatch, serial_wgmma_api_mismatch,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--staged-descriptor", action="store_true")
    group.add_argument("--staged-single-cta", action="store_true")
    group.add_argument("--staged-mma-v2", action="store_true")
    group.add_argument("--bridge-v4", action="store_true")
    group.add_argument("--coalesced-v5", action="store_true")
    group.add_argument("--bridge-v5", action="store_true")
    group.add_argument("--bridge-v5-retry", action="store_true")
    group.add_argument("--bridge-v6", action="store_true")
    group.add_argument("--bridge-v6-retry", action="store_true")
    group.add_argument("--descriptor-v8", action="store_true")
    group.add_argument("--wgmma-v9", action="store_true")
    group.add_argument("--wgmma-v10", action="store_true")
    group.add_argument("--wgmma-v11", action="store_true")
    group.add_argument("--wgmma-v13", action="store_true")
    group.add_argument("--wgmma-v14", action="store_true")
    group.add_argument("--wgmma-v15", action="store_true")
    group.add_argument("--wgmma-v16", action="store_true")
    group.add_argument("--staged-two-cta", action="store_true")
    args = parser.parse_args()
    if args.wgmma_v9 or args.wgmma_v10 or args.wgmma_v11 or args.wgmma_v13 or args.wgmma_v14 or args.wgmma_v15 or args.wgmma_v16:
        catalog_name = ("triton_h100_gluon_bridge_local_v16.json" if args.wgmma_v16
                        else "triton_h100_gluon_bridge_local_v15.json" if args.wgmma_v15
                        else "triton_h100_gluon_bridge_local_v14.json" if args.wgmma_v14
                        else "triton_h100_gluon_bridge_local_v13.json" if args.wgmma_v13
                        else "triton_h100_gluon_bridge_local_v11.json" if args.wgmma_v11
                        else "triton_h100_gluon_bridge_local_v10.json" if args.wgmma_v10
                        else "triton_h100_gluon_bridge_local_v9.json")
        strategy_id = "single_cta_gluon_tma_wgmma_serial"
        if args.wgmma_v13 or args.wgmma_v14 or args.wgmma_v15 or args.wgmma_v16:
            strategy_id = "single_cta_gluon_tma_wgmma_serial_guarded"
        output = HERE / ("v16-wgmma-serial-probe" if args.wgmma_v16
                         else "v15-wgmma-serial-probe" if args.wgmma_v15
                         else "v14-wgmma-serial-probe" if args.wgmma_v14
                         else "v13-wgmma-serial-probe" if args.wgmma_v13
                         else "v11-wgmma-serial-probe" if args.wgmma_v11
                         else "v10-wgmma-serial-probe" if args.wgmma_v10
                         else "v9-wgmma-serial-probe")
        parent_path = HERE / "v8-packaged-root-mcts-bgen4-complete-gluon.py"
        parent = KernelProgram(parent_path.read_text(), backend="triton_h100")
        parent_median_us = 1718.960
        cublas_median_us = 181.040
    elif args.descriptor_v8:
        catalog_name = "triton_h100_gluon_bridge_local_v8.json"
        strategy_id = "descriptor_tma_staging"
        output = HERE / "v8-descriptor-probe"
        parent = load_triton_bf16_gemm_root()
        parent_median_us = 832.784
        cublas_median_us = 183.568
    elif args.bridge_v5 or args.bridge_v5_retry or args.bridge_v6 or args.bridge_v6_retry:
        catalog_name = ("triton_h100_gluon_bridge_v6.json" if args.bridge_v6 or args.bridge_v6_retry
                        else "triton_h100_gluon_bridge_local_v5.json")
        strategy_id = "gluon_mma_v2_pointer_port"
        output = HERE / ("v6-mma-v2-bridge-retry" if args.bridge_v6_retry else
                         "v6-mma-v2-bridge-probe" if args.bridge_v6 else
                         "v5-mma-v2-bridge-retry" if args.bridge_v5_retry
                         else "v5-mma-v2-bridge-repair")
        parent_path = HERE / "v3-descriptor-repair/attempt-1-candidate.py"
        parent = KernelProgram(parent_path.read_text(), backend="triton_h100")
        parent_median_us = 300.448
        cublas_median_us = 181.536
    elif args.coalesced_v5:
        catalog_name = "triton_h100_gluon_bridge_local_v5.json"
        strategy_id = "gluon_mma_v2_coalesced_layout"
        output = HERE / "v5-coalesced-layout-repair"
        parent_path = HERE / "v4-mma-v2-thread-layout-repair/candidate.py"
        parent = KernelProgram(parent_path.read_text(), backend="triton_h100")
        parent_median_us = 3302.400
        cublas_median_us = 182.064
    elif args.bridge_v4:
        catalog_name = "triton_h100_gluon_bridge_v4.json"
        strategy_id = "gluon_mma_v2_pointer_port"
        output = HERE / "v4-mma-v2-pointer-repair"
        parent_path = HERE / "v3-descriptor-repair/attempt-1-candidate.py"
        parent = KernelProgram(parent_path.read_text(), backend="triton_h100")
        parent_median_us = 300.448
        cublas_median_us = 181.536
    elif args.staged_two_cta:
        catalog_name = "triton_h100_strategies_v3_staged.json"
        strategy_id = "two_cta_cluster_partition"
        output = HERE / "v3-two-cta-repair"
        parent_path = HERE / "single_cta_gluon_fixture.py"
        parent = KernelProgram(parent_path.read_text(), backend="triton_h100")
        parent_median_us = 249.632
        cublas_median_us = 181.424
    elif args.staged_single_cta or args.staged_mma_v2:
        catalog_name = "triton_h100_strategies_v3_staged.json"
        strategy_id = (
            "gluon_mma_v2_port" if args.staged_mma_v2
            else "single_cta_gluon_wgmma_tma"
        )
        output = HERE / (
            "v3-mma-v2-repair" if args.staged_mma_v2 else "v3-single-cta-repair"
        )
        parent_path = HERE / "v3-descriptor-repair/attempt-1-candidate.py"
        parent = KernelProgram(parent_path.read_text(), backend="triton_h100")
        parent_median_us = 300.448
        cublas_median_us = 181.536
    elif args.staged_descriptor:
        catalog_name = "triton_h100_strategies_v3_staged.json"
        strategy_id = "descriptor_tma_staging"
        output = HERE / "v3-descriptor-repair"
        parent = load_triton_bf16_gemm_root()
        parent_median_us = 832.784
        cublas_median_us = 183.568
    else:
        catalog_name = "triton_h100_strategies_v2.json"
        strategy_id = "clustered_gluon_wgmma_tma"
        output = HERE / "v2-model-repair"
        parent = load_triton_bf16_gemm_root()
        parent_median_us = 832.784
        cublas_median_us = 183.568
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    base = os.environ.get("ENMAAS_BASE_URL", "").rstrip("/")
    if not base.startswith("https://") or not os.environ.get("ENMAAS_KEY"):
        raise RuntimeError("ENMAAS_BASE_URL and ENMAAS_KEY must be set")
    catalog_path = ROOT / "configs" / catalog_name
    strategy = next(item for item in parse_strategies(load_data(catalog_path))
                    if item.id == strategy_id)
    if strategy.scaffold_resource is not None or strategy.candidate_stage is None:
        raise ValueError("expected scaffold-free staged action")
    if strategy not in applicable_strategies(
        (strategy,), parent, BF16_GEMM_WORKLOAD, {"gpu_model": "H100"}
    ):
        raise ValueError("strategy does not apply to selected parent stage")
    request = GenerationRequest(
        parent=parent, strategy=strategy,
        workload=BF16_GEMM_WORKLOAD, hardware={"gpu_model": "H100"}, profile=None,
        current_benchmark={"median_us": parent_median_us, "measurement_count": 30},
        reference_baselines={"cublas_median_us": cublas_median_us},
    )
    client = MessagesClient(MessagesConfig(
        model="rits/zai-org/glm-5-3", url=base + "/messages",
        max_output_tokens=8192, timeout_seconds=180, thinking_mode="disabled",
    ))
    generator = LLMKernelGenerator(client)
    output.mkdir()
    report = {
        "schema_version": 1,
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "strategy_id": strategy.id,
        "parent_source_sha256": hashlib.sha256(parent.source.encode()).hexdigest(),
        "expected_candidate_stage": strategy.candidate_stage,
        "scaffold": False,
        "max_b_gen": 2,
        "max_repairs": 1,
        "attempts": [],
        "complete": False,
    }

    def save() -> None:
        (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    save()
    for index in range(2):
        try:
            result = generator.generate(request)
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"[-1000:]
            report["failed_attempt"] = index
            save()
            raise
        prefix = f"attempt-{index}"
        (output / f"{prefix}-prompt.json").write_text(result.prompt_text or "")
        (output / f"{prefix}-raw.txt").write_text(result.raw_output)
        source = result.program.source if result.program else None
        if source is not None:
            (output / f"{prefix}-candidate.py").write_text(source)
        stage = triton_parent_stage(source) if source else "missing"
        edit_mismatch = (
            coalesced_layout_edit_mismatch(parent.source, source)
            if args.coalesced_v5 and source else
            serial_wgmma_api_mismatch(parent.source, source)
            if (args.wgmma_v13 or args.wgmma_v14 or args.wgmma_v15 or args.wgmma_v16) and source else None
        )
        row = {
            "attempt": index, "b_gen": index + 1,
            "generation_id": result.generation_id,
            "model": (result.metadata or {}).get("model"),
            "prompt_sha256": result.prompt_hash,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_seconds": result.latency_seconds,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest() if source else None,
            "candidate_stage": stage,
            "edit_mismatch": edit_mismatch,
            "stage_accepted": stage == strategy.candidate_stage and edit_mismatch is None,
        }
        report["attempts"].append(row)
        save()
        print(json.dumps(row, sort_keys=True), flush=True)
        if row["stage_accepted"]:
            break
        evaluation = EvaluationResult(
            ProposalStatus.INVALID,
            program=result.program,
            invalid_reason=(InvalidReason.STRATEGY_MISMATCH if source else InvalidReason.OTHER),
            metadata={"expected_candidate_stage": strategy.candidate_stage,
                      "observed_candidate_stage": stage},
        )
        repair_strategy = (
            replace(strategy, prompts={"triton_h100": strategy.prompt_for("triton_h100")
                                        + " Static edit guard: " + edit_mismatch})
            if edit_mismatch else strategy
        )
        request = replace(request, strategy=repair_strategy, attempt=index + 1,
                          previous_program=result.program, previous_result=evaluation)
    report["b_gen_used"] = len(report["attempts"])
    report["stage_accepted"] = bool(report["attempts"][-1]["stage_accepted"])
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
