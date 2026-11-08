"""One bounded model probe for the v2 clustered-Gluon structural action."""

from __future__ import annotations

import ast
import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD, load_triton_bf16_gemm_root
from kernel_mcts.config import load_data, parse_strategies
from kernel_mcts.generation import GenerationRequest
from kernel_mcts.llm_generation import LLMKernelGenerator
from kernel_mcts.messages_client import MessagesClient, MessagesConfig
from kernel_mcts.strategy_applicability import triton_parent_stage


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaffold", action="store_true")
    args = parser.parse_args()
    output = HERE / ("v2-model-probe-scaffold" if args.scaffold else "v2-model-probe")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    base = os.environ.get("ENMAAS_BASE_URL", "").rstrip("/")
    if not base.startswith("https://") or not os.environ.get("ENMAAS_KEY"):
        raise RuntimeError("ENMAAS_BASE_URL and ENMAAS_KEY must be set")
    catalog_path = ROOT / "configs/triton_h100_strategies_v2.json"
    strategy = next(item for item in parse_strategies(load_data(catalog_path))
                    if item.id == "clustered_gluon_wgmma_tma")
    if args.scaffold:
        strategy = replace(
            strategy,
            scaffold_resource="bf16_gemm_gluon_wait1.py",
            prompts={**strategy.prompts, "triton_h100": (
                strategy.prompt_for("triton_h100")
                + " A validated source scaffold is supplied in strategy.source_scaffold; "
                "use it as the structural basis for the complete replacement source."
            )},
        )
    request = GenerationRequest(
        parent=load_triton_bf16_gemm_root(), strategy=strategy,
        workload=BF16_GEMM_WORKLOAD, hardware={"gpu_model": "H100"}, profile=None,
        current_benchmark={"median_us": 832.784, "measurement_count": 30},
        reference_baselines={"cublas_median_us": 183.568},
    )
    client = MessagesClient(MessagesConfig(
        model="rits/zai-org/glm-5-3", url=base + "/messages",
        max_output_tokens=8192, timeout_seconds=180, thinking_mode="disabled",
    ))
    result = LLMKernelGenerator(client).generate(request)
    output.mkdir()
    (output / "prompt.json").write_text(result.prompt_text or "")
    (output / "raw.txt").write_text(result.raw_output)
    source = result.program.source if result.program else None
    if source is not None:
        (output / "candidate.py").write_text(source)
    syntax_ok = False
    if source is not None:
        try:
            ast.parse(source)
            syntax_ok = True
        except SyntaxError:
            pass
    summary = {
        "schema_version": 1,
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "scaffold_override": args.scaffold,
        "prompt_sha256": result.prompt_hash,
        "response_id": result.generation_id,
        "model": (result.metadata or {}).get("model"),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_seconds": result.latency_seconds,
        "source_extracted": source is not None,
        "syntax_ok": syntax_ok,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest() if source else None,
        "classified_stage": triton_parent_stage(source) if syntax_ok and source else None,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
