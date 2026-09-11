from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .benchmarks import BF16_GEMM_WORKLOAD, load_bf16_gemm_root
from .config import load_data, parse_strategies
from .domain import Strategy
from .generation import GenerationRequest
from .llm_generation import LLMKernelGenerator
from .openai_client import OpenAIResponsesClient, OpenAIResponsesConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Make one guarded OpenAI API request for a BF16 GEMM kernel candidate."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--strategies", type=Path, required=True)
    parser.add_argument("--strategy-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-output-tokens", type=int, default=16_384)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--confirm-api-call",
        action="store_true",
        help="required because this command makes one potentially billable API call",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if not arguments.confirm_api_call:
        parser.error("--confirm-api-call is required")
    if arguments.output.exists():
        parser.error(f"refusing to overwrite existing output: {arguments.output}")

    strategies = parse_strategies(load_data(arguments.strategies))
    strategy = _select_strategy(parser, strategies, arguments.strategy_id)
    client = OpenAIResponsesClient(
        OpenAIResponsesConfig(
            model=arguments.model,
            reasoning_effort=arguments.reasoning_effort,
            max_output_tokens=arguments.max_output_tokens,
            timeout_seconds=arguments.timeout,
            api_key_env=arguments.api_key_env,
            store=False,
        )
    )
    generation = LLMKernelGenerator(client).generate(
        GenerationRequest(
            parent=load_bf16_gemm_root(),
            strategy=strategy,
            workload=BF16_GEMM_WORKLOAD,
            hardware={
                "gpu_model": "NVIDIA H100 80GB HBM3",
                "form_factor": "SXM",
                "minimum_compute_capability": "9.0",
            },
            profile=None,
        )
    )
    if generation.program is None:
        print(
            "The API response contained no extractable kernel source; no file was written.",
            file=sys.stderr,
        )
        return 1

    arguments.output.write_text(generation.program.source, encoding="utf-8")
    metadata = generation.metadata or {}
    print(
        f"Generated candidate: response_id={generation.generation_id}, "
        f"model={metadata.get('model', arguments.model)}, "
        f"input_tokens={generation.input_tokens}, "
        f"output_tokens={generation.output_tokens}, "
        f"latency_seconds={generation.latency_seconds:.3f}"
    )
    print(f"Candidate source: {arguments.output.resolve()}")
    return 0


def _select_strategy(
    parser: argparse.ArgumentParser,
    strategies: tuple[Strategy, ...],
    strategy_id: str,
) -> Strategy:
    matches = [strategy for strategy in strategies if strategy.id == strategy_id]
    if len(matches) != 1:
        available = ", ".join(strategy.id for strategy in strategies) or "none"
        parser.error(
            f"strategy {strategy_id!r} must match exactly once; available: {available}"
        )
    return matches[0]


if __name__ == "__main__":
    raise SystemExit(main())
