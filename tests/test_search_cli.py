from __future__ import annotations

import pytest

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD, load_bf16_gemm_root
from kernel_mcts.domain import Strategy
from kernel_mcts.generation import GenerationRequest
from kernel_mcts.search_cli import build_parser, main
from kernel_mcts.smoke import SmokeKernelGenerator


def test_smoke_generator_is_deterministic_and_one_shot() -> None:
    generator = SmokeKernelGenerator()
    request = GenerationRequest(
        parent=load_bf16_gemm_root(),
        strategy=Strategy("smoke", "smoke", {"cuda_cpp": "smoke"}),
        workload=BF16_GEMM_WORKLOAD,
        hardware={"gpu_model": "H100"},
        profile=None,
    )

    result = generator.generate(request)

    assert result.generation_id == "smoke-generation-1"
    assert result.program is not None
    assert result.program != request.parent
    assert result.metadata == {"generator": "deterministic-smoke", "llm_call": False}
    with pytest.raises(RuntimeError, match="exactly one"):
        generator.generate(request)


def test_search_cli_rejects_non_smoke_generation_budget(capsys) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--image",
                "worker:v1",
                "--trace",
                "trace.sqlite",
                "--generation-budget",
                "2",
                "--confirm-create-and-terminate",
            ]
        )

    assert "requires --generation-budget=1" in capsys.readouterr().err


def test_search_cli_parser_accepts_manual_volume_pair() -> None:
    arguments = build_parser().parse_args(
        [
            "--image",
            "worker:v1",
            "--trace",
            "trace.sqlite",
            "--network-volume-id",
            "volume-1",
            "--data-center-id",
            "EUR-IS-3",
        ]
    )

    assert arguments.network_volume_id == "volume-1"
    assert arguments.data_center_id == "EUR-IS-3"
