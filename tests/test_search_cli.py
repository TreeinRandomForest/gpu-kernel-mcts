from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD, load_bf16_gemm_root
from kernel_mcts.domain import KernelProgram, Strategy
from kernel_mcts.generation import GenerationRequest, GenerationResult
from kernel_mcts.llm_generation import LLMKernelGenerator
from kernel_mcts.search_cli import (
    ProgressKernelGenerator,
    SearchCLIProgress,
    build_parser,
    main,
)
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


def test_search_cli_parser_accepts_ephemeral_storage() -> None:
    arguments = build_parser().parse_args(
        [
            "--image",
            "worker:v1",
            "--trace",
            "trace.sqlite",
            "--ephemeral-storage",
        ]
    )

    assert arguments.ephemeral_storage


def test_openai_search_requires_model_before_provisioning(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        "kernel_mcts.search_cli.create_runpod_provider",
        lambda *args, **kwargs: pytest.fail("must not provision"),
    )

    with pytest.raises(SystemExit):
        main(
            [
                "--image",
                "worker:v1",
                "--trace",
                str(tmp_path / "trace.sqlite"),
                "--generator",
                "openai",
                "--confirm-create-and-terminate",
            ]
        )

    assert "--model is required" in capsys.readouterr().err


def test_openai_search_wires_configured_generator_and_exports_best(
    tmp_path, monkeypatch, capsys
) -> None:
    strategies = tmp_path / "strategies.json"
    strategies.write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "id": "coalescing",
                        "description": "Improve coalescing",
                        "prompts": {"cuda_cpp": "Coalesce global loads."},
                    }
                ]
            }
        )
    )
    trace = tmp_path / "trace.sqlite"
    best_output = tmp_path / "best.cu"
    captured = {}

    class FakeOpenAIClient:
        def __init__(self, config):
            captured["openai_config"] = config

    def fake_search(**values):
        captured["search"] = values
        node = SimpleNamespace(
            program=KernelProgram("best source"),
            reward=0.5,
        )
        result = SimpleNamespace(
            root=node,
            best=node,
            nodes=(node,),
            iterations=3,
            generations=3,
            prior_calls=0,
            profile_calls=2,
        )
        return SimpleNamespace(run_id="llm-run", result=result)

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("RUNPOD_API_KEY", "test-runpod-key")
    monkeypatch.setattr("kernel_mcts.search_cli.OpenAIResponsesClient", FakeOpenAIClient)
    monkeypatch.setattr(
        "kernel_mcts.search_cli.resolve_reusable_volume",
        lambda *args: ("volume-1", "EUR-IS-3"),
    )
    monkeypatch.setattr(
        "kernel_mcts.search_cli.create_runpod_provider", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr("kernel_mcts.search_cli.run_mcts_search", fake_search)

    result = main(
        [
            "--image",
            "worker:v1",
            "--trace",
            str(trace),
            "--generator",
            "openai",
            "--model",
            "test-model",
            "--strategies",
            str(strategies),
            "--generation-budget",
            "3",
            "--network-volume-id",
            "volume-1",
            "--data-center-id",
            "EUR-IS-3",
            "--best-output",
            str(best_output),
            "--confirm-create-and-terminate",
        ]
    )

    assert result == 0
    assert best_output.read_text() == "best source"
    assert captured["openai_config"].model == "test-model"
    search = captured["search"]
    assert isinstance(search["generator"], ProgressKernelGenerator)
    assert isinstance(search["generator"]._generator, LLMKernelGenerator)
    assert [strategy.id for strategy in search["strategies"]] == ["coalescing"]
    assert search["generation_budget"] == 3
    assert search["model_name"] == "test-model"
    assert search["mcts_config"].max_repairs == 1
    assert search["mcts_config"].max_infrastructure_retries == 1
    assert search["hardware"].required_profilers == ("ncu",)
    assert "OpenAI search completed" in capsys.readouterr().out


def test_search_progress_finishes_readiness_and_reports_events() -> None:
    class Trace:
        def __init__(self) -> None:
            self.events = []

        def start_run(self, *args) -> None:
            pass

        def emit(self, event_type, payload) -> None:
            self.events.append((event_type, payload))

    class Readiness:
        def __init__(self) -> None:
            self.finished = 0

        def finish(self) -> None:
            self.finished += 1

    trace = Trace()
    readiness = Readiness()
    stream = io.StringIO()
    progress = SearchCLIProgress(trace, readiness, stream)

    progress.emit("environment_manifest", {"worker_id": "worker-1"})
    progress.emit("profiling_started", {"node_id": "node-1", "profile_call": 1})
    progress.emit("node_profiled", {"profile_call": 1})
    progress.emit(
        "generation",
        {"b_gen": 3, "proposal_status": "VALID", "strategy_id": "coalescing"},
    )
    progress.emit(
        "iteration_completed",
        {
            "iteration": 2,
            "b_gen": 3,
            "status": "VALID",
            "backed_up_reward": 0.5,
        },
    )
    progress.emit("new_global_best", {"iteration": 2, "reward": 0.5})
    progress.emit(
        "run_completed", {"iterations": 2, "b_gen": 3, "best_reward": 0.5}
    )

    assert readiness.finished == 1
    assert [event for event, _ in trace.events] == [
        "environment_manifest",
        "profiling_started",
        "node_profiled",
        "generation",
        "iteration_completed",
        "new_global_best",
        "run_completed",
    ]
    output = stream.getvalue()
    assert "Worker ready; starting root evaluation" in output
    assert "Starting lightweight profile 1 for node node-1" in output
    assert "Lightweight profile completed: profile_call=1" in output
    assert "B_gen=3, status=VALID, strategy=coalescing" in output
    assert "iteration=2, B_gen=3, status=VALID, backed_up_reward=0.5" in output
    assert "New best: iteration=2, reward=0.5" in output
    assert "Search completed; terminating worker" in output


def test_progress_generator_reports_start_and_evaluation_handoff() -> None:
    class Generator:
        def generate(self, request) -> GenerationResult:
            return GenerationResult(
                "generation-1",
                "source",
                KernelProgram("source"),
                "prompt-hash",
            )

    stream = io.StringIO()
    generator = ProgressKernelGenerator(Generator(), stream)
    request = GenerationRequest(
        parent=KernelProgram("root"),
        strategy=Strategy("coalescing", "Coalesce loads", {}),
        workload=BF16_GEMM_WORKLOAD,
        hardware={"gpu_model": "H100"},
        profile=None,
    )

    result = generator.generate(request)

    assert result.generation_id == "generation-1"
    assert stream.getvalue().splitlines() == [
        "Starting generation call 1: strategy=coalescing",
        "Generation call 1 returned; evaluating proposal",
    ]
