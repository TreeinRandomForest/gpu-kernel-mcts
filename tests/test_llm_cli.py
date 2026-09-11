from __future__ import annotations

import json

import pytest

from kernel_mcts.llm import LLMCompletion
from kernel_mcts.llm_cli import main


def write_strategies(path) -> None:
    path.write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "id": "shared_memory",
                        "description": "Improve data reuse",
                        "prompts": {"cuda_cpp": "Use shared-memory staging."},
                    }
                ]
            }
        )
    )


class FakeOpenAIClient:
    configs = []
    prompts = []

    def __init__(self, config) -> None:
        self.configs.append(config)

    def complete(self, prompt):
        self.prompts.append(prompt)
        return LLMCompletion(
            "response-1",
            "generated kernel",
            "resolved-model",
            input_tokens=100,
            output_tokens=200,
            latency_seconds=1.25,
        )


def test_guarded_cli_makes_one_call_and_writes_candidate(
    tmp_path, monkeypatch, capsys
) -> None:
    strategies = tmp_path / "strategies.json"
    output = tmp_path / "candidate.cu"
    write_strategies(strategies)
    FakeOpenAIClient.configs.clear()
    FakeOpenAIClient.prompts.clear()
    monkeypatch.setattr("kernel_mcts.llm_cli.OpenAIResponsesClient", FakeOpenAIClient)

    result = main(
        [
            "--model",
            "requested-model",
            "--strategies",
            str(strategies),
            "--strategy-id",
            "shared_memory",
            "--output",
            str(output),
            "--reasoning-effort",
            "high",
            "--max-output-tokens",
            "4096",
            "--confirm-api-call",
        ]
    )

    assert result == 0
    assert output.read_text() == "generated kernel"
    assert len(FakeOpenAIClient.prompts) == 1
    prompt = json.loads(FakeOpenAIClient.prompts[0])
    assert prompt["strategy"]["id"] == "shared_memory"
    assert "bf16_gemm_root" in prompt["parent_kernel"]
    config = FakeOpenAIClient.configs[0]
    assert config.model == "requested-model"
    assert config.reasoning_effort == "high"
    assert config.max_output_tokens == 4096
    assert config.store is False
    stdout = capsys.readouterr().out
    assert "resolved-model" in stdout
    assert "input_tokens=100" in stdout
    assert "generated kernel" not in stdout


def test_cli_requires_explicit_confirmation(tmp_path, capsys) -> None:
    strategies = tmp_path / "strategies.json"
    write_strategies(strategies)

    with pytest.raises(SystemExit):
        main(
            [
                "--model",
                "test-model",
                "--strategies",
                str(strategies),
                "--strategy-id",
                "shared_memory",
                "--output",
                str(tmp_path / "candidate.cu"),
            ]
        )

    assert "--confirm-api-call is required" in capsys.readouterr().err


def test_cli_refuses_to_overwrite_before_api_call(
    tmp_path, monkeypatch, capsys
) -> None:
    strategies = tmp_path / "strategies.json"
    output = tmp_path / "candidate.cu"
    write_strategies(strategies)
    output.write_text("existing")
    FakeOpenAIClient.configs.clear()
    monkeypatch.setattr("kernel_mcts.llm_cli.OpenAIResponsesClient", FakeOpenAIClient)

    with pytest.raises(SystemExit):
        main(
            [
                "--model",
                "test-model",
                "--strategies",
                str(strategies),
                "--strategy-id",
                "shared_memory",
                "--output",
                str(output),
                "--confirm-api-call",
            ]
        )

    assert "refusing to overwrite" in capsys.readouterr().err
    assert FakeOpenAIClient.configs == []
    assert output.read_text() == "existing"


def test_cli_rejects_unknown_strategy_before_api_call(
    tmp_path, monkeypatch, capsys
) -> None:
    strategies = tmp_path / "strategies.json"
    write_strategies(strategies)
    FakeOpenAIClient.configs.clear()
    monkeypatch.setattr("kernel_mcts.llm_cli.OpenAIResponsesClient", FakeOpenAIClient)

    with pytest.raises(SystemExit):
        main(
            [
                "--model",
                "test-model",
                "--strategies",
                str(strategies),
                "--strategy-id",
                "missing",
                "--output",
                str(tmp_path / "candidate.cu"),
                "--confirm-api-call",
            ]
        )

    assert "available: shared_memory" in capsys.readouterr().err
    assert FakeOpenAIClient.configs == []
