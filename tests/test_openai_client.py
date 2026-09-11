from __future__ import annotations

from types import SimpleNamespace

import pytest

from kernel_mcts.openai_client import OpenAIResponsesClient, OpenAIResponsesConfig


class FakeResponses:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **values):
        self.calls.append(values)
        return SimpleNamespace(
            id="response-1",
            output_text="kernel source",
            model="test-model-2026-09-11",
            status="completed",
            usage=SimpleNamespace(input_tokens=123, output_tokens=45),
        )


def test_responses_client_makes_independent_unstored_request() -> None:
    responses = FakeResponses()
    sdk = SimpleNamespace(responses=responses)
    subject = OpenAIResponsesClient(
        OpenAIResponsesConfig(
            model="test-model",
            reasoning_effort="high",
            max_output_tokens=4096,
        ),
        client=sdk,
    )

    result = subject.complete("optimize this kernel")

    assert result.response_id == "response-1"
    assert result.output_text == "kernel source"
    assert result.model == "test-model-2026-09-11"
    assert result.input_tokens == 123
    assert result.output_tokens == 45
    assert result.latency_seconds is not None
    assert result.metadata["provider"] == "openai"
    assert responses.calls == [
        {
            "model": "test-model",
            "instructions": (
                "Return only the complete replacement kernel source. "
                "Do not use Markdown fences or explanatory prose."
            ),
            "input": "optimize this kernel",
            "max_output_tokens": 4096,
            "reasoning": {"effort": "high"},
            "store": False,
        }
    ]
    assert "conversation" not in responses.calls[0]
    assert "previous_response_id" not in responses.calls[0]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"model": ""}, "model"),
        ({"model": "test", "max_output_tokens": 0}, "max_output_tokens"),
        ({"model": "test", "timeout_seconds": 0}, "timeout_seconds"),
    ],
)
def test_openai_config_rejects_invalid_values(values, message) -> None:
    with pytest.raises(ValueError, match=message):
        OpenAIResponsesConfig(**values)


def test_client_requires_configured_api_key(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_TEST_OPENAI_KEY", raising=False)

    with pytest.raises(RuntimeError, match="MISSING_TEST_OPENAI_KEY"):
        OpenAIResponsesClient(
            OpenAIResponsesConfig(
                model="test-model",
                api_key_env="MISSING_TEST_OPENAI_KEY",
            )
        )
