from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from kernel_mcts.openai_client import (
    OPENAI_GENERATION_INSTRUCTIONS,
    OpenAIResponsesClient,
    OpenAIResponsesConfig,
)


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
            usage=SimpleNamespace(
                input_tokens=123,
                output_tokens=45,
                input_tokens_details=SimpleNamespace(cached_tokens=64, cache_write_tokens=32),
            ),
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
    assert result.metadata["cached_input_tokens"] == 64
    assert result.metadata["cache_write_input_tokens"] == 32
    assert result.instructions_text == OPENAI_GENERATION_INSTRUCTIONS
    assert responses.calls == [
        {
            "model": "test-model",
            "instructions": OPENAI_GENERATION_INSTRUCTIONS,
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
        ({"model": "test", "base_url": "http://gateway.example/v1"}, "HTTPS"),
        ({"model": "test", "base_url": "https://user:pass@gateway.example/v1"}, "HTTPS"),
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


def test_client_uses_custom_responses_gateway_without_exposing_key(monkeypatch) -> None:
    captured = {}

    class FakeOpenAI:
        def __init__(self, **options):
            captured.update(options)
            self.responses = FakeResponses()

    monkeypatch.setenv("TEST_ENMAAS_KEY", "test-secret")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    subject = OpenAIResponsesClient(OpenAIResponsesConfig(
        model="rits/zai-org/glm-5-3",
        api_key_env="TEST_ENMAAS_KEY",
        base_url="https://gateway.example/v1",
    ))

    result = subject.complete("test prompt")

    assert captured["api_key"] == "test-secret"
    assert captured["base_url"] == "https://gateway.example/v1"
    assert result.metadata["provider"] == "openai_compatible"
    assert "test-secret" not in repr(result.metadata)


def test_bounded_local_conversation_replays_only_recent_text_turn():
    responses = FakeResponses()
    subject = OpenAIResponsesClient(
        OpenAIResponsesConfig(model="test-model", history_turns=1, instructions="JSON only"),
        client=SimpleNamespace(responses=responses),
    )
    subject.complete("first")
    subject.complete("second")
    third = subject.complete("third")

    assert responses.calls[0]["input"] == [{"role": "user", "content": "first"}]
    assert responses.calls[1]["input"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "kernel source"},
        {"role": "user", "content": "second"},
    ]
    assert responses.calls[2]["input"][0]["content"] == "second"
    assert responses.calls[2]["instructions"] == "JSON only"
    assert responses.calls[2]["store"] is False
    assert third.metadata["conversation_input"] == responses.calls[2]["input"]


def test_full_history_bound_keeps_first_turn_visible():
    responses = FakeResponses()
    subject = OpenAIResponsesClient(
        OpenAIResponsesConfig(model="test-model", history_turns=50),
        client=SimpleNamespace(responses=responses),
    )
    for number in range(4):
        subject.complete(f"prompt-{number}")
    messages = responses.calls[3]["input"]
    assert len(messages) == 7
    assert messages[0] == {"role": "user", "content": "prompt-0"}
    assert messages[-1] == {"role": "user", "content": "prompt-3"}
