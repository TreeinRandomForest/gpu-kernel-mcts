from __future__ import annotations

import json
from urllib.error import HTTPError

import pytest

from kernel_mcts.messages_client import MessagesClient, MessagesConfig


def test_messages_client_sends_anthropic_request_and_maps_usage(monkeypatch) -> None:
    monkeypatch.setenv("TEST_ENMAAS_KEY", "test-secret")
    calls = []

    def send(request, timeout):
        calls.append((request, timeout))
        return json.dumps({
            "id": "msg-1", "model": "rits/zai-org/glm-5-3",
            "stop_reason": "end_turn",
            "content": [{"type": "thinking", "thinking": "hidden"},
                        {"type": "text", "text": "kernel source"}],
            "usage": {"input_tokens": 100, "cache_creation_input_tokens": 10,
                      "cache_read_input_tokens": 20, "output_tokens": 45},
        }).encode()

    client = MessagesClient(MessagesConfig(
        model="rits/zai-org/glm-5-3", url="https://gateway.example/v1/messages",
        api_key_env="TEST_ENMAAS_KEY", instructions="Return CUDA only",
    ), send=send)
    result = client.complete("optimize")

    request, timeout = calls[0]
    assert timeout == 180.0
    assert request.get_method() == "POST"
    assert dict((key.lower(), value) for key, value in request.header_items()) == {
        "x-api-key": "test-secret", "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    assert json.loads(request.data) == {
        "model": "rits/zai-org/glm-5-3", "max_tokens": 16_384,
        "system": "Return CUDA only",
        "messages": [{"role": "user", "content": "optimize"}],
    }
    assert result.output_text == "kernel source"
    assert result.response_id == "msg-1"
    assert result.input_tokens == 130
    assert result.output_tokens == 45
    assert result.metadata["cached_input_tokens"] == 20
    assert result.metadata["cache_write_input_tokens"] == 10
    assert result.metadata["provider"] == "messages"
    assert "test-secret" not in repr(result)


def test_messages_client_replays_only_bounded_visible_turns(monkeypatch) -> None:
    monkeypatch.setenv("TEST_ENMAAS_KEY", "test-secret")
    payloads = []

    def send(request, _timeout):
        payloads.append(json.loads(request.data))
        return json.dumps({
            "id": f"msg-{len(payloads)}", "content": [{"type": "text", "text": "answer"}],
            "usage": {"input_tokens": 2, "output_tokens": 1},
        }).encode()

    client = MessagesClient(MessagesConfig(
        model="glm", url="https://gateway.example/v1/messages",
        api_key_env="TEST_ENMAAS_KEY", history_turns=1,
    ), send=send)
    client.complete("first")
    client.complete("second")
    third = client.complete("third")

    assert payloads[0]["messages"] == [{"role": "user", "content": "first"}]
    assert payloads[2]["messages"] == [
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "third"},
    ]
    assert third.metadata["conversation_input"] == payloads[2]["messages"]
    assert third.metadata["history_turns"] == 1


def test_messages_client_does_not_expose_http_error_body(monkeypatch) -> None:
    monkeypatch.setenv("TEST_ENMAAS_KEY", "test-secret")

    def send(_request, _timeout):
        raise HTTPError("https://gateway.example/v1/messages", 401, "secret detail", None, None)

    client = MessagesClient(MessagesConfig(
        model="glm", url="https://gateway.example/v1/messages",
        api_key_env="TEST_ENMAAS_KEY",
    ), send=send)
    with pytest.raises(RuntimeError, match="HTTP 401") as error:
        client.complete("prompt")
    assert "secret detail" not in str(error.value)


def test_messages_client_requires_key_and_https(monkeypatch) -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        MessagesConfig(model="glm", url="http://gateway.example/v1/messages")
    monkeypatch.delenv("MISSING_ENMAAS_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MISSING_ENMAAS_KEY"):
        MessagesClient(MessagesConfig(
            model="glm", url="https://gateway.example/v1/messages",
            api_key_env="MISSING_ENMAAS_KEY",
        ))
