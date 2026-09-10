from __future__ import annotations

import json
from dataclasses import replace

import pytest

from kernel_mcts.providers import RunPodPodRequest
from kernel_mcts.runpod_api import (
    HTTPResponse,
    NetworkVolume,
    RunPodAPIError,
    RunPodRESTClient,
)


REQUEST = RunPodPodRequest(
    gpu_type="NVIDIA H100 80GB HBM3",
    gpu_count=1,
    image="registry.example/kernel-mcts:cuda-12.4",
    container_disk_gb=50,
    interruptible=False,
)


class FakeHTTP:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append((method, url, dict(headers), body, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def response(status: int, value=None) -> HTTPResponse:
    body = b"" if value is None else json.dumps(value).encode()
    return HTTPResponse(status, body)


def test_create_pod_uses_official_rest_payload_without_secret_in_body() -> None:
    secret = "runpod-secret"
    http = FakeHTTP([response(201, {"id": "pod-1"})])
    client = RunPodRESTClient(
        secret,
        http=http,
        worker_token_factory=lambda: "worker-secret",
    )

    pod = client.create_pod(REQUEST)

    method, url, headers, body, timeout = http.calls[0]
    assert pod.pod_id == "pod-1"
    assert method == "POST"
    assert url == "https://rest.runpod.io/v1/pods"
    assert headers["Authorization"] == f"Bearer {secret}"
    assert headers["User-Agent"] == "gpu-kernel-mcts/0.1.0"
    assert timeout == 30.0
    assert json.loads(body) == {
        "cloudType": "SECURE",
        "computeType": "GPU",
        "containerDiskInGb": 50,
        "gpuCount": 1,
        "gpuTypeIds": ["NVIDIA H100 80GB HBM3"],
        "gpuTypePriority": "custom",
        "imageName": "registry.example/kernel-mcts:cuda-12.4",
        "interruptible": False,
        "name": "gpu-kernel-mcts",
        "ports": ["8000/http"],
        "volumeInGb": 0,
        "env": {
            "KERNEL_MCTS_CONTAINER_IMAGE": "registry.example/kernel-mcts:cuda-12.4",
            "KERNEL_MCTS_WORKER_TOKEN": "worker-secret",
        },
    }
    assert secret.encode() not in body
    assert secret not in repr(client)


def test_create_pod_attaches_network_volume_in_matching_data_center() -> None:
    http = FakeHTTP([response(201, {"id": "pod-1"})])
    client = RunPodRESTClient(
        "api-secret",
        http=http,
        worker_token_factory=lambda: "worker-secret",
    )
    request = replace(
        REQUEST,
        network_volume_id="volume-1",
        data_center_ids=("US-TX-3",),
    )

    client.create_pod(request)

    payload = json.loads(http.calls[0][3])
    assert payload["networkVolumeId"] == "volume-1"
    assert payload["dataCenterIds"] == ["US-TX-3"]
    assert payload["dataCenterPriority"] == "custom"
    assert "volumeInGb" not in payload


def test_wait_polls_until_http_proxy_endpoint_is_running() -> None:
    http = FakeHTTP(
        [
            response(201, {"podId": "pod-1"}),
            response(200, {"desiredStatus": "EXITED_PENDING"}),
            response(200, {"desiredStatus": "RUNNING"}),
            response(200, {"status": "ready"}),
        ]
    )
    clock = FakeClock()
    progress = []
    client = RunPodRESTClient(
        "secret",
        http=http,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        poll_interval_seconds=2.0,
        worker_token_factory=lambda: "worker-secret",
    )
    client.create_pod(REQUEST)

    endpoint = client.wait_until_ready(
        "pod-1", 10.0, progress=lambda status, elapsed: progress.append((status, elapsed))
    )

    assert endpoint.worker_id == "pod-1"
    assert endpoint.address == "https://pod-1-8000.proxy.runpod.net"
    assert endpoint.auth_token == "worker-secret"
    assert "worker-secret" not in repr(endpoint)
    assert clock.sleeps == [2.0]
    assert progress == [("EXITED_PENDING", 0.0), ("RUNNING", 2.0)]
    assert [call[0] for call in http.calls] == ["POST", "GET", "GET", "GET"]


def test_http_readiness_waits_for_authenticated_worker_health() -> None:
    http = FakeHTTP(
        [
            response(201, {"id": "pod-1"}),
            response(200, {"desiredStatus": "RUNNING"}),
            response(503, {"status": "starting"}),
            response(200, {"desiredStatus": "RUNNING"}),
            response(200, {"status": "ready"}),
        ]
    )
    clock = FakeClock()
    client = RunPodRESTClient(
        "api-secret",
        http=http,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        worker_token_factory=lambda: "worker-secret",
    )
    client.create_pod(REQUEST)

    client.wait_until_ready("pod-1", 10.0)

    health_calls = [call for call in http.calls if call[1].endswith("/health")]
    assert len(health_calls) == 2
    assert health_calls[0][2]["Authorization"] == "Bearer worker-secret"
    assert health_calls[0][2]["User-Agent"] == "gpu-kernel-mcts/0.1.0"
    assert clock.sleeps == [2.0]


def test_tcp_endpoint_waits_for_public_port_mapping() -> None:
    request = replace(REQUEST, worker_port=22, worker_protocol="tcp")
    http = FakeHTTP(
        [
            response(201, {"id": "pod-1"}),
            response(200, {"desiredStatus": "RUNNING", "portMappings": {}}),
            response(
                200,
                {
                    "desiredStatus": "RUNNING",
                    "publicIp": "192.0.2.10",
                    "portMappings": {"22": 30123},
                },
            ),
        ]
    )
    clock = FakeClock()
    client = RunPodRESTClient(
        "secret", http=http, sleep=clock.sleep, monotonic=clock.monotonic
    )
    client.create_pod(request)

    endpoint = client.wait_until_ready("pod-1", 10.0)

    assert endpoint.address == "tcp://192.0.2.10:30123"


def test_wait_timeout_is_bounded_and_does_not_expose_secret() -> None:
    secret = "never-log-me"
    http = FakeHTTP(
        [
            response(201, {"id": "pod-1"}),
            response(200, {"desiredStatus": "CREATED"}),
            response(200, {"desiredStatus": "CREATED"}),
            response(200, {"desiredStatus": "CREATED"}),
        ]
    )
    clock = FakeClock()
    client = RunPodRESTClient(
        secret,
        http=http,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        poll_interval_seconds=1.0,
    )
    client.create_pod(REQUEST)

    with pytest.raises(TimeoutError, match="within 2 seconds") as captured:
        client.wait_until_ready("pod-1", 2.0)

    assert secret not in str(captured.value)
    assert clock.now == 2.0


def test_terminal_pod_state_fails_without_waiting() -> None:
    http = FakeHTTP(
        [response(201, {"id": "pod-1"}), response(200, {"desiredStatus": "TERMINATED"})]
    )
    client = RunPodRESTClient("secret", http=http)
    client.create_pod(REQUEST)

    with pytest.raises(RunPodAPIError, match="became TERMINATED"):
        client.wait_until_ready("pod-1", 10.0)


def test_terminate_uses_delete_and_accepts_empty_204() -> None:
    http = FakeHTTP([response(201, {"id": "pod-1"}), response(204)])
    client = RunPodRESTClient("secret", http=http)
    client.create_pod(REQUEST)

    client.terminate_pod("pod-1")

    assert http.calls[-1][0:2] == (
        "DELETE",
        "https://rest.runpod.io/v1/pods/pod-1",
    )


@pytest.mark.parametrize(
    "bad_response",
    [response(401, {"error": "secret echoed by server"}), response(200, ["not", "object"])],
)
def test_api_errors_are_sanitized(bad_response: HTTPResponse) -> None:
    secret = "secret-credential"
    http = FakeHTTP([bad_response])
    client = RunPodRESTClient(secret, http=http)

    with pytest.raises(RunPodAPIError) as captured:
        client.create_pod(REQUEST)

    assert secret not in str(captured.value)
    assert "secret echoed by server" not in str(captured.value)


def test_unknown_pod_cannot_be_polled() -> None:
    client = RunPodRESTClient("secret", http=FakeHTTP([]))

    with pytest.raises(ValueError, match="not created by this client"):
        client.wait_until_ready("pod-1", 10.0)


def test_controller_credentials_cannot_be_forwarded_to_worker() -> None:
    request = replace(REQUEST, environment={"RUNPOD_API_KEY": "must-not-forward"})
    client = RunPodRESTClient("controller-secret", http=FakeHTTP([]))

    with pytest.raises(ValueError, match="unsupported keys: RUNPOD_API_KEY"):
        client.create_pod(request)


@pytest.mark.parametrize(
    "changes",
    [
        {"network_volume_id": "volume-1"},
        {"data_center_ids": ("US-TX-3",)},
    ],
)
def test_network_volume_and_data_center_must_be_configured_together(changes) -> None:
    with pytest.raises(ValueError, match="configured together"):
        replace(REQUEST, **changes)


def test_list_and_create_network_volumes() -> None:
    http = FakeHTTP(
        [
            response(
                200,
                [
                    {
                        "id": "existing",
                        "name": "gpu-kernel-mcts",
                        "size": 50,
                        "dataCenterId": "US-TX-3",
                    }
                ],
            ),
            response(
                200,
                {
                    "id": "created",
                    "name": "gpu-kernel-mcts",
                    "size": 100,
                    "dataCenterId": "US-GA-1",
                },
            ),
        ]
    )
    client = RunPodRESTClient("secret", http=http)

    listed = client.list_network_volumes()
    created = client.create_network_volume(
        name="gpu-kernel-mcts",
        size_gb=100,
        data_center_id="US-GA-1",
    )

    assert listed == (NetworkVolume("existing", "gpu-kernel-mcts", 50, "US-TX-3"),)
    assert created == NetworkVolume("created", "gpu-kernel-mcts", 100, "US-GA-1")
    assert http.calls[0][0:2] == (
        "GET",
        "https://rest.runpod.io/v1/networkvolumes",
    )
    assert json.loads(http.calls[1][3]) == {
        "name": "gpu-kernel-mcts",
        "size": 100,
        "dataCenterId": "US-GA-1",
    }
