from __future__ import annotations

import json
import secrets
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol

from .providers import RunPodClient, RunPodPod, RunPodPodRequest, WorkerEndpoint


_ALLOWED_WORKER_ENV = {
    "KERNEL_MCTS_ARTIFACT_ROOT",
    "KERNEL_MCTS_CONTAINER_DIGEST",
    "KERNEL_MCTS_GIT_COMMIT",
    "KERNEL_MCTS_WORKER_PORT",
}


class RunPodAPIError(RuntimeError):
    """A sanitized RunPod lifecycle API failure."""


@dataclass(frozen=True, slots=True)
class HTTPResponse:
    status: int
    body: bytes


class HTTPTransport(Protocol):
    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> HTTPResponse: ...


class RunPodRESTClient(RunPodClient):
    """Minimal adapter for the official RunPod Pods REST API."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://rest.runpod.io/v1",
        request_timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 2.0,
        http: HTTPTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        worker_token_factory: Callable[[], str] = lambda: secrets.token_urlsafe(32),
    ) -> None:
        if not api_key:
            raise ValueError("RunPod API key is required")
        if request_timeout_seconds <= 0 or poll_interval_seconds <= 0:
            raise ValueError("RunPod request timeout and poll interval must be positive")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._request_timeout_seconds = request_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._http = http or _urllib_http
        self._sleep = sleep
        self._monotonic = monotonic
        self._worker_token_factory = worker_token_factory
        self._requests: dict[str, RunPodPodRequest] = {}
        self._worker_tokens: dict[str, str] = {}

    def __repr__(self) -> str:
        return (
            f"RunPodRESTClient(base_url={self._base_url!r}, "
            f"request_timeout_seconds={self._request_timeout_seconds!r})"
        )

    def create_pod(self, request: RunPodPodRequest) -> RunPodPod:
        unexpected_environment = set(request.environment) - _ALLOWED_WORKER_ENV
        if unexpected_environment:
            names = ", ".join(sorted(unexpected_environment))
            raise ValueError(f"worker environment contains unsupported keys: {names}")
        worker_token = self._worker_token_factory()
        if not worker_token:
            raise ValueError("worker token factory returned an empty token")
        payload = {
            "name": request.name,
            "cloudType": request.cloud_type,
            "computeType": "GPU",
            "gpuTypeIds": [request.gpu_type],
            "gpuTypePriority": "custom",
            "gpuCount": request.gpu_count,
            "imageName": request.image,
            "containerDiskInGb": request.container_disk_gb,
            "volumeInGb": 0,
            "interruptible": request.interruptible,
            "ports": [f"{request.worker_port}/{request.worker_protocol}"],
            "env": {
                **request.environment,
                "KERNEL_MCTS_WORKER_TOKEN": worker_token,
                "KERNEL_MCTS_CONTAINER_IMAGE": request.image,
            },
        }
        response = self._request("POST", "/pods", payload)
        pod_id = response.get("id") or response.get("podId")
        if not isinstance(pod_id, str) or not pod_id:
            raise RunPodAPIError("RunPod create response did not contain a pod ID")
        self._requests[pod_id] = request
        self._worker_tokens[pod_id] = worker_token
        return RunPodPod(pod_id)

    def wait_until_ready(self, pod_id: str, timeout_seconds: float) -> WorkerEndpoint:
        if timeout_seconds <= 0:
            raise ValueError("RunPod startup timeout must be positive")
        request = self._requests.get(pod_id)
        if request is None:
            raise ValueError("cannot wait for a pod not created by this client")
        deadline = self._monotonic() + timeout_seconds
        last_status = "UNKNOWN"
        while True:
            pod = self._request("GET", f"/pods/{pod_id}")
            status = pod.get("desiredStatus")
            last_status = status if isinstance(status, str) else "UNKNOWN"
            if last_status == "RUNNING":
                address = _endpoint_address(pod_id, request, pod)
                if address is not None and (
                    request.worker_protocol != "http"
                    or self._worker_is_ready(address, self._worker_tokens[pod_id])
                ):
                    return WorkerEndpoint(
                        worker_id=pod_id,
                        address=address,
                        auth_token=self._worker_tokens[pod_id],
                    )
            if last_status in {"EXITED", "TERMINATED"}:
                raise RunPodAPIError(
                    f"RunPod pod became {last_status} before its worker endpoint was ready"
                )
            if self._monotonic() >= deadline:
                raise TimeoutError(
                    f"RunPod pod was not ready within {timeout_seconds:g} seconds "
                    f"(last status: {last_status})"
                )
            self._sleep(min(self._poll_interval_seconds, max(0.0, deadline - self._monotonic())))

    def _worker_is_ready(self, address: str, token: str) -> bool:
        try:
            response = self._http(
                "GET",
                f"{address}/health",
                {
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                },
                None,
                self._request_timeout_seconds,
            )
            if response.status != 200:
                return False
            payload = json.loads(response.body)
            return isinstance(payload, dict) and payload.get("status") == "ready"
        except (OSError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError):
            return False

    def terminate_pod(self, pod_id: str) -> None:
        self._request("DELETE", f"/pods/{pod_id}", expected_statuses={200, 202, 204})
        self._requests.pop(pod_id, None)
        self._worker_tokens.pop(pod_id, None)

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None = None,
        *,
        expected_statuses: set[int] | None = None,
    ) -> dict[str, object]:
        body = (
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            if payload is not None
            else None
        )
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        try:
            response = self._http(
                method,
                f"{self._base_url}{path}",
                headers,
                body,
                self._request_timeout_seconds,
            )
        except (OSError, TimeoutError) as error:
            raise RunPodAPIError(f"RunPod {method} request failed") from error
        allowed = expected_statuses or {200, 201}
        if response.status not in allowed:
            raise RunPodAPIError(
                f"RunPod {method} request failed with HTTP {response.status}"
            )
        if response.status == 204 or not response.body:
            return {}
        try:
            value = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RunPodAPIError("RunPod returned malformed JSON") from error
        if not isinstance(value, dict):
            raise RunPodAPIError("RunPod response must be a JSON object")
        return value


def _endpoint_address(
    pod_id: str,
    request: RunPodPodRequest,
    pod: Mapping[str, object],
) -> str | None:
    if request.worker_protocol == "http":
        return f"https://{pod_id}-{request.worker_port}.proxy.runpod.net"
    public_ip = pod.get("publicIp")
    mappings = pod.get("portMappings")
    if not isinstance(public_ip, str) or not isinstance(mappings, Mapping):
        return None
    public_port = mappings.get(str(request.worker_port))
    if not isinstance(public_port, int):
        return None
    return f"tcp://{public_ip}:{public_port}"


def _urllib_http(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout: float,
) -> HTTPResponse:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HTTPResponse(response.status, response.read())
    except urllib.error.HTTPError as error:
        return HTTPResponse(error.code, error.read())
