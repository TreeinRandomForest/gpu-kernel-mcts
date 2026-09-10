from __future__ import annotations

import hashlib
import hmac
import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Mapping, Protocol

from .domain import EvaluationResult, KernelProgram, WorkloadContract
from .interfaces import KernelEvaluator
from .providers import EnvironmentManifest, WorkerEndpoint, WorkerTransport
from .serialization import (
    deserialize_environment_manifest,
    deserialize_evaluation,
    deserialize_workload,
    serialize_environment_manifest,
    serialize_evaluation,
    serialize_program,
    serialize_workload,
)


class WorkerProtocolError(RuntimeError):
    """A sanitized remote-worker protocol failure."""


@dataclass(frozen=True, slots=True)
class WorkerResponse:
    status: int
    payload: Mapping[str, object]


class WorkerHTTP(Protocol):
    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]: ...


class WorkerApplication:
    """Authenticated, sequential, idempotent worker protocol application."""

    def __init__(
        self,
        *,
        auth_token: str,
        manifest: EnvironmentManifest,
        evaluator: KernelEvaluator,
        max_request_bytes: int = 2_000_000,
    ) -> None:
        if not auth_token:
            raise ValueError("worker authentication token is required")
        self._auth_token = auth_token
        self._manifest = manifest
        self._evaluator = evaluator
        self._max_request_bytes = max_request_bytes
        self._lock = threading.Lock()
        self._evaluations: dict[str, tuple[str, dict[str, object]]] = {}

    def handle(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str],
        body: bytes = b"",
    ) -> WorkerResponse:
        if not self._authorized(headers):
            return WorkerResponse(401, {"error": "unauthorized"})
        if method == "GET" and path == "/health":
            return WorkerResponse(200, {"status": "ready"})
        if method == "GET" and path == "/manifest":
            return WorkerResponse(200, serialize_environment_manifest(self._manifest))
        if method != "POST" or path != "/evaluate":
            return WorkerResponse(404, {"error": "not_found"})
        if len(body) > self._max_request_bytes:
            return WorkerResponse(413, {"error": "request_too_large"})
        try:
            payload = json.loads(body)
            evaluation_id = payload["evaluation_id"]
            program_value = payload["program"]
            if not isinstance(evaluation_id, str) or not evaluation_id:
                raise ValueError
            if not isinstance(program_value, Mapping):
                raise ValueError
            program = KernelProgram(
                str(program_value["source"]),
                str(program_value["backend"]),
            )
            workload = deserialize_workload(payload["workload"])
            if payload.get("profile_level") != "tier0":
                return WorkerResponse(400, {"error": "unsupported_profile_level"})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return WorkerResponse(400, {"error": "invalid_request"})

        request_hash = hashlib.sha256(body).hexdigest()
        with self._lock:
            cached = self._evaluations.get(evaluation_id)
            if cached is not None:
                if cached[0] != request_hash:
                    return WorkerResponse(409, {"error": "evaluation_id_conflict"})
                return WorkerResponse(200, cached[1])
            result = self._evaluator.evaluate(program, workload)
            serialized = serialize_evaluation(result)
            self._evaluations[evaluation_id] = (request_hash, serialized)
            return WorkerResponse(200, serialized)

    def _authorized(self, headers: Mapping[str, str]) -> bool:
        authorization = next(
            (value for key, value in headers.items() if key.casefold() == "authorization"),
            "",
        )
        return hmac.compare_digest(authorization, f"Bearer {self._auth_token}")


class HTTPWorkerTransport(WorkerTransport):
    def __init__(
        self,
        endpoint: WorkerEndpoint,
        *,
        timeout_seconds: float = 180.0,
        http: WorkerHTTP | None = None,
    ) -> None:
        if not endpoint.auth_token:
            raise ValueError("worker endpoint is missing its authentication token")
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds
        self._http = http or _urllib_worker_http
        self._closed = False

    def get_environment_manifest(self) -> EnvironmentManifest:
        return deserialize_environment_manifest(self._request("GET", "/manifest"))

    def evaluate(
        self,
        evaluation_id: str,
        program: KernelProgram,
        workload: WorkloadContract,
        profile_level: str,
    ) -> EvaluationResult:
        payload = {
            "evaluation_id": evaluation_id,
            "program": serialize_program(program),
            "workload": serialize_workload(workload),
            "profile_level": profile_level,
        }
        return deserialize_evaluation(self._request("POST", "/evaluate", payload))

    def close(self) -> None:
        self._closed = True

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        if self._closed:
            raise WorkerProtocolError("worker transport is closed")
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() if payload else None
        headers = {
            "Authorization": f"Bearer {self._endpoint.auth_token}",
            "Accept": "application/json",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        try:
            status, response_body = self._http(
                method,
                f"{self._endpoint.address}{path}",
                headers,
                body,
                self._timeout_seconds,
            )
        except (OSError, TimeoutError) as error:
            raise WorkerProtocolError("remote worker request failed") from error
        if status != 200:
            raise WorkerProtocolError(f"remote worker returned HTTP {status}")
        try:
            value = json.loads(response_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise WorkerProtocolError("remote worker returned malformed JSON") from error
        if not isinstance(value, Mapping):
            raise WorkerProtocolError("remote worker response must be an object")
        return value


def _urllib_worker_http(method, url, headers, body, timeout):
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
