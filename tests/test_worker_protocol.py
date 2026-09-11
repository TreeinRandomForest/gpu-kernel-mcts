from __future__ import annotations

import json

import pytest

from kernel_mcts.domain import (
    BenchmarkResult,
    CompileStatus,
    CorrectnessStatus,
    EvaluationResult,
    KernelProgram,
    ProposalStatus,
    ShapeCase,
    WorkloadContract,
)
from kernel_mcts.evaluation import EvaluationInfrastructureError
from kernel_mcts.providers import EnvironmentManifest, WorkerEndpoint
from kernel_mcts.serialization import serialize_evaluation
from kernel_mcts.worker_protocol import (
    HTTPWorkerTransport,
    WorkerApplication,
    WorkerProtocolError,
)


WORKLOAD = WorkloadContract(
    "toy",
    "gemm",
    "bfloat16",
    (ShapeCase({"M": 1, "N": 1, "K": 1}, 1.0),),
    0.02,
    0.02,
)
MANIFEST = EnvironmentManifest(
    worker_id="pod-1",
    provider="runpod",
    pod_id="pod-1",
    gpu_model="NVIDIA H100 80GB HBM3",
    compute_capability="9.0",
    form_factor="SXM",
    captured_at="2026-09-09T00:00:00+00:00",
)


class FakeEvaluator:
    def __init__(self) -> None:
        self.calls = []

    def evaluate(self, program, workload):
        self.calls.append((program, workload))
        return EvaluationResult(
            ProposalStatus.VALID,
            program,
            "state",
            0.5,
            BenchmarkResult((1.0,), 1.0),
            compile_status=CompileStatus.SUCCESS,
            correctness_status=CorrectnessStatus.PASS,
            worker_id="pod-1",
            environment_manifest_id=MANIFEST.manifest_id,
        )


class FakeProfiler:
    def __init__(self) -> None:
        self.calls = []

    def lightweight_profile(self, evaluation, workload):
        self.calls.append((evaluation, workload))
        return {
            "schema_version": 1,
            "profiler": "ncu",
            "metrics": {"occupancy": {"value": 50.0, "unit": "%"}},
        }


def request_body(evaluation_id="evaluation-1", source="kernel"):
    return json.dumps(
        {
            "evaluation_id": evaluation_id,
            "program": {"source": source, "backend": "cuda_cpp"},
            "workload": {
                "benchmark_id": "toy",
                "operation": "gemm",
                "dtype": "bfloat16",
                "shapes": [{"dimensions": {"M": 1, "N": 1, "K": 1}, "weight": 1.0}],
                "rtol": 0.02,
                "atol": 0.02,
                "metadata": {},
            },
            "profile_level": "tier0",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def test_worker_requires_authentication_for_every_endpoint() -> None:
    app = WorkerApplication(auth_token="token", manifest=MANIFEST, evaluator=FakeEvaluator())

    for method, path in [
        ("GET", "/health"),
        ("GET", "/manifest"),
        ("GET", "/calibration"),
        ("POST", "/evaluate"),
        ("POST", "/profile"),
    ]:
        response = app.handle(method, path, {}, request_body())
        assert response.status == 401


def test_worker_evaluation_is_idempotent_by_request_id_and_payload() -> None:
    evaluator = FakeEvaluator()
    app = WorkerApplication(auth_token="token", manifest=MANIFEST, evaluator=evaluator)
    headers = {"Authorization": "Bearer token"}
    body = request_body()

    first = app.handle("POST", "/evaluate", headers, body)
    second = app.handle("POST", "/evaluate", headers, body)
    conflict = app.handle(
        "POST", "/evaluate", headers, request_body(source="different kernel")
    )

    assert first.status == second.status == 200
    assert first.payload == second.payload
    assert len(evaluator.calls) == 1
    assert conflict.status == 409


def test_worker_profiles_cached_evaluation_idempotently() -> None:
    evaluator = FakeEvaluator()
    profiler = FakeProfiler()
    app = WorkerApplication(
        auth_token="token",
        manifest=MANIFEST,
        evaluator=evaluator,
        profiler=profiler,
    )
    headers = {"Authorization": "Bearer token"}
    evaluated = app.handle("POST", "/evaluate", headers, request_body())
    body = json.dumps(
        {"evaluation_id": "evaluation-1", "profile_level": "lightweight"}
    ).encode()

    first = app.handle("POST", "/profile", headers, body)
    second = app.handle("POST", "/profile", headers, body)

    assert evaluated.status == 200
    assert first.status == second.status == 200
    assert first.payload == second.payload
    assert len(evaluator.calls) == 1
    assert len(profiler.calls) == 1


def test_worker_rejects_profile_without_cached_evaluation() -> None:
    app = WorkerApplication(
        auth_token="token",
        manifest=MANIFEST,
        evaluator=FakeEvaluator(),
        profiler=FakeProfiler(),
    )
    body = json.dumps(
        {"evaluation_id": "missing", "profile_level": "lightweight"}
    ).encode()

    response = app.handle("POST", "/profile", {"Authorization": "Bearer token"}, body)

    assert response.status == 404
    assert response.payload == {"error": "evaluation_not_found"}


def test_worker_returns_bounded_profiling_diagnostic() -> None:
    class FailedProfiler:
        def lightweight_profile(self, evaluation, workload):
            raise EvaluationInfrastructureError("Nsight Compute failed: metric unavailable")

    app = WorkerApplication(
        auth_token="token",
        manifest=MANIFEST,
        evaluator=FakeEvaluator(),
        profiler=FailedProfiler(),
    )
    headers = {"Authorization": "Bearer token"}
    app.handle("POST", "/evaluate", headers, request_body())
    body = json.dumps(
        {"evaluation_id": "evaluation-1", "profile_level": "lightweight"}
    ).encode()

    response = app.handle("POST", "/profile", headers, body)

    assert response.status == 500
    assert response.payload["error"] == "profiling_failed"
    assert response.payload["diagnostic"] == (
        "Nsight Compute failed: metric unavailable"
    )


def test_worker_rejects_large_invalid_and_non_tier0_requests() -> None:
    app = WorkerApplication(
        auth_token="token",
        manifest=MANIFEST,
        evaluator=FakeEvaluator(),
        max_request_bytes=10,
    )
    headers = {"authorization": "Bearer token"}
    assert app.handle("POST", "/evaluate", headers, b"x" * 11).status == 413

    normal = WorkerApplication(auth_token="token", manifest=MANIFEST, evaluator=FakeEvaluator())
    assert normal.handle("POST", "/evaluate", headers, b"not json").status == 400
    payload = json.loads(request_body())
    payload["profile_level"] = "full"
    assert normal.handle("POST", "/evaluate", headers, json.dumps(payload).encode()).status == 400


class AppHTTP:
    def __init__(self, app) -> None:
        self.app = app
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append((method, url, dict(headers), body, timeout))
        path = "/" + url.split("/", 3)[-1]
        response = self.app.handle(method, path, headers, body or b"")
        return response.status, json.dumps(response.payload).encode()


def test_http_transport_round_trips_manifest_and_evaluation() -> None:
    evaluator = FakeEvaluator()
    calibration = {"benchmark_id": "toy", "benchmark": {"median_us": 1.0}}
    profiler = FakeProfiler()
    app = WorkerApplication(
        auth_token="token",
        manifest=MANIFEST,
        evaluator=evaluator,
        profiler=profiler,
        calibration=calibration,
    )
    http = AppHTTP(app)
    transport = HTTPWorkerTransport(
        WorkerEndpoint("pod-1", "https://worker.invalid", "token"),
        http=http,
    )

    manifest = transport.get_environment_manifest()
    returned_calibration = transport.get_calibration()
    result = transport.evaluate("evaluation-1", KernelProgram("kernel"), WORKLOAD, "tier0")
    profile = transport.profile("evaluation-1", "lightweight")

    assert manifest == MANIFEST
    assert returned_calibration == calibration
    assert result == evaluator.evaluate(KernelProgram("kernel"), WORKLOAD)
    assert profile["profiler"] == "ncu"
    assert len(profiler.calls) == 1
    assert http.calls[0][2]["Authorization"] == "Bearer token"
    assert all(call[2]["User-Agent"] == "gpu-kernel-mcts/0.1.0" for call in http.calls)
    assert "token" not in repr(transport._endpoint)


def test_transport_errors_are_sanitized_and_close_is_enforced() -> None:
    secret = "worker-secret"

    def failed_http(method, url, headers, body, timeout):
        return 500, json.dumps({"error": secret}).encode()

    transport = HTTPWorkerTransport(
        WorkerEndpoint("pod-1", "https://worker.invalid", secret),
        http=failed_http,
    )
    with pytest.raises(WorkerProtocolError) as captured:
        transport.get_environment_manifest()
    assert secret not in str(captured.value)

    def profiling_failure_http(method, url, headers, body, timeout):
        return 500, json.dumps(
            {
                "error": "profiling_failed",
                "error_type": "EvaluationInfrastructureError",
                "diagnostic": "Nsight Compute failed: metric unavailable",
            }
        ).encode()

    profiling_transport = HTTPWorkerTransport(
        WorkerEndpoint("pod-1", "https://worker.invalid", secret),
        http=profiling_failure_http,
    )
    with pytest.raises(WorkerProtocolError, match="metric unavailable"):
        profiling_transport.profile("evaluation-1", "lightweight")

    transport.close()
    with pytest.raises(WorkerProtocolError, match="closed"):
        transport.get_environment_manifest()
