from __future__ import annotations

from kernel_mcts.worker_service import WorkerBootstrap, capture_environment_manifest


def test_bootstrap_reports_stage_and_sanitized_failure_category() -> None:
    bootstrap = WorkerBootstrap("token")

    def fail(progress):
        progress("vendor_baselines")
        raise RuntimeError("arbitrary details are not returned")

    bootstrap.initialize(fail)
    response = bootstrap.handle(
        "GET",
        "/health",
        {"Authorization": "Bearer token"},
    )

    assert response.status == 500
    assert response.payload == {
        "status": "failed",
        "stage": "vendor_baselines",
        "error_code": "RuntimeError",
    }


def test_bootstrap_health_requires_worker_authentication() -> None:
    bootstrap = WorkerBootstrap("token")

    response = bootstrap.handle("GET", "/health", {})

    assert response.status == 401


def test_manifest_uses_provider_neutral_worker_identity(monkeypatch) -> None:
    def command(arguments):
        if arguments[0] == "nvidia-smi":
            return "NVIDIA H100 80GB HBM3, GPU-1, 9.0, 81559"
        if arguments[0] == "nvcc":
            return "Cuda compilation tools, release 12.8, V12.8.93"
        raise AssertionError(arguments)

    monkeypatch.setattr("kernel_mcts.worker_service._command", command)
    monkeypatch.setattr(
        "kernel_mcts.worker_service._optional_command", lambda args: "2025.1"
    )

    result = capture_environment_manifest(
        {
            "KERNEL_MCTS_WORKER_ID": "instance-1",
            "KERNEL_MCTS_PROVIDER": "nebius",
            "KERNEL_MCTS_CONTAINER_IMAGE": "worker:v1",
        }
    )

    assert result.worker_id == result.pod_id == "instance-1"
    assert result.provider == "nebius"
    assert result.form_factor == "SXM"
