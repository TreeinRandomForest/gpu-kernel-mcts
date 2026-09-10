from __future__ import annotations

from kernel_mcts.worker_service import WorkerBootstrap


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
