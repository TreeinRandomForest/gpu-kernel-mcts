from __future__ import annotations

from kernel_mcts.worker_service import (
    WorkerBootstrap,
    _build_backend,
    capture_environment_manifest,
)
from kernel_mcts import cute_entrypoint
from kernel_mcts.cute_backend import CuTeDSLBackend
from kernel_mcts.cuda_backend import CudaCppBackend
from kernel_mcts.providers import EnvironmentManifest


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


def test_cute_image_entrypoint_dispatches_worker_when_token_is_present(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setenv("KERNEL_MCTS_WORKER_TOKEN", "token")
    monkeypatch.setattr(cute_entrypoint.worker_service, "main", lambda: calls.append("worker"))
    monkeypatch.setattr(
        cute_entrypoint.cute_baseline_cli,
        "main",
        lambda argv: calls.append("baseline"),
    )

    assert cute_entrypoint.main(["--mode", "comparison"]) == 0
    assert calls == ["worker"]


def test_cute_image_entrypoint_preserves_standalone_cli(monkeypatch) -> None:
    calls = []
    monkeypatch.delenv("KERNEL_MCTS_WORKER_TOKEN", raising=False)
    monkeypatch.setattr(
        cute_entrypoint.cute_baseline_cli,
        "main",
        lambda argv: calls.append(tuple(argv)) or 7,
    )

    assert cute_entrypoint.main(["--mode", "backend"]) == 7
    assert calls == [("--mode", "backend")]


def test_worker_builds_backend_selected_by_environment(tmp_path) -> None:
    manifest = EnvironmentManifest(
        worker_id="worker-1",
        provider="test",
        pod_id="worker-1",
        gpu_model="NVIDIA H100 80GB HBM3",
        compute_capability="9.0",
        form_factor="SXM",
        captured_at="2026-09-20T00:00:00+00:00",
        profiler_versions={"ncu": "2025.1"},
    )
    common = {"KERNEL_MCTS_ARTIFACT_ROOT": str(tmp_path)}

    cuda_name, cuda_backend, cuda_root = _build_backend(common, manifest)
    cute_name, cute_backend, cute_root = _build_backend(
        {**common, "KERNEL_MCTS_BACKEND": "cute_dsl"}, manifest
    )

    assert cuda_name == cuda_root.backend == "cuda_cpp"
    assert isinstance(cuda_backend, CudaCppBackend)
    assert cute_name == cute_root.backend == "cute_dsl"
    assert isinstance(cute_backend, CuTeDSLBackend)
