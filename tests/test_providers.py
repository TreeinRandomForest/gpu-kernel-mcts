from __future__ import annotations

from dataclasses import replace

import pytest

from kernel_mcts.domain import EvaluationResult, KernelProgram, ProposalStatus, ShapeCase, WorkloadContract
from kernel_mcts.providers import (
    EnvironmentManifest,
    HardwareSpec,
    RunPodConfig,
    RunPodPod,
    RunPodProvider,
    WorkerEndpoint,
    validate_environment,
)


def manifest(**changes) -> EnvironmentManifest:
    values = {
        "worker_id": "worker-1",
        "provider": "runpod",
        "pod_id": "pod-1",
        "gpu_model": "NVIDIA H100 80GB HBM3",
        "gpu_count": 1,
        "gpu_uuid": "GPU-123",
        "compute_capability": "9.0",
        "form_factor": "SXM",
        "captured_at": "2026-09-09T12:00:00+00:00",
        "toolchain_versions": {"cuda_toolkit": "12.4", "nvcc": "12.4.131"},
        "profiler_versions": {"ncu": "2024.1"},
        "library_versions": {"pytorch": "2.8.0"},
        "host_information": {"os": "Linux"},
        "container_image": "kernel-mcts:cuda-12.4",
        "container_digest": "sha256:abc",
        "operating_state": {"power_limit_watts": 700},
        "project_git_commit": "abc123",
        "dirty_tree": False,
        "config_hashes": {"search": "search-hash"},
    }
    values.update(changes)
    return EnvironmentManifest(**values)


def test_environment_manifest_id_is_deterministic_and_ignores_capture_time() -> None:
    first = manifest()
    recaptured = replace(first, captured_at="2026-09-09T12:05:00+00:00")
    changed_toolchain = replace(first, toolchain_versions={"cuda_toolkit": "12.5"})

    assert len(first.manifest_id) == 64
    assert recaptured.manifest_id == first.manifest_id
    assert changed_toolchain.manifest_id != first.manifest_id


def test_environment_validation_accepts_matching_constraints() -> None:
    validate_environment(
        HardwareSpec(
            "H100",
            form_factor="sxm",
            minimum_compute_capability="9.0",
            minimum_tool_versions={"cuda_toolkit": "12.3", "ncu": "2024.1"},
            required_profilers=("ncu",),
        ),
        manifest(),
    )


@pytest.mark.parametrize(
    ("requested", "observed", "message"),
    [
        (HardwareSpec("H100", form_factor="SXM"), manifest(form_factor="PCIe"), "form factor"),
        (HardwareSpec("H100", gpu_count=2), manifest(gpu_count=1), "below requested"),
        (
            HardwareSpec("H100", minimum_compute_capability="9.1"),
            manifest(),
            "compute capability",
        ),
        (
            HardwareSpec("H100", minimum_tool_versions={"cuda_toolkit": "12.5"}),
            manifest(),
            "below required",
        ),
        (
            HardwareSpec("H100", minimum_tool_versions={"nvdisasm": "12.4"}),
            manifest(),
            "missing required tool",
        ),
        (
            HardwareSpec("H100", required_profilers=("nsys",)),
            manifest(),
            "missing required profiler",
        ),
    ],
)
def test_environment_validation_rejects_constraint_mismatch(
    requested: HardwareSpec,
    observed: EnvironmentManifest,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_environment(requested, observed)


def test_hardware_and_manifest_gpu_counts_must_be_positive() -> None:
    with pytest.raises(ValueError, match="gpu_count"):
        HardwareSpec("H100", gpu_count=0)
    with pytest.raises(ValueError, match="gpu_count"):
        manifest(gpu_count=0)


@pytest.mark.parametrize(
    "changes",
    [
        {"network_volume_id": "volume-1"},
        {"data_center_ids": ("US-TX-3",)},
    ],
)
def test_runpod_config_requires_volume_and_region_together(changes) -> None:
    with pytest.raises(ValueError, match="configured together"):
        RunPodConfig(image="worker:latest", **changes)


WORKLOAD = WorkloadContract(
    "toy",
    "toy",
    "fp32",
    (ShapeCase({"n": 1}, 1.0),),
    0.0,
    0.0,
)


class FakeClient:
    def __init__(self, *, startup_error: Exception | None = None) -> None:
        self.startup_error = startup_error
        self.requests = []
        self.waits = []
        self.terminated = []

    def create_pod(self, request):
        self.requests.append(request)
        return RunPodPod("pod-1")

    def wait_until_ready(self, pod_id, timeout_seconds):
        self.waits.append((pod_id, timeout_seconds))
        if self.startup_error is not None:
            raise self.startup_error
        return WorkerEndpoint("worker-1", "https://worker.invalid")

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)


class FakeTransport:
    def __init__(
        self,
        environment: EnvironmentManifest | None = None,
        evaluation_error: Exception | None = None,
    ) -> None:
        self.environment = environment or manifest()
        self.evaluation_error = evaluation_error
        self.manifest_calls = 0
        self.evaluations = []
        self.close_calls = 0

    def get_environment_manifest(self):
        self.manifest_calls += 1
        return self.environment

    def evaluate(self, evaluation_id, program, workload, profile_level):
        self.evaluations.append((evaluation_id, program, workload, profile_level))
        if self.evaluation_error is not None:
            raise self.evaluation_error
        return EvaluationResult(ProposalStatus.VALID, program, "state", 1.0)

    def close(self):
        self.close_calls += 1


class RecordingEvents:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event_type, payload):
        self.events.append((event_type, payload))


def provider(client, transport, environ=None, **config_changes) -> RunPodProvider:
    config = {"image": "kernel-mcts:cuda-12.4", **config_changes}
    return RunPodProvider(
        RunPodConfig(**config),
        client_factory=lambda api_key: client,
        transport_factory=lambda endpoint: transport,
        environ=environ if environ is not None else {"RUNPOD_API_KEY": "secret"},
    )


def test_runpod_missing_key_fails_before_api_activity() -> None:
    client = FakeClient()
    transport = FakeTransport()
    runpod = provider(client, transport, environ={})

    with pytest.raises(ValueError, match="RUNPOD_API_KEY"):
        runpod.acquire_worker(HardwareSpec("H100"))

    assert client.requests == []


def test_runpod_acquires_once_reuses_worker_and_releases_idempotently() -> None:
    client = FakeClient()
    transport = FakeTransport()
    runpod = provider(client, transport)

    worker = runpod.acquire_worker(HardwareSpec("H100", form_factor="SXM"))
    first = worker.evaluate("evaluation-1", KernelProgram("one"), WORKLOAD, "tier0")
    second = worker.evaluate("evaluation-2", KernelProgram("two"), WORKLOAD, "tier0")
    runpod.release_worker(worker)
    runpod.release_worker(worker)

    assert len(client.requests) == 1
    assert client.requests[0].environment == {"KERNEL_MCTS_WORKER_PORT": "8000"}
    assert transport.manifest_calls == 1
    assert len(transport.evaluations) == 2
    assert first.worker_id == second.worker_id == "worker-1"
    assert first.environment_manifest_id == manifest().manifest_id
    assert transport.close_calls == 1
    assert client.terminated == ["pod-1"]
    assert worker.released
    with pytest.raises(RuntimeError, match="released"):
        worker.evaluate("evaluation-3", KernelProgram("three"), WORKLOAD, "tier0")


def test_runpod_provider_passes_network_volume_affinity() -> None:
    client = FakeClient()
    transport = FakeTransport()
    runpod = provider(
        client,
        transport,
        network_volume_id="volume-1",
        data_center_ids=("US-TX-3",),
    )

    worker = runpod.acquire_worker(HardwareSpec("H100"))

    assert client.requests[0].network_volume_id == "volume-1"
    assert client.requests[0].data_center_ids == ("US-TX-3",)
    runpod.release_worker(worker)


def test_runpod_context_logs_manifest_and_releases_after_exception() -> None:
    client = FakeClient()
    transport = FakeTransport()
    runpod = provider(client, transport)
    events = RecordingEvents()

    with pytest.raises(RuntimeError, match="search failed"):
        with runpod.worker_for_run(HardwareSpec("H100"), events) as worker:
            assert worker.get_environment_manifest() is transport.environment
            raise RuntimeError("search failed")

    assert events.events == [("environment_manifest", transport.environment.as_dict())]
    assert transport.close_calls == 1
    assert client.terminated == ["pod-1"]


def test_invalid_manifest_terminates_pod_before_returning_worker() -> None:
    client = FakeClient()
    transport = FakeTransport(manifest(form_factor="PCIe"))
    runpod = provider(client, transport)

    with pytest.raises(ValueError, match="form factor"):
        runpod.acquire_worker(HardwareSpec("H100", form_factor="SXM"))

    assert transport.close_calls == 1
    assert client.terminated == ["pod-1"]


@pytest.mark.parametrize(
    ("invalid_manifest", "message"),
    [
        (manifest(pod_id="another-pod"), "pod mismatch"),
        (manifest(worker_id="another-worker"), "identity mismatch"),
    ],
)
def test_manifest_identity_mismatch_terminates_pod(
    invalid_manifest: EnvironmentManifest,
    message: str,
) -> None:
    client = FakeClient()
    transport = FakeTransport(invalid_manifest)
    runpod = provider(client, transport)

    with pytest.raises(ValueError, match=message):
        runpod.acquire_worker(HardwareSpec("H100"))

    assert transport.close_calls == 1
    assert client.terminated == ["pod-1"]


def test_startup_failure_terminates_created_pod() -> None:
    client = FakeClient(startup_error=RuntimeError("startup failed"))
    transport = FakeTransport()
    runpod = provider(client, transport)

    with pytest.raises(RuntimeError, match="startup failed"):
        runpod.acquire_worker(HardwareSpec("H100"))

    assert transport.close_calls == 0
    assert client.terminated == ["pod-1"]


def test_transport_error_becomes_infrastructure_failure_without_secret() -> None:
    secret = "do-not-log-this-key"
    client = FakeClient()
    transport = FakeTransport(evaluation_error=RuntimeError(f"failure {secret}"))
    runpod = provider(client, transport, environ={"RUNPOD_API_KEY": secret})
    worker = runpod.acquire_worker(HardwareSpec("H100"))

    result = worker.evaluate("evaluation", KernelProgram("kernel"), WORKLOAD, "tier0")

    assert result.status == ProposalStatus.INFRASTRUCTURE_FAILURE
    assert result.metadata == {"evaluation_id": "evaluation", "error_type": "RuntimeError"}
    assert secret not in repr(runpod)
    assert secret not in repr(result)
    runpod.release_worker(worker)
