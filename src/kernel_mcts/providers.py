from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Iterator, Mapping, Protocol

from .domain import EvaluationResult, KernelProgram, ProposalStatus, WorkloadContract


@dataclass(frozen=True, slots=True)
class HardwareSpec:
    gpu_model: str
    gpu_count: int = 1
    form_factor: str | None = None
    minimum_compute_capability: str | None = None
    minimum_tool_versions: Mapping[str, str] = field(default_factory=dict)
    required_profilers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.gpu_count < 1:
            raise ValueError("gpu_count must be positive")


@dataclass(frozen=True, slots=True)
class EnvironmentManifest:
    worker_id: str
    provider: str
    gpu_model: str
    compute_capability: str
    captured_at: str
    pod_id: str | None = None
    gpu_count: int = 1
    gpu_uuid: str | None = None
    sm_count: int | None = None
    total_memory_bytes: int | None = None
    form_factor: str | None = None
    mig_configuration: Mapping[str, object] = field(default_factory=dict)
    toolchain_versions: Mapping[str, str] = field(default_factory=dict)
    profiler_versions: Mapping[str, str] = field(default_factory=dict)
    library_versions: Mapping[str, str] = field(default_factory=dict)
    host_information: Mapping[str, object] = field(default_factory=dict)
    container_image: str | None = None
    container_digest: str | None = None
    container_runtime: str | None = None
    operating_state: Mapping[str, object] = field(default_factory=dict)
    project_git_commit: str | None = None
    dirty_tree: bool | None = None
    config_hashes: Mapping[str, str] = field(default_factory=dict)
    backend_implementation_hash: str | None = None
    manifest_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.gpu_count < 1:
            raise ValueError("manifest gpu_count must be positive")
        object.__setattr__(self, "manifest_id", _manifest_id(self.identity_payload()))

    def identity_payload(self) -> dict[str, object]:
        """Return environment identity fields, excluding the capture timestamp."""
        return {
            "worker_id": self.worker_id,
            "provider": self.provider,
            "pod_id": self.pod_id,
            "gpu_model": self.gpu_model,
            "gpu_count": self.gpu_count,
            "gpu_uuid": self.gpu_uuid,
            "compute_capability": self.compute_capability,
            "sm_count": self.sm_count,
            "total_memory_bytes": self.total_memory_bytes,
            "form_factor": self.form_factor,
            "mig_configuration": dict(self.mig_configuration),
            "toolchain_versions": dict(self.toolchain_versions),
            "profiler_versions": dict(self.profiler_versions),
            "library_versions": dict(self.library_versions),
            "host_information": dict(self.host_information),
            "container_image": self.container_image,
            "container_digest": self.container_digest,
            "container_runtime": self.container_runtime,
            "operating_state": dict(self.operating_state),
            "project_git_commit": self.project_git_commit,
            "dirty_tree": self.dirty_tree,
            "config_hashes": dict(self.config_hashes),
            "backend_implementation_hash": self.backend_implementation_hash,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "manifest_id": self.manifest_id,
            "captured_at": self.captured_at,
            **self.identity_payload(),
        }


class GPUWorker(Protocol):
    @property
    def worker_id(self) -> str: ...

    def get_environment_manifest(self) -> EnvironmentManifest: ...

    def evaluate(
        self,
        evaluation_id: str,
        program: KernelProgram,
        workload: WorkloadContract,
        profile_level: str,
    ) -> EvaluationResult: ...


class GPUProvider(Protocol):
    def acquire_worker(self, hardware: HardwareSpec) -> GPUWorker: ...

    def release_worker(self, worker: GPUWorker) -> None: ...


@dataclass(frozen=True, slots=True)
class RunPodConfig:
    image: str
    api_key_env: str = "RUNPOD_API_KEY"
    gpu_type: str = "H100_SXM"
    gpu_count: int = 1
    container_disk_gb: int = 50
    interruptible: bool = False
    startup_timeout_seconds: float = 600.0
    terminate_after_run: bool = True

    def __post_init__(self) -> None:
        if not self.image:
            raise ValueError("RunPod worker image is required")
        if self.gpu_count < 1 or self.container_disk_gb < 1:
            raise ValueError("RunPod GPU count and container disk must be positive")
        if self.startup_timeout_seconds <= 0:
            raise ValueError("RunPod startup timeout must be positive")


@dataclass(frozen=True, slots=True)
class RunPodPodRequest:
    gpu_type: str
    gpu_count: int
    image: str
    container_disk_gb: int
    interruptible: bool


@dataclass(frozen=True, slots=True)
class RunPodPod:
    pod_id: str


@dataclass(frozen=True, slots=True)
class WorkerEndpoint:
    worker_id: str
    address: str


class RunPodClient(Protocol):
    def create_pod(self, request: RunPodPodRequest) -> RunPodPod: ...

    def wait_until_ready(self, pod_id: str, timeout_seconds: float) -> WorkerEndpoint: ...

    def terminate_pod(self, pod_id: str) -> None: ...


class RunPodClientFactory(Protocol):
    def __call__(self, api_key: str) -> RunPodClient: ...


class WorkerTransport(Protocol):
    def get_environment_manifest(self) -> EnvironmentManifest: ...

    def evaluate(
        self,
        evaluation_id: str,
        program: KernelProgram,
        workload: WorkloadContract,
        profile_level: str,
    ) -> EvaluationResult: ...

    def close(self) -> None: ...


class WorkerTransportFactory(Protocol):
    def __call__(self, endpoint: WorkerEndpoint) -> WorkerTransport: ...


class ManifestEventSink(Protocol):
    def emit(self, event_type: str, payload: Mapping[str, object]) -> None: ...


class RunPodWorker:
    def __init__(
        self,
        *,
        pod_id: str,
        endpoint: WorkerEndpoint,
        transport: WorkerTransport,
        manifest: EnvironmentManifest,
        owner_token: object,
    ) -> None:
        self.pod_id = pod_id
        self.endpoint = endpoint
        self._transport = transport
        self._manifest = manifest
        self._owner_token = owner_token
        self._released = False

    @property
    def worker_id(self) -> str:
        return self.endpoint.worker_id

    @property
    def released(self) -> bool:
        return self._released

    def get_environment_manifest(self) -> EnvironmentManifest:
        return self._manifest

    def evaluate(
        self,
        evaluation_id: str,
        program: KernelProgram,
        workload: WorkloadContract,
        profile_level: str,
    ) -> EvaluationResult:
        if self._released:
            raise RuntimeError("cannot evaluate on a released RunPod worker")
        try:
            result = self._transport.evaluate(
                evaluation_id,
                program,
                workload,
                profile_level,
            )
        except Exception as error:
            return EvaluationResult(
                status=ProposalStatus.INFRASTRUCTURE_FAILURE,
                worker_id=self.worker_id,
                environment_manifest_id=self._manifest.manifest_id,
                metadata={
                    "evaluation_id": evaluation_id,
                    "error_type": type(error).__name__,
                },
            )
        return replace(
            result,
            worker_id=result.worker_id or self.worker_id,
            environment_manifest_id=(
                result.environment_manifest_id or self._manifest.manifest_id
            ),
        )

    def _close(self) -> None:
        if not self._released:
            self._transport.close()


class RunPodProvider:
    def __init__(
        self,
        config: RunPodConfig,
        *,
        client_factory: RunPodClientFactory,
        transport_factory: WorkerTransportFactory,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self._client_factory = client_factory
        self._transport_factory = transport_factory
        self._environ = environ if environ is not None else os.environ
        self._owner_token = object()
        self._active: dict[int, tuple[RunPodWorker, RunPodClient]] = {}

    def acquire_worker(self, hardware: HardwareSpec) -> RunPodWorker:
        api_key = self._environ.get(self.config.api_key_env)
        if not api_key:
            raise ValueError(
                f"required RunPod credential environment variable "
                f"{self.config.api_key_env!r} is not set"
            )
        client = self._client_factory(api_key)
        pod: RunPodPod | None = None
        transport: WorkerTransport | None = None
        try:
            pod = client.create_pod(
                RunPodPodRequest(
                    gpu_type=self.config.gpu_type,
                    gpu_count=self.config.gpu_count,
                    image=self.config.image,
                    container_disk_gb=self.config.container_disk_gb,
                    interruptible=self.config.interruptible,
                )
            )
            endpoint = client.wait_until_ready(
                pod.pod_id,
                self.config.startup_timeout_seconds,
            )
            transport = self._transport_factory(endpoint)
            manifest = transport.get_environment_manifest()
            if manifest.pod_id != pod.pod_id:
                raise ValueError(
                    f"worker manifest pod mismatch: expected {pod.pod_id!r}, "
                    f"observed {manifest.pod_id!r}"
                )
            if manifest.worker_id != endpoint.worker_id:
                raise ValueError(
                    f"worker manifest identity mismatch: expected {endpoint.worker_id!r}, "
                    f"observed {manifest.worker_id!r}"
                )
            validate_environment(hardware, manifest)
            worker = RunPodWorker(
                pod_id=pod.pod_id,
                endpoint=endpoint,
                transport=transport,
                manifest=manifest,
                owner_token=self._owner_token,
            )
            self._active[id(worker)] = (worker, client)
            return worker
        except Exception:
            if transport is not None:
                try:
                    transport.close()
                except Exception:
                    pass
            if pod is not None:
                try:
                    client.terminate_pod(pod.pod_id)
                except Exception:
                    pass
            raise

    def release_worker(self, worker: GPUWorker) -> None:
        if not isinstance(worker, RunPodWorker) or worker._owner_token is not self._owner_token:
            raise ValueError("worker is not owned by this RunPod provider")
        if worker.released:
            return
        entry = self._active.get(id(worker))
        if entry is None or entry[0] is not worker:
            raise ValueError("worker is not active in this RunPod provider")
        client = entry[1]
        cleanup_error: Exception | None = None
        try:
            worker._close()
        except Exception as error:
            cleanup_error = error
        try:
            if self.config.terminate_after_run:
                client.terminate_pod(worker.pod_id)
        except Exception as error:
            if cleanup_error is None:
                cleanup_error = error
        finally:
            worker._released = True
            self._active.pop(id(worker), None)
        if cleanup_error is not None:
            raise cleanup_error

    @contextmanager
    def worker_for_run(
        self,
        hardware: HardwareSpec,
        events: ManifestEventSink | None = None,
    ) -> Iterator[RunPodWorker]:
        worker = self.acquire_worker(hardware)
        try:
            if events is not None:
                events.emit("environment_manifest", worker.get_environment_manifest().as_dict())
            yield worker
        finally:
            self.release_worker(worker)


def validate_environment(requested: HardwareSpec, observed: EnvironmentManifest) -> None:
    if requested.gpu_model.casefold() not in observed.gpu_model.casefold():
        raise ValueError(
            f"worker GPU mismatch: requested {requested.gpu_model!r}, observed {observed.gpu_model!r}"
        )
    if observed.gpu_count < requested.gpu_count:
        raise ValueError(
            f"worker has {observed.gpu_count} GPUs, below requested {requested.gpu_count}"
        )
    if requested.form_factor is not None:
        if observed.form_factor is None or (
            requested.form_factor.casefold() != observed.form_factor.casefold()
        ):
            raise ValueError(
                f"worker form factor mismatch: requested {requested.form_factor!r}, "
                f"observed {observed.form_factor!r}"
            )
    if requested.minimum_compute_capability is not None:
        _require_minimum_version(
            "compute capability",
            observed.compute_capability,
            requested.minimum_compute_capability,
        )

    available_versions = {
        **observed.toolchain_versions,
        **observed.profiler_versions,
    }
    for tool, minimum in requested.minimum_tool_versions.items():
        actual = available_versions.get(tool)
        if actual is None:
            raise ValueError(f"worker is missing required tool {tool!r}")
        _require_minimum_version(tool, actual, minimum)
    for profiler in requested.required_profilers:
        if not observed.profiler_versions.get(profiler):
            raise ValueError(f"worker is missing required profiler {profiler!r}")


def _manifest_id(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_minimum_version(name: str, actual: str, minimum: str) -> None:
    actual_version = _numeric_version(actual)
    minimum_version = _numeric_version(minimum)
    width = max(len(actual_version), len(minimum_version))
    actual_version += (0,) * (width - len(actual_version))
    minimum_version += (0,) * (width - len(minimum_version))
    if actual_version < minimum_version:
        raise ValueError(f"worker {name} version {actual!r} is below required {minimum!r}")


def _numeric_version(value: str) -> tuple[int, ...]:
    numbers = tuple(int(item) for item in re.findall(r"\d+", value))
    if not numbers:
        raise ValueError(f"cannot parse version {value!r}")
    return numbers
