from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Mapping, Protocol

from .domain import EvaluationResult, KernelProgram, WorkloadContract


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
