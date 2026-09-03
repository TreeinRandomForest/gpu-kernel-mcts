from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from .domain import EvaluationResult, KernelProgram, WorkloadContract


@dataclass(frozen=True, slots=True)
class HardwareSpec:
    gpu_model: str
    gpu_count: int = 1
    form_factor: str | None = None
    minimum_compute_capability: str | None = None


@dataclass(frozen=True, slots=True)
class EnvironmentManifest:
    manifest_id: str
    worker_id: str
    provider: str
    gpu_model: str
    gpu_uuid: str | None
    compute_capability: str
    versions: Mapping[str, str]
    container: Mapping[str, str]
    operating_state: Mapping[str, object]


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
    if requested.minimum_compute_capability is not None:
        wanted = tuple(map(int, requested.minimum_compute_capability.split(".")))
        actual = tuple(map(int, observed.compute_capability.split(".")))
        if actual < wanted:
            raise ValueError(f"worker compute capability {actual} is below required {wanted}")

