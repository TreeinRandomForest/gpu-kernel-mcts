from __future__ import annotations

from dataclasses import replace

import pytest

from kernel_mcts.providers import EnvironmentManifest, HardwareSpec, validate_environment


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
