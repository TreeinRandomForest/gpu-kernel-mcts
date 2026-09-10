from __future__ import annotations

import json
import subprocess

import pytest

from kernel_mcts.runpod_api import NetworkVolume
from kernel_mcts.runpod_discovery import RunPodDiscovery, RunPodDiscoveryError


DATACENTERS = [
    {
        "id": "US-LOW-1",
        "gpuAvailability": [
            {
                "gpuId": "NVIDIA H100 80GB HBM3",
                "displayName": "H100 SXM",
                "stockStatus": "Low",
            }
        ],
    },
    {
        "id": "US-HIGH-1",
        "gpuAvailability": [
            {
                "gpuId": "NVIDIA H100 80GB HBM3",
                "displayName": "H100 SXM",
                "stockStatus": "High",
            },
            {
                "gpuId": "NVIDIA H100 PCIe",
                "displayName": "H100 PCIe",
                "stockStatus": "High",
            },
        ],
    },
    {
        "id": "US-NONE-1",
        "gpuAvailability": [
            {
                "gpuId": "NVIDIA H100 80GB HBM3",
                "stockStatus": "Unavailable",
            }
        ],
    },
]


class FakeREST:
    def __init__(self, volumes=()) -> None:
        self.volumes = tuple(volumes)
        self.created = []

    def list_network_volumes(self):
        return self.volumes

    def create_network_volume(self, **values):
        self.created.append(values)
        return NetworkVolume(
            "new-volume",
            values["name"],
            values["size_gb"],
            values["data_center_id"],
        )


def runner_for(value, returncode=0):
    def run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            returncode,
            json.dumps(value) if not isinstance(value, str) else value,
            "",
        )

    return run


def discovery(rest, value=DATACENTERS):
    return RunPodDiscovery(
        rest,
        runner=runner_for(value),
        environ={"HOME": "/tmp/test-home", "PATH": "/test/bin"},
    )


def test_availability_matches_exact_gpu_and_ranks_stock() -> None:
    available = discovery(FakeREST()).available_data_centers(
        "NVIDIA H100 80GB HBM3"
    )

    assert [(item.data_center_id, item.stock_status) for item in available] == [
        ("US-HIGH-1", "High"),
        ("US-LOW-1", "Low"),
    ]


def test_existing_managed_volume_is_preferred_over_creating_one() -> None:
    rest = FakeREST(
        [
            NetworkVolume("unrelated", "personal", 100, "US-HIGH-1"),
            NetworkVolume("managed", "gpu-kernel-mcts", 50, "US-LOW-1"),
        ]
    )

    selection = discovery(rest).select_volume(
        gpu_id="NVIDIA H100 80GB HBM3",
        volume_name="gpu-kernel-mcts",
        size_gb=50,
        allow_create=False,
    )

    assert selection.volume.volume_id == "managed"
    assert not selection.created
    assert rest.created == []


def test_volume_creation_requires_explicit_permission() -> None:
    rest = FakeREST()

    with pytest.raises(RunPodDiscoveryError, match="explicit.*confirmation"):
        discovery(rest).select_volume(
            gpu_id="NVIDIA H100 80GB HBM3",
            volume_name="gpu-kernel-mcts",
            size_gb=50,
            allow_create=False,
        )

    selection = discovery(rest).select_volume(
        gpu_id="NVIDIA H100 80GB HBM3",
        volume_name="gpu-kernel-mcts",
        size_gb=50,
        allow_create=True,
    )
    assert selection.created
    assert rest.created == [
        {
            "name": "gpu-kernel-mcts",
            "size_gb": 50,
            "data_center_id": "US-HIGH-1",
        }
    ]


def test_preferred_data_center_restricts_volume_creation() -> None:
    rest = FakeREST()

    selection = discovery(rest).select_volume(
        gpu_id="NVIDIA H100 80GB HBM3",
        volume_name="gpu-kernel-mcts",
        size_gb=50,
        allow_create=True,
        preferred_data_center_id="US-LOW-1",
    )

    assert selection.created
    assert rest.created == [
        {
            "name": "gpu-kernel-mcts",
            "size_gb": 50,
            "data_center_id": "US-LOW-1",
        }
    ]


def test_unavailable_preferred_data_center_fails_before_volume_operations() -> None:
    rest = FakeREST()

    with pytest.raises(RunPodDiscoveryError, match="preferred data center.*availability"):
        discovery(rest).select_volume(
            gpu_id="NVIDIA H100 80GB HBM3",
            volume_name="gpu-kernel-mcts",
            size_gb=50,
            allow_create=True,
            preferred_data_center_id="US-NONE-1",
        )

    assert rest.created == []


def test_small_managed_volume_is_not_silently_reused() -> None:
    rest = FakeREST(
        [NetworkVolume("small", "gpu-kernel-mcts", 25, "US-HIGH-1")]
    )

    with pytest.raises(RunPodDiscoveryError, match="too small"):
        discovery(rest).select_volume(
            gpu_id="NVIDIA H100 80GB HBM3",
            volume_name="gpu-kernel-mcts",
            size_gb=50,
            allow_create=True,
        )
    assert rest.created == []


@pytest.mark.parametrize("output", ["not json", {"not": "a list"}])
def test_invalid_discovery_output_is_sanitized(output) -> None:
    subject = discovery(FakeREST(), output)

    with pytest.raises(RunPodDiscoveryError, match="invalid discovery JSON"):
        subject.available_data_centers("NVIDIA H100 80GB HBM3")


def test_no_h100_availability_fails_before_volume_operations() -> None:
    rest = FakeREST()
    subject = discovery(rest, [])

    with pytest.raises(RunPodDiscoveryError, match="no data center"):
        subject.select_volume(
            gpu_id="NVIDIA H100 80GB HBM3",
            volume_name="gpu-kernel-mcts",
            size_gb=50,
            allow_create=True,
        )
    assert rest.created == []
