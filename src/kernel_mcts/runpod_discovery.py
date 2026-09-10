from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Callable, Mapping

from .runpod_api import NetworkVolume, RunPodRESTClient


class RunPodDiscoveryError(RuntimeError):
    """A sanitized GPU/data-center discovery failure."""


@dataclass(frozen=True, slots=True)
class DataCenterAvailability:
    data_center_id: str
    gpu_id: str
    stock_status: str


@dataclass(frozen=True, slots=True)
class VolumeSelection:
    volume: NetworkVolume
    created: bool


class RunPodDiscovery:
    def __init__(
        self,
        rest_client: RunPodRESTClient,
        *,
        runpodctl: str = "runpodctl",
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._rest_client = rest_client
        self._runpodctl = runpodctl
        self._runner = runner
        self._environ = os.environ if environ is None else environ

    def available_data_centers(
        self,
        gpu_id: str,
    ) -> tuple[DataCenterAvailability, ...]:
        environment = {
            "PATH": self._environ.get("PATH", os.defpath),
            "HOME": self._environ.get("HOME", ""),
        }
        if self._environ.get("RUNPOD_API_KEY"):
            environment["RUNPOD_API_KEY"] = self._environ["RUNPOD_API_KEY"]
        try:
            result = self._runner(
                [self._runpodctl, "datacenter", "list", "--output", "json"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RunPodDiscoveryError("runpodctl data-center discovery failed") from error
        if result.returncode != 0:
            raise RunPodDiscoveryError("runpodctl data-center discovery failed")
        try:
            data = json.loads(result.stdout)
            if not isinstance(data, list):
                raise TypeError
            matches = []
            for center in data:
                if not isinstance(center, Mapping) or not isinstance(center.get("id"), str):
                    raise TypeError
                for availability in center.get("gpuAvailability", []):
                    if not isinstance(availability, Mapping):
                        raise TypeError
                    status = str(availability.get("stockStatus", "Unknown"))
                    if availability.get("gpuId") == gpu_id and _has_stock(status):
                        matches.append(DataCenterAvailability(center["id"], gpu_id, status))
            return tuple(sorted(matches, key=_availability_rank))
        except (json.JSONDecodeError, TypeError) as error:
            raise RunPodDiscoveryError("runpodctl returned invalid discovery JSON") from error

    def select_volume(
        self,
        *,
        gpu_id: str,
        volume_name: str,
        size_gb: int,
        allow_create: bool,
        preferred_data_center_id: str | None = None,
    ) -> VolumeSelection:
        if not volume_name or size_gb < 1 or size_gb > 4000:
            raise ValueError("managed volume requires a name and size from 1-4000 GB")
        available = self.available_data_centers(gpu_id)
        if not available:
            raise RunPodDiscoveryError(f"no data center currently reports availability for {gpu_id}")
        if preferred_data_center_id is not None:
            available = tuple(
                item
                for item in available
                if item.data_center_id == preferred_data_center_id
            )
            if not available:
                raise RunPodDiscoveryError(
                    f"preferred data center {preferred_data_center_id!r} does not "
                    f"currently report availability for {gpu_id}"
                )
        center_ids = {item.data_center_id for item in available}
        named = [
            volume
            for volume in self._rest_client.list_network_volumes()
            if volume.name == volume_name and volume.data_center_id in center_ids
        ]
        adequate = [volume for volume in named if volume.size_gb >= size_gb]
        if adequate:
            by_center = {item.data_center_id: index for index, item in enumerate(available)}
            selected = min(
                adequate,
                key=lambda volume: (by_center[volume.data_center_id], volume.volume_id),
            )
            return VolumeSelection(selected, created=False)
        if named:
            raise RunPodDiscoveryError(
                f"managed volume {volume_name!r} exists in an available data center but is too small"
            )
        if not allow_create:
            raise RunPodDiscoveryError(
                "no reusable managed volume exists; explicit volume-creation confirmation is required"
            )
        selected_center = available[0].data_center_id
        volume = self._rest_client.create_network_volume(
            name=volume_name,
            size_gb=size_gb,
            data_center_id=selected_center,
        )
        return VolumeSelection(volume, created=True)


def _has_stock(status: str) -> bool:
    return status.casefold() in {"high", "medium", "low", "available"}


def _availability_rank(item: DataCenterAvailability) -> tuple[int, str]:
    rank = {"high": 0, "medium": 1, "low": 2}.get(item.stock_status.casefold(), 3)
    return rank, item.data_center_id
