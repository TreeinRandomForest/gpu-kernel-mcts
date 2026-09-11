from __future__ import annotations

import argparse

from .runpod_api import RunPodRESTClient
from .runpod_discovery import RunPodDiscovery


def resolve_reusable_volume(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
    api_key: str,
) -> tuple[str | None, str | None]:
    """Resolve explicit, managed-network, or ephemeral pod storage."""
    manual = bool(arguments.network_volume_id or arguments.data_center_id)
    ephemeral = bool(arguments.ephemeral_storage)
    if bool(arguments.network_volume_id) != bool(arguments.data_center_id):
        parser.error("--network-volume-id and --data-center-id must be supplied together")
    if arguments.auto_volume and manual:
        parser.error("--auto-volume cannot be combined with explicit volume/data-center IDs")
    if ephemeral and (arguments.auto_volume or manual):
        parser.error(
            "--ephemeral-storage cannot be combined with network-volume options"
        )
    if ephemeral and arguments.preferred_data_center_id:
        parser.error(
            "--preferred-data-center-id cannot be used with --ephemeral-storage"
        )
    if arguments.preferred_data_center_id and not arguments.auto_volume:
        parser.error("--preferred-data-center-id requires --auto-volume")
    if not arguments.auto_volume and not manual and not ephemeral:
        parser.error(
            "provide explicit volume/data-center IDs, --auto-volume, "
            "or --ephemeral-storage"
        )
    if ephemeral:
        return None, None
    if manual:
        return arguments.network_volume_id, arguments.data_center_id
    selection = RunPodDiscovery(RunPodRESTClient(api_key)).select_volume(
        gpu_id=arguments.gpu_type,
        volume_name=arguments.volume_name,
        size_gb=1,
        allow_create=False,
        preferred_data_center_id=arguments.preferred_data_center_id,
    )
    print(
        f"Reusing managed volume {selection.volume.volume_id} "
        f"in {selection.volume.data_center_id}"
    )
    return selection.volume.volume_id, selection.volume.data_center_id
