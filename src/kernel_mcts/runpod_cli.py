from __future__ import annotations

import argparse
import os

from .providers import RunPodPodRequest
from .runpod_api import RunPodRESTClient
from .runpod_discovery import RunPodDiscovery


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Provision a RunPod worker, confirm readiness, and terminate it."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--api-key-env", default="RUNPOD_API_KEY")
    parser.add_argument("--gpu-type", default="NVIDIA H100 80GB HBM3")
    parser.add_argument(
        "--network-volume-id",
        help="existing RunPod network volume attached to the H100 pod",
    )
    parser.add_argument(
        "--data-center-id",
        help="RunPod data-center ID matching the network volume region",
    )
    parser.add_argument(
        "--auto-volume",
        action="store_true",
        help="discover H100 availability and select a managed network volume",
    )
    parser.add_argument("--volume-name", default="gpu-kernel-mcts")
    parser.add_argument("--volume-size-gb", type=int, default=50)
    parser.add_argument(
        "--confirm-create-volume",
        action="store_true",
        help="allow creation of a persistent billable volume if no reusable one exists",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--confirm-create-and-terminate",
        action="store_true",
        help="required because this command creates a billable Pod",
    )
    arguments = parser.parse_args()
    if not arguments.confirm_create_and_terminate:
        parser.error("--confirm-create-and-terminate is required")
    api_key = os.environ.get(arguments.api_key_env)
    if not api_key:
        parser.error(f"environment variable {arguments.api_key_env!r} is not set")

    client = RunPodRESTClient(api_key)
    manual_volume = bool(arguments.network_volume_id or arguments.data_center_id)
    if bool(arguments.network_volume_id) != bool(arguments.data_center_id):
        parser.error("--network-volume-id and --data-center-id must be supplied together")
    if arguments.auto_volume and manual_volume:
        parser.error("--auto-volume cannot be combined with explicit volume/data-center IDs")
    if not arguments.auto_volume and not manual_volume:
        parser.error("provide explicit volume/data-center IDs or use --auto-volume")
    if arguments.auto_volume:
        selection = RunPodDiscovery(client).select_volume(
            gpu_id=arguments.gpu_type,
            volume_name=arguments.volume_name,
            size_gb=arguments.volume_size_gb,
            allow_create=arguments.confirm_create_volume,
        )
        network_volume_id = selection.volume.volume_id
        data_center_id = selection.volume.data_center_id
        action = "created" if selection.created else "reusing"
        print(
            f"{action.capitalize()} managed volume {network_volume_id} "
            f"in {data_center_id}"
        )
    else:
        network_volume_id = arguments.network_volume_id
        data_center_id = arguments.data_center_id
    pod = client.create_pod(
        RunPodPodRequest(
            gpu_type=arguments.gpu_type,
            gpu_count=1,
            image=arguments.image,
            container_disk_gb=50,
            interruptible=False,
            network_volume_id=network_volume_id,
            data_center_ids=(data_center_id,),
        )
    )
    try:
        endpoint = client.wait_until_ready(pod.pod_id, arguments.timeout)
        print(f"Pod {pod.pod_id} is running; worker endpoint: {endpoint.address}")
    finally:
        client.terminate_pod(pod.pod_id)
        print(f"Pod {pod.pod_id} terminated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
