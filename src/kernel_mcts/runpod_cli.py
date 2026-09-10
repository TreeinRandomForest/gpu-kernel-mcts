from __future__ import annotations

import argparse
import os

from .providers import RunPodPodRequest
from .runpod_api import RunPodRESTClient


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Provision a RunPod worker, confirm readiness, and terminate it."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--api-key-env", default="RUNPOD_API_KEY")
    parser.add_argument("--gpu-type", default="NVIDIA H100 80GB HBM3")
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
    pod = client.create_pod(
        RunPodPodRequest(
            gpu_type=arguments.gpu_type,
            gpu_count=1,
            image=arguments.image,
            container_disk_gb=50,
            interruptible=False,
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
