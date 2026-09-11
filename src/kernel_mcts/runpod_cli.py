from __future__ import annotations

import argparse
import os
import sys
from typing import TextIO

from .providers import RunPodPodRequest
from .runpod_api import RunPodRESTClient
from .runpod_discovery import RunPodDiscovery
from .worker_protocol import HTTPWorkerTransport


class ReadinessProgress:
    _FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, label: str, stream: TextIO = sys.stdout) -> None:
        self._label = label
        self._stream = stream
        self._frame = 0
        self._last_status: str | None = None
        self._active_line = False

    def __call__(self, status: str, elapsed_seconds: float) -> None:
        if self._stream.isatty():
            frame = self._FRAMES[self._frame % len(self._FRAMES)]
            self._frame += 1
            message = (
                f"{frame} Waiting for {self._label} — status: {status} "
                f"— {elapsed_seconds:.0f}s elapsed"
            )
            self._stream.write(f"\r\033[2K{message}")
            self._stream.flush()
            self._active_line = True
        elif status != self._last_status:
            self._stream.write(
                f"Waiting for {self._label} — status: {status} "
                f"— {elapsed_seconds:.0f}s elapsed\n"
            )
            self._stream.flush()
        self._last_status = status

    def finish(self) -> None:
        if self._active_line:
            self._stream.write("\n")
            self._stream.flush()
            self._active_line = False


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
    parser.add_argument(
        "--preferred-data-center-id",
        help="restrict automatic volume reuse or creation to this available data center",
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
    if arguments.preferred_data_center_id and not arguments.auto_volume:
        parser.error("--preferred-data-center-id requires --auto-volume")
    if not arguments.auto_volume and not manual_volume:
        parser.error("provide explicit volume/data-center IDs or use --auto-volume")
    if arguments.auto_volume:
        selection = RunPodDiscovery(client).select_volume(
            gpu_id=arguments.gpu_type,
            volume_name=arguments.volume_name,
            size_gb=arguments.volume_size_gb,
            allow_create=arguments.confirm_create_volume,
            preferred_data_center_id=arguments.preferred_data_center_id,
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
    progress = ReadinessProgress(f"pod {pod.pod_id}")
    try:
        endpoint = client.wait_until_ready(
            pod.pod_id,
            arguments.timeout,
            progress=progress,
        )
        progress.finish()
        print(f"Pod {pod.pod_id} is running; worker endpoint: {endpoint.address}")
        transport = HTTPWorkerTransport(endpoint, timeout_seconds=arguments.timeout)
        manifest = transport.get_environment_manifest()
        calibration = transport.get_calibration()
        _print_calibration(manifest, calibration)
        transport.close()
    finally:
        progress.finish()
        client.terminate_pod(pod.pod_id)
        print(f"Pod {pod.pod_id} terminated")
    return 0


def _print_calibration(manifest, calibration) -> None:
    compilation = calibration.get("compile", {})
    correctness = calibration.get("correctness", {})
    benchmark = calibration.get("benchmark", {})
    print(
        f"Worker hardware: {manifest.gpu_model} ({manifest.form_factor or 'unknown form factor'}), "
        f"compute capability {manifest.compute_capability}"
    )
    print(
        f"Root calibration {calibration.get('benchmark_id', 'unknown')}: "
        f"compile={compilation.get('success')}, "
        f"correctness={correctness.get('success')}, "
        f"max_error={correctness.get('maximum_error')}, "
        f"mean_error={correctness.get('mean_error')}"
    )
    print(
        f"Benchmark: median={benchmark.get('median_us')} us, "
        f"mean={benchmark.get('mean_us')} us, min={benchmark.get('min_us')} us, "
        f"max={benchmark.get('max_us')} us, samples={len(benchmark.get('timings_us', []))}"
    )
    for name, result in calibration.get("vendor_baselines", {}).items():
        correctness = result.get("correctness", {})
        benchmark = result.get("benchmark", {})
        print(
            f"{name} baseline: correctness={correctness.get('success')}, "
            f"max_error={correctness.get('maximum_error')}, "
            f"median={benchmark.get('median_us')} us, "
            f"mean={benchmark.get('mean_us')} us, "
            f"samples={len(benchmark.get('timings_us', []))}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
