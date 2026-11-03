"""Same-worker timing and diagnostic NCU comparison of fast Triton B and cuBLAS."""

from __future__ import annotations

import csv
import hashlib
import io
import ipaddress
import json
import random
import shlex
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from uuid import uuid4

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD
from kernel_mcts.cuda_backend import _ncu_csv_output, _parse_ncu_csv
from kernel_mcts.domain import KernelProgram, ProposalStatus
from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.profiling import resolve_profile_metric_set
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.provenance import capture_repository_state, image_digest_from_reference
from kernel_mcts.serialization import serialize_evaluation


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "docs/experiments/triton-h100-bd-crossover/trace_B.py"
TRACE = ROOT / "docs/experiments/triton-h100-bd-crossover/trace_manifest.json"
RESULTS = HERE / "results.json"
IMAGE = (
    "docker.io/saarora/gpu-kernel-mcts@sha256:"
    "d0cd8e7b784ac8828691aa0add0e41d1802e22ef11745780de018afaa12c0469"
)
VENDOR_BINARY = "/tmp/kernel-mcts-artifacts/bf16-gemm-vendor-baselines"


def save(report: dict) -> None:
    temporary = RESULTS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESULTS)


def checked(command: list[str], *, timeout: float = 900) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): " + (result.stderr or result.stdout)[-1200:]
        )
    return result.stdout


def package_bundle() -> bytes:
    """Overlay current worker code for diagnostic_v2 on the pinned image."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        archive.add(ROOT / "src/kernel_mcts", arcname="kernel_mcts",
                    filter=lambda info: None if "__pycache__" in info.name else info)
    return stream.getvalue()


def overlay_worker(worker, ssh_private_key: Path, bundle: bytes) -> None:
    ip = instance_public_ip(worker.instance.instance_id)
    ssh = [
        "ssh", "-F", "/dev/null", "-i", str(ssh_private_key),
        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        f"sanjay@{ip}",
    ]
    package_path = checked([*ssh, shlex.join([
        "sudo", "docker", "exec", "kernel-mcts-worker", "python", "-c",
        "import kernel_mcts,pathlib; print(pathlib.Path(kernel_mcts.__file__).parent.parent)",
    ])], timeout=60).strip()
    if not package_path.startswith("/opt/venv/") or not package_path.endswith("site-packages"):
        raise RuntimeError("unexpected worker package location")
    result = subprocess.run(
        [*ssh, "sudo", "docker", "exec", "-i", "kernel-mcts-worker", "tar",
         "-xz", "-C", package_path], input=bundle, capture_output=True, timeout=120,
    )
    if result.returncode:
        raise RuntimeError("worker package overlay failed: "
                           + result.stderr.decode(errors="replace")[-1200:])
    checked([*ssh, "sudo", "docker", "restart", "kernel-mcts-worker"], timeout=120)
    for _ in range(30):
        try:
            worker._transport.get_environment_manifest()
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError("overlaid worker did not become ready")


def instance_public_ip(instance_id: str) -> str:
    instance = json.loads(checked([
        "nebius", "compute", "instance", "get", "--id", instance_id,
        "--format", "json", "--no-check-update",
    ], timeout=30))
    for interface in instance.get("status", {}).get("network_interfaces", []):
        address = interface.get("public_ip_address", {}).get("address")
        if isinstance(address, str):
            return str(ipaddress.ip_interface(address).ip)
    raise RuntimeError("Nebius instance has no public IP")


def ssh_prefix(ip: str) -> list[str]:
    return [
        "ssh", "-F", "/dev/null", "-i", str(Path.home() / ".ssh/nebius"),
        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10", f"sanjay@{ip}",
        "sudo", "docker", "exec", "kernel-mcts-worker",
    ]


def vendor_target(*, warmups: int, measurements: int) -> list[str]:
    return [
        VENDOR_BINARY, "--implementation=cublas", "--M=4096", "--N=4096",
        "--K=4096", "--rtol=0.02", "--atol=0.02", "--seed=0",
        f"--warmups={warmups}", f"--measurements={measurements}",
    ]


def cublas_profile_command(metrics: tuple[str, ...]) -> list[str]:
    # The baseline executable's first GPU kernel is its initial cuBLAS GEMM.
    # Verify the selected kernel name from NCU before accepting the metrics.
    return [
        "ncu", "--csv", "--page", "raw", "--log-file", "stdout",
        "--kernel-name-base", "function", "--launch-count", "1",
        "--metrics", ",".join(metrics),
        *vendor_target(warmups=1, measurements=1),
    ]


def parse_cublas_profile(output: str, metrics: tuple[str, ...]) -> dict:
    csv_text = _ncu_csv_output(output, output, metrics)
    rows = list(csv.reader(csv_text.splitlines()))
    header = next((row for row in rows if "Kernel Name" in row and
                   ("Metric Name" in row or all(metric in row for metric in metrics))), None)
    if header is None:
        raise ValueError("NCU did not identify the selected kernel")
    name_index = header.index("Kernel Name")
    start = rows.index(header) + 1
    launch = {}
    if "Metric Name" in header:
        metric_index = header.index("Metric Name")
        names = {row[name_index] for row in rows[start:]
                 if len(row) > max(name_index, metric_index) and row[metric_index] in metrics}
        if len(names) != 1:
            raise ValueError(f"expected one profiled kernel, got {sorted(names)}")
        name = names.pop()
    else:
        data = [row for row in rows[start + 1:]
                if len(row) == len(header) and row[0].isdigit()]
        if len(data) != 1:
            raise ValueError(f"expected one profiled kernel, got {len(data)}")
        values = dict(zip(header, data[0]))
        if values.get("Process Name") != "bf16-gemm-vendor-baselines":
            raise ValueError("profiled process is not the cuBLAS vendor baseline")
        name = values["Kernel Name"]
        launch = {key: values.get(key) for key in (
            "Block Size", "Grid Size", "launch__registers_per_thread",
            "launch__shared_mem_per_block", "launch__cluster_size",
            "launch__waves_per_multiprocessor", "gpu__time_duration.sum",
        )}
    if not name:
        raise ValueError("selected kernel has no name")
    parsed = _parse_ncu_csv(csv_text, metrics)
    return {"kernel_name": name, "launch": launch, "metrics": parsed}


def finish_saved_capture() -> None:
    """Complete a run whose two profiles were saved before CSV parsing failed."""
    report = json.loads(RESULTS.read_text())
    if report.get("complete") or len(report.get("observations", [])) != 6 or not report.get("triton_profile"):
        raise ValueError("saved run is not an eligible captured comparison")
    definition = resolve_profile_metric_set(
        "diagnostic_v2", "sm_90", report["environment_manifest"]["profiler_versions"]["ncu"]
    )
    raw = (HERE / "cublas.ncu.csv").read_text()
    profile = parse_cublas_profile(raw, definition.metrics)
    report["cublas_profile"] = {
        "metric_set": definition.id, "kernel_name": profile["kernel_name"],
        "launch": profile["launch"], "resolved_metrics": list(definition.metrics),
        "metrics": profile["metrics"],
        "summary": {alias: profile["metrics"][metric]["value"]
                    for metric, alias in definition.aliases.items()},
    }
    report["complete"] = True
    save(report)


def main() -> None:
    if RESULTS.exists():
        raise FileExistsError(f"refusing to overwrite {RESULTS}")
    source = SOURCE.read_text()
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    trace = json.loads(TRACE.read_text())
    expected = next(arm["source_sha256"] for arm in trace["arms"] if arm["arm"] == "B")
    if source_hash != expected:
        raise ValueError("fast B source no longer matches corrected seed-0 trace")
    repository = capture_repository_state(ROOT)
    bundle = package_bundle()
    report = {
        "schema_version": 1, "run_id": str(uuid4()), "image": IMAGE,
        "git_commit": repository.commit, "dirty_tree": repository.dirty,
        "source_bundle_sha256": hashlib.sha256(bundle).hexdigest(),
        "triton_source_sha256": source_hash, "workload": BF16_GEMM_WORKLOAD.benchmark_id,
        "observations": [], "complete": False,
    }
    config = NebiusConfig(
        image=IMAGE, project_id="project-e00aymjmpr00grzcky46ba",
        subnet_id="vpcsubnet-e00a30bhyztjvq7yaz",
        ssh_public_key=Path.home() / ".ssh/nebius.pub",
        ssh_private_key=Path.home() / ".ssh/nebius", username="sanjay",
        startup_timeout_seconds=1200, backend="triton_h100",
        project_git_commit=repository.commit, project_dirty_tree=repository.dirty,
        container_digest=image_digest_from_reference(IMAGE),
    )
    provider = create_nebius_provider(
        config, readiness_progress=lambda stage, elapsed: print(
            f"worker {stage}: {elapsed:.1f}s", flush=True),
    )
    hardware = HardwareSpec("H100", form_factor="SXM",
                            minimum_compute_capability="9.0", required_profilers=("ncu",))
    save(report)
    try:
        with provider.worker_for_run(hardware) as worker:
            overlay_worker(worker, config.ssh_private_key, bundle)
            manifest = worker.get_environment_manifest().as_dict()
            report["environment_manifest"] = manifest
            report["startup_calibration"] = worker.get_calibration()
            save(report)
            ip = instance_public_ip(worker.instance.instance_id)
            ssh = ssh_prefix(ip)
            first_evaluation_id = None
            for round_index in range(3):
                order = ["triton", "cublas"]
                random.Random(200 + round_index).shuffle(order)
                for name in order:
                    if name == "triton":
                        evaluation_id = f"cublas-compare-{round_index}-{uuid4()}"
                        result = worker.evaluate(
                            evaluation_id, KernelProgram(source, backend="triton_h100"),
                            BF16_GEMM_WORKLOAD, "tier0",
                        )
                        if result.status != ProposalStatus.VALID:
                            raise RuntimeError(f"Triton verification failed: {result.status.value}")
                        payload = serialize_evaluation(result)
                        observation = {
                            "round": round_index, "implementation": name,
                            "evaluation_id": evaluation_id,
                            "benchmark": payload["benchmark"],
                            "correctness": payload.get("correctness"),
                            "compiler_resources": result.metadata["compiler_context"]["resources"],
                        }
                        if first_evaluation_id is None:
                            first_evaluation_id = evaluation_id
                        median = payload["benchmark"]["median_us"]
                    else:
                        vendor = json.loads(checked([
                            *ssh, *vendor_target(warmups=10, measurements=30)
                        ], timeout=180))
                        if vendor.get("status") != "ok" or vendor.get("success") is not True:
                            raise RuntimeError("cuBLAS benchmark failed")
                        samples = vendor.get("timings_us")
                        if not isinstance(samples, list) or len(samples) != 30:
                            raise RuntimeError("cuBLAS benchmark returned the wrong sample count")
                        median = statistics.median(samples)
                        observation = {
                            "round": round_index, "implementation": name,
                            "median_us": median, "timings_us": samples,
                            "correctness": {
                                "success": vendor["success"],
                                "maximum_error": vendor["maximum_error"],
                                "mean_error": vendor["mean_error"],
                            },
                        }
                    report["observations"].append(observation)
                    save(report)
                    print(f"round {round_index} {name}: {median:.3f} us", flush=True)
            if first_evaluation_id is None:
                raise RuntimeError("no Triton evaluation to profile")
            print("collecting Triton diagnostic_v2 NCU profile", flush=True)
            report["triton_profile"] = worker.profile(
                first_evaluation_id, "lightweight", "diagnostic_v2"
            )
            save(report)
            ncu_version = manifest["profiler_versions"]["ncu"]
            definition = resolve_profile_metric_set("diagnostic_v2", "sm_90", ncu_version)
            print("collecting cuBLAS diagnostic_v2 NCU profile", flush=True)
            raw = checked([*ssh, *cublas_profile_command(definition.metrics)], timeout=900)
            (HERE / "cublas.ncu.csv").write_text(raw)
            profile = parse_cublas_profile(raw, definition.metrics)
            report["cublas_profile"] = {
                "metric_set": definition.id, "kernel_name": profile["kernel_name"],
                "launch": profile["launch"],
                "resolved_metrics": list(definition.metrics),
                "metrics": profile["metrics"],
                "summary": {alias: profile["metrics"][metric]["value"]
                            for metric, alias in definition.aliases.items()},
            }
            report["complete"] = True
            save(report)
    finally:
        save(report)


if __name__ == "__main__":
    if sys.argv[1:] == ["--recover"]:
        finish_saved_capture()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit("usage: run.py [--recover]")
