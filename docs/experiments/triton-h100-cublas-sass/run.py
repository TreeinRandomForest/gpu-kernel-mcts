"""Collect matched-input timings and NCU SASS reports on one guarded H100."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import subprocess
import time
from pathlib import Path
from uuid import uuid4

from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.provenance import capture_repository_state, image_digest_from_reference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "docs/experiments/triton-h100-bd-crossover/trace_B.py"
HARNESS = HERE / "paired_harness.py"
RESULTS = HERE / "results.json"
OUTPUT = HERE / "output"
IMAGE = (
    "docker.io/saarora/gpu-kernel-mcts@sha256:"
    "d0cd8e7b784ac8828691aa0add0e41d1802e22ef11745780de018afaa12c0469"
)
REMOTE_SOURCE = "/tmp/paired-trace-b.py"
REMOTE_HARNESS = "/tmp/paired_harness.py"


def save(report: dict) -> None:
    temporary = RESULTS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESULTS)


def checked(command: list[str], *, input_bytes: bytes | None = None,
            timeout: float = 180) -> bytes:
    result = subprocess.run(command, input=input_bytes, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): "
                           + (result.stderr or result.stdout).decode(errors="replace")[-1600:])
    return result.stdout


def instance_public_ip(instance_id: str) -> str:
    value = json.loads(checked([
        "nebius", "compute", "instance", "get", "--id", instance_id,
        "--format", "json", "--no-check-update",
    ], timeout=30))
    for interface in value.get("status", {}).get("network_interfaces", []):
        address = interface.get("public_ip_address", {}).get("address")
        if isinstance(address, str):
            return str(ipaddress.ip_interface(address).ip)
    raise RuntimeError("Nebius instance has no public IP")


def ssh_prefix(ip: str) -> list[str]:
    return [
        "ssh", "-F", "/dev/null", "-i", str(Path.home() / ".ssh/nebius"),
        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3", f"sanjay@{ip}",
        "sudo", "docker", "exec", "kernel-mcts-worker",
    ]


def target_command(mode: str, name: str | None = None) -> list[str]:
    command = ["python", REMOTE_HARNESS, "--source", REMOTE_SOURCE, "--mode", mode]
    if name is not None:
        command += ["--target", name]
    return command


def upload(ssh: list[str], remote: str, payload: bytes) -> None:
    command = [*ssh[:-1], "-i", ssh[-1], "tee", remote]
    for attempt in range(4):
        try:
            checked(command, input_bytes=payload, timeout=60)
            return
        except RuntimeError as error:
            if "Connection reset" not in str(error) or attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


def source_profile_command(name: str, prefix: str) -> list[str]:
    return [
        "ncu", "--set", "full", "--import-source", "yes",
        "--nvtx", "--nvtx-include", f"PAIRED_{name.upper()}/",
        "--launch-count", "1", "--export", prefix,
        *target_command("profile", name),
    ]


def main() -> None:
    if RESULTS.exists():
        raise FileExistsError(f"refusing to overwrite {RESULTS}")
    source = SOURCE.read_bytes()
    harness = HARNESS.read_bytes()
    source_hash = hashlib.sha256(source).hexdigest()
    if source_hash != "b9b4574de16a817477990842f3e1e4d4dbd95b662e17f375f5c82981a463fba4":
        raise ValueError("fast B source identity changed")
    repository = capture_repository_state(ROOT)
    report = {
        "schema_version": 1, "run_id": str(uuid4()), "image": IMAGE,
        "git_commit": repository.commit, "dirty_tree": repository.dirty,
        "source_sha256": source_hash,
        "harness_sha256": hashlib.sha256(harness).hexdigest(),
        "benchmark": None, "profiles": {}, "complete": False,
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
    OUTPUT.mkdir(exist_ok=True)
    save(report)
    try:
        with provider.worker_for_run(hardware) as worker:
            report["environment_manifest"] = worker.get_environment_manifest().as_dict()
            save(report)
            ssh = ssh_prefix(instance_public_ip(worker.instance.instance_id))
            for remote, payload in ((REMOTE_SOURCE, source), (REMOTE_HARNESS, harness)):
                upload(ssh, remote, payload)
            print("running paired same-input benchmark", flush=True)
            benchmark = json.loads(checked([*ssh, *target_command("benchmark")],
                                           timeout=360))
            if benchmark.get("status") != "ok" or len(benchmark.get("observations", [])) != 6:
                raise RuntimeError("paired benchmark failed")
            report["benchmark"] = benchmark
            save(report)
            for item in benchmark["observations"]:
                print(item["round"], item["implementation"],
                      round(item["median_us"], 3), flush=True)
            for name in ("cublas", "triton"):
                prefix = f"/tmp/paired-{name}-source"
                print(f"collecting {name} NCU Source/SASS", flush=True)
                checked([*ssh, *source_profile_command(name, prefix)], timeout=1800)
                remote_report = prefix + ".ncu-rep"
                report_bytes = checked([*ssh, "cat", remote_report], timeout=240)
                (OUTPUT / f"{name}.ncu-rep").write_bytes(report_bytes)
                exports = {}
                for view in ("sass", "ptx"):
                    result = subprocess.run([
                        *ssh, "ncu", "--import", remote_report, "--page", "source",
                        "--print-source", view, "--csv",
                    ], capture_output=True, timeout=240)
                    if result.returncode == 0:
                        (OUTPUT / f"{name}.{view}.csv").write_bytes(result.stdout)
                        exports[view] = "captured"
                    else:
                        exports[view] = "unavailable"
                raw = checked([*ssh, "ncu", "--import", remote_report,
                               "--page", "raw", "--csv"], timeout=240)
                (OUTPUT / f"{name}.raw.csv").write_bytes(raw)
                report["profiles"][name] = {
                    "ncu_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                    "ncu_report_size_bytes": len(report_bytes),
                    "source_exports": exports,
                }
                save(report)
            report["complete"] = True
            save(report)
    finally:
        save(report)


if __name__ == "__main__":
    main()
