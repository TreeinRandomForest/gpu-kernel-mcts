"""Run access-order/grid-size sensitivity on one pinned H100 worker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.provenance import capture_repository_state, image_digest_from_reference

from run import HERE, IMAGE, OUTPUT, REMOTE, ROOT, checked


SOURCES = ("bench_inside.py", "sensitivity_inside.py")
PROFILES = (
    ("sensitivity_l2_contiguous_dram", 8 << 20, 528, "contiguous",
     "dram__bytes_read.sum"),
    ("sensitivity_l2_strided_dram", 8 << 20, 528, "stride_32_words",
     "dram__bytes_read.sum"),
    ("sensitivity_hbm_contiguous_dram", 256 << 20, 528, "contiguous",
     "dram__bytes_read.sum"),
    ("sensitivity_hbm_strided_dram", 256 << 20, 528, "stride_32_words",
     "dram__bytes_read.sum"),
)


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    for name in ("sensitivity_timings.json", "sensitivity_provenance.json"):
        if (OUTPUT / name).exists():
            raise FileExistsError(OUTPUT / name)
    repository = capture_repository_state(ROOT)
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
    with provider.worker_for_run(hardware) as worker:
        instance = json.loads(checked([
            "nebius", "compute", "instance", "get", "--id", worker.instance.instance_id,
            "--format", "json", "--no-check-update",
        ], timeout=30))
        ip = instance["status"]["network_interfaces"][0]["public_ip_address"]["address"].split("/")[0]
        ssh = ["ssh", "-F", "/dev/null", "-i", str(config.ssh_private_key),
               "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
               f"{config.username}@{ip}"]
        docker = [*ssh, "sudo", "docker", "exec", "kernel-mcts-worker"]
        checked([*docker, "mkdir", "-p", REMOTE], timeout=30)
        for name in SOURCES:
            checked([*ssh, "sudo", "docker", "exec", "-i", "kernel-mcts-worker",
                     "tee", f"{REMOTE}/{name}"],
                    input_bytes=(HERE / name).read_bytes(), timeout=30)
        print("measuring access-pattern and grid-size sensitivity", flush=True)
        print(checked([*docker, "python", f"{REMOTE}/sensitivity_inside.py",
                       "--output", f"{REMOTE}/sensitivity_timings.json"],
                      timeout=3600).decode().strip(), flush=True)
        (OUTPUT / "sensitivity_timings.json").write_bytes(
            checked([*docker, "cat", f"{REMOTE}/sensitivity_timings.json"], timeout=30)
        )
        for label, size, grid, pattern, metric in PROFILES:
            print(f"profiling {label}: {metric}", flush=True)
            command = [*docker, "ncu", "--csv", "--replay-mode", "application",
                       "--cache-control", "none", "--clock-control", "none",
                       "--target-processes", "all", "--kernel-name",
                       "regex:.*sensitivity_read_probe.*", "--launch-skip", "10",
                       "--launch-count", "1", "--metrics", metric, "python",
                       f"{REMOTE}/sensitivity_inside.py", "--mode", "profile",
                       "--size", str(size), "--grid", str(grid),
                       "--pattern", pattern]
            try:
                (OUTPUT / f"{label}.ncu.csv").write_bytes(checked(command, timeout=900))
            except Exception as error:
                (OUTPUT / f"{label}.error.json").write_text(json.dumps({
                    "error_type": type(error).__name__, "message": str(error),
                }, indent=2) + "\n")
                print(f"{label} profile failed: {type(error).__name__}", flush=True)
        (OUTPUT / "sensitivity_provenance.json").write_text(json.dumps({
            "worker_image": IMAGE, "repo_commit": repository.commit,
            "dirty_tree": repository.dirty,
            "worker_manifest": worker.get_environment_manifest().as_dict(),
            "experiment": "standalone_h100_read_sensitivity_v1",
            "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                              for name in SOURCES},
            "ncu_profiles": [{"label": label, "metric": metric,
                              "replay_mode": "application"}
                             for label, _, _, _, metric in PROFILES],
        }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
