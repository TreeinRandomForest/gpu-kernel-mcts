"""Provision one H100 for read-only cache-path probes, then delete it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.provenance import capture_repository_state, image_digest_from_reference

from run import HERE, IMAGE, OUTPUT, REMOTE, ROOT, checked


METRICS = ("dram__bytes_read.sum", "dram__bytes_write.sum",
           "lts__t_sector_hit_rate.pct")
PROFILES = (("read_cg_8m", 8 << 20, ".cg"),
            ("read_ca_8m", 8 << 20, ".ca"),
            ("read_cg_256m", 256 << 20, ".cg"))
APP_PROFILES = (("read_cg_1m_app", 1 << 20, ".cg"),
                ("read_cg_8m_app", 8 << 20, ".cg"),
                ("read_ca_8m_app", 8 << 20, ".ca"),
                ("read_cg_256m_app", 256 << 20, ".cg"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles-only", action="store_true",
                        help="rerun NCU with application replay; retain existing timings")
    args = parser.parse_args()
    provenance_name = ("read_paths_app_provenance.json" if args.profiles_only
                       else "read_paths_provenance.json")
    OUTPUT.mkdir(exist_ok=True)
    names = [provenance_name]
    if not args.profiles_only:
        names.append("read_paths_timings.json")
    for name in names:
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
        for name in ("bench_inside.py", "read_paths_inside.py"):
            checked([*ssh, "sudo", "docker", "exec", "-i", "kernel-mcts-worker",
                     "tee", f"{REMOTE}/{name}"],
                    input_bytes=(HERE / name).read_bytes(), timeout=30)
        if not args.profiles_only:
            print("measuring read-only .cg/.ca paths", flush=True)
            print(checked([*docker, "python", f"{REMOTE}/read_paths_inside.py",
                           "--output", f"{REMOTE}/read_paths_timings.json"],
                          timeout=1800).decode().strip(), flush=True)
            (OUTPUT / "read_paths_timings.json").write_bytes(
                checked([*docker, "cat", f"{REMOTE}/read_paths_timings.json"], timeout=30)
            )
        profiles = APP_PROFILES if args.profiles_only else PROFILES
        for label, size, modifier in profiles:
            print(f"profiling {label} traffic and L2 hit rate", flush=True)
            command = [*docker, "ncu", "--csv", "--cache-control", "none",
                       "--clock-control", "none", "--target-processes", "all",
                       "--kernel-name", "regex:.*read_reduce_probe.*",
                       "--launch-skip", "10", "--launch-count", "1",
                       "--metrics", ",".join(METRICS), "python",
                       f"{REMOTE}/read_paths_inside.py", "--mode", "profile",
                       "--size", str(size), "--cache-modifier", modifier]
            if args.profiles_only:
                command.insert(len(docker) + 2, "application")
                command.insert(len(docker) + 2, "--replay-mode")
            try:
                (OUTPUT / f"{label}.ncu.csv").write_bytes(checked(command, timeout=600))
            except Exception as error:
                (OUTPUT / f"{label}.error.json").write_text(json.dumps({
                    "error_type": type(error).__name__, "message": str(error),
                }, indent=2) + "\n")
                print(f"{label} profile failed: {type(error).__name__}", flush=True)
        (OUTPUT / provenance_name).write_text(json.dumps({
            "worker_image": IMAGE, "repo_commit": repository.commit,
            "dirty_tree": repository.dirty,
            "worker_manifest": worker.get_environment_manifest().as_dict(),
            "experiment": "standalone_h100_read_paths_v1",
            "ncu_metrics": METRICS,
            "ncu_cache_control": "none",
            "ncu_clock_control": "none",
            "ncu_replay_mode": "application" if args.profiles_only else "kernel",
        }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
