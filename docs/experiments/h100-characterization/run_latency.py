"""Run dependent-read latency probes on one pinned H100, then delete it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.provenance import capture_repository_state, image_digest_from_reference

from run import HERE, IMAGE, OUTPUT, REMOTE, ROOT, checked


SOURCE = "latency_inside.cu"
PROFILES = (
    ("latency_l1_ca_4k_l1", "global_chase_ca", "l1_ca_4k",
     "l1tex__t_sector_hit_rate.pct"),
    ("latency_l1_ca_4k_dram", "global_chase_ca", "l1_ca_4k",
     "dram__bytes_read.sum"),
    ("latency_l2_cg_1m_l2", "global_chase_cg", "l2_cg_1m",
     "lts__t_sector_hit_rate.pct"),
    ("latency_l2_cg_1m_dram", "global_chase_cg", "l2_cg_1m",
     "dram__bytes_read.sum"),
    ("latency_l2_cg_8m_dram", "global_chase_cg", "l2_cg_8m",
     "dram__bytes_read.sum"),
    ("latency_hbm_cg_128m_dram", "global_chase_cg", "hbm_cg_128m",
     "dram__bytes_read.sum"),
    ("latency_hbm_cg_128m_l2", "global_chase_cg", "hbm_cg_128m",
     "lts__t_sector_hit_rate.pct"),
    ("latency_shared_4k_wavefronts", "shared_chase_probe", "shared_4k",
     "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"),
)


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    for name in ("latency_timings.json", "latency_provenance.json"):
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
        checked([*ssh, "sudo", "docker", "exec", "-i", "kernel-mcts-worker",
                 "tee", f"{REMOTE}/{SOURCE}"],
                input_bytes=(HERE / SOURCE).read_bytes(), timeout=30)
        print("compiling dependent-load latency probes", flush=True)
        checked([*docker, "nvcc", "-std=c++17", "-O3", "-arch=sm_90", "-lineinfo",
                 "-o", f"{REMOTE}/latency_probe", f"{REMOTE}/{SOURCE}"], timeout=300)
        before = checked([*docker, "nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu",
                          "--format=csv,noheader,nounits"], timeout=30).decode().strip()
        print("measuring dependent-read chains", flush=True)
        print(checked([*docker, f"{REMOTE}/latency_probe", "--mode", "bench",
                       "--output", f"{REMOTE}/latency_timings.json"],
                      timeout=1800).decode().strip(), flush=True)
        after = checked([*docker, "nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu",
                         "--format=csv,noheader,nounits"], timeout=30).decode().strip()
        (OUTPUT / "latency_timings.json").write_bytes(
            checked([*docker, "cat", f"{REMOTE}/latency_timings.json"], timeout=30)
        )
        try:
            (OUTPUT / "latency.sass.txt").write_bytes(
                checked([*docker, "cuobjdump", "--dump-sass", f"{REMOTE}/latency_probe"],
                        timeout=120)
            )
        except Exception as error:
            (OUTPUT / "latency_sass.error.json").write_text(json.dumps({
                "error_type": type(error).__name__, "message": str(error),
            }, indent=2) + "\n")
        for label, kernel, case, metric in PROFILES:
            print(f"profiling {label}: {metric}", flush=True)
            command = [*docker, "ncu", "--csv", "--replay-mode", "application",
                       "--cache-control", "none", "--clock-control", "none",
                       "--target-processes", "all", "--kernel-name", f"regex:.*{kernel}.*",
                       "--launch-skip", "10", "--launch-count", "1",
                       "--metrics", metric, f"{REMOTE}/latency_probe", "--mode",
                       "profile", "--label", case]
            try:
                (OUTPUT / f"{label}.ncu.csv").write_bytes(checked(command, timeout=900))
            except Exception as error:
                (OUTPUT / f"{label}.error.json").write_text(json.dumps({
                    "error_type": type(error).__name__, "message": str(error),
                }, indent=2) + "\n")
                print(f"{label} profile failed: {type(error).__name__}", flush=True)
        (OUTPUT / "latency_provenance.json").write_text(json.dumps({
            "worker_image": IMAGE, "repo_commit": repository.commit,
            "dirty_tree": repository.dirty,
            "worker_manifest": worker.get_environment_manifest().as_dict(),
            "experiment": "standalone_h100_dependent_read_latency_v1",
            "source_sha256": hashlib.sha256((HERE / SOURCE).read_bytes()).hexdigest(),
            "gpu_state_before": before, "gpu_state_after": after,
            "ncu_profiles": [{"label": label, "metric": metric, "replay_mode": "application"}
                             for label, _, _, metric in PROFILES],
        }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
