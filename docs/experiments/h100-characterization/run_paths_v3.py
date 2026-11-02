"""Run matched cache controls and explicit on-chip probes on one H100."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.provenance import capture_repository_state, image_digest_from_reference

from run import HERE, IMAGE, OUTPUT, REMOTE, ROOT, checked


SOURCES = ("bench_inside.py", "clean_paths_inside.py", "onchip_inside.cu")
PROFILES = (
    ("clean_l2_8m_x64_dram", "controlled_read_probe", "dram__bytes_read.sum",
     "python", ("clean_paths_inside.py", "--mode", "profile", "--size", str(8 << 20),
                "--repeats", "64", "--policy", "cyclic"), "application"),
    ("clean_l2_8m_x64_l2", "controlled_read_probe", "lts__t_sector_hit_rate.pct",
     "python", ("clean_paths_inside.py", "--mode", "profile", "--size", str(8 << 20),
                "--repeats", "64", "--policy", "cyclic"), "application"),
    ("clean_hbm_256m_x4_dram", "controlled_read_probe", "dram__bytes_read.sum",
     "python", ("clean_paths_inside.py", "--mode", "profile", "--size", str(256 << 20),
                "--repeats", "4", "--policy", "disjoint"), "application"),
    ("onchip_shared_x256_wavefronts", "shared_load_probe",
     "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
     "binary", ("--mode", "profile", "--kind", "shared", "--repeats", "256"), "kernel"),
    ("onchip_register_x256_wavefronts", "register_control_probe",
     "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
     "binary", ("--mode", "profile", "--kind", "register", "--repeats", "256"), "kernel"),
)


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    for name in ("clean_paths_timings.json", "onchip_timings.json", "paths_v3_provenance.json"):
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
        print("measuring matched L2/HBM read controls", flush=True)
        print(checked([*docker, "python", f"{REMOTE}/clean_paths_inside.py",
                       "--output", f"{REMOTE}/clean_paths_timings.json"],
                      timeout=1800).decode().strip(), flush=True)
        (OUTPUT / "clean_paths_timings.json").write_bytes(
            checked([*docker, "cat", f"{REMOTE}/clean_paths_timings.json"], timeout=30)
        )
        print("compiling and measuring explicit shared/register controls", flush=True)
        checked([*docker, "nvcc", "-std=c++17", "-O3", "-arch=sm_90", "-lineinfo",
                 "-o", f"{REMOTE}/onchip_probe", f"{REMOTE}/onchip_inside.cu"],
                timeout=300)
        print(checked([*docker, f"{REMOTE}/onchip_probe", "--mode", "bench",
                       "--output", f"{REMOTE}/onchip_timings.json"],
                      timeout=1800).decode().strip(), flush=True)
        (OUTPUT / "onchip_timings.json").write_bytes(
            checked([*docker, "cat", f"{REMOTE}/onchip_timings.json"], timeout=30)
        )
        try:
            (OUTPUT / "onchip.sass.txt").write_bytes(
                checked([*docker, "cuobjdump", "--dump-sass", f"{REMOTE}/onchip_probe"],
                        timeout=120)
            )
        except Exception as error:
            (OUTPUT / "onchip_sass.error.json").write_text(json.dumps({
                "error_type": type(error).__name__, "message": str(error),
            }, indent=2) + "\n")
        for label, kernel, metric, executable, arguments, replay_mode in PROFILES:
            print(f"profiling {label}: {metric}", flush=True)
            if executable == "python":
                target = ["python", f"{REMOTE}/{arguments[0]}", *arguments[1:]]
            else:
                target = [f"{REMOTE}/onchip_probe", *arguments]
            command = [*docker, "ncu", "--csv", "--replay-mode", replay_mode,
                       "--cache-control", "none", "--clock-control", "none",
                       "--target-processes", "all", "--kernel-name", f"regex:.*{kernel}.*",
                       "--launch-skip", "10", "--launch-count", "1",
                       "--metrics", metric, *target]
            try:
                (OUTPUT / f"{label}.ncu.csv").write_bytes(checked(command, timeout=600))
            except Exception as error:
                (OUTPUT / f"{label}.error.json").write_text(json.dumps({
                    "error_type": type(error).__name__, "message": str(error),
                }, indent=2) + "\n")
                print(f"{label} profile failed: {type(error).__name__}", flush=True)
        (OUTPUT / "paths_v3_provenance.json").write_text(json.dumps({
            "worker_image": IMAGE, "repo_commit": repository.commit,
            "dirty_tree": repository.dirty,
            "worker_manifest": worker.get_environment_manifest().as_dict(),
            "experiment": "standalone_h100_paths_v3",
            "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                              for name in SOURCES},
            "ncu_profiles": [{"label": label, "metric": metric, "replay_mode": mode}
                             for label, _, metric, _, _, mode in PROFILES],
        }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
