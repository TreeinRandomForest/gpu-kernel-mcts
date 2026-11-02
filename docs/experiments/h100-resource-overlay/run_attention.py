"""Compile, verify, and profile attention on one disposable Nebius H100."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.provenance import capture_repository_state, image_digest_from_reference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = HERE / "output"
IMAGE = (
    "docker.io/saarora/gpu-kernel-mcts@sha256:"
    "d0cd8e7b784ac8828691aa0add0e41d1802e22ef11745780de018afaa12c0469"
)
REMOTE = "/tmp/attention-resource-overlay"
METRICS = ("dram__bytes_read.sum", "dram__bytes_write.sum",
           "lts__t_sector_hit_rate.pct", "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
           "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum")


def checked(command: list[str], *, input_bytes: bytes | None = None,
            timeout: float = 900) -> bytes:
    result = subprocess.run(command, input=input_bytes, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): "
                           + result.stderr.decode(errors="replace")[-1000:])
    return result.stdout


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    for name in ("attention_result.json", "attention.ptx", "attention_provenance.json"):
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
                 "tee", f"{REMOTE}/attention_inside.py"],
                input_bytes=(HERE / "attention_inside.py").read_bytes(), timeout=30)
        print("compiling and verifying real attention", flush=True)
        print(checked([*docker, "python", f"{REMOTE}/attention_inside.py",
                       "--mode", "compile", "--output", REMOTE], timeout=900).decode().strip(),
              flush=True)
        (OUTPUT / "attention_result.json").write_bytes(
            checked([*docker, "cat", f"{REMOTE}/result.json"], timeout=30))
        (OUTPUT / "attention.ptx").write_bytes(
            checked([*docker, "cat", f"{REMOTE}/attention.ptx"], timeout=30))
        print("profiling aggregate traffic", flush=True)
        try:
            command = [*docker, "ncu", "--csv", "--replay-mode", "application",
                       "--cache-control", "none", "--clock-control", "none",
                       "--target-processes", "all", "--kernel-name",
                       "regex:.*causal_attention_probe.*", "--launch-skip", "1",
                       "--launch-count", "1", "--metrics", ",".join(METRICS),
                       "python", f"{REMOTE}/attention_inside.py", "--mode", "profile"]
            (OUTPUT / "attention.ncu.csv").write_bytes(checked(command, timeout=900))
        except Exception as error:
            (OUTPUT / "attention_ncu_error.json").write_text(json.dumps({
                "type": type(error).__name__, "message": str(error),
            }, indent=2) + "\n")
            print(f"NCU unavailable: {type(error).__name__}", flush=True)
        (OUTPUT / "attention_provenance.json").write_text(json.dumps({
            "schema_version": 1, "experiment": "real_triton_attention_overlay_v1",
            "worker_image": IMAGE, "repo_commit": repository.commit,
            "dirty_tree": repository.dirty,
            "worker_manifest": worker.get_environment_manifest().as_dict(),
            "source_sha256": hashlib.sha256((HERE / "attention_inside.py").read_bytes()).hexdigest(),
            "ptx_sha256": hashlib.sha256((OUTPUT / "attention.ptx").read_bytes()).hexdigest(),
            "ncu_metrics": METRICS,
        }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
