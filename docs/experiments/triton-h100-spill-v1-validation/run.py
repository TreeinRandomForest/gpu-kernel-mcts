"""Provision one H100 and run the current package as a local worker overlay."""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
from pathlib import Path

from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.provenance import capture_repository_state, image_digest_from_reference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CROSSOVER = ROOT / "docs/experiments/triton-h100-winner-crossover"
OUTPUT = HERE / "output"
IMAGE = (
    "docker.io/saarora/gpu-kernel-mcts@sha256:"
    "d0cd8e7b784ac8828691aa0add0e41d1802e22ef11745780de018afaa12c0469"
)
REMOTE = "/tmp/spill-v1/docs/experiments/triton-h100-spill-v1-validation"


def checked(command: list[str], *, input_bytes: bytes | None = None,
            timeout: float = 900.0) -> bytes:
    result = subprocess.run(command, input=input_bytes, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): "
                           + result.stderr.decode(errors="replace")[-1200:])
    return result.stdout


def bundle() -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        archive.add(ROOT / "src/kernel_mcts", arcname="src/kernel_mcts",
                    filter=lambda info: None if "__pycache__" in info.name else info)
        archive.add(HERE / "run_inside.py",
                    arcname="docs/experiments/triton-h100-spill-v1-validation/run_inside.py")
        for name in ("grouped_bn256_w4", "grouped_bn256_w8"):
            archive.add(CROSSOVER / f"{name}.py",
                        arcname=f"docs/experiments/triton-h100-spill-v1-validation/{name}.py")
    return stream.getvalue()


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    for name in ("spill-v1-validation.sqlite", "summary.json", "provenance.json"):
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
            f"worker {stage}: {elapsed:.1f}s", flush=True
        ),
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
        checked([*docker, "mkdir", "-p", "/tmp/spill-v1", REMOTE], timeout=30)
        checked([*ssh, "sudo", "docker", "exec", "-i", "kernel-mcts-worker",
                 "tar", "-xz", "-C", "/tmp/spill-v1"], input_bytes=bundle(), timeout=120)
        for name, payload in (
            ("manifest.json", worker.get_environment_manifest().as_dict()),
            ("calibration.json", worker.get_calibration()),
        ):
            checked([*ssh, "sudo", "docker", "exec", "-i", "kernel-mcts-worker",
                     "tee", f"{REMOTE}/{name}"],
                    input_bytes=json.dumps(payload).encode(), timeout=30)
        print("running spill_v1 MCTS validation on H100", flush=True)
        status = checked([*ssh, "sudo", "docker", "exec", "-e",
                          "PYTHONPATH=/tmp/spill-v1/src", "kernel-mcts-worker",
                          "python", f"{REMOTE}/run_inside.py"], timeout=1800)
        print(status.decode().strip(), flush=True)
        for name in ("spill-v1-validation.sqlite", "summary.json"):
            raw = checked([*docker, "cat", f"{REMOTE}/output/{name}"], timeout=120)
            (OUTPUT / name).write_bytes(raw)
        (OUTPUT / "provenance.json").write_text(json.dumps({
            "base_image": IMAGE, "repo_commit": repository.commit,
            "dirty_tree": repository.dirty,
            "execution": "current source overlay in separate Python process",
        }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
