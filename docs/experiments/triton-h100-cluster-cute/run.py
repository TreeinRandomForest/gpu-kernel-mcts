"""Run paired CuTe single-/two-CTA H100 GEMM on one Nebius worker."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import tarfile
from pathlib import Path
from uuid import uuid4

from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.provenance import capture_repository_state, image_digest_from_reference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PAIRED = HERE / "paired.py"
RESULTS = HERE / "results.json"
OUTPUT = HERE / "output"
IMAGE = "docker.io/saarora/gpu-kernel-mcts:cutedsl-wide-swizzle-v71"
REMOTE_SCRIPT = "/tmp/cluster-cute-paired.py"
REMOTE_OUTPUT = "/tmp/cluster-cute-output"


def load_helpers():
    path = ROOT / "docs/experiments/triton-h100-context-factorial/run.py"
    spec = importlib.util.spec_from_file_location("cluster_cute_helpers", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import H100 worker helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(report: dict) -> None:
    temporary = RESULTS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESULTS)


def main() -> None:
    if RESULTS.exists():
        raise FileExistsError(f"refusing to overwrite {RESULTS}")
    repository = capture_repository_state(ROOT)
    report = {"schema_version": 1, "run_id": str(uuid4()),
              "image": IMAGE, "git_commit": repository.commit,
              "dirty_tree": repository.dirty,
              "harness_sha256": hashlib.sha256(PAIRED.read_bytes()).hexdigest(),
              "complete": False}
    config = NebiusConfig(
        image=IMAGE, project_id="project-e00aymjmpr00grzcky46ba",
        subnet_id="vpcsubnet-e00a30bhyztjvq7yaz",
        ssh_public_key=Path.home() / ".ssh/nebius.pub",
        ssh_private_key=Path.home() / ".ssh/nebius", username="sanjay",
        startup_timeout_seconds=1200, backend="cute_dsl",
        project_git_commit=repository.commit, project_dirty_tree=repository.dirty,
        container_digest=image_digest_from_reference(IMAGE),
    )
    provider = create_nebius_provider(
        config, readiness_progress=lambda stage, elapsed: print(
            f"worker {stage}: {elapsed:.1f}s", flush=True),
    )
    hardware = HardwareSpec("H100", form_factor="SXM",
                            minimum_compute_capability="9.0", required_profilers=("ncu",))
    helpers = load_helpers()
    OUTPUT.mkdir(exist_ok=True)
    save(report)
    try:
        with provider.worker_for_run(hardware) as worker:
            report["environment_manifest"] = worker.get_environment_manifest().as_dict()
            save(report)
            bundle = helpers.package_bundle()
            helpers.overlay_worker(worker, config.ssh_private_key, bundle)
            instance = json.loads(helpers.checked([
                "nebius", "compute", "instance", "get", "--id", worker.instance.instance_id,
                "--format", "json", "--no-check-update",
            ], timeout=30))
            ip = instance["status"]["network_interfaces"][0]["public_ip_address"]["address"].split("/")[0]
            ssh = ["ssh", "-F", "/dev/null", "-i", str(config.ssh_private_key),
                   "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                   f"sanjay@{ip}"]
            remote_host_script = "/tmp/cluster-cute-paired-host.py"
            helpers.checked([*ssh, "tee", remote_host_script], input_bytes=PAIRED.read_bytes())
            helpers.checked([*ssh, "sudo", "docker", "cp", remote_host_script,
                             "kernel-mcts-worker:" + REMOTE_SCRIPT])
            command = [*ssh, "timeout", "2400", "sudo", "docker", "exec", "kernel-mcts-worker",
                       "python", REMOTE_SCRIPT, "--output", f"{REMOTE_OUTPUT}/paired.json"]
            print("running paired CuTe cluster experiment", flush=True)
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            try:
                assert process.stdout is not None
                last_lines: list[str] = []
                for line in process.stdout:
                    print(line.rstrip(), flush=True)
                    last_lines.append(line)
                    last_lines = last_lines[-20:]
                if process.wait(timeout=2400) != 0:
                    raise RuntimeError("remote paired CuTe run failed: "
                                       + "".join(last_lines)[-1500:])
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
            archive = helpers.checked([*ssh, "sudo", "docker", "exec", "kernel-mcts-worker",
                                       "tar", "-cz", "-C", REMOTE_OUTPUT, "."], timeout=240)
            archive_path = OUTPUT / "remote-artifacts.tar.gz"
            archive_path.write_bytes(archive)
            with tarfile.open(archive_path, "r:gz") as tar:
                for member in tar.getmembers():
                    target = (OUTPUT / member.name).resolve()
                    if not target.is_relative_to(OUTPUT.resolve()):
                        raise ValueError("unsafe artifact path")
                tar.extractall(OUTPUT, filter="data")
            paired = json.loads((OUTPUT / "paired.json").read_text())
            if not paired.get("complete") or len(paired.get("observations", [])) != 6:
                raise ValueError("paired result incomplete")
            report["remote_artifacts_sha256"] = hashlib.sha256(archive).hexdigest()
            report["complete"] = True
            save(report)
    finally:
        save(report)


if __name__ == "__main__":
    main()
