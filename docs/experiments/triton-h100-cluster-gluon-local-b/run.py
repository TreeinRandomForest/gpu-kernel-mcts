"""Run a three-arm B-descriptor layout ablation on one Nebius H100."""

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
SOURCE = HERE / "kernel.py"
HARNESS = HERE / "paired.py"
RESULTS = HERE / "results.json"
OUTPUT = HERE / "output"
IMAGE = "docker.io/saarora/gpu-kernel-mcts:triton-gluon-3.9-cluster-v1"
REMOTE_SOURCE = "/tmp/gluon-cluster-local-b-kernel.py"
REMOTE_HARNESS = "/tmp/gluon-cluster-local-b-paired.py"
REMOTE_OUTPUT = "/tmp/gluon-cluster-local-b-output"


def helpers():
    path = ROOT / "docs/experiments/triton-h100-cublas-sass/run.py"
    spec = importlib.util.spec_from_file_location("gluon_worker_helpers", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import worker helpers")
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
              "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "harness_sha256": hashlib.sha256(HARNESS.read_bytes()).hexdigest(),
              "complete": False}
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
    helper = helpers()
    OUTPUT.mkdir(exist_ok=True)
    save(report)
    try:
        with provider.worker_for_run(hardware) as worker:
            report["environment_manifest"] = worker.get_environment_manifest().as_dict()
            save(report)
            ssh = helper.ssh_prefix(helper.instance_public_ip(worker.instance.instance_id))
            helper.upload(ssh, REMOTE_SOURCE, SOURCE.read_bytes())
            helper.upload(ssh, REMOTE_HARNESS, HARNESS.read_bytes())
            command = [*ssh, "python", REMOTE_HARNESS, "--source", REMOTE_SOURCE,
                       "--output", f"{REMOTE_OUTPUT}/paired.json"]
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            try:
                assert process.stdout is not None
                lines = []
                for line in process.stdout:
                    print(line.rstrip(), flush=True)
                    lines.append(line)
                    lines = lines[-30:]
                if process.wait(timeout=1800) != 0:
                    raise RuntimeError("paired Gluon run failed: " + "".join(lines)[-1800:])
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
            archive = helper.checked([*ssh, "tar", "-cz", "-C", REMOTE_OUTPUT, "."],
                                     timeout=240)
            archive_path = OUTPUT / "remote-artifacts.tar.gz"
            archive_path.write_bytes(archive)
            with tarfile.open(archive_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if not (OUTPUT / member.name).resolve().is_relative_to(OUTPUT.resolve()):
                        raise ValueError("unsafe artifact path")
                tar.extractall(OUTPUT, filter="data")
            paired = json.loads((OUTPUT / "paired.json").read_text())
            report["paired_complete"] = paired.get("complete", False)
            report["remote_artifacts_sha256"] = hashlib.sha256(archive).hexdigest()
            report["complete"] = True
            save(report)
    finally:
        save(report)


if __name__ == "__main__":
    main()
