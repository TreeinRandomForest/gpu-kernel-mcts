"""Run a patched PTX B-TMA control alongside compiler-generated cubins."""

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
SOURCE = ROOT / "docs/experiments/triton-h100-cluster-gluon-v2/kernel.py"
HARNESS = HERE / "paired.py"
ORIGINAL = ROOT / "docs/experiments/triton-h100-cluster-gluon-schedule/output/multicast_original_v87.cubin"
PATCHED = ROOT / "docs/experiments/triton-h100-cluster-gluon-schedule/output/ordinary_patched_from_multicast_v87.cubin"
RESULTS = HERE / "results.json"
OUTPUT = HERE / "output"
IMAGE = "docker.io/saarora/gpu-kernel-mcts:triton-gluon-3.9-cluster-v1"
REMOTE_SOURCE = "/tmp/gluon-cluster-v2-kernel.py"
REMOTE_HARNESS = "/tmp/gluon-cluster-ptx-control-paired.py"
REMOTE_ORIGINAL = "/tmp/gluon-cluster-ptx-control-original.cubin"
REMOTE_PATCHED = "/tmp/gluon-cluster-ptx-control-patched.cubin"
REMOTE_OUTPUT = "/tmp/gluon-cluster-ptx-control-output"


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
    if not ORIGINAL.exists() or not PATCHED.exists():
        raise FileNotFoundError("assemble both pinned PTX controls before provisioning")
    previous = json.loads((SOURCE.parent / "results.json").read_text())
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if not previous["complete"] or previous["source_sha256"] != source_hash:
        raise ValueError("previous Gluon source is incomplete or changed")
    repository = capture_repository_state(ROOT)
    report = {"schema_version": 1, "run_id": str(uuid4()),
              "image": IMAGE, "git_commit": repository.commit,
              "dirty_tree": repository.dirty,
              "source_sha256": source_hash,
              "harness_sha256": hashlib.sha256(HARNESS.read_bytes()).hexdigest(),
              "original_cubin_sha256": hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
              "patched_cubin_sha256": hashlib.sha256(PATCHED.read_bytes()).hexdigest(),
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
            helper.upload(ssh, REMOTE_ORIGINAL, ORIGINAL.read_bytes())
            helper.upload(ssh, REMOTE_PATCHED, PATCHED.read_bytes())
            command = [*ssh, "timeout", "120", "python", REMOTE_HARNESS,
                       "--source", REMOTE_SOURCE,
                       "--original", REMOTE_ORIGINAL, "--patched", REMOTE_PATCHED,
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
                return_code = process.wait(timeout=180)
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
            report["remote_exit_code"] = return_code
            report["last_output"] = "".join(lines)[-1800:]
            report["paired_complete"] = paired.get("complete", False)
            report["remote_artifacts_sha256"] = hashlib.sha256(archive).hexdigest()
            report["complete"] = return_code == 0 and report["paired_complete"]
            save(report)
    finally:
        save(report)


if __name__ == "__main__":
    main()
