"""Remeasure wait-two and wait-one with each arm in each order position."""

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
CANDIDATE = HERE / "kernel.py"
CONTROL = ROOT / "docs/experiments/triton-h100-gluon-wait1/kernel.py"
FAST_B = ROOT / "docs/experiments/triton-h100-bd-crossover/trace_B.py"
HELPER = ROOT / "docs/experiments/triton-h100-gluon-source-schedule/paired.py"
HARNESS = HERE / "paired_balanced.py"
FIRST = HERE / "output/paired.json"
RESULTS = HERE / "balanced_results.json"
OUTPUT = HERE / "balanced_output"
IMAGE = "docker.io/saarora/gpu-kernel-mcts:triton-gluon-3.9-cluster-v1"
REMOTES = {"candidate": "/tmp/gluon-wait2-candidate.py",
           "control": "/tmp/gluon-wait2-control.py",
           "fast_b": "/tmp/gluon-wait2-fast-b.py",
           "helper": "/tmp/gluon-wait2-helper.py",
           "harness": "/tmp/gluon-wait2-balanced.py"}


def helpers():
    path = ROOT / "docs/experiments/triton-h100-cublas-sass/run.py"
    spec = importlib.util.spec_from_file_location("balanced_worker_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(report):
    temporary = RESULTS.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESULTS)


def main():
    if RESULTS.exists():
        raise FileExistsError(RESULTS)
    first = json.loads(FIRST.read_text())
    sources = {"candidate": CANDIDATE, "control": CONTROL,
               "fast_b": FAST_B, "helper": HELPER, "harness": HARNESS}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in sources.items()}
    if (not first["complete"] or first["source_sha256"]["candidate"] != hashes["candidate"] or
            first["source_sha256"]["control"] != hashes["control"]):
        raise ValueError("first run source identity changed or was incomplete")
    repository = capture_repository_state(ROOT)
    report = {"schema_version": 1, "run_id": str(uuid4()), "complete": False,
              "image": IMAGE, "git_commit": repository.commit,
              "dirty_tree": repository.dirty, "source_sha256": hashes,
              "first_run_cubin_sha256": {name: first["compiler"][name]["cubin_sha256"]
                                             for name in ("candidate", "control")}}
    config = NebiusConfig(image=IMAGE, project_id="project-e00aymjmpr00grzcky46ba",
                          subnet_id="vpcsubnet-e00a30bhyztjvq7yaz",
                          ssh_public_key=Path.home() / ".ssh/nebius.pub",
                          ssh_private_key=Path.home() / ".ssh/nebius", username="sanjay",
                          startup_timeout_seconds=1200, backend="triton_h100",
                          project_git_commit=repository.commit,
                          project_dirty_tree=repository.dirty,
                          container_digest=image_digest_from_reference(IMAGE))
    provider = create_nebius_provider(config, readiness_progress=lambda stage, elapsed:
                                      print(f"worker {stage}: {elapsed:.1f}s", flush=True))
    helper = helpers()
    hardware = HardwareSpec("H100", form_factor="SXM", minimum_compute_capability="9.0")
    OUTPUT.mkdir(exist_ok=True)
    save(report)
    try:
        with provider.worker_for_run(hardware) as worker:
            report["environment_manifest"] = worker.get_environment_manifest().as_dict()
            save(report)
            ssh = helper.ssh_prefix(helper.instance_public_ip(worker.instance.instance_id))
            for name, path in sources.items():
                helper.upload(ssh, REMOTES[name], path.read_bytes())
            command = [*ssh, "timeout", "300", "python", REMOTES["harness"],
                       "--candidate", REMOTES["candidate"], "--control", REMOTES["control"],
                       "--fast-b", REMOTES["fast_b"], "--helper", REMOTES["helper"],
                       "--output", "/tmp/gluon-wait2-balanced-output/paired.json"]
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            lines = []
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    print(line.rstrip(), flush=True)
                    lines.append(line)
                    lines = lines[-30:]
                code = process.wait(timeout=360)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
            archive = helper.checked([*ssh, "tar", "-cz", "-C",
                                      "/tmp/gluon-wait2-balanced-output", "."], timeout=240)
            (OUTPUT / "remote-artifacts.tar.gz").write_bytes(archive)
            with tarfile.open(OUTPUT / "remote-artifacts.tar.gz", "r:gz") as tar:
                for member in tar.getmembers():
                    if not (OUTPUT / member.name).resolve().is_relative_to(OUTPUT.resolve()):
                        raise ValueError("unsafe artifact path")
                tar.extractall(OUTPUT, filter="data")
            paired = json.loads((OUTPUT / "paired.json").read_text())
            report["remote_exit_code"] = code
            report["last_output"] = "".join(lines)[-1800:]
            report["paired_complete"] = paired["complete"]
            report["balanced_cubin_sha256"] = paired.get("cubin_sha256")
            report["complete"] = (code == 0 and paired["complete"] and
                                  paired["cubin_sha256"] == report["first_run_cubin_sha256"])
            save(report)
    finally:
        save(report)


if __name__ == "__main__":
    main()
