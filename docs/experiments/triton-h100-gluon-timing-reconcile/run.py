"""Run the Gluon timing-contract crossover on one Nebius H100."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from uuid import uuid4

from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.provenance import capture_repository_state, image_digest_from_reference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SINGLE = ROOT / "docs/experiments/triton-h100-gluon-search-adapter/single_cta_gluon_fixture.py"
PERSISTENT = ROOT / "docs/experiments/triton-h100-gluon-search-adapter/persistent_factorial_fixture.py"
MEASURE = HERE / "measure.py"
RESULTS = HERE / "results.json"
PAIRED = HERE / "paired.json"
IMAGE = "docker.io/saarora/gpu-kernel-mcts@sha256:22eb914d5041a988653d9d19a298d6bdb64f178f175563743d0af4591cd75ee5"
REMOTE = {"single": "/tmp/timing-reconcile-single.py",
          "persistent": "/tmp/timing-reconcile-persistent.py",
          "measure": "/tmp/timing-reconcile-measure.py"}
REMOTE_OUTPUT = "/tmp/timing-reconcile-paired.json"


def helpers():
    path = ROOT / "docs/experiments/triton-h100-cublas-sass/run.py"
    spec = importlib.util.spec_from_file_location("timing_reconcile_helpers", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load worker helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(report: dict):
    temporary = RESULTS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESULTS)


def main():
    if RESULTS.exists() or PAIRED.exists():
        raise FileExistsError("refusing to overwrite timing results")
    paths = {"single": SINGLE, "persistent": PERSISTENT, "measure": MEASURE}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in paths.items()}
    repository = capture_repository_state(ROOT)
    report = {"schema_version": 1, "run_id": str(uuid4()), "complete": False,
              "image": IMAGE, "git_commit": repository.commit,
              "dirty_tree": repository.dirty, "source_sha256": hashes}
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
            f"worker {stage}: {elapsed:.1f}s", flush=True))
    hardware = HardwareSpec("H100", form_factor="SXM",
                            minimum_compute_capability="9.0",
                            minimum_tool_versions={"triton": "3.9.0"})
    helper = helpers()
    save(report)
    try:
        with provider.worker_for_run(hardware) as worker:
            report["environment_manifest"] = worker.get_environment_manifest().as_dict()
            save(report)
            ssh = helper.ssh_prefix(helper.instance_public_ip(worker.instance.instance_id))
            for name, path in paths.items():
                helper.upload(ssh, REMOTE[name], path.read_bytes())
            cmd = [*ssh, "timeout", "1200", "python", REMOTE["measure"],
                   "--single", REMOTE["single"], "--persistent", REMOTE["persistent"],
                   "--output", REMOTE_OUTPUT]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            tail = []
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    print(line.rstrip(), flush=True)
                    tail.append(line)
                    tail = tail[-40:]
                code = process.wait(timeout=1260)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
            report["remote_exit_code"] = code
            report["last_output"] = "".join(tail)[-3000:]
            save(report)
            PAIRED.write_bytes(helper.checked([*ssh, "cat", REMOTE_OUTPUT], timeout=120))
            paired = json.loads(PAIRED.read_text())
            report["paired_complete"] = paired.get("complete", False)
            report["complete"] = code == 0 and report["paired_complete"]
            save(report)
    finally:
        save(report)


if __name__ == "__main__":
    main()
