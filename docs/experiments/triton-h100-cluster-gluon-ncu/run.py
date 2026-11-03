"""Profile three matched double-buffered Gluon arms on one Nebius H100."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from uuid import uuid4

from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.provenance import capture_repository_state, image_digest_from_reference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "docs/experiments/triton-h100-cluster-gluon-v2/kernel.py"
HARNESS = HERE / "profile_one.py"
RESULTS = HERE / "results.json"
OUTPUT = HERE / "output"
IMAGE = "docker.io/saarora/gpu-kernel-mcts:triton-gluon-3.9-cluster-v1"
REMOTE_SOURCE = "/tmp/gluon-cluster-v2-kernel.py"
REMOTE_HARNESS = "/tmp/gluon-cluster-ncu-profile.py"
ARMS = ("single_pipeline", "cluster_pipeline_ordinary", "cluster_pipeline_multicast")


def helpers():
    path = ROOT / "docs/experiments/triton-h100-cublas-sass/run.py"
    spec = importlib.util.spec_from_file_location("gluon_ncu_worker_helpers", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import worker helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(report: dict) -> None:
    temporary = RESULTS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESULTS)


def profile_command(name: str) -> list[str]:
    return ["ncu", "--set", "full", "--import-source", "yes", "--cache-control", "all",
            "--nvtx", "--nvtx-include", f"GLUON_PROFILE_{name.upper()}/",
            "--launch-count", "1", "--export", f"/tmp/gluon-profile-{name}",
            "python", REMOTE_HARNESS, "--source", REMOTE_SOURCE, "--arm", name]


def main() -> None:
    if RESULTS.exists():
        raise FileExistsError(f"refusing to overwrite {RESULTS}")
    previous = json.loads((SOURCE.parent / "results.json").read_text())
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if not previous["complete"] or source_hash != previous["source_sha256"]:
        raise ValueError("prior six-arm source is incomplete or changed")
    repository = capture_repository_state(ROOT)
    report = {"schema_version": 1, "run_id": str(uuid4()), "image": IMAGE,
              "git_commit": repository.commit, "dirty_tree": repository.dirty,
              "source_sha256": source_hash,
              "harness_sha256": hashlib.sha256(HARNESS.read_bytes()).hexdigest(),
              "profiles": {}, "complete": False}
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
            for name in ARMS:
                print("profiling", name, flush=True)
                try:
                    log = helper.checked([*ssh, *profile_command(name)], timeout=1800)
                    (OUTPUT / f"{name}.log").write_bytes(log)
                    records = [json.loads(line) for line in log.decode().splitlines()
                               if line.startswith("{") and '"arm"' in line]
                    if len(records) != 1 or records[0]["arm"] != name:
                        raise ValueError("profile harness summary missing")
                    record = records[0]
                    if record["source_sha256"] != source_hash or not record["correctness"]["exact"]:
                        raise ValueError("profile source or correctness mismatch")
                    remote = f"/tmp/gluon-profile-{name}.ncu-rep"
                    profile = helper.checked([*ssh, "cat", remote], timeout=240)
                    (OUTPUT / f"{name}.ncu-rep").write_bytes(profile)
                    raw = helper.checked([*ssh, "ncu", "--import", remote, "--page", "raw",
                                          "--csv"], timeout=240)
                    (OUTPUT / f"{name}.raw.csv").write_bytes(raw)
                    sass = helper.checked([*ssh, "ncu", "--import", remote,
                                           "--page", "source", "--print-source", "sass",
                                           "--csv"], timeout=240)
                    (OUTPUT / f"{name}.sass.csv").write_bytes(sass)
                    record.update({"status": "ok", "ncu_report_sha256": hashlib.sha256(profile).hexdigest(),
                                   "raw_csv_sha256": hashlib.sha256(raw).hexdigest(),
                                   "sass_csv_sha256": hashlib.sha256(sass).hexdigest()})
                except Exception as error:
                    record = {"arm": name, "status": "failed", "error": str(error)[-1800:]}
                report["profiles"][name] = record
                save(report)
                print(name, record["status"], flush=True)
            report["complete"] = all(report["profiles"][name]["status"] == "ok" for name in ARMS)
            save(report)
    finally:
        save(report)


if __name__ == "__main__":
    main()
