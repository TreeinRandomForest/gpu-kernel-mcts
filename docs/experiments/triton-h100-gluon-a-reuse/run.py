"""Run matched two-tile A-reuse controls on one H100 worker."""

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
KERNEL = HERE / "kernel.py"
BASELINE = ROOT / "docs/experiments/triton-h100-gluon-factorial/kernel.py"
HELPER = ROOT / "docs/experiments/triton-h100-gluon-source-schedule/paired.py"
PAIRED = HERE / "paired.py"
PROFILE = HERE / "profile_one.py"
PRIOR = ROOT / "docs/experiments/triton-h100-gluon-factorial/results.json"
RESULTS = HERE / "results.json"
OUTPUT = HERE / "output"
IMAGE = "docker.io/saarora/gpu-kernel-mcts@sha256:22eb914d5041a988653d9d19a298d6bdb64f178f175563743d0af4591cd75ee5"
REMOTE = {"kernel": "/tmp/gluon-a-reuse-kernel.py",
          "baseline": "/tmp/gluon-a-reuse-baseline.py",
          "helper": "/tmp/gluon-a-reuse-helper.py",
          "paired": "/tmp/gluon-a-reuse-paired.py",
          "profile": "/tmp/gluon-a-reuse-profile.py"}
REMOTE_OUTPUT = "/tmp/gluon-a-reuse-output"


def worker_helpers():
    path = ROOT / "docs/experiments/triton-h100-cublas-sass/run.py"
    spec = importlib.util.spec_from_file_location("a_reuse_worker_helpers", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load worker helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def names():
    spec = importlib.util.spec_from_file_location("a_reuse_names", PAIRED)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load paired harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NAMES


def save(report: dict):
    temporary = RESULTS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESULTS)


def main():
    if RESULTS.exists():
        raise FileExistsError(f"refusing to overwrite {RESULTS}")
    prior = json.loads(PRIOR.read_text())
    if not prior["complete"] or prior["sha256"]["kernel"] != hashlib.sha256(BASELINE.read_bytes()).hexdigest():
        raise ValueError("validated baseline source changed")
    repository = capture_repository_state(ROOT)
    paths = {"kernel": KERNEL, "baseline": BASELINE, "helper": HELPER,
             "paired": PAIRED, "profile": PROFILE}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in paths.items()}
    report = {"schema_version": 1, "run_id": str(uuid4()),
              "complete": False, "image": IMAGE, "git_commit": repository.commit,
              "dirty_tree": repository.dirty, "sha256": hashes,
              "profiles": {}, "expected_arms": names()}
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
                            minimum_compute_capability="9.0", required_profilers=("ncu",), minimum_tool_versions={"triton": "3.9.0"})
    helper = worker_helpers()
    OUTPUT.mkdir(exist_ok=True)
    save(report)
    try:
        with provider.worker_for_run(hardware) as worker:
            report["environment_manifest"] = worker.get_environment_manifest().as_dict()
            save(report)
            ssh = helper.ssh_prefix(helper.instance_public_ip(worker.instance.instance_id))
            for name, path in paths.items():
                helper.upload(ssh, REMOTE[name], path.read_bytes())
            common = ["--kernel", REMOTE["kernel"], "--baseline", REMOTE["baseline"],
                      "--helper", REMOTE["helper"]]
            command = [*ssh, "timeout", "1200", "python", REMOTE["paired"],
                       *common, "--output", f"{REMOTE_OUTPUT}/paired.json"]
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            try:
                assert process.stdout is not None
                tail = []
                for line in process.stdout:
                    print(line.rstrip(), flush=True)
                    tail.append(line)
                    tail = tail[-40:]
                return_code = process.wait(timeout=1260)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
            report["remote_exit_code"] = return_code
            report["last_output"] = "".join(tail)[-2500:]
            save(report)
            archive = helper.checked([*ssh, "tar", "-cz", "-C", REMOTE_OUTPUT, "."],
                                     timeout=300)
            (OUTPUT / "remote-artifacts.tar.gz").write_bytes(archive)
            with tarfile.open(OUTPUT / "remote-artifacts.tar.gz", "r:gz") as tar:
                for member in tar.getmembers():
                    if not (OUTPUT / member.name).resolve().is_relative_to(OUTPUT.resolve()):
                        raise ValueError("unsafe artifact path")
                tar.extractall(OUTPUT, filter="data")
            paired = json.loads((OUTPUT / "paired.json").read_text())
            report["paired_complete"] = paired.get("complete", False)
            save(report)
            if return_code == 0 and report["paired_complete"]:
                for name in report["expected_arms"]:
                    try:
                        log = helper.checked([*ssh, "ncu", "--set", "full",
                            "--import-source", "yes", "--cache-control", "all",
                            "--nvtx", "--nvtx-include", f"AREUSE_{name.upper()}/",
                            "--launch-count", "1", "--export", f"/tmp/gluon-a-reuse-{name}",
                            "python", REMOTE["profile"], *common,
                            "--paired", REMOTE["paired"], "--arm", name], timeout=1800)
                        (OUTPUT / f"{name}.ncu.log").write_bytes(log)
                        records = [json.loads(line) for line in log.decode().splitlines()
                                   if line.startswith("{") and '"arm"' in line]
                        if len(records) != 1 or records[0]["arm"] != name:
                            raise ValueError("profile harness summary missing")
                        record = records[0]
                        if (not record["correctness_exact"] or
                                any(record["sha256"][key] != hashes[key]
                                    for key in ("kernel", "baseline", "helper", "paired")) or
                                (name not in ("cublas",) and
                                 record["cubin_sha256"] !=
                                 paired["variants"][name]["compiler"]["cubin_sha256"])):
                            raise ValueError("profile source, correctness, or cubin mismatch")
                        remote = f"/tmp/gluon-a-reuse-{name}.ncu-rep"
                        for suffix, cmd in (("ncu-rep", ["cat", remote]),
                            ("raw.csv", ["ncu", "--import", remote, "--page", "raw", "--csv"]),
                            ("sass.csv", ["ncu", "--import", remote, "--page", "source",
                                          "--print-source", "sass", "--csv"])):
                            (OUTPUT / f"{name}.{suffix}").write_bytes(
                                helper.checked([*ssh, *cmd], timeout=300))
                        record["status"] = "ok"
                    except Exception as error:
                        record = {"arm": name, "status": "failed", "error": str(error)[-1600:]}
                    report["profiles"][name] = record
                    save(report)
                    print("profile", name, record["status"], flush=True)
            report["complete"] = (return_code == 0 and report["paired_complete"] and
                                  all(report["profiles"].get(name, {}).get("status") == "ok"
                                      for name in report["expected_arms"]))
            save(report)
    finally:
        save(report)


if __name__ == "__main__":
    main()
