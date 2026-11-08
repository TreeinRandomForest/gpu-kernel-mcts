"""Validate the model Gluon bridge and four-warp local arm on one H100."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.provenance import capture_repository_state, image_digest_from_reference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
IMAGE = "docker.io/saarora/gpu-kernel-mcts:triton-gluon-3.9-cluster-v1"
SOURCES = {
    "bridge": HERE / "v4-mma-v2-gpu-repair/candidate.py",
    "single_cta": HERE / "single_cta_gluon_fixture.py",
    "four_warp": HERE / "single_cta_four_warp_fixture.py",
    "smoke": HERE / "smoke.py",
    "paired": HERE / "paired.py",
}
REMOTE = {name: f"/tmp/gluon-v4-{name}.py" for name in SOURCES}


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def save(report: dict, results: Path) -> None:
    temporary = results.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(results)


def remote_run(ssh: list[str], command: list[str]) -> dict:
    process = subprocess.Popen([*ssh, "timeout", "1200", "python", *command],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line)
            if not line.startswith("RESULT_JSON="):
                print(line.rstrip(), flush=True)
        code = process.wait(timeout=1260)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    records = [json.loads(line.removeprefix("RESULT_JSON=")) for line in lines
               if line.startswith("RESULT_JSON=")]
    return {"exit_code": code, "measurement": records[0] if len(records) == 1 else None,
            "log_tail": "".join(line for line in lines
                                 if not line.startswith("RESULT_JSON="))[-4000:]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="v4-bridge-four-warp")
    parser.add_argument("--warp-variant", type=Path, default=SOURCES["four_warp"])
    parser.add_argument("--debug-bridge", action="store_true")
    args = parser.parse_args()
    if not args.label.replace("-", "").isalnum():
        parser.error("label must be alphanumeric with hyphens")
    results = HERE / f"{args.label}-results.json"
    if results.exists():
        raise FileExistsError(results)
    sources = {**SOURCES, "four_warp": args.warp_variant}
    remote_paths = dict(REMOTE)
    if args.debug_bridge:
        sources["debug"] = HERE / "debug_compile.py"
        remote_paths["debug"] = "/tmp/gluon-v4-debug.py"
    smoke_runner = module(HERE / "run.py", "gluon_adapter_runner_v4")
    helper = module(ROOT / "docs/experiments/triton-h100-cublas-sass/run.py",
                    "gluon_worker_helpers_v4")
    overlay = module(ROOT / "docs/experiments/triton-h100-context-factorial/run.py",
                     "gluon_overlay_v4")
    repository = capture_repository_state(ROOT)
    bundle = smoke_runner.package_bundle()
    report = {
        "schema_version": 1, "complete": False, "image": IMAGE,
        "git_commit": repository.commit, "dirty_tree": repository.dirty,
        "bundle_sha256": hashlib.sha256(bundle).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for name, path in sources.items()},
    }
    save(report, results)
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
                            minimum_compute_capability="9.0", required_profilers=("ncu",))
    try:
        with provider.worker_for_run(hardware) as worker:
            report["environment_manifest"] = worker.get_environment_manifest().as_dict()
            save(report, results)
            overlay.overlay_worker(worker, Path.home() / ".ssh/nebius", bundle)
            ssh = helper.ssh_prefix(helper.instance_public_ip(worker.instance.instance_id))
            for name, path in sources.items():
                helper.upload(ssh, remote_paths[name], path.read_bytes())
            print("validating Gluon bridge", flush=True)
            report["bridge"] = remote_run(
                ssh, [remote_paths["smoke"], "--candidate", remote_paths["bridge"]])
            save(report, results)
            if args.debug_bridge and report["bridge"]["exit_code"] != 0:
                print("collecting uncaught bridge traceback", flush=True)
                report["bridge_debug"] = remote_run(
                    ssh, [remote_paths["debug"], remote_paths["bridge"]])
                save(report, results)
            print("paired warp-count crossover", flush=True)
            report["warp_variant"] = remote_run(
                ssh, [remote_paths["paired"], "--single-cta", remote_paths["single_cta"],
                      "--variant", f"warp_variant={remote_paths['four_warp']}",
                      "--include-cublas"])
            report["complete"] = (
                report["bridge"]["exit_code"] == 0
                and report["warp_variant"]["exit_code"] == 0
                and (report["bridge"]["measurement"] or {}).get("complete") is True
                and (report["warp_variant"]["measurement"] or {}).get("complete") is True)
            save(report, results)
    finally:
        save(report, results)


if __name__ == "__main__":
    main()
