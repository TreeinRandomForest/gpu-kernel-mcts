"""Run the four isolated TMA address/mask variants on one pinned H100."""

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
OUTPUT = HERE / "output"
RESULTS = HERE / "results.json"
IMAGE = (
    "docker.io/saarora/gpu-kernel-mcts@sha256:"
    "d0cd8e7b784ac8828691aa0add0e41d1802e22ef11745780de018afaa12c0469"
)
REMOTE = "/tmp/tma-address"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(report: dict) -> None:
    temporary = RESULTS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESULTS)


def harness_command(mode: str, target: str | None = None) -> list[str]:
    command = ["python", f"{REMOTE}/harness.py", "--variants-dir", REMOTE,
               "--mode", mode]
    if target:
        command += ["--target", target]
    return command


def profile_command(name: str, prefix: str, names: tuple[str, ...]) -> list[str]:
    if name not in names:
        raise ValueError("unknown variant")
    return ["ncu", "--set", "full", "--import-source", "yes",
            "--cache-control", "all", "--replay-mode", "kernel",
            "--nvtx", "--nvtx-include", f"TMA_ADDRESS_{name.upper()}/",
            "--launch-count", "1", "--export", prefix,
            *harness_command("profile", name)]


def main() -> None:
    if RESULTS.exists():
        raise FileExistsError(f"refusing to overwrite {RESULTS}")
    variants = load_module("tma_address_variants", HERE / "variants.py")
    helpers = load_module("matched_sass_helpers", ROOT / "docs/experiments/triton-h100-cublas-sass/run.py")
    manifest = variants.write_variants()
    baseline_sha = hashlib.sha256(variants.BASELINE.read_bytes()).hexdigest()
    previous = json.loads((variants.BASELINE.parent / "results.json").read_text())
    if baseline_sha != previous["source_sha256"]["tma"]:
        raise ValueError("TMA baseline changed")
    harness = (HERE / "harness.py").read_bytes()
    repository = capture_repository_state(ROOT)
    report = {"schema_version": 1, "run_id": str(uuid4()), "image": IMAGE,
              "git_commit": repository.commit, "dirty_tree": repository.dirty,
              "baseline_sha256": baseline_sha,
              "harness_sha256": hashlib.sha256(harness).hexdigest(),
              "variant_sha256": {name: item["sha256"] for name, item in manifest.items()},
              "benchmark": None, "profiles": {}, "complete": False,
              "cache_control": "all", "replay_mode": "kernel"}
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
    OUTPUT.mkdir(exist_ok=True)
    save(report)
    try:
        with provider.worker_for_run(hardware) as worker:
            report["environment_manifest"] = worker.get_environment_manifest().as_dict()
            save(report)
            ssh = helpers.ssh_prefix(helpers.instance_public_ip(worker.instance.instance_id))
            helpers.checked([*ssh, "mkdir", "-p", REMOTE])
            helpers.upload(ssh, f"{REMOTE}/harness.py", harness)
            for name, item in manifest.items():
                helpers.upload(ssh, f"{REMOTE}/{name}.py", Path(item["path"]).read_bytes())
            print("running paired correctness and CUDA-event timing", flush=True)
            benchmark = json.loads(helpers.checked(
                [*ssh, *harness_command("benchmark")], timeout=360))
            if (benchmark.get("status") != "ok"
                    or benchmark.get("source_sha256") != report["variant_sha256"]
                    or len(benchmark.get("observations", [])) != 15):
                raise RuntimeError("benchmark validation failed")
            report["benchmark"] = benchmark
            save(report)
            for item in benchmark["observations"]:
                print(item["round"], item["implementation"],
                      round(item["median_us"], 3), flush=True)
            for name in variants.NAMES:
                prefix = f"{REMOTE}/{name}"
                print("profiling", name, flush=True)
                helpers.checked([*ssh, *profile_command(name, prefix, variants.NAMES)],
                                timeout=1800)
                remote_report = prefix + ".ncu-rep"
                report_bytes = helpers.checked([*ssh, "cat", remote_report], timeout=240)
                (OUTPUT / f"{name}.ncu-rep").write_bytes(report_bytes)
                for view in ("sass", "ptx"):
                    export = subprocess.run([
                        *ssh, "ncu", "--import", remote_report, "--page", "source",
                        "--print-source", view, "--csv",
                    ], capture_output=True, timeout=240)
                    if export.returncode:
                        raise RuntimeError(f"NCU {name} {view} export failed: "
                                           + export.stderr.decode(errors="replace")[-800:])
                    (OUTPUT / f"{name}.{view}.csv").write_bytes(export.stdout)
                raw = helpers.checked([*ssh, "ncu", "--import", remote_report,
                                       "--page", "raw", "--csv"], timeout=240)
                (OUTPUT / f"{name}.raw.csv").write_bytes(raw)
                report["profiles"][name] = {
                    "ncu_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                    "ncu_report_size_bytes": len(report_bytes),
                }
                save(report)
            report["complete"] = True
            save(report)
    finally:
        save(report)


if __name__ == "__main__":
    main()
