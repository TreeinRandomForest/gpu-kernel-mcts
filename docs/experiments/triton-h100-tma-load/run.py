"""Provision one H100 for the isolated Triton TMA-load comparison."""

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
BASELINE = ROOT / "docs/experiments/triton-h100-bd-crossover/trace_B.py"
TMA = HERE / "tma_kernel.py"
HARNESS = HERE / "paired_harness.py"
RESULTS = HERE / "results.json"
OUTPUT = HERE / "output"
IMAGE = (
    "docker.io/saarora/gpu-kernel-mcts@sha256:"
    "d0cd8e7b784ac8828691aa0add0e41d1802e22ef11745780de018afaa12c0469"
)
REMOTE_BASELINE = "/tmp/tma-test-baseline.py"
REMOTE_TMA = "/tmp/tma-test-kernel.py"
REMOTE_HARNESS = "/tmp/tma-test-harness.py"


def _load_helpers():
    path = ROOT / "docs/experiments/triton-h100-cublas-sass/run.py"
    spec = importlib.util.spec_from_file_location("matched_sass_helpers", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import matched-input runner helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(report: dict) -> None:
    temporary = RESULTS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESULTS)


def target_command(mode: str, name: str | None = None) -> list[str]:
    command = ["python", REMOTE_HARNESS, "--baseline", REMOTE_BASELINE,
               "--tma", REMOTE_TMA, "--mode", mode]
    if name is not None:
        command += ["--target", name]
    return command


def profile_command(name: str, prefix: str) -> list[str]:
    return ["ncu", "--set", "full", "--import-source", "yes", "--nvtx",
            "--nvtx-include", f"TMA_TEST_{name.upper()}/", "--launch-count", "1",
            "--export", prefix, *target_command("profile", name)]


def main() -> None:
    if RESULTS.exists():
        raise FileExistsError(f"refusing to overwrite {RESULTS}")
    baseline = BASELINE.read_bytes()
    if hashlib.sha256(baseline).hexdigest() != (
        "b9b4574de16a817477990842f3e1e4d4dbd95b662e17f375f5c82981a463fba4"
    ):
        raise ValueError("fast B source identity changed")
    files = {REMOTE_BASELINE: baseline, REMOTE_TMA: TMA.read_bytes(),
             REMOTE_HARNESS: HARNESS.read_bytes()}
    source_hashes = {"baseline": hashlib.sha256(baseline).hexdigest(),
                     "tma": hashlib.sha256(files[REMOTE_TMA]).hexdigest()}
    repository = capture_repository_state(ROOT)
    report = {"schema_version": 1, "run_id": str(uuid4()), "image": IMAGE,
              "git_commit": repository.commit, "dirty_tree": repository.dirty,
              "source_sha256": source_hashes,
              "harness_sha256": hashlib.sha256(files[REMOTE_HARNESS]).hexdigest(),
              "benchmark": None, "profiles": {}, "complete": False}
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
    helpers = _load_helpers()
    OUTPUT.mkdir(exist_ok=True)
    save(report)
    try:
        with provider.worker_for_run(hardware) as worker:
            report["environment_manifest"] = worker.get_environment_manifest().as_dict()
            save(report)
            ssh = helpers.ssh_prefix(helpers.instance_public_ip(worker.instance.instance_id))
            for remote, payload in files.items():
                helpers.upload(ssh, remote, payload)
            print("running matched-input benchmark", flush=True)
            benchmark = json.loads(helpers.checked(
                [*ssh, *target_command("benchmark")], timeout=360))
            if (benchmark.get("status") != "ok"
                    or benchmark.get("source_sha256") != source_hashes
                    or len(benchmark.get("observations", [])) != 9):
                raise RuntimeError("benchmark failed validation")
            report["benchmark"] = benchmark
            save(report)
            for item in benchmark["observations"]:
                print(item["round"], item["implementation"],
                      round(item["median_us"], 3), flush=True)
            for name in ("baseline", "tma"):
                prefix = f"/tmp/tma-test-{name}-source"
                print(f"collecting {name} NCU Source/SASS", flush=True)
                helpers.checked([*ssh, *profile_command(name, prefix)], timeout=1800)
                remote_report = prefix + ".ncu-rep"
                report_bytes = helpers.checked([*ssh, "cat", remote_report], timeout=240)
                (OUTPUT / f"{name}.ncu-rep").write_bytes(report_bytes)
                for view in ("sass", "ptx"):
                    result = subprocess.run([
                        *ssh, "ncu", "--import", remote_report, "--page", "source",
                        "--print-source", view, "--csv",
                    ], capture_output=True, timeout=240)
                    if result.returncode:
                        raise RuntimeError(f"NCU {name} {view} export failed: "
                                           + result.stderr.decode(errors="replace")[-800:])
                    (OUTPUT / f"{name}.{view}.csv").write_bytes(result.stdout)
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
