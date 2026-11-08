"""Provision one H100, overlay local source, and run the backend smoke."""

from __future__ import annotations

import hashlib
import argparse
import importlib.util
import io
import json
import subprocess
import re
import sys
import tarfile
from pathlib import Path

from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.provenance import capture_repository_state, image_digest_from_reference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
IMAGE = "docker.io/saarora/gpu-kernel-mcts:triton-gluon-3.9-cluster-v1"
REMOTE_SCRIPT = "/tmp/gluon-search-adapter-smoke.py"
REMOTE_CANDIDATE = "/tmp/gluon-search-adapter-candidate.py"
REMOTE_VARIANTS = {"n128": "/tmp/gluon-search-adapter-n128.py",
                   "k32": "/tmp/gluon-search-adapter-k32.py",
                   "model": "/tmp/gluon-search-adapter-model.py"}


def helper(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save(report: dict, results: Path) -> None:
    temporary = results.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(results)


def package_bundle() -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        archive.add(ROOT / "src/kernel_mcts", arcname="kernel_mcts",
                    filter=lambda info: None if "__pycache__" in info.name else info)
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--paired-candidate", type=Path)
    parser.add_argument("--paired-base", type=Path)
    parser.add_argument("--include-cublas", action="store_true")
    parser.add_argument("--label", default="")
    parser.add_argument("--paired-crossover", action="store_true")
    parser.add_argument("--typed-sweep", action="store_true")
    args = parser.parse_args()
    if args.include_cublas and args.paired_candidate is None:
        parser.error("--include-cublas requires --paired-candidate")
    if args.paired_base is not None and args.paired_candidate is None:
        parser.error("--paired-base requires --paired-candidate")
    if (args.paired_crossover or args.typed_sweep) and (
        args.candidate is not None or args.paired_candidate is not None or args.label
        or (args.paired_crossover and args.typed_sweep)
    ):
        parser.error("crossover modes cannot be combined with candidate, label or each other")
    if args.candidate is not None and args.paired_candidate is not None:
        parser.error("--candidate and --paired-candidate are mutually exclusive")
    if (args.candidate is not None or args.paired_candidate is not None) and not re.fullmatch(
        r"[a-z0-9-]+", args.label
    ):
        parser.error("candidate runs require a lowercase --label")
    results = HERE / ("typed-sweep-results.json" if args.typed_sweep else
                      "paired-crossover-results.json" if args.paired_crossover
                      else f"{args.label}-results.json" if args.label else "results.json")
    if results.exists():
        raise FileExistsError(f"refusing to overwrite {results}")
    candidate_path = (args.paired_base if args.paired_base is not None else
                      HERE / "single_cta_gluon_fixture.py"
                      if args.paired_crossover or args.typed_sweep or args.paired_candidate
                      else args.candidate)
    candidate = candidate_path.read_bytes() if candidate_path is not None else None
    variants = {}
    if args.typed_sweep:
        typed = helper(HERE / "typed_shapes.py", "gluon_typed_shapes")
        variants = {
            "n128": typed.render(typed.SingleCtaShape(cta_n=128)).encode(),
            "k32": typed.render(typed.SingleCtaShape(block_k=32)).encode(),
        }
    elif args.paired_candidate is not None:
        variants = {"model": args.paired_candidate.read_bytes()}
    repository = capture_repository_state(ROOT)
    bundle = package_bundle()
    script = (HERE / ("paired.py" if args.paired_crossover or args.typed_sweep
                      or args.paired_candidate is not None
                      else "smoke.py")).read_bytes()
    report = {
        "schema_version": 1, "image": IMAGE, "git_commit": repository.commit,
        "dirty_tree": repository.dirty,
        "bundle_sha256": hashlib.sha256(bundle).hexdigest(),
        "harness_sha256": hashlib.sha256(script).hexdigest(), "complete": False,
        "mode": "paired_candidate" if args.paired_candidate is not None else
                "typed_sweep" if args.typed_sweep else
                "paired_crossover" if args.paired_crossover else "smoke",
    }
    if not args.paired_crossover and not args.typed_sweep and args.paired_candidate is None:
        report["smoke_sha256"] = report["harness_sha256"]
    if candidate is not None:
        report["candidate_sha256"] = hashlib.sha256(candidate).hexdigest()
    if args.paired_base is not None:
        report["paired_base_source"] = str(args.paired_base)
    if args.typed_sweep:
        report["renderer_sha256"] = hashlib.sha256((HERE / "typed_shapes.py").read_bytes()).hexdigest()
    if variants:
        report["variant_sha256"] = {name: hashlib.sha256(source).hexdigest()
                                    for name, source in variants.items()}
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
            f"worker {stage}: {elapsed:.1f}s", flush=True),
    )
    hardware = HardwareSpec("H100", form_factor="SXM",
                            minimum_compute_capability="9.0", required_profilers=("ncu",))
    remote = helper(ROOT / "docs/experiments/triton-h100-cublas-sass/run.py",
                    "gluon_smoke_remote")
    overlay = helper(ROOT / "docs/experiments/triton-h100-context-factorial/run.py",
                     "gluon_smoke_overlay")
    try:
        with provider.worker_for_run(hardware) as worker:
            report["environment_manifest"] = worker.get_environment_manifest().as_dict()
            save(report, results)
            print("overlaying local package source", flush=True)
            overlay.overlay_worker(worker, Path.home() / ".ssh/nebius", bundle)
            ssh = remote.ssh_prefix(remote.instance_public_ip(worker.instance.instance_id))
            remote.upload(ssh, REMOTE_SCRIPT, script)
            if candidate is not None:
                remote.upload(ssh, REMOTE_CANDIDATE, candidate)
            for name, source in variants.items():
                remote.upload(ssh, REMOTE_VARIANTS[name], source)
            print("running paired model candidate" if args.paired_candidate is not None else
                  "running typed tile sweep" if args.typed_sweep else
                  "running paired crossover" if args.paired_crossover else
                  "running backend smoke", flush=True)
            command = [*ssh, "python", REMOTE_SCRIPT]
            if args.paired_crossover or args.typed_sweep or args.paired_candidate is not None:
                command += ["--single-cta", REMOTE_CANDIDATE]
                for name in variants:
                    command += ["--variant", f"{name}={REMOTE_VARIANTS[name]}"]
                if args.include_cublas:
                    command.append("--include-cublas")
            elif candidate is not None:
                command += ["--candidate", REMOTE_CANDIDATE]
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            lines: list[str] = []
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    lines.append(line)
                    if not line.startswith("RESULT_JSON="):
                        print(line.rstrip(), flush=True)
                code = process.wait(timeout=1800)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
            report["remote_exit_code"] = code
            report["remote_log_tail"] = "".join(
                line for line in lines if not line.startswith("RESULT_JSON="))[-4000:]
            records = [json.loads(line.removeprefix("RESULT_JSON=")) for line in lines
                       if line.startswith("RESULT_JSON=")]
            if len(records) == 1:
                report["measurement"] = records[0]
                report["complete"] = code == 0 and records[0].get("complete") is True
            save(report, results)
            if not report["complete"]:
                raise RuntimeError(f"backend smoke did not complete; see {results}")
    finally:
        save(report, results)


if __name__ == "__main__":
    main()
