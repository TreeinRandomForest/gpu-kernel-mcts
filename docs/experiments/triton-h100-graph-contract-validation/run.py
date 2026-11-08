"""Overlay the local timing implementation and validate on one Nebius H100."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import shlex
import subprocess
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from uuid import uuid4

from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.orchestration import _validate_timing_calibration
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.provenance import capture_repository_state, image_digest_from_reference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SINGLE = ROOT / "docs/experiments/triton-h100-gluon-search-adapter/single_cta_gluon_fixture.py"
PERSISTENT = ROOT / "docs/experiments/triton-h100-gluon-search-adapter/persistent_factorial_fixture.py"
VALIDATE = HERE / "validate.py"
RESULTS = HERE / "results.json"
VALIDATION = HERE / "validation.json"
IMAGE = "docker.io/saarora/gpu-kernel-mcts@sha256:22eb914d5041a988653d9d19a298d6bdb64f178f175563743d0af4591cd75ee5"
REMOTE = {"single": "/tmp/graph-contract-single.py",
          "persistent": "/tmp/graph-contract-persistent.py",
          "validate": "/tmp/graph-contract-validate.py"}
REMOTE_OUTPUT = "/tmp/graph-contract-validation.json"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bundle() -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        archive.add(ROOT / "src/kernel_mcts", arcname="kernel_mcts",
                    filter=lambda info: None if "__pycache__" in info.name else info)
    return output.getvalue()


def save(report: dict):
    temporary = RESULTS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESULTS)


def diagnose_worker(worker, ssh: list[str]) -> dict[str, object]:
    request = urllib.request.Request(
        worker.endpoint.address + "/health",
        headers={"Authorization": f"Bearer {worker.endpoint.auth_token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            health = response.read().decode(errors="replace")
    except urllib.error.HTTPError as error:
        health = error.read().decode(errors="replace")
    except Exception as error:
        health = repr(error)
    code = "from kernel_mcts.worker_service import build_application; build_application(progress=print)"
    command = [*ssh, "python", "-c", shlex.quote(code)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=240)
        detail = {"exit_code": str(result.returncode),
                  "output_tail": (result.stdout + result.stderr)[-5000:]}
    except Exception as error:
        detail = {"error": repr(error)}
    baselines = {}
    executable = "/tmp/kernel-mcts-artifacts/bf16-gemm-vendor-baselines"
    for implementation in ("cublas", "cutlass"):
        try:
            result = subprocess.run(
                [*ssh, executable, f"--implementation={implementation}",
                 "--M=4096", "--N=4096", "--K=4096", "--measurements=1",
                 "--timing-contract=triton_h100_cuda_graph_v1"],
                capture_output=True, text=True, timeout=120,
            )
            baselines[implementation] = {"exit_code": result.returncode,
                                         "output_tail": (result.stdout + result.stderr)[-1500:]}
        except Exception as error:
            baselines[implementation] = {"error": repr(error)}
    return {"health": health, **detail, "baselines": baselines}


def main():
    if RESULTS.exists() or VALIDATION.exists():
        raise FileExistsError("refusing to overwrite validation results")
    repository = capture_repository_state(ROOT)
    paths = {"single": SINGLE, "persistent": PERSISTENT, "validate": VALIDATE}
    sources = {name: hashlib.sha256(path.read_bytes()).hexdigest()
               for name, path in paths.items()}
    archive = bundle()
    report = {"schema_version": 1, "run_id": str(uuid4()), "complete": False,
              "image": IMAGE, "git_commit": repository.commit,
              "dirty_tree": repository.dirty, "source_sha256": sources,
              "bundle_sha256": hashlib.sha256(archive).hexdigest()}
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
    helpers = load(ROOT / "docs/experiments/triton-h100-cublas-sass/run.py",
                   "graph_contract_helpers")
    overlay = load(ROOT / "docs/experiments/triton-h100-context-factorial/run.py",
                   "graph_contract_overlay")
    save(report)
    try:
        with provider.worker_for_run(hardware) as worker:
            report["environment_manifest"] = worker.get_environment_manifest().as_dict()
            save(report)
            print("overlaying local timing package", flush=True)
            try:
                overlay.overlay_worker(worker, config.ssh_private_key, archive)
            except Exception:
                ssh = helpers.ssh_prefix(helpers.instance_public_ip(worker.instance.instance_id))
                report["bootstrap_diagnostic"] = diagnose_worker(worker, ssh)
                save(report)
                print(json.dumps(report["bootstrap_diagnostic"], indent=2), flush=True)
                raise
            calibration = worker.get_calibration()
            report["calibration"] = calibration
            save(report)
            _validate_timing_calibration(calibration, "triton_h100")
            ssh = helpers.ssh_prefix(helpers.instance_public_ip(worker.instance.instance_id))
            for name, path in paths.items():
                helpers.upload(ssh, REMOTE[name], path.read_bytes())
            cmd = [*ssh, "timeout", "1200", "python", REMOTE["validate"],
                   "--single", REMOTE["single"],
                   "--persistent", REMOTE["persistent"], "--output", REMOTE_OUTPUT]
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
            report["last_output"] = "".join(tail)[-2500:]
            save(report)
            VALIDATION.write_bytes(helpers.checked([*ssh, "cat", REMOTE_OUTPUT], timeout=120))
            validation = json.loads(VALIDATION.read_text())
            report["complete"] = code == 0 and validation.get("complete") is True
            save(report)
    finally:
        save(report)


if __name__ == "__main__":
    main()
