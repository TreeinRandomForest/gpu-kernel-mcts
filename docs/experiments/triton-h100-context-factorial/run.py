"""Run paired Triton H100 NCU/compiled-context arms on one Nebius worker."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shlex
import signal
import sqlite3
import subprocess
import tarfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD, load_triton_bf16_gemm_root
from kernel_mcts.config import load_data, parse_strategies
from kernel_mcts.generation import GenerationResult
from kernel_mcts.llm_generation import LLMKernelGenerator, build_generation_prompt
from kernel_mcts.messages_client import MessagesClient, MessagesConfig
from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.orchestration import run_mcts_search
from kernel_mcts.persistence import SQLiteTraceStore
from kernel_mcts.priors import UniformStrategyPrior
from kernel_mcts.provenance import capture_repository_state, image_digest_from_reference
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.search.mcts import MCTSConfig


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = HERE / "output"
IMAGE = (
    "docker.io/saarora/gpu-kernel-mcts@sha256:"
    "d0cd8e7b784ac8828691aa0add0e41d1802e22ef11745780de018afaa12c0469"
)
ARMS = {"A": (False, False), "B": (True, False),
        "C": (False, True), "D": (True, True)}
MODEL = "rits/zai-org/glm-5-3"


def checked(command: list[str], *, input_bytes: bytes | None = None,
            timeout: float = 180.0) -> bytes:
    result = subprocess.run(command, input=input_bytes, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): "
                           + result.stderr.decode(errors="replace")[-1200:])
    return result.stdout


def package_bundle() -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        archive.add(ROOT / "src/kernel_mcts", arcname="kernel_mcts",
                    filter=lambda info: None if "__pycache__" in info.name else info)
    return stream.getvalue()


def completed_trace(path: Path, budget: int) -> bool:
    """Only a search with a closed run and the full budget may be skipped."""
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            rows = connection.execute(
                "SELECT ended_at, final_b_gen FROM search_runs"
            ).fetchall()
            return len(rows) == 1 and rows[0][0] is not None \
                and rows[0][1] == budget \
                and connection.execute("SELECT COUNT(*) FROM generations").fetchone()[0] == budget
    except sqlite3.Error:
        return False


def archive_interrupted_trace(path: Path) -> Path:
    archive = OUTPUT / "interrupted"
    archive.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = archive / f"{path.stem}-{stamp}{path.suffix}"
    if target.exists():
        raise FileExistsError(target)
    path.rename(target)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{path}{suffix}")
        if sidecar.exists():
            sidecar.rename(Path(f"{target}{suffix}"))
    summary = path.with_suffix(".json")
    if summary.exists():
        summary.rename(target.with_suffix(".json"))
    return target


class ReusedWorkerProvider:
    def __init__(self, worker):
        self.worker = worker

    def acquire_worker(self, hardware):
        return self.worker

    def release_worker(self, worker):
        if worker is not self.worker:
            raise ValueError("unexpected worker")


class GatewayDeadlineExceeded(Exception):
    pass


@contextmanager
def request_deadline(seconds: float):
    """Enforce a wall-clock bound even when a socket read never completes."""
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def on_alarm(signum, frame):
        raise GatewayDeadlineExceeded

    signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


class BoundedGatewayGenerator:
    """Record transient gateway failures as budgeted invalid generations."""

    def __init__(self, client, *, maximum_consecutive_failures: int = 3,
                 request_deadline_seconds: float = 120):
        self.generator = LLMKernelGenerator(client)
        self.consecutive_failures = 0
        self.maximum_consecutive_failures = maximum_consecutive_failures
        self.request_deadline_seconds = request_deadline_seconds

    def generate(self, request):
        started = perf_counter()
        try:
            with request_deadline(self.request_deadline_seconds):
                result = self.generator.generate(request)
        except (RuntimeError, GatewayDeadlineExceeded) as error:
            message = ("Messages API hard deadline exceeded"
                       if isinstance(error, GatewayDeadlineExceeded) else str(error))
            if message not in {"Messages API returned HTTP 502",
                               "Messages API returned HTTP 503",
                               "Messages API returned HTTP 504",
                               "Messages API transport failed",
                               "Messages API hard deadline exceeded"}:
                raise
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.maximum_consecutive_failures:
                raise RuntimeError("Messages gateway had repeated transient failures") from None
            prompt = build_generation_prompt(request)
            return GenerationResult(
                generation_id=f"gateway-error-{uuid4()}", raw_output="", program=None,
                prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
                prompt_text=prompt, latency_seconds=perf_counter() - started,
                metadata={
                    "generator": "llm", "llm_call": True, "provider": "messages",
                    "model": MODEL, "thinking_mode": "disabled",
                    "gateway_error": message, "max_output_tokens": 16_384,
                },
            )
        self.consecutive_failures = 0
        return result


def overlay_worker(worker, ssh_private_key: Path, bundle: bytes) -> None:
    instance = json.loads(checked([
        "nebius", "compute", "instance", "get", "--id", worker.instance.instance_id,
        "--format", "json", "--no-check-update",
    ], timeout=30))
    ip = instance["status"]["network_interfaces"][0]["public_ip_address"]["address"].split("/")[0]
    ssh = ["ssh", "-F", "/dev/null", "-i", str(ssh_private_key),
           "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
           f"sanjay@{ip}"]
    package_path = checked([*ssh, shlex.join([
        "sudo", "docker", "exec", "kernel-mcts-worker", "python", "-c",
        "import kernel_mcts,pathlib; print(pathlib.Path(kernel_mcts.__file__).parent.parent)",
    ])]).decode().strip()
    if not package_path.startswith("/opt/venv/") or not package_path.endswith("site-packages"):
        raise RuntimeError("unexpected worker package location")
    checked([*ssh, "sudo", "docker", "exec", "-i", "kernel-mcts-worker", "tar", "-xz",
             "-C", package_path], input_bytes=bundle)
    checked([*ssh, "sudo", "docker", "restart", "kernel-mcts-worker"], timeout=120)
    for _ in range(30):
        try:
            worker._transport.get_environment_manifest()
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError("overlaid worker did not become ready")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--arms", choices=tuple(ARMS), nargs="+", default=list(ARMS))
    parser.add_argument("--label", default="factorial")
    parser.add_argument("--resume", action="store_true",
                        help="skip complete traces and archive incomplete ones before rerunning")
    args = parser.parse_args()
    if args.budget < 1 or len(set(args.seeds)) != len(args.seeds) \
            or len(set(args.arms)) != len(args.arms):
        parser.error("budget must be positive and seeds/arms unique")
    base = os.environ.get("ENMAAS_BASE_URL", "").rstrip("/")
    if not base.startswith("https://") or not os.environ.get("ENMAAS_KEY"):
        parser.error("ENMAAS_BASE_URL and ENMAAS_KEY must be set in the local shell")
    OUTPUT.mkdir(exist_ok=True)
    combinations = []
    for seed in args.seeds:
        for arm in args.arms:
            path = OUTPUT / f"{args.label}-b{args.budget}-s{seed}-{arm}.sqlite"
            if path.exists():
                if not args.resume:
                    parser.error("refusing to overwrite an existing SQLite trace")
                if completed_trace(path, args.budget):
                    print(f"skipping complete seed={seed} arm={arm}", flush=True)
                    continue
                archived = archive_interrupted_trace(path)
                print(f"archived interrupted trace: {archived}", flush=True)
            combinations.append((seed, arm))
    if not combinations:
        print("all requested arms are complete", flush=True)
        return
    repository = capture_repository_state(ROOT)
    bundle = package_bundle()
    bundle_sha256 = hashlib.sha256(bundle).hexdigest()
    strategies_path = ROOT / "configs/triton_h100_strategies.json"
    strategy_sha256 = hashlib.sha256(strategies_path.read_bytes()).hexdigest()
    strategies = parse_strategies(load_data(strategies_path))
    root = load_triton_bf16_gemm_root()
    root_sha256 = hashlib.sha256(root.source.encode()).hexdigest()
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
    with provider.worker_for_run(hardware) as worker:
        overlay_worker(worker, config.ssh_private_key, bundle)
        reused = ReusedWorkerProvider(worker)
        for seed, arm in combinations:
            show_profile, show_compiler = ARMS[arm]
            trace_path = OUTPUT / f"{args.label}-b{args.budget}-s{seed}-{arm}.sqlite"
            client = MessagesClient(MessagesConfig(
                model=MODEL, url=f"{base}/messages", max_output_tokens=16_384,
                timeout_seconds=300, thinking_mode="disabled",
            ))
            search_config = MCTSConfig(
                max_repairs=1, include_parent_profile=show_profile,
                profile_prompt_view="ncu_only", include_compiler_summary=show_compiler,
                profile_metric_set="spill_v1",
            )
            print(f"starting seed={seed} arm={arm} B_gen={args.budget}", flush=True)
            with SQLiteTraceStore(trace_path) as trace:
                execution = run_mcts_search(
                    provider=reused, hardware=hardware, workload=BF16_GEMM_WORKLOAD,
                    root_program=root, strategies=strategies,
                    generator=BoundedGatewayGenerator(client),
                    prior_provider=UniformStrategyPrior(),
                    generation_budget=args.budget, trace=trace,
                    mcts_config=search_config, seed=seed, model_name=MODEL,
                    run_metadata={
                        "ablation_arm": arm, "worker_image": IMAGE,
                        "worker_code_overlay_sha256": bundle_sha256,
                        "worker_code_overlay": True,
                        "root_source_sha256": root_sha256,
                        "strategy_catalog_sha256": strategy_sha256,
                        "messages_thinking": "disabled",
                        "messages_request_deadline_seconds": 120,
                        "messages_endpoint_sha256": hashlib.sha256(f"{base}/messages".encode()).hexdigest(),
                        "git_commit": repository.commit, "dirty_tree": repository.dirty,
                    },
                )
            result = {"seed": seed, "arm": arm, "budget": args.budget,
                      "run_id": execution.run_id,
                      "generations": execution.result.generations,
                      "valid_nodes": len(execution.result.nodes),
                      "best_median_us": execution.result.best.evaluation.benchmark.median_us,
                      "trace": trace_path.name}
            (OUTPUT / f"{args.label}-b{args.budget}-s{seed}-{arm}.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
            print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
