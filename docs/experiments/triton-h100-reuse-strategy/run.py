"""Smoke-check reuse_v1, then compare old/new Triton MCTS strategy catalogs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD, load_triton_bf16_gemm_root
from kernel_mcts.config import load_data, parse_strategies
from kernel_mcts.llm_generation import build_generation_prompt
from kernel_mcts.messages_client import MessagesClient, MessagesConfig
from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.orchestration import run_mcts_search
from kernel_mcts.persistence import SQLiteTraceStore
from kernel_mcts.priors import UniformStrategyPrior
from kernel_mcts.provenance import capture_repository_state, image_digest_from_reference
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.search.mcts import MCTSConfig
from kernel_mcts.smoke import TritonSmokeKernelGenerator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = HERE / "output"
OLD_CATALOG = HERE / "old_catalog.json"
NEW_CATALOG = ROOT / "configs/triton_h100_strategies.json"
IMAGE = (
    "docker.io/saarora/gpu-kernel-mcts@sha256:"
    "d0cd8e7b784ac8828691aa0add0e41d1802e22ef11745780de018afaa12c0469"
)
MODEL = "rits/zai-org/glm-5-3"


def _load_factorial_helpers():
    path = ROOT / "docs/experiments/triton-h100-context-factorial/run.py"
    spec = importlib.util.spec_from_file_location("triton_factorial_helpers", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import existing H100 MCTS runner helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_catalogs(old, new) -> None:
    old_ids = [item.id for item in old]
    new_ids = [item.id for item in new]
    if new_ids != old_ids + ["improve_cross_cta_reuse"]:
        raise ValueError("new catalog must add only the cross-CTA reuse action")
    if any(a != b for a, b in zip(old, new)):
        raise ValueError("existing four strategy definitions changed")


def verify_smoke_prompt(payload: dict) -> dict:
    if payload.get("strategy", {}).get("id") != "improve_cross_cta_reuse":
        raise ValueError("smoke did not select the reuse action")
    profile = payload.get("parent_profile")
    if not isinstance(profile, dict) or profile.get("metric_set") != "reuse_v1":
        raise ValueError("reuse_v1 parent profile missing from prompt")
    if profile.get("cache_control") != "all" or profile.get("replay_mode") != "kernel":
        raise ValueError("profile cache/replay provenance missing")
    summary = profile.get("summary")
    required = {
        "dram_bytes_read", "l2_tex_read_sectors",
        "l2_tex_read_lookup_hit_sectors", "l2_tex_read_lookup_miss_sectors",
        "l2_read_sector_hit_rate_pct", "compiler_spills",
        "local_load_l1_hit_sectors", "local_load_l1_miss_sectors",
    }
    if not isinstance(summary, dict) or not required.issubset(summary):
        raise ValueError("reuse/spill counters missing from prompt summary")
    if "BM and BN" not in payload["strategy"]["backend_prompt"]:
        raise ValueError("tile reasoning instruction missing")
    return {"strategy_id": payload["strategy"]["id"],
            "profile_metric_set": profile["metric_set"],
            "cache_control": profile["cache_control"],
            "replay_mode": profile["replay_mode"],
            "summary_keys": sorted(summary)}


class PromptRecordingSmoke:
    def __init__(self) -> None:
        self.inner = TritonSmokeKernelGenerator()

    def generate(self, request):
        prompt = build_generation_prompt(request)
        result = self.inner.generate(request)
        return replace(result, prompt_text=prompt,
                       prompt_hash=hashlib.sha256(prompt.encode()).hexdigest())


def archive_incomplete(path: Path) -> Path:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.budget < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("budget must be positive and seeds unique")
    base = os.environ.get("ENMAAS_BASE_URL", "").rstrip("/")
    if not base.startswith("https://") or not os.environ.get("ENMAAS_KEY"):
        parser.error("ENMAAS_BASE_URL and ENMAAS_KEY must be set locally")
    old, new = (parse_strategies(load_data(path)) for path in (OLD_CATALOG, NEW_CATALOG))
    verify_catalogs(old, new)
    OUTPUT.mkdir(exist_ok=True)
    helpers = _load_factorial_helpers()
    combinations = []
    for seed in args.seeds:
        for arm in (("old", "new") if seed % 2 == 0 else ("new", "old")):
            path = OUTPUT / f"b{args.budget}-s{seed}-{arm}.sqlite"
            if path.exists():
                if not args.resume:
                    parser.error(f"refusing to overwrite {path}")
                if helpers.completed_trace(path, args.budget):
                    print(f"skipping completed seed={seed} arm={arm}", flush=True)
                    continue
                print(f"archived incomplete: {archive_incomplete(path)}", flush=True)
            combinations.append((seed, arm))
    smoke_path = OUTPUT / "reuse-v1-smoke.sqlite"
    smoke_needed = not (smoke_path.exists() and helpers.completed_trace(smoke_path, 1))
    if smoke_path.exists() and smoke_needed:
        if not args.resume:
            parser.error(f"refusing to overwrite {smoke_path}")
        print(f"archived incomplete: {archive_incomplete(smoke_path)}", flush=True)
    if not combinations and not smoke_needed:
        print("all requested arms are complete", flush=True)
        return
    repository = capture_repository_state(ROOT)
    bundle = helpers.package_bundle()
    bundle_hash = hashlib.sha256(bundle).hexdigest()
    catalog_hashes = {arm: hashlib.sha256(path.read_bytes()).hexdigest()
                      for arm, path in (("old", OLD_CATALOG), ("new", NEW_CATALOG))}
    root = load_triton_bf16_gemm_root()
    root_hash = hashlib.sha256(root.source.encode()).hexdigest()
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
        helpers.overlay_worker(worker, config.ssh_private_key, bundle)
        reused = helpers.ReusedWorkerProvider(worker)
        if smoke_needed:
            print("running reuse_v1 prompt/profile smoke", flush=True)
            with SQLiteTraceStore(smoke_path) as trace:
                smoke = run_mcts_search(
                    provider=reused, hardware=hardware, workload=BF16_GEMM_WORKLOAD,
                    root_program=root, strategies=(new[-1],),
                    generator=PromptRecordingSmoke(),
                    prior_provider=UniformStrategyPrior(), generation_budget=1,
                    trace=trace, mcts_config=MCTSConfig(
                        max_repairs=0, include_parent_profile=True,
                        profile_prompt_view="full", profile_metric_set="reuse_v1",
                    ), seed=0, model_name="deterministic-smoke",
                    run_metadata={"experiment": "reuse_v1_prompt_smoke",
                                  "worker_code_overlay_sha256": bundle_hash,
                                  "strategy_catalog_sha256": catalog_hashes["new"],
                                  "root_source_sha256": root_hash},
                )
            with sqlite3.connect(smoke_path) as connection:
                prompts = [row[0] for row in connection.execute(
                    "SELECT prompt_text FROM generations WHERE prompt_text IS NOT NULL"
                )]
            if len(prompts) != 1:
                raise RuntimeError("smoke did not persist exactly one prompt")
            smoke_audit = verify_smoke_prompt(json.loads(prompts[0]))
            smoke_audit.update({"run_id": smoke.run_id,
                                "prompt_sha256": hashlib.sha256(prompts[0].encode()).hexdigest(),
                                "worker_code_overlay_sha256": bundle_hash})
            (OUTPUT / "reuse-v1-smoke.json").write_text(
                json.dumps(smoke_audit, indent=2, sort_keys=True) + "\n"
            )
            print("reuse_v1 prompt smoke passed", flush=True)
        for seed, arm in combinations:
            trace_path = OUTPUT / f"b{args.budget}-s{seed}-{arm}.sqlite"
            client = MessagesClient(MessagesConfig(
                model=MODEL, url=f"{base}/messages", max_output_tokens=16_384,
                timeout_seconds=300, thinking_mode="disabled",
            ))
            strategies = old if arm == "old" else new
            print(f"starting seed={seed} arm={arm} B_gen={args.budget}", flush=True)
            with SQLiteTraceStore(trace_path) as trace:
                execution = run_mcts_search(
                    provider=reused, hardware=hardware, workload=BF16_GEMM_WORKLOAD,
                    root_program=root, strategies=strategies,
                    generator=helpers.BoundedGatewayGenerator(client),
                    prior_provider=UniformStrategyPrior(),
                    generation_budget=args.budget, trace=trace,
                    mcts_config=MCTSConfig(
                        max_repairs=1, include_parent_profile=True,
                        profile_prompt_view="full", include_compiler_summary=False,
                        profile_metric_set="reuse_v1",
                    ), seed=seed, model_name=MODEL,
                    run_metadata={
                        "experiment": "cross_cta_reuse_strategy_pilot", "arm": arm,
                        "worker_image": IMAGE, "worker_code_overlay_sha256": bundle_hash,
                        "root_source_sha256": root_hash,
                        "strategy_catalog_sha256": catalog_hashes[arm],
                        "strategy_ids": [item.id for item in strategies],
                        "profile_metric_set": "reuse_v1", "messages_thinking": "disabled",
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
                      "strategy_catalog_sha256": catalog_hashes[arm],
                      "trace": trace_path.name}
            (OUTPUT / f"b{args.budget}-s{seed}-{arm}.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
            print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
