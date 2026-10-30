"""One model-free MCTS step on H100 to validate spill_v1 through trace and prompt."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from uuid import uuid4

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD
from kernel_mcts.domain import BenchmarkResult, KernelProgram, Strategy
from kernel_mcts.evaluation import BackendKernelEvaluator, EvaluationContext
from kernel_mcts.generation import GenerationResult
from kernel_mcts.llm_generation import build_generation_prompt
from kernel_mcts.orchestration import run_mcts_search
from kernel_mcts.persistence import SQLiteTraceStore
from kernel_mcts.priors import UniformStrategyPrior
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.search.mcts import MCTSConfig
from kernel_mcts.serialization import deserialize_environment_manifest
from kernel_mcts.triton_backend import TritonBackendConfig, TritonH100Backend


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"


class FixedGenerator:
    def __init__(self, child: KernelProgram) -> None:
        self.child = child

    def generate(self, request):
        prompt = build_generation_prompt(request)
        return GenerationResult(
            generation_id=str(uuid4()), raw_output=self.child.source,
            program=self.child, prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
            prompt_text=prompt, metadata={"generator": "fixed_validation"},
        )


class LocalWorker:
    def __init__(self, evaluator, manifest, calibration) -> None:
        self.evaluator = evaluator
        self.manifest = manifest
        self.calibration = calibration
        self.results = {}

    @property
    def worker_id(self):
        return self.manifest.worker_id

    def get_environment_manifest(self):
        return self.manifest

    def get_calibration(self):
        return self.calibration

    def evaluate(self, evaluation_id, program, workload, profile_level):
        assert profile_level == "tier0"
        if evaluation_id not in self.results:
            self.results[evaluation_id] = self.evaluator.evaluate(program, workload)
        return self.results[evaluation_id]

    def profile(self, evaluation_id, profile_level, metric_set="lightweight_v1"):
        assert profile_level == "lightweight"
        return self.evaluator.lightweight_profile(
            self.results[evaluation_id], BF16_GEMM_WORKLOAD, metric_set
        )

    def fetch_compiler_artifact(self, evaluation_id, sha256):
        path = self.results[evaluation_id].compiled_artifact.ptx_path
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha256:
            raise ValueError("PTX hash mismatch")
        return raw


class LocalProvider:
    def __init__(self, worker):
        self.worker = worker

    def acquire_worker(self, hardware):
        return self.worker

    def release_worker(self, worker):
        assert worker is self.worker


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    trace_path = OUTPUT / "spill-v1-validation.sqlite"
    if trace_path.exists():
        raise FileExistsError(trace_path)
    manifest = deserialize_environment_manifest(json.loads((HERE / "manifest.json").read_text()))
    calibration = json.loads((HERE / "calibration.json").read_text())
    backend = TritonH100Backend(TritonBackendConfig(
        artifact_root=OUTPUT / "artifacts", ncu_version="2025.1.1.0",
        profile_timeout_seconds=900,
    ))
    evaluator = BackendKernelEvaluator(
        backend=backend, root_benchmark=BenchmarkResult((1000.0,), 1000.0),
        context=EvaluationContext(
            manifest.worker_id, manifest.manifest_id,
            {"dynamic": True, "entry_point": "launch"},
            {"gpu_model": manifest.gpu_model, "compute_capability": manifest.compute_capability},
        ),
    )
    worker = LocalWorker(evaluator, manifest, calibration)
    root = KernelProgram((HERE / "grouped_bn256_w4.py").read_text(), "triton_h100")
    child = KernelProgram((HERE / "grouped_bn256_w8.py").read_text(), "triton_h100")
    with SQLiteTraceStore(trace_path) as trace:
        execution = run_mcts_search(
            provider=LocalProvider(worker),
            hardware=HardwareSpec("H100", form_factor="SXM", minimum_compute_capability="9.0",
                                  required_profilers=("ncu",)),
            workload=BF16_GEMM_WORKLOAD,
            root_program=root,
            strategies=(Strategy("change_warps", "Change warp count",
                                 {"triton_h100": "Use eight warps."}),),
            generator=FixedGenerator(child), prior_provider=UniformStrategyPrior(),
            generation_budget=1, trace=trace,
            mcts_config=MCTSConfig(max_repairs=0, profile_metric_set="spill_v1"),
            model_name="fixed_validation",
            run_metadata={"worker_image": "pinned_image_with_local_source_overlay",
                          "validation_kind": "model_free_spill_v1"},
        )
    with sqlite3.connect(trace_path) as connection:
        profiles = [(node_id, json.loads(profile)) for node_id, profile in connection.execute(
            "SELECT node_id, profile_json FROM nodes WHERE profile_json IS NOT NULL"
        )]
        prompts = [(json.loads(text), prompt_hash) for text, prompt_hash in connection.execute(
            "SELECT prompt_text, prompt_hash FROM generations WHERE prompt_text IS NOT NULL"
        )]
    if len(profiles) != 2 or len(prompts) != 1:
        raise RuntimeError("expected two persisted profiles and one prompt")
    parent_profile = prompts[0][0]["parent_profile"]
    if parent_profile["metric_set"] != "spill_v1" or parent_profile["summary"]["compiler_spills"] != 256:
        raise RuntimeError("spill_v1 parent profile was not delivered to prompt")
    summary = {
        "run_id": execution.run_id,
        "generations": execution.result.generations,
        "profile_calls": execution.result.profile_calls,
        "profiles": [{"node_id": node_id, "metric_set": profile["metric_set"],
                      "summary": profile["summary"]} for node_id, profile in profiles],
        "prompt_parent_profile": parent_profile["summary"],
        "prompt_sha256": prompts[0][1],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"run_id": execution.run_id, "profiles": len(profiles),
                      "prompt_metric_set": parent_profile["metric_set"]}))


if __name__ == "__main__":
    main()
