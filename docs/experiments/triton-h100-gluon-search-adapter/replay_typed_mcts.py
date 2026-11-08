"""Replay measured tile variants through a two-mutation MCTS smoke.

This validates search routing and budget accounting; it does not remeasure a GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD
from kernel_mcts.budget import GenerationBudget, MutationBudget
from kernel_mcts.config import load_data, parse_strategies
from kernel_mcts.domain import BenchmarkResult, EvaluationResult, KernelProgram, ProposalStatus
from kernel_mcts.priors import UniformStrategyPrior
from kernel_mcts.search import MCTS, MCTSConfig

from typed_shapes import (SingleCtaShape, SingleCtaShapeMutationGenerator, TEMPLATE,
                          render)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MEASUREMENT = HERE / "typed-sweep-results.json"
OUTPUT = HERE / "typed-mcts-replay.json"


class RecordedEvaluator:
    def __init__(self, medians: dict[str, float], root_median: float) -> None:
        self.medians = medians
        self.root_median = root_median

    def evaluate(self, program: KernelProgram, workload) -> EvaluationResult:
        if workload != BF16_GEMM_WORKLOAD:
            raise ValueError("unexpected replay workload")
        digest = hashlib.sha256(program.source.encode()).hexdigest()
        median = self.medians[digest]
        return EvaluationResult(
            ProposalStatus.VALID, program=program, state_key=digest,
            reward=math.log(self.root_median / median),
            benchmark=BenchmarkResult((median,), median),
            metadata={"measurement_source": "typed-sweep-results.json",
                      "replay_only": True},
        )


class Events:
    def __init__(self) -> None:
        self.generations = []

    def emit(self, kind: str, payload) -> None:
        if kind == "generation":
            self.generations.append(payload)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    recorded = json.loads(MEASUREMENT.read_text())
    if not recorded.get("complete") or not recorded["measurement"]["complete"]:
        raise ValueError("typed H100 sweep is incomplete")
    measured = recorded["measurement"]
    source_by_name = {
        "single_cta": TEMPLATE.read_text(),
        "n128": render(SingleCtaShape(cta_n=128)),
        "k32": render(SingleCtaShape(block_k=32)),
    }
    medians = {}
    for name, source in source_by_name.items():
        row = measured["candidates"][name]
        digest = hashlib.sha256(source.encode()).hexdigest()
        if (row["source_sha256"] != digest or not row["compile_success"]
                or not row["correctness_success"]):
            raise ValueError(f"{name} lacks matching valid H100 evidence")
        if name == "single_cta":
            median = measured["summary"]["n128"]["single_cta_median_of_round_medians_us"]
        else:
            median = measured["summary"][name]["contender_median_of_round_medians_us"]
        medians[digest] = median
    root_program = KernelProgram(source_by_name["single_cta"], backend="triton_h100")
    root_median = medians[hashlib.sha256(root_program.source.encode()).hexdigest()]
    evaluator = RecordedEvaluator(medians, root_median)
    strategies = parse_strategies(load_data(ROOT / "configs/triton_h100_strategies_v3_staged.json"))
    action = next(item for item in strategies if item.id == "single_cta_tile_shape")
    events = Events()
    search = MCTS(
        strategies=(action,), workload=BF16_GEMM_WORKLOAD,
        generator=SingleCtaShapeMutationGenerator(), evaluator=evaluator,
        prior_provider=UniformStrategyPrior(), budget=GenerationBudget(0),
        mutation_budget=MutationBudget(2), hardware={"gpu_model": "H100"},
        config=MCTSConfig(max_depth=1, k_max=2, c_pw=2.0, max_repairs=0),
        events=events,
    )
    outcome = search.run(evaluator.evaluate(root_program, BF16_GEMM_WORKLOAD))
    report = {
        "schema_version": 1, "mode": "measured_result_replay_without_gpu",
        "measurement_sha256": hashlib.sha256(MEASUREMENT.read_bytes()).hexdigest(),
        "b_gen": outcome.generations, "b_mut": outcome.mutations,
        "iterations": outcome.iterations, "node_count": len(outcome.nodes),
        "best_source_sha256": hashlib.sha256(outcome.best.program.source.encode()).hexdigest(),
        "best_median_us": outcome.best.evaluation.benchmark.median_us,
        "nodes": [{"source_sha256": hashlib.sha256(node.program.source.encode()).hexdigest(),
                   "median_us": node.evaluation.benchmark.median_us,
                   "reward": node.reward} for node in outcome.nodes],
        "generation_events": [{"b_gen": item.get("b_gen"), "b_mut": item.get("b_mut"),
                               "proposal_mechanism": item.get("proposal_mechanism")}
                              for item in events.generations],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("mode", "b_gen", "b_mut", "node_count", "best_median_us")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
