"""Stage-aware pure-Triton actions in the opt-in combined catalog."""

from pathlib import Path

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD
from kernel_mcts.budget import GenerationBudget
from kernel_mcts.config import load_data, parse_strategies
from kernel_mcts.domain import (
    BenchmarkResult, EvaluationResult, InvalidReason, KernelProgram, ProposalStatus,
)
from kernel_mcts.generation import GenerationRequest, GenerationResult
from kernel_mcts.proposals import run_proposal
from kernel_mcts.strategy_applicability import applicable_strategies, triton_parent_stage
from kernel_mcts.triton_action_fidelity import (
    triton_constexpr_actionable, triton_constexpr_mismatch,
    triton_pointer_loop_mismatch, triton_pointer_loop_ready,
    triton_wide_tile_actionable, triton_wide_tile_mismatch,
)


ROOT = Path("src/kernel_mcts/benchmarks/kernels/bf16_gemm_triton.py")
NARROW = Path("docs/experiments/triton-h100-winner-crossover/ungrouped_bn128_w8.py")
WIDE = Path("docs/experiments/triton-h100-winner-crossover/ungrouped_bn256_w8.py")
CATALOG = Path("configs/triton_h100_combined_v19.json")
CONSTEXPR_CATALOG = Path("configs/triton_h100_combined_v20.json")
PURE_CATALOG = Path("configs/triton_h100_pure_local_v20.json")
RUNTIME_WINNER = Path("docs/experiments/triton-h100-combined-strategy/"
                      "v19-packaged-root-bgen14-seed0-best.py")
CONSTEXPR_WINNER = Path("docs/experiments/triton-h100-combined-strategy/"
                        "v19-packaged-root-bgen14-seed0-best-constexpr.py")


def test_pure_triton_actions_have_staged_applicability() -> None:
    catalog = parse_strategies(load_data(CATALOG))
    root = KernelProgram(ROOT.read_text(), "triton_h100")
    narrow = KernelProgram(NARROW.read_text(), "triton_h100")
    wide = KernelProgram(WIDE.read_text(), "triton_h100")
    hardware = {"gpu_model": "H100"}
    eligible = lambda program: {item.id for item in applicable_strategies(
        catalog, program, BF16_GEMM_WORKLOAD, hardware)}
    assert triton_parent_stage(root.source) == "pointer_triton"
    assert "triton_pointer_advance_tile_128" in eligible(root)
    assert "triton_widen_bn_256_eight_warps" not in eligible(root)
    assert "descriptor_tma_staging" in eligible(root)
    assert "triton_pointer_advance_tile_128" not in eligible(narrow)
    assert "triton_widen_bn_256_eight_warps" in eligible(narrow)
    assert "triton_widen_bn_256_eight_warps" not in eligible(wide)


def test_pure_control_catalog_is_exact_subset_of_combined_catalog() -> None:
    combined = parse_strategies(load_data(CONSTEXPR_CATALOG))
    pure = parse_strategies(load_data(PURE_CATALOG))
    assert [item.id for item in pure] == [
        "triton_pointer_advance_tile_128",
        "triton_widen_bn_256_eight_warps",
        "triton_specialize_gemm_dims",
    ]
    assert pure == combined[:3]
    assert all(item.candidate_stage == "pointer_triton" for item in pure)


def test_pointer_loop_guard_accepts_measured_family_and_rejects_missing_advance() -> None:
    root, narrow = ROOT.read_text(), NARROW.read_text()
    assert not triton_pointer_loop_ready(root)
    assert triton_pointer_loop_ready(narrow)
    assert triton_pointer_loop_mismatch(root, narrow) is None
    broken = narrow.replace("b_ptrs += BK", "pass")
    assert "advance both a_ptrs and b_ptrs" in triton_pointer_loop_mismatch(root, broken)
    four_warps = narrow.replace("num_warps=8", "num_warps=4")
    assert "num_warps=8" in triton_pointer_loop_mismatch(root, four_warps)


def test_wide_tile_guard_allows_only_bn_literal_change() -> None:
    narrow, wide = NARROW.read_text(), WIDE.read_text()
    assert triton_wide_tile_actionable(narrow)
    assert not triton_wide_tile_actionable(wide)
    assert triton_wide_tile_mismatch(narrow, wide) is None
    changed_warps = wide.replace("num_warps=8", "num_warps=4")
    assert "change only BN" in triton_wide_tile_mismatch(narrow, changed_warps)
    changed_pointer = wide.replace("b_ptrs += BK", "b_ptrs += 2 * BK")
    assert "change only BN" in triton_wide_tile_mismatch(narrow, changed_pointer)


def test_wide_tile_guard_charges_repair_before_gpu_evaluation() -> None:
    parent = KernelProgram(NARROW.read_text(), "triton_h100")
    accepted = KernelProgram(WIDE.read_text(), "triton_h100")
    broken = KernelProgram(accepted.source.replace("num_warps=8", "num_warps=4"),
                           "triton_h100")
    strategy = next(item for item in parse_strategies(load_data(CATALOG))
                    if item.id == "triton_widen_bn_256_eight_warps")

    class Generator:
        calls = 0

        def generate(self, request):
            self.calls += 1
            program = broken if self.calls == 1 else accepted
            return GenerationResult(str(self.calls), program.source, program, "prompt")

    class Evaluator:
        calls = 0

        def evaluate(self, program, workload):
            self.calls += 1
            assert program == accepted
            return EvaluationResult(ProposalStatus.VALID, program, "synthetic", 0.0,
                                    BenchmarkResult((1.0,), 1.0))

    generator, evaluator, budget = Generator(), Evaluator(), GenerationBudget(2)
    outcome = run_proposal(
        generator=generator, evaluator=evaluator, budget=budget,
        request=GenerationRequest(parent, strategy, BF16_GEMM_WORKLOAD,
                                  {"gpu_model": "H100"}, None),
        max_repairs=1, max_infrastructure_retries=0,
    )
    assert budget.snapshot().used == 2
    assert evaluator.calls == 1
    assert outcome.attempts[0].evaluation.invalid_reason == InvalidReason.STRATEGY_MISMATCH
    assert outcome.result.status == ProposalStatus.VALID


def test_constexpr_action_is_local_and_only_eligible_for_runtime_wide_parent() -> None:
    runtime, specialized = RUNTIME_WINNER.read_text(), CONSTEXPR_WINNER.read_text()
    assert triton_constexpr_actionable(runtime)
    assert not triton_constexpr_actionable(specialized)
    assert triton_constexpr_mismatch(runtime, specialized) is None
    changed_body = specialized.replace("a_ptrs += BK", "a_ptrs += 2 * BK")
    assert "preserve all other source" in triton_constexpr_mismatch(runtime, changed_body)
    missing = specialized.replace("N: tl.constexpr", "N")
    assert "annotate exactly" in triton_constexpr_mismatch(runtime, missing)
    catalog = parse_strategies(load_data(CONSTEXPR_CATALOG))
    eligible = lambda source: {item.id for item in applicable_strategies(
        catalog, KernelProgram(source, "triton_h100"), BF16_GEMM_WORKLOAD,
        {"gpu_model": "H100"})}
    assert "triton_specialize_gemm_dims" in eligible(runtime)
    assert "triton_specialize_gemm_dims" not in eligible(specialized)


def test_constexpr_guard_charges_failed_repair_before_gpu_evaluation() -> None:
    parent = KernelProgram(RUNTIME_WINNER.read_text(), "triton_h100")
    accepted = KernelProgram(CONSTEXPR_WINNER.read_text(), "triton_h100")
    broken = KernelProgram(accepted.source.replace("a_ptrs += BK", "a_ptrs += 2 * BK"),
                           "triton_h100")
    strategy = next(item for item in parse_strategies(load_data(CONSTEXPR_CATALOG))
                    if item.id == "triton_specialize_gemm_dims")

    class Generator:
        calls = 0

        def generate(self, request):
            self.calls += 1
            program = broken if self.calls == 1 else accepted
            return GenerationResult(str(self.calls), program.source, program, "prompt")

    class Evaluator:
        calls = 0

        def evaluate(self, program, workload):
            self.calls += 1
            assert program == accepted
            return EvaluationResult(ProposalStatus.VALID, program, "synthetic", 0.0,
                                    BenchmarkResult((1.0,), 1.0))

    generator, evaluator, budget = Generator(), Evaluator(), GenerationBudget(2)
    outcome = run_proposal(
        generator=generator, evaluator=evaluator, budget=budget,
        request=GenerationRequest(parent, strategy, BF16_GEMM_WORKLOAD,
                                  {"gpu_model": "H100"}, None),
        max_repairs=1, max_infrastructure_retries=0,
    )
    assert budget.snapshot().used == 2
    assert evaluator.calls == 1
    assert outcome.attempts[0].evaluation.invalid_reason == InvalidReason.STRATEGY_MISMATCH
    assert outcome.result.status == ProposalStatus.VALID
