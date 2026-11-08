"""Bounded source rendering preserves the validated one-CTA kernel family."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD
from kernel_mcts.budget import GenerationBudget, MutationBudget
from kernel_mcts.domain import BenchmarkResult, EvaluationResult, KernelProgram, ProposalStatus, Strategy
from kernel_mcts.generation import GenerationRequest, ProposalBudgetKind
from kernel_mcts.priors import UniformStrategyPrior
from kernel_mcts.search import MCTS, MCTSConfig


MODULE = Path("docs/experiments/triton-h100-gluon-search-adapter/typed_shapes.py")


def typed_shapes():
    spec = importlib.util.spec_from_file_location("gluon_typed_shapes", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_render_changes_only_declared_shape_sites() -> None:
    shapes = typed_shapes()
    base = shapes.TEMPLATE.read_text()
    assert shapes.render(shapes.SingleCtaShape()) == base
    smaller = shapes.render(shapes.SingleCtaShape(cta_n=128, block_k=32))
    expected = (base.replace("CTA_N = 256", "CTA_N = 128")
                .replace("BLOCK_K = 64", "BLOCK_K = 32")
                .replace("instr_shape=[16, 256, 16]", "instr_shape=[16, 128, 16]"))
    assert smaller == expected
    assert "cluster_size = 1" in smaller
    assert "num_ctas=cluster_size" in smaller


def test_render_rejects_unsupported_shape_and_template_drift() -> None:
    shapes = typed_shapes()
    with pytest.raises(ValueError, match="unsupported"):
        shapes.SingleCtaShape(cta_n=64)
    with pytest.raises(ValueError, match="template changed"):
        shapes.render(shapes.SingleCtaShape(), template=shapes.TEMPLATE.read_text() + "\n")


def test_typed_shape_mutations_use_b_mut_and_keep_slower_valid_nodes() -> None:
    shapes = typed_shapes()
    root = KernelProgram(shapes.TEMPLATE.read_text(), backend="triton_h100")
    action = Strategy(shapes.SingleCtaShapeMutationGenerator.strategy_id,
                      "bounded tile shape", {"triton_h100": "tile shape"},
                      applicability={"backend": "triton_h100", "operation": "gemm",
                                     "gpu_model": "H100", "parent_stages": ["single_cta_gluon"]},
                      candidate_stage="single_cta_gluon")
    generator = shapes.SingleCtaShapeMutationGenerator()
    request = GenerationRequest(root, action, BF16_GEMM_WORKLOAD,
                                {"gpu_model": "H100"}, None)
    assert generator.proposal_budget_kind(request) == ProposalBudgetKind.MUTATION
    assert generator.can_generate(request)

    class ValidSlowerEvaluator:
        def evaluate(self, program, workload):
            is_root = program.source == root.source
            latency = 250.0 if is_root else 300.0
            return EvaluationResult(ProposalStatus.VALID, program,
                                    str(hash(program.source)),
                                    0.0 if is_root else -0.2,
                                    BenchmarkResult((latency,), latency))

    evaluator = ValidSlowerEvaluator()
    outcome = MCTS(
        strategies=(action,), workload=BF16_GEMM_WORKLOAD,
        generator=generator, evaluator=evaluator,
        prior_provider=UniformStrategyPrior(), budget=GenerationBudget(0),
        mutation_budget=MutationBudget(2), hardware={"gpu_model": "H100"},
        config=MCTSConfig(max_depth=1, k_max=2, c_pw=2.0, max_repairs=0),
    ).run(evaluator.evaluate(root, BF16_GEMM_WORKLOAD))
    assert outcome.generations == 0
    assert outcome.mutations == 2
    assert len(outcome.nodes) == 3
    assert outcome.best.program == root
