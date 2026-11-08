"""The opt-in traversal action changes only the program-ID mapping and grid."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD
from kernel_mcts.budget import GenerationBudget
from kernel_mcts.config import load_data, parse_strategies
from kernel_mcts.domain import BenchmarkResult, EvaluationResult, InvalidReason, KernelProgram, ProposalStatus
from kernel_mcts.generation import GenerationRequest, GenerationResult
from kernel_mcts.llm_generation import build_generation_prompt
from kernel_mcts.priors import UniformStrategyPrior
from kernel_mcts.proposals import run_proposal
from kernel_mcts.search import MCTS, MCTSConfig
from kernel_mcts.triton_action_fidelity import (
    coalesced_layout_actionable, coalesced_layout_edit_mismatch,
    pipelined_wgmma_api_mismatch, serial_wgmma_api_mismatch,
    traversal_edit_mismatch, traversal_group_m,
)


ROOT = Path("docs/experiments/triton-h100-gluon-search-adapter/single_cta_gluon_fixture.py")
CATALOG = Path("configs/triton_h100_gluon_local_v1.json")
BRIDGE_CATALOG = Path("configs/triton_h100_gluon_bridge_local_v5.json")
MMA_PARENT = Path("docs/experiments/triton-h100-gluon-search-adapter/"
                  "v4-mma-v2-thread-layout-repair/candidate.py")
MMA_COALESCED = Path("docs/experiments/triton-h100-gluon-search-adapter/"
                     "v4_mma_v2_coalesced_layout_fixture.py")
WGMMA_VALID = Path("docs/experiments/triton-h100-gluon-search-adapter/"
                   "v11-wgmma-serial-static-repair/candidate.py")
WGMMA_CATALOG = Path("configs/triton_h100_gluon_bridge_local_v12.json")
PIPELINE_CATALOG = Path("configs/triton_h100_gluon_bridge_local_v17.json")
SERIAL_FULL_CHAIN = Path("docs/experiments/triton-h100-gluon-search-adapter/"
                         "v16-packaged-root-full-chain-bgen10-serial-wgmma.py")
OLD_PIDS = "    pid_m = gl.program_id(0)\n    pid_n = gl.program_id(1)"
NEW_PIDS = """    GROUP_M: gl.constexpr = 8
    pid = gl.program_id(0)
    num_pid_m = gl.cdiv(a_desc.shape[0], block_m)
    num_pid_n = gl.cdiv(b_desc.shape[0], block_n)
    group_id = pid // (GROUP_M * num_pid_n)
    first_pid_m = group_id * GROUP_M
    group_size_m = gl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % (GROUP_M * num_pid_n)) % group_size_m
    pid_n = (pid % (GROUP_M * num_pid_n)) // group_size_m"""
OLD_GRID = "grid = (triton.cdiv(M, block_m), triton.cdiv(N, CTA_N))"
NEW_GRID = "grid = (triton.cdiv(M, block_m) * triton.cdiv(N, CTA_N),)"


def candidate() -> str:
    source = ROOT.read_text()
    assert source.count(OLD_PIDS) == source.count(OLD_GRID) == 1
    return source.replace(OLD_PIDS, NEW_PIDS).replace(OLD_GRID, NEW_GRID)


def test_traversal_guard_accepts_local_mapping_and_rejects_other_edits() -> None:
    parent = ROOT.read_text()
    valid = candidate()
    assert traversal_edit_mismatch(parent, valid) is None
    assert traversal_edit_mismatch(parent, parent) == "program-ID mapping must be bounded assignments"
    assert "outside program-ID mapping" in traversal_edit_mismatch(
        parent, valid.replace("BLOCK_K = 64", "BLOCK_K = 32"))
    assert "one-dimensional" in traversal_edit_mismatch(
        parent, valid.replace(NEW_GRID, OLD_GRID))
    assert "integer // division" in traversal_edit_mismatch(
        parent, valid.replace("gl.cdiv(a_desc.shape[0], block_m)",
                              "a_desc.shape[0] / block_m"))
    assert traversal_group_m(valid) == 8
    assert traversal_group_m(parent) is None
    assert traversal_edit_mismatch(parent, valid.replace(
        '"""One-CTA diagnostic variant of the validated Gluon wait-one kernel."""\n', "")) == (
        "preserve the top-level module docstring exactly"
    )


def test_coalesced_layout_guard_bounds_source_edit() -> None:
    parent, candidate = MMA_PARENT.read_text(), MMA_COALESCED.read_text()
    assert coalesced_layout_edit_mismatch(parent, candidate) is None
    assert "did not change" in coalesced_layout_edit_mismatch(candidate, candidate)
    assert "outside A/B/C" in coalesced_layout_edit_mismatch(
        parent, candidate.replace("BM, BN, BK = 64, 128, 64",
                                  "BM, BN, BK = 32, 64, 64"))
    assert "a_layout order" in coalesced_layout_edit_mismatch(
        parent, candidate.replace("[2, 2], [1, 0]", "[2, 2], [0, 1]", 1))


def test_coalesced_layout_guard_accepts_keyword_order_sites() -> None:
    parent = MMA_PARENT.read_text()
    candidate = MMA_COALESCED.read_text()
    for source, replacement in (
        ("[2, 2], [0, 1])", "[2, 2], order=[0, 1])"),
        ("[2, 2], [1, 0])", "[2, 2], order=[1, 0])"),
        ("[4, 1], [0, 1])", "[4, 1], order=[0, 1])"),
        ("[4, 1], [1, 0])", "[4, 1], order=[1, 0])"),
    ):
        parent = parent.replace(source, replacement)
        candidate = candidate.replace(source, replacement)
    assert coalesced_layout_actionable(parent)
    assert coalesced_layout_edit_mismatch(parent, candidate) is None
    assert not coalesced_layout_actionable(candidate)


def test_serial_wgmma_api_guard_accepts_validated_source_and_aggregates_errors() -> None:
    source = WGMMA_VALID.read_text()
    assert serial_wgmma_api_mismatch("", source) is None
    broken = source.replace("gl.allocate_shared_memory(", "gl.allocate_shared(", 1)
    broken = broken.replace("torch.bfloat16", "triton.bfloat16", 1)
    detail = serial_wgmma_api_mismatch("", broken)
    assert detail is not None
    assert "allocate_shared_memory" in detail
    assert "torch.bfloat16" in detail
    dropped_acc = source.replace("acc = warpgroup_mma(a_smem", "warpgroup_mma(a_smem")
    assert "assign acc = warpgroup_mma" in serial_wgmma_api_mismatch("", dropped_acc)
    shared_conversion = source.replace("c_smem.store(acc.to(gl.bfloat16))",
                                       "c_smem.store(gl.convert_layout(acc, layout=c_desc.layout).to(gl.bfloat16))")
    assert "do not convert_layout to descriptor.layout" in serial_wgmma_api_mismatch(
        "", shared_conversion)
    invalid_barrier = source.replace("mbarrier.invalidate(bar)", "mbarrier.inval(bar)")
    assert "mbarrier.invalidate(bar)" in serial_wgmma_api_mismatch("", invalid_barrier)
    invalid_bytes = source.replace("a_desc.nbytes_per_cta + b_desc.nbytes_per_cta",
                                   "a_smem.nbytes + b_smem.nbytes")
    assert "mbarrier.expect(bar, a_desc.nbytes_per_cta" in serial_wgmma_api_mismatch(
        "", invalid_bytes)


def test_serial_wgmma_guard_matches_descriptor_aliases_used_by_tma_loads() -> None:
    source = WGMMA_VALID.read_text()
    kernel, launch = source.split("\ndef launch(", 1)
    for old, new in (("a_desc", "left_desc"), ("b_desc", "right_desc"),
                     ("c_desc", "out_desc")):
        kernel = kernel.replace(old, new)
    renamed = kernel + "\ndef launch(" + launch
    assert serial_wgmma_api_mismatch("", renamed) is None

    wrong_bytes = renamed.replace(
        "left_desc.nbytes_per_cta + right_desc.nbytes_per_cta",
        "left_desc.nbytes_per_cta + out_desc.nbytes_per_cta",
    )
    assert "matching descriptor aliases" in serial_wgmma_api_mismatch("", wrong_bytes)

    shared_bytes = renamed.replace(
        "left_desc.nbytes_per_cta + right_desc.nbytes_per_cta",
        "a_smem.nbytes + b_smem.nbytes",
    )
    assert "matching descriptor aliases" in serial_wgmma_api_mismatch("", shared_bytes)


def test_wait_one_pipeline_guard_distinguishes_serial_from_validated_reference() -> None:
    serial = SERIAL_FULL_CHAIN.read_text()
    reference = ROOT.read_text().replace("multicast=True", "multicast=False")
    assert pipelined_wgmma_api_mismatch(serial, reference) is None
    detail = pipelined_wgmma_api_mismatch(serial, serial)
    assert detail is not None
    assert "three-slot" in detail and "num_outstanding=1" in detail
    broken = reference.replace("future = step + 2", "future = step + 1")
    assert "future=step+2" in pipelined_wgmma_api_mismatch(serial, broken)
    broken_bytes = reference.replace(
        "a_desc.nbytes_per_cta + b_desc.nbytes_per_cta",
        "a_tile.nbytes + b_tile.nbytes",
    )
    assert "descriptor nbytes_per_cta" in pipelined_wgmma_api_mismatch(
        serial, broken_bytes)
    broken_index = reference.replace("a_smem.index(pre)", "a_smem.index([pre])")
    assert "scalar pre/slot" in pipelined_wgmma_api_mismatch(serial, broken_index)
    explicit_init = reference.replace(
        "for initial_slot in gl.static_range(3):\n        mbarrier.init(bars.index(initial_slot), count=1)",
        "mbarrier.init(bars.index(0), count=1)\n"
        "    mbarrier.init(bars.index(1), count=1)\n"
        "    mbarrier.init(bars.index(2), count=1)",
    )
    assert pipelined_wgmma_api_mismatch(serial, explicit_init) is None


def test_wait_one_pipeline_action_is_staged_from_single_cta_gluon() -> None:
    action = next(item for item in parse_strategies(load_data(PIPELINE_CATALOG))
                  if item.id == "single_cta_gluon_tma_wgmma_wait_one")
    assert action.candidate_stage == "single_cta_gluon"
    assert action.applicability["parent_stages"] == ["single_cta_gluon"]


def test_wait_one_pipeline_guard_charges_repair_before_evaluation() -> None:
    parent = KernelProgram(SERIAL_FULL_CHAIN.read_text(), "triton_h100")
    accepted = KernelProgram(
        ROOT.read_text().replace("multicast=True", "multicast=False"),
        "triton_h100",
    )
    strategy = next(item for item in parse_strategies(load_data(PIPELINE_CATALOG))
                    if item.id == "single_cta_gluon_tma_wgmma_wait_one")

    class Generator:
        calls = 0

        def generate(self, request):
            self.calls += 1
            program = parent if self.calls == 1 else accepted
            return GenerationResult(str(self.calls), program.source, program, "prompt")

    class Evaluator:
        calls = 0

        def evaluate(self, program, workload):
            self.calls += 1
            assert program == accepted
            return EvaluationResult(ProposalStatus.VALID, program, "valid-synthetic", 0.0,
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
    assert "three-slot" in outcome.attempts[0].evaluation.metadata["detail"]
    assert outcome.result.status == ProposalStatus.VALID


def test_serial_wgmma_api_guard_charges_repair_before_evaluation() -> None:
    parent = KernelProgram(Path(
        "docs/experiments/triton-h100-gluon-search-adapter/"
        "v8-packaged-root-mcts-bgen4-complete-gluon.py"
    ).read_text(), "triton_h100")
    accepted = KernelProgram(WGMMA_VALID.read_text(), "triton_h100")
    broken_source = accepted.source.replace("gl.allocate_shared_memory(",
                                            "gl.allocate_shared(", 1)
    broken = KernelProgram(broken_source, "triton_h100")
    strategy = next(item for item in parse_strategies(load_data(WGMMA_CATALOG))
                    if item.id == "single_cta_gluon_tma_wgmma_serial_guarded")

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
            return EvaluationResult(ProposalStatus.VALID, program, "valid-synthetic", 0.0,
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
    assert "allocate_shared_memory" in outcome.attempts[0].evaluation.metadata["detail"]
    assert outcome.result.status == ProposalStatus.VALID


def test_coalesced_layout_guard_charges_repair_before_evaluation() -> None:
    parent = KernelProgram(MMA_PARENT.read_text(), "triton_h100")
    accepted = KernelProgram(MMA_COALESCED.read_text(), "triton_h100")
    strategy = next(item for item in parse_strategies(load_data(BRIDGE_CATALOG))
                    if item.id == "gluon_mma_v2_coalesced_layout")

    class Generator:
        def __init__(self):
            self.calls = 0

        def generate(self, request):
            self.calls += 1
            program = parent if self.calls == 1 else accepted
            return GenerationResult(str(self.calls), program.source, program, "prompt")

    class Evaluator:
        def __init__(self):
            self.calls = 0

        def evaluate(self, program, workload):
            self.calls += 1
            assert program == accepted
            return EvaluationResult(ProposalStatus.VALID, program, "coalesced-synthetic", 0.0,
                                    BenchmarkResult((1.0,), 1.0))

    generator, evaluator, budget = Generator(), Evaluator(), GenerationBudget(2)
    result = run_proposal(
        generator=generator, evaluator=evaluator, budget=budget,
        request=GenerationRequest(parent, strategy, BF16_GEMM_WORKLOAD,
                                  {"gpu_model": "H100"}, None),
        max_repairs=1, max_infrastructure_retries=0,
    )
    assert budget.snapshot().used == 2
    assert evaluator.calls == 1
    assert result.attempts[0].evaluation.invalid_reason == InvalidReason.STRATEGY_MISMATCH
    assert result.attempts[0].evaluation.metadata["strategy_edit_guard"] == strategy.id
    assert result.result.status == ProposalStatus.VALID


def test_traversal_guard_rejects_before_evaluator_and_charges_repair() -> None:
    parent = KernelProgram(ROOT.read_text(), backend="triton_h100")
    accepted = KernelProgram(candidate(), backend="triton_h100")
    strategy, = parse_strategies(load_data(CATALOG))

    class Generator:
        def __init__(self):
            self.calls = 0

        def generate(self, request):
            self.calls += 1
            program = parent if self.calls == 1 else accepted
            return GenerationResult(str(self.calls), program.source, program, "prompt")

    class Evaluator:
        def __init__(self):
            self.calls = 0

        def evaluate(self, program, workload):
            self.calls += 1
            assert program == accepted
            return EvaluationResult(ProposalStatus.VALID, program,
                                    "valid-synthetic", 0.0,
                                    BenchmarkResult((1.0,), 1.0))

    generator, evaluator, budget = Generator(), Evaluator(), GenerationBudget(2)
    result = run_proposal(
        generator=generator, evaluator=evaluator, budget=budget,
        request=GenerationRequest(parent, strategy, BF16_GEMM_WORKLOAD,
                                  {"gpu_model": "H100"}, None),
        max_repairs=1, max_infrastructure_retries=0,
    )
    assert budget.snapshot().used == 2
    assert evaluator.calls == 1
    assert result.attempts[0].evaluation.invalid_reason == InvalidReason.STRATEGY_MISMATCH
    assert result.attempts[0].evaluation.metadata["strategy_edit_guard"] == strategy.id
    assert result.result.status == ProposalStatus.VALID


def test_mcts_traversal_prompt_lists_existing_group_size() -> None:
    parent = KernelProgram(ROOT.read_text(), backend="triton_h100")
    strategy, = parse_strategies(load_data(CATALOG))

    class Generator:
        def __init__(self):
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            group = 8 if len(self.requests) == 1 else 4
            source = candidate().replace("GROUP_M: gl.constexpr = 8",
                                         f"GROUP_M: gl.constexpr = {group}")
            return GenerationResult(str(group), source,
                                    KernelProgram(source, "triton_h100"), "prompt")

    class Evaluator:
        def evaluate(self, program, workload):
            digest = hashlib.sha256(program.source.encode()).hexdigest()
            return EvaluationResult(ProposalStatus.VALID, program, digest, 0.0,
                                    BenchmarkResult((250.0,), 250.0))

    generator, evaluator = Generator(), Evaluator()
    outcome = MCTS(
        strategies=(strategy,), workload=BF16_GEMM_WORKLOAD,
        generator=generator, evaluator=evaluator,
        prior_provider=UniformStrategyPrior(), budget=GenerationBudget(2),
        hardware={"gpu_model": "H100"},
        config=MCTSConfig(max_depth=1, k_max=2, c_pw=2.0, max_repairs=0),
    ).run(evaluator.evaluate(parent, BF16_GEMM_WORKLOAD))
    assert outcome.generations == 2
    assert len(generator.requests) == 2
    assert generator.requests[0].existing_realizations == ()
    assert generator.requests[1].existing_realizations[0]["group_m"] == 8
    prompt = json.loads(build_generation_prompt(generator.requests[1]))
    assert prompt["existing_realizations"][0]["group_m"] == 8
