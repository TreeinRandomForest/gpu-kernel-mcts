from __future__ import annotations

from kernel_mcts.autotuning import (
    PostSearchAutotuner,
    TuningConfig,
    parse_tuning_parameters,
    render_configuration,
)
from kernel_mcts.domain import (
    BenchmarkResult,
    EvaluationResult,
    KernelProgram,
    ProposalStatus,
    ShapeCase,
    WorkloadContract,
)


WORKLOAD = WorkloadContract(
    "toy", "toy", "fp32", (ShapeCase({"n": 1}, 1.0),), 0.0, 0.0
)
SOURCE = """// KERNEL_MCTS_TUNE BLOCK_M=64,128
// KERNEL_MCTS_TUNE STAGES=1,2
#define BLOCK_M 64
#define STAGES 1
kernel
"""


class Events:
    def __init__(self):
        self.events = []

    def emit(self, event_type, payload):
        self.events.append((event_type, payload))


class Evaluator:
    def evaluate(self, program, workload):
        block = 128 if "#define BLOCK_M 128" in program.source else 64
        stages = 2 if "#define STAGES 2" in program.source else 1
        reward = block / 100.0 + stages / 10.0
        return EvaluationResult(
            ProposalStatus.VALID,
            program,
            f"{block}:{stages}",
            reward,
            BenchmarkResult((1.0 / reward,), 1.0 / reward),
        )


def baseline(source=SOURCE):
    return EvaluationResult(
        ProposalStatus.VALID,
        KernelProgram(source),
        "baseline",
        0.0,
        BenchmarkResult((10.0,), 10.0),
    )


def test_parses_and_renders_explicit_integer_tuning_parameters() -> None:
    parameters = parse_tuning_parameters(KernelProgram(SOURCE))
    rendered = render_configuration(
        KernelProgram(SOURCE), {"BLOCK_M": 128, "STAGES": 2}
    )

    assert [(item.name, item.choices) for item in parameters] == [
        ("BLOCK_M", (64, 128)),
        ("STAGES", (1, 2)),
    ]
    assert "#define BLOCK_M 128" in rendered.source
    assert "#define STAGES 2" in rendered.source


def test_parses_and_renders_zero_valued_tuning_choice() -> None:
    source = """// KERNEL_MCTS_TUNE SMEM_PADDING=0,8,16
#define SMEM_PADDING 8
kernel
"""

    parameters = parse_tuning_parameters(KernelProgram(source))
    rendered = render_configuration(KernelProgram(source), {"SMEM_PADDING": 0})

    assert [(item.name, item.choices) for item in parameters] == [
        ("SMEM_PADDING", (0, 8, 16))
    ]
    assert "#define SMEM_PADDING 0" in rendered.source


def test_post_search_tuner_has_separate_bounded_trials_and_best() -> None:
    events = Events()
    result = PostSearchAutotuner(
        Evaluator(), WORKLOAD, TuningConfig(3, "grid", seed=0), events
    ).run(baseline())

    assert result.used == 3
    assert result.best is not None
    assert result.best_parameters == {"BLOCK_M": 128, "STAGES": 1}
    assert [event for event, _ in events.events] == [
        "tuning_started",
        "tuning_trial",
        "tuning_trial",
        "tuning_trial",
        "tuning_completed",
    ]


def test_post_search_tuner_skips_unannotated_kernel() -> None:
    events = Events()
    result = PostSearchAutotuner(
        Evaluator(), WORKLOAD, TuningConfig(3), events
    ).run(baseline("kernel"))

    assert result.used == 0
    assert result.skipped_reason is not None
    assert events.events[0][0] == "tuning_skipped"
