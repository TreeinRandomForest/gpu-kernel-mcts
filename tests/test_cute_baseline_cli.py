from kernel_mcts.cute_baseline_cli import build_parser


def test_cli_accepts_standalone_tuning_mode(tmp_path) -> None:
    output = tmp_path / "tuning.json"

    arguments = build_parser().parse_args(
        ["--mode", "tune", "--output", str(output)]
    )

    assert arguments.mode == "tune"
    assert arguments.output == output


def test_cli_accepts_backend_evaluation_mode() -> None:
    arguments = build_parser().parse_args(["--mode", "backend"])

    assert arguments.mode == "backend"
