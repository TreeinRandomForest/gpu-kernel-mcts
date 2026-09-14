from kernel_mcts.config import load_data, parse_strategies


def test_production_strategies_include_tensor_core_structural_actions() -> None:
    strategies = parse_strategies(load_data("configs/strategies.yaml"))
    by_id = {strategy.id: strategy for strategy in strategies}

    assert len(by_id) == len(strategies)
    assert {
        "tensor_core_output_tiling",
        "pipeline_tensor_core_data_movement",
    } <= by_id.keys()
    for strategy_id in (
        "tensor_core_output_tiling",
        "pipeline_tensor_core_data_movement",
    ):
        assert by_id[strategy_id].prompt_for("cuda_cpp").strip()
