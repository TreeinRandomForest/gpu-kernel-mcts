import pytest

from kernel_mcts.budget import BudgetExhausted, GenerationBudget


def test_budget_accounts_exactly() -> None:
    budget = GenerationBudget(2)
    assert budget.reserve() == 1
    assert budget.reserve() == 2
    assert budget.exhausted
    with pytest.raises(BudgetExhausted):
        budget.reserve()

