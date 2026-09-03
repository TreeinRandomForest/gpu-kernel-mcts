from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


class BudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    limit: int
    used: int

    @property
    def remaining(self) -> int:
        return self.limit - self.used


class GenerationBudget:
    """Thread-safe accounting for all candidate and repair generations."""

    def __init__(self, limit: int) -> None:
        if limit < 0:
            raise ValueError("generation budget cannot be negative")
        self._limit = limit
        self._used = 0
        self._lock = Lock()

    def reserve(self) -> int:
        with self._lock:
            if self._used >= self._limit:
                raise BudgetExhausted("candidate-generation budget exhausted")
            self._used += 1
            return self._used

    @property
    def exhausted(self) -> bool:
        return self.snapshot().remaining == 0

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(self._limit, self._used)

