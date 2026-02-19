# DOE/executor/execution_budget.py

from dataclasses import dataclass

@dataclass
class ExecutionBudget:
    total: int
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(self.total - self.used, 0)

    def consume(self, n: int) -> int:
        """
        요청한 실행 수 n 중 실제 허용된 실행 수를 반환
        """
        allowed = min(n, self.remaining)
        self.used += allowed
        return allowed

    def exhausted(self) -> bool:
        return self.remaining <= 0
