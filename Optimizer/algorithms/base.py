from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from Optimizer.config import OptimizerConfig
    from Optimizer.algorithms.result import OptimizerAlgorithmResult
    from Optimizer.executor.input_workflow import ResolvedOptimizerInputs


class OptimizerAlgorithm(Protocol):
    """Common contract for replaceable Optimizer algorithms.

    Algorithm implementations should not resolve task inputs or write task
    outputs directly. They receive already-resolved Optimizer inputs and return
    the common result object consumed by `output_workflow`.
    """

    algorithm_id: str

    def run(
        self,
        *,
        config: OptimizerConfig,
        resolved: ResolvedOptimizerInputs,
    ) -> OptimizerAlgorithmResult:
        ...
