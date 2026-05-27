from __future__ import annotations

from typing import Type

from Optimizer.algorithms.base import OptimizerAlgorithm
from Optimizer.algorithms.focus_bo.algorithm import FocusBOAlgorithm


OPTIMIZER_ALGORITHM_REGISTRY: dict[str, Type[OptimizerAlgorithm]] = {
    "focus_bo": FocusBOAlgorithm,
    "default": FocusBOAlgorithm,
}


def register_optimizer_algorithm(name: str, algorithm_cls: Type[OptimizerAlgorithm]) -> None:
    key = str(name or "").strip().lower()
    if not key:
        raise ValueError("Optimizer algorithm name must be non-empty.")
    if not hasattr(algorithm_cls, "run"):
        raise TypeError("Optimizer algorithm class must define run(...).")
    OPTIMIZER_ALGORITHM_REGISTRY[key] = algorithm_cls


def list_optimizer_algorithms() -> list[str]:
    return sorted(OPTIMIZER_ALGORITHM_REGISTRY.keys())


def get_optimizer_algorithm(name: str | None) -> OptimizerAlgorithm:
    key = str(name or "focus_bo").strip().lower()
    if key not in OPTIMIZER_ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unsupported optimizer algorithm: {name}. "
            f"Available algorithms: {list_optimizer_algorithms()}"
        )
    return OPTIMIZER_ALGORITHM_REGISTRY[key]()
