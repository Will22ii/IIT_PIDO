# doe/doe_algorithm/registry.py

from typing import Callable, Dict
import numpy as np

from DOE.doe_algorithm.lhs import latin_hypercube_sampling
from DOE.doe_algorithm.random import random_sampling


# 모든 DOE 알고리즘은 아래 시그니처를 따라야 함
# algo(n_samples: int, bounds: list[tuple], rng: np.random.Generator | None) -> np.ndarray
DOEAlgorithm = Callable[..., np.ndarray]


# -------------------------------------------------
# DOE Algorithm Registry
# -------------------------------------------------
DOE_ALGORITHM_REGISTRY: Dict[str, DOEAlgorithm] = {
    "lhs": latin_hypercube_sampling,
    "random": random_sampling,
}


def get_doe_algorithm(name: str) -> DOEAlgorithm:
    """
    Retrieve DOE algorithm by name.

    NOTE:
    Returned algorithm MUST accept `rng` argument.
    """
    try:
        return DOE_ALGORITHM_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unsupported DOE algorithm: '{name}'. "
            f"Available algorithms: {list(DOE_ALGORITHM_REGISTRY.keys())}"
        )
