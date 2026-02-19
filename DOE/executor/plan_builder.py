# DOE/executor/plan_builder.py

import numpy as np
from typing import Sequence, Tuple

from DOE.doe_algorithm.lhs import latin_hypercube_sampling


class PlanBuilder:
    """
    Build X_plan (candidate set) using LHS only.

    Responsibilities:
    - Generate candidate points (X_plan)
    - NO deduplication
    - NO policy / gate / selection logic
    """

    def __init__(
        self,
        *,
        bounds: Sequence[Tuple[float, float]],
        rng: np.random.Generator,
    ):
        self.bounds = bounds
        self.rng = rng
        self.dim = len(bounds)

    def build(self, *, n_plan: int, n_divisions: int) -> np.ndarray:
        """
        Generate X_plan using LHS.

        Parameters
        ----------
        n_plan : int
            Number of candidate points
        n_divisions : int
            LHS divisions for the current stage
        """
        return latin_hypercube_sampling(
            n_samples=n_plan,
            bounds=self.bounds,
            rng=self.rng,
            n_divisions=n_divisions,
        )
