from typing import Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base import BaseModel


class RandomForestModel(BaseModel):
    """
    RandomForest regression model wrapper
    """

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        max_features: str | float | None = "sqrt",
        random_state: Optional[int] = None,
        n_jobs: int = -1,
    ):
        super().__init__(model_name="RandomForest")

        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._estimator = RandomForestRegressor(**self.params)
        self._estimator.fit(X, y)
        self.is_fitted = True
