# MODELER/Models/xgboost.py

from typing import Optional, Dict
import numpy as np
from xgboost import XGBRegressor

from .base import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost regression model wrapper (HPO-compatible)

    NOTE:
    - feature_names는 trainer가 책임진다
    - 이 클래스에서는 feature 이름을 생성/추론하지 않는다
    """

    def __init__(
        self,
        *,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        **xgb_params,
    ):
        super().__init__(model_name="XGBoost")

        params: Dict = dict(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=n_jobs,
        )

        params.update(xgb_params)

        self.params = params
        self._estimator: Optional[XGBRegressor] = None

        # trainer에서 반드시 주입됨
        self.feature_names: Optional[list[str]] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._estimator = XGBRegressor(**self.params)
        self._estimator.fit(X, y)

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._estimator.predict(X)

    @property
    def model(self):
        return self._estimator
