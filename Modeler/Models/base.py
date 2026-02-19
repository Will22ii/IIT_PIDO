# Modeler/Models/base.py

from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np


class BaseModel(ABC):
    """
    Minimal base wrapper for ML models.

    Responsibilities:
    - Hold a trained estimator
    - Provide unified fit / predict interface
    - Expose raw estimator for analysis (SHAP, importance, etc.)

    Non-responsibilities:
    - Uncertainty computation
    - Feature importance computation
    - SHAP computation
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._estimator: Optional[Any] = None
        self.is_fitted: bool = False

    # -------------------------------------------------
    # Required interface
    # -------------------------------------------------

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the underlying estimator.
        Must set self._estimator and self.is_fitted.
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained estimator.
        """
        if not self.is_fitted or self._estimator is None:
            raise RuntimeError(
                f"Model '{self.model_name}' is not fitted yet."
            )
        return self._estimator.predict(X)

    # -------------------------------------------------
    # Raw estimator access (for SHAP / importance)
    # -------------------------------------------------

    def get_estimator(self) -> Any:
        """
        Return the underlying trained estimator.

        Executor / Analyzer may use this for:
        - SHAP
        - feature importance
        - advanced inspection
        """
        if not self.is_fitted or self._estimator is None:
            raise RuntimeError(
                f"Estimator for model '{self.model_name}' is not available."
            )
        return self._estimator

    # -------------------------------------------------
    # Metadata helpers
    # -------------------------------------------------

    def get_model_name(self) -> str:
        return self.model_name

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return f"<{self.__class__.__name__} name={self.model_name} status={status}>"
