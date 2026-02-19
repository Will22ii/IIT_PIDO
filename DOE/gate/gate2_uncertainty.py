# DOE/gate/gate2_uncertainty.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
from scipy.stats import chi2


@dataclass(frozen=True)
class Gate2Result:
    passed: bool
    score: float
    per_point_std: np.ndarray
    info: Dict[str, float]


class Gate2Uncertainty:
    """
    Gate2: Uncertainty gate using chi-square fitted distribution.

    Design:
    - Compute per-point uncertainty via ensemble std.
    - Square to variance and fit σ²·χ²(k).
    - Use theoretical chi-square CDF threshold (q-level).
    - Pass if sufficient fraction lies below the threshold.

    Notes:
    - Models are assumed to be already trained (Bootstrap handled upstream).
    - This gate DOES NOT care about top-k or objective ranking.
    """

    def __init__(
        self,
        *,
        k: int = 2,
        cdf_level: float = 0.8,
        ratio_threshold: float = 0.8,
        relax_factor: float = 1.1,
    ):
        if k <= 0:
            raise ValueError("k must be positive")
        if not (0 < cdf_level < 1):
            raise ValueError("cdf_level must be in (0,1)")
        if not (0 < ratio_threshold <= 1):
            raise ValueError("ratio_threshold must be in (0,1]")
        if relax_factor <= 0:
            raise ValueError("relax_factor must be positive")

        self.k = int(k)
        self.cdf_level = float(cdf_level)
        self.ratio_threshold = float(ratio_threshold)
        self.relax_factor = float(relax_factor)

    def evaluate(
        self,
        *,
        models: Sequence[Any],
        X_candidate: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Returns dict for compatibility with GateManager:
          {
            "passed": bool,
            "score": float,
            "per_point_std": np.ndarray,
            "info": Dict[str, float],
          }
        """
        models = list(models)
        if len(models) < 2:
            raise ValueError("Gate2 requires at least 2 models")

        X_candidate = np.asarray(X_candidate, dtype=float)
        if X_candidate.ndim != 2 or X_candidate.shape[0] == 0:
            raise ValueError("X_candidate must be a non-empty 2D array")

        # 1) Collect predictions: shape (n_models, n_points)
        preds = [self._predict(m, X_candidate) for m in models]
        Y = np.vstack(preds)  # (M, N)

        # 2) Per-point uncertainty
        per_point_std = np.std(Y, axis=0)
        per_point_var = per_point_std ** 2

        # 3) Fit scale σ² (MLE)
        sigma2_hat = float(np.mean(per_point_var) / self.k)

        # 4) Chi-square threshold
        chi2_q = float(chi2.ppf(self.cdf_level, df=self.k))
        threshold = self.relax_factor * sigma2_hat * chi2_q

        # 5) Empirical ratio
        ratio = float(np.mean(per_point_var <= threshold))

        passed = bool(ratio >= self.ratio_threshold)

        return {
            "passed": passed,
            "score": ratio,
            "per_point_std": per_point_std,
            "info": {
                "k": float(self.k),
                "cdf_level": float(self.cdf_level),
                "ratio_threshold": float(self.ratio_threshold),
                "relax_factor": float(self.relax_factor),
                "sigma2_hat": sigma2_hat,
                "chi2_q": chi2_q,
                "threshold": threshold,
                "empirical_ratio": ratio,
            },
        }

    # -------------------------
    # Internals
    # -------------------------

    def _predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """
        Supports:
        - model.predict(X) -> (n,) or (n,1)
        - callable model(X) -> (n,)
        """
        if hasattr(model, "predict") and callable(getattr(model, "predict")):
            y = model.predict(X)
        elif callable(model):
            y = model(X)
        else:
            raise TypeError("Model must have .predict(X) or be callable")

        y = np.asarray(y, dtype=float).reshape(-1)
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"Model prediction length mismatch: got {y.shape[0]}, expected {X.shape[0]}"
            )
        return y
