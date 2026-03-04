# DOE/gate/gate1_topk_stability.py

from __future__ import annotations

from typing import Any, Dict, List, Literal, Sequence

import numpy as np

from utils.bounds_utils import normalize_with_bounds


class Gate1TopKStability:
    """
    Gate1: Spatial stability (soft-Jaccard on top-k)

    목적:
    - 서로 다른 surrogate 모델들이 같은 X_candidate 집합에 대해
      "좋다고 판단하는(top-k)" 점들이 유사한지 평가한다.

    구현:
    - 기준 모델(가장 많은 데이터)의 top-k 집합과
      다른 모델 top-k 집합 간의 soft-Jaccard 유사도를 계산
    - eps 이내면 "같은 점"으로 매칭
    """

    def __init__(
        self,
        *,
        k_ratio: float,
        pass_ratio: float = 0.5,
        objective_sense: Literal["min", "max"] = "min",
    ):
        if not (0.0 < k_ratio <= 1.0):
            raise ValueError("k_ratio must be in (0, 1]")
        if not (0.0 <= pass_ratio <= 1.0):
            raise ValueError("pass_ratio must be in [0, 1]")
        if objective_sense not in ("min", "max"):
            raise ValueError("objective_sense must be 'min' or 'max'")

        self.k_ratio = float(k_ratio)
        self.pass_ratio = float(pass_ratio)
        self.objective_sense = objective_sense

    def evaluate(
        self,
        *,
        models: Sequence[Any],
        X_candidate: np.ndarray,
        bounds: Sequence[tuple[float, float]],
        eps: float | None = None,
    ) -> Dict[str, Any]:
        """
        Returns dict for compatibility with GateManager:
          {
            "passed": bool,
            "score": float,
            "pair_scores": { ... },
            "topk_indices": [ [...], [...], ... ],
          }
        """
        models = list(models)
        if len(models) < 2:
            raise ValueError("Gate1 requires at least 2 models")

        X_candidate = np.asarray(X_candidate, dtype=float)
        if X_candidate.ndim != 2 or X_candidate.shape[0] == 0:
            raise ValueError("X_candidate must be a non-empty 2D array")

        k = self._resolve_k(X_candidate.shape[0])

        # 기준 모델 = 가장 많은 데이터 모델(마지막)
        ref_model = models[-1]
        ref_y = self._predict(ref_model, X_candidate)
        ref_idx = self._topk_index(ref_y, k=min(k, X_candidate.shape[0]))

        X_ref = X_candidate[ref_idx]
        X_ref_norm = self._normalize(X_ref, bounds)

        eps_use = float(eps) if eps is not None else 0.0
        if eps_use <= 0.0:
            raise ValueError("Gate1 soft-Jaccard requires eps > 0")

        scores = []
        for m in models[:-1]:
            yhat = self._predict(m, X_candidate)
            idx = self._topk_index(yhat, k=min(k, X_candidate.shape[0]))
            X_top = X_candidate[idx]
            X_top_norm = self._normalize(X_top, bounds)
            scores.append(self._soft_jaccard(X_ref_norm, X_top_norm, eps=eps_use))

        # 두 모델을 가정 (small, medium)
        score_small = scores[0] if len(scores) > 0 else 0.0
        score_medium = scores[1] if len(scores) > 1 else 0.0

        score = float(min(score_medium, score_small))
        passed = bool(score >= self.pass_ratio)

        return {
            "passed": passed,
            "score": score,
            "coverage_ref": float(score),
            "coverage_small": float(score_small),
            "coverage_medium": float(score_medium),
            "thresholds": {
                "threshold_small": float(self.pass_ratio),
                "threshold_medium": float(self.pass_ratio),
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
            raise ValueError(f"Model prediction length mismatch: got {y.shape[0]}, expected {X.shape[0]}")
        return y

    def _topk_index(self, y: np.ndarray, *, k: int) -> List[int]:
        """
        objective_sense = "min": smallest y is best
        objective_sense = "max": largest y is best
        """
        if k <= 0:
            return []

        if self.objective_sense == "min":
            order = np.argsort(y)  # ascending
        else:
            order = np.argsort(-y)  # descending
        return order[:k].astype(int).tolist()

    def _resolve_k(self, n: int) -> int:
        if n <= 0:
            return 1
        return max(1, int(round(n * self.k_ratio)))

    def _normalize(
        self,
        X: np.ndarray,
        bounds: Sequence[tuple[float, float]],
    ) -> np.ndarray:
        return normalize_with_bounds(X, bounds)

    def _soft_jaccard(self, A: np.ndarray, B: np.ndarray, *, eps: float) -> float:
        if A.size == 0 and B.size == 0:
            return 1.0
        if A.size == 0 or B.size == 0:
            return 0.0

        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)

        used_b = np.zeros(B.shape[0], dtype=bool)
        matches = 0
        for a in A:
            dists = np.linalg.norm(B - a.reshape(1, -1), axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] <= eps and not used_b[idx]:
                used_b[idx] = True
                matches += 1

        union = A.shape[0] + B.shape[0] - matches
        if union <= 0:
            return 0.0
        return float(matches / union)

