import numpy as np
import pandas as pd
from typing import List, Dict


class ImportanceAnalyzer:
    """
    Generate permutation effect deltas from trained models.
    """

    def __init__(self, *, perm_sample_size: int | None = None):
        self.perm_sample_size = perm_sample_size

    # =================================================
    # Public API
    # =================================================

    def run_perm_effect(
        self,
        *,
        models: List,
        X_ref: pd.DataFrame,
        problem_name: str,
        random_seed: int | None = None,
    ) -> Dict[str, pd.DataFrame]:
        rows = []
        X_used = self._subsample(X_ref)

        for fold, model in enumerate(models):
            rng = np.random.default_rng(
                None if random_seed is None else random_seed + fold
            )
            X_num = self._prepare_input(model, X_used)
            base = X_num.to_numpy()
            columns = list(X_num.columns)

            pred_base = model.model.predict(X_num)

            for idx, col in enumerate(columns):
                X_perm = base.copy()
                X_perm[:, idx] = rng.permutation(X_perm[:, idx])
                pred_perm = model.model.predict(X_perm)
                delta = np.mean((pred_base - pred_perm) ** 2)
                rows.append(
                    {
                        "problem": problem_name,
                        "method": "PERM",
                        "fold": fold,
                        "feature": col,
                        "delta": float(delta),
                    }
                )

        return {
            "perm_effect_raw": pd.DataFrame(rows),
        }

    # =================================================
    # Internal helpers
    # =================================================

    def _subsample(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.perm_sample_size is None:
            return X
        if len(X) <= self.perm_sample_size:
            return X
        return X.sample(self.perm_sample_size, random_state=42)

    def _prepare_input(self, model, X: pd.DataFrame) -> pd.DataFrame:
        # keep only the features used by the model (and in correct order)
        missing = [f for f in model.feature_names if f not in X.columns]
        if missing:
            raise RuntimeError(
                "SHAP input missing model features: "
                + ", ".join(missing)
            )
        X_used = X.loc[:, model.feature_names].copy()

        # force numeric and validate shape
        for col in X_used.columns:
            if not np.issubdtype(X_used[col].dtype, np.number):
                X_used[col] = pd.to_numeric(X_used[col], errors="coerce")
        X_num = X_used.astype(float)

        # guard against NaN/inf after coercion
        if not np.isfinite(X_num.to_numpy()).all():
            bad_cols = [
                col
                for col in X_num.columns
                if not np.isfinite(X_num[col].to_numpy()).all()
            ]
            raise RuntimeError(
                "SHAP input contains NaN/inf after numeric coercion in: "
                + ", ".join(bad_cols)
            )

        return X_num
