# DOE/executor/surrogate_factory.py

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from Modeler.Models.xgboost import XGBoostModel


class SurrogateFactory:
    """
    Build surrogate ensembles for Gate evaluation.

    Design:
    - Gate1 models: "data-size effect" using exec size deltas:
        N-2*exec, N-exec, N
      with FIXED permutation seed + FIXED model random_state.
      -> isolates stability improvements attributable to more data.

    - Gate2 models: Bootstrap ensemble for uncertainty estimation.
      -> each model trained on bootstrap-resampled dataset.

    Notes:
    - We intentionally DO NOT run HPO here (fast, deterministic, gate-only policy).
    - Feature columns are restricted to x_* (avoid leakage from feasible/success).
    """

    def __init__(
        self,
        *,
        target_col: str = "objective",
        feature_prefix: str = "x_",
        n_models: int = 3,
        gate1_fixed_seed: int = 123,      # fixed sampling seed for Gate1
        gate1_fixed_model_seed: int = 123,  # fixed model seed for Gate1
        gate2_bootstrap_seeds: Tuple[int, int, int] = (0, 1, 2),
        xgb_params: Optional[dict] = None,
    ):
        self.target_col = target_col
        self.feature_prefix = feature_prefix
        self.n_models = int(n_models)

        self.gate1_fixed_seed = int(gate1_fixed_seed)
        self.gate1_fixed_model_seed = int(gate1_fixed_model_seed)

        if len(gate2_bootstrap_seeds) != self.n_models:
            raise ValueError("gate2_bootstrap_seeds length must equal n_models")
        self.gate2_bootstrap_seeds = gate2_bootstrap_seeds

        # Reasonable deterministic defaults (can be tuned later)
        default_params = {
            "n_estimators": 600,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        }
        self.xgb_params = {**default_params, **(xgb_params or {})}

    # -----------------------------
    # Public API
    # -----------------------------

    def build(self, *, df: pd.DataFrame, round_idx: int, exec_size: int | None = None) -> Dict[str, List[XGBoostModel]]:
        """
        Returns:
          {
            "gate1": [model0, model1, model2],
            "gate2": [model0, model1, model2],
          }

        Orchestrator should pass:
          - gate1_models to Gate1
          - gate2_models to Gate2
        """
        gate1_models = self.build_gate1(df=df, round_idx=round_idx, exec_size=exec_size)
        gate2_models = self.build_gate2(df=df)
        return {"gate1": gate1_models, "gate2": gate2_models}

    def build_gate1(self, *, df: pd.DataFrame, round_idx: int, exec_size: int | None = None) -> List[XGBoostModel]:
        """
        Gate1: 3 models with increasing data ratios.
        - exec_size provided: use [N-2*exec, N-exec, N] (clipped)
        - if exec_size is None: fall back to [N, N, N]
        - sampling uses ONE fixed permutation (seed fixed), making subsets nested:
            idx_perm = permutation(N)
            subset_n = idx_perm[:n]
        - model random_state also fixed to isolate data-size effect.
        """
        self._validate_df(df)

        X, y, feature_cols = self._split_xy(df)

        N = X.shape[0]
        if exec_size is None or exec_size <= 0:
            sizes = [N, N, N]
        else:
            n_exec = int(exec_size)
            sizes = [max(1, N - 2 * n_exec), max(1, N - n_exec), N]
            # ensure non-decreasing and <= N
            sizes = [min(s, N) for s in sizes]
            for i in range(1, len(sizes)):
                if sizes[i] < sizes[i - 1]:
                    sizes[i] = sizes[i - 1]

        # fixed permutation -> nested subsets
        rng = np.random.default_rng(self.gate1_fixed_seed)
        perm = rng.permutation(N)

        models: List[XGBoostModel] = []
        for n in sizes:
            idx = perm[:n]

            m = self._new_xgb_model(random_state=self.gate1_fixed_model_seed)
            m.feature_names = feature_cols
            m.fit(X[idx], y[idx])
            models.append(m)

        return models

    def build_gate2(self, *, df: pd.DataFrame) -> List[XGBoostModel]:
        """
        Gate2: 3 bootstrap models for uncertainty.
        - Each model trained on bootstrap sample of full available df.
        - Model random_state is tied to bootstrap seed for diversity.
        """
        self._validate_df(df)

        X, y, feature_cols = self._split_xy(df)
        N = X.shape[0]

        models: List[XGBoostModel] = []
        for seed in self.gate2_bootstrap_seeds:
            rng = np.random.default_rng(seed)
            boot_idx = rng.choice(N, size=N, replace=True)

            m = self._new_xgb_model(random_state=seed)
            m.feature_names = feature_cols
            m.fit(X[boot_idx], y[boot_idx])
            models.append(m)

        return models

    # -----------------------------
    # Internals
    # -----------------------------

    def _new_xgb_model(self, *, random_state: int) -> XGBoostModel:
        params = dict(self.xgb_params)
        params["random_state"] = int(random_state)
        return XGBoostModel(**params)


    def _split_xy(self, df: pd.DataFrame):
        feature_cols = [c for c in df.columns if c.startswith(self.feature_prefix)]
        if self.target_col not in df.columns:
            raise ValueError(f"target_col '{self.target_col}' not found in df columns")
        if not feature_cols:
            raise ValueError(f"No feature columns found with prefix '{self.feature_prefix}'")

        X = df[feature_cols].values.astype(float)
        y = df[self.target_col].values.astype(float)
        return X, y, feature_cols

    @staticmethod
    def _validate_df(df: pd.DataFrame) -> None:
        if df is None or len(df) == 0:
            raise ValueError("Empty dataset for surrogate training")
