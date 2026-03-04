# MODELER/executor/trainer.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from Modeler.executor.splitter import FixedKFoldSplitter
from Modeler.Models.registry import get_model_class


class ModelTrainer:
    """
    Trainer for Modeler stage.

    - Fixed/Repeated K-Fold
    - Model registry based selection
    - Supports external model parameters (e.g. HPO results)
    - Collect fold-level predictions and models
    """

    def __init__(
        self,
        *,
        base_random_seed: int,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        model_params: Optional[Dict] = None,   # ✅ HPO 결과 주입
        model_name: str = "xgb",
        kfold_splits: int = 5,
        kfold_repeats: int = 1,
    ):
        self.base_random_seed = base_random_seed
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.model_name = model_name
        self.kfold_splits = kfold_splits
        self.kfold_repeats = kfold_repeats

        # HPO 결과 or None
        self.model_params = model_params or {}

        self.splitter = FixedKFoldSplitter(
            base_random_seed=self.base_random_seed,
            n_splits=self.kfold_splits,
            n_repeats=self.kfold_repeats,
        )

    # -------------------------------------------------
    # Main entry
    # -------------------------------------------------

    def run(self, df: pd.DataFrame) -> Dict:
        """
        Execute K-Fold training.

        Returns
        -------
        dict with:
          - models
          - fold_predictions
          - oof_prediction
          - y_true
          - feature_cols
        """

        # -----------------------------
        # Feature / target split
        # -----------------------------
        if self.feature_cols is None:
            feature_cols = [
                c for c in df.columns
                if c != self.target_col
            ]
        else:
            feature_cols = self.feature_cols

        X = df[feature_cols].values
        y = df[self.target_col].values

        # -----------------------------
        # Containers
        # -----------------------------
        models: List = []
        fold_predictions = []
        oof_sum = np.zeros(len(df), dtype=float)
        oof_count = np.zeros(len(df), dtype=int)

        # -----------------------------
        # K-Fold loop
        # -----------------------------
        for run_id, train_idx, valid_idx in self.splitter.split(X):
            model_seed = self.splitter.get_model_seed(run_id)


            model_cls = get_model_class(self.model_name)
            model = model_cls(
                **self.model_params,
                random_state=model_seed,
            )

            model.fit(
                X[train_idx],
                y[train_idx],
            )

            model.feature_names = feature_cols

            y_valid_pred = model.predict(
                X[valid_idx]
            )
            y_valid_pred = np.asarray(y_valid_pred, dtype=float).reshape(-1)

            # store
            models.append(model)
            fold_predictions.append({
                "run_id": run_id,
                "repeat_id": self.splitter.get_repeat_id(run_id),
                "fold_id": self.splitter.get_fold_id(run_id),
                "valid_idx": valid_idx,
                "y_pred": y_valid_pred,
            })

            oof_sum[valid_idx] += y_valid_pred
            oof_count[valid_idx] += 1

        oof_pred = np.divide(
            oof_sum,
            np.maximum(oof_count, 1),
            out=np.zeros_like(oof_sum, dtype=float),
            where=oof_count > 0,
        )

        return {
            "models": models,
            "fold_predictions": fold_predictions,
            "oof_prediction": oof_pred,
            "oof_prediction_count": oof_count,
            "y_true": y,
            "feature_cols": feature_cols,
        }
