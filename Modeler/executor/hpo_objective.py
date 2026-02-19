# Modeler/executor/hpo_objective.py

from __future__ import annotations

from typing import Callable

import numpy as np
import optuna

from Modeler.executor.splitter import FixedKFoldSplitter
from Modeler.Models.xgboost import XGBoostModel


def make_robust_objective(
    *,
    X: np.ndarray,
    y: np.ndarray,
    base_random_seed: int,
    search_space_fn: Callable,
    lambda_std: float,
    kfold_splits: int = 5,
):
    """
    Robust objective:
    score = mean(RMSE) + lambda_std * std(RMSE)

    - Uses FixedKFoldSplitter (MODELER policy)
    - Fold-level seed = base_seed + (run_id + 1)
    """
    splitter = FixedKFoldSplitter(
        base_random_seed=base_random_seed,
        n_splits=kfold_splits,
    )

    def objective(trial: optuna.Trial) -> float:
        params = search_space_fn(trial)

        valid_rmses = []
        train_rmses = []

        for run_id, train_idx, valid_idx in splitter.split(X):
            model_seed = base_random_seed + (run_id + 1)

            model = XGBoostModel(
                **params,
                random_state=model_seed,
            )

            model.fit(
                X[train_idx],
                y[train_idx],
            )

            # -------------------------
            # Train RMSE
            # -------------------------
            y_train_pred = model.predict(X[train_idx])
            train_err = y[train_idx] - y_train_pred
            train_rmse = float(np.sqrt(np.mean(train_err ** 2)))
            train_rmses.append(train_rmse)

            # -------------------------
            # Validation RMSE
            # -------------------------
            y_valid_pred = model.predict(X[valid_idx])
            valid_err = y[valid_idx] - y_valid_pred
            valid_rmse = float(np.sqrt(np.mean(valid_err ** 2)))
            valid_rmses.append(valid_rmse)

        # -------------------------
        # Aggregate
        # -------------------------
        mean_valid_rmse = float(np.mean(valid_rmses))
        std_valid_rmse = float(np.std(valid_rmses))

        mean_train_rmse = float(np.mean(train_rmses))
        std_train_rmse = float(np.std(train_rmses))

        score = mean_valid_rmse + lambda_std * std_valid_rmse

        # -------------------------
        # Logging (diagnostics)
        # -------------------------
        trial.set_user_attr("valid_rmse_mean", mean_valid_rmse)
        trial.set_user_attr("valid_rmse_std", std_valid_rmse)
        trial.set_user_attr("valid_rmse_per_fold", valid_rmses)

        trial.set_user_attr("train_rmse_mean", mean_train_rmse)
        trial.set_user_attr("train_rmse_std", std_train_rmse)
        trial.set_user_attr("train_rmse_per_fold", train_rmses)

        trial.set_user_attr(
            "generalization_gap",
            mean_valid_rmse - mean_train_rmse
        )

        return score

    return objective
