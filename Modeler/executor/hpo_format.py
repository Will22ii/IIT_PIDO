# Modeler/executor/hpo_format.py

from __future__ import annotations

from datetime import datetime
from typing import Dict

import optuna


def format_hpo_result(
    *,
    best_trial: optuna.Trial,
    base_random_seed: int,
    n_trials: int,
    lambda_std: float,
) -> Dict:
    """
    Build a consistent result dict for both DOE additional and MODELER.
    """
    valid_mean = best_trial.user_attrs.get("valid_rmse_mean")
    valid_std = best_trial.user_attrs.get("valid_rmse_std")
    valid_per_fold = best_trial.user_attrs.get("valid_rmse_per_fold")

    return {
        "model": "XGBoost",
        "best_params": best_trial.params,
        "metric": "robust_rmse",
        "score": best_trial.value,

        # validation 기준 (현재 표준)
        "valid_rmse_mean": valid_mean,
        "valid_rmse_std": valid_std,
        "valid_rmse_per_fold": valid_per_fold,

        # legacy keys (Modeler compatibility)
        "rmse_mean": valid_mean,
        "rmse_std": valid_std,
        "rmse_per_fold": valid_per_fold,

        "train_rmse_mean": best_trial.user_attrs.get("train_rmse_mean"),
        "train_rmse_std": best_trial.user_attrs.get("train_rmse_std"),
        "train_rmse_per_fold": best_trial.user_attrs.get("train_rmse_per_fold"),
        "generalization_gap": best_trial.user_attrs.get("generalization_gap"),

        "n_trials": n_trials,
        "lambda_std": lambda_std,
        "base_random_seed": base_random_seed,
        "created_at": datetime.now().isoformat(),
    }
