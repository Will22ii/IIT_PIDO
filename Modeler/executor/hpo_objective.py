# Modeler/executor/hpo_objective.py

from __future__ import annotations

from typing import Callable

import numpy as np
import optuna

from Modeler.executor.splitter import FixedKFoldSplitter
from Modeler.Models.xgboost import XGBoostModel


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.asarray(y_true, dtype=float).reshape(-1)
    y_p = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_t.size <= 1 or y_t.size != y_p.size:
        return 0.0
    ss_res = float(np.sum((y_t - y_p) ** 2))
    y_mean = float(np.mean(y_t))
    ss_tot = float(np.sum((y_t - y_mean) ** 2))
    if not np.isfinite(ss_tot) or ss_tot <= 1e-12:
        return 0.0
    r2 = 1.0 - (ss_res / ss_tot)
    if not np.isfinite(r2):
        return 0.0
    return float(r2)


def make_robust_objective(
    *,
    X: np.ndarray,
    y: np.ndarray,
    base_random_seed: int,
    search_space_fn: Callable,
    lambda_std: float,
    lambda_gap: float = 0.0,
    kfold_splits: int = 5,
    pruning_enabled: bool = False,
    pruning_warmup_steps: int = 1,
):
    """
    Robust objective:
    score = mean(RMSE) + lambda_std * std(RMSE) + lambda_gap * max(train_r2 - valid_r2, 0)

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
        valid_r2s = []
        train_r2s = []
        prune_after_step = max(int(pruning_warmup_steps), 1)

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
            train_r2s.append(_safe_r2(y[train_idx], y_train_pred))

            # -------------------------
            # Validation RMSE
            # -------------------------
            y_valid_pred = model.predict(X[valid_idx])
            valid_err = y[valid_idx] - y_valid_pred
            valid_rmse = float(np.sqrt(np.mean(valid_err ** 2)))
            valid_rmses.append(valid_rmse)
            valid_r2s.append(_safe_r2(y[valid_idx], y_valid_pred))

            if bool(pruning_enabled):
                step = int(run_id) + 1
                partial_mean = float(np.mean(valid_rmses))
                partial_std = float(np.std(valid_rmses))
                partial_train_r2 = float(np.mean(train_r2s)) if train_r2s else 0.0
                partial_valid_r2 = float(np.mean(valid_r2s)) if valid_r2s else 0.0
                partial_gap_r2 = max(partial_train_r2 - partial_valid_r2, 0.0)
                partial_score = (
                    partial_mean
                    + float(lambda_std) * partial_std
                    + float(lambda_gap) * float(partial_gap_r2)
                )
                trial.report(partial_score, step=step)
                if step >= prune_after_step and trial.should_prune():
                    raise optuna.TrialPruned()

        # -------------------------
        # Aggregate
        # -------------------------
        mean_valid_rmse = float(np.mean(valid_rmses))
        std_valid_rmse = float(np.std(valid_rmses))

        mean_train_rmse = float(np.mean(train_rmses))
        std_train_rmse = float(np.std(train_rmses))
        mean_train_r2 = float(np.mean(train_r2s)) if train_r2s else 0.0
        mean_valid_r2 = float(np.mean(valid_r2s)) if valid_r2s else 0.0
        overfit_gap_r2 = max(mean_train_r2 - mean_valid_r2, 0.0)
        gap_penalty = float(lambda_gap) * float(overfit_gap_r2)

        score = mean_valid_rmse + lambda_std * std_valid_rmse + gap_penalty

        # -------------------------
        # Logging (diagnostics)
        # -------------------------
        trial.set_user_attr("valid_rmse_mean", mean_valid_rmse)
        trial.set_user_attr("valid_rmse_std", std_valid_rmse)
        trial.set_user_attr("valid_rmse_per_fold", valid_rmses)

        trial.set_user_attr("train_rmse_mean", mean_train_rmse)
        trial.set_user_attr("train_rmse_std", std_train_rmse)
        trial.set_user_attr("train_rmse_per_fold", train_rmses)
        trial.set_user_attr("train_r2_mean", mean_train_r2)
        trial.set_user_attr("train_r2_per_fold", train_r2s)
        trial.set_user_attr("valid_r2_mean", mean_valid_r2)
        trial.set_user_attr("valid_r2_per_fold", valid_r2s)

        trial.set_user_attr(
            "generalization_gap",
            mean_valid_rmse - mean_train_rmse
        )
        trial.set_user_attr("overfit_gap_r2", overfit_gap_r2)
        trial.set_user_attr("gap_penalty", gap_penalty)
        trial.set_user_attr("score_rmse_component", mean_valid_rmse)
        trial.set_user_attr("score_std_component", float(lambda_std) * std_valid_rmse)
        trial.set_user_attr("score_gap_component", gap_penalty)

        return score

    return objective
