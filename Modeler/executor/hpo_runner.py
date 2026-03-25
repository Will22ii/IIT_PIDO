# MODELER/executor/hpo_runner.py

from typing import Any, Dict

import numpy as np

import optuna

from Modeler.executor.hpo_objective import make_robust_objective
from Modeler.executor.hpo_format import format_hpo_result


DEFAULT_XGB_SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "n_estimators": (300, 800),
    "learning_rate": (0.01, 0.1),
    "max_depth": (3, 9),
    "min_child_weight": (1, 10),
    "subsample": (0.7, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "gamma": (0.0, 0.3),
}


def _resolve_bounds(
    *,
    space: dict[str, Any] | None,
    key: str,
    default_low: float,
    default_high: float,
) -> tuple[float, float]:
    if not isinstance(space, dict):
        return float(default_low), float(default_high)
    raw = space.get(key)
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        low_raw, high_raw = raw
    elif isinstance(raw, dict):
        low_raw = raw.get("low", default_low)
        high_raw = raw.get("high", default_high)
    else:
        return float(default_low), float(default_high)
    try:
        low = float(low_raw)
        high = float(high_raw)
    except (TypeError, ValueError):
        return float(default_low), float(default_high)
    if not np.isfinite(low) or not np.isfinite(high):
        return float(default_low), float(default_high)
    if high < low:
        low, high = high, low
    return float(low), float(high)


def make_xgb_search_space(
    *,
    search_space: dict[str, Any] | None = None,
):
    n_est_low, n_est_high = _resolve_bounds(
        space=search_space,
        key="n_estimators",
        default_low=DEFAULT_XGB_SEARCH_SPACE["n_estimators"][0],
        default_high=DEFAULT_XGB_SEARCH_SPACE["n_estimators"][1],
    )
    lr_low, lr_high = _resolve_bounds(
        space=search_space,
        key="learning_rate",
        default_low=DEFAULT_XGB_SEARCH_SPACE["learning_rate"][0],
        default_high=DEFAULT_XGB_SEARCH_SPACE["learning_rate"][1],
    )
    depth_low, depth_high = _resolve_bounds(
        space=search_space,
        key="max_depth",
        default_low=DEFAULT_XGB_SEARCH_SPACE["max_depth"][0],
        default_high=DEFAULT_XGB_SEARCH_SPACE["max_depth"][1],
    )
    child_low, child_high = _resolve_bounds(
        space=search_space,
        key="min_child_weight",
        default_low=DEFAULT_XGB_SEARCH_SPACE["min_child_weight"][0],
        default_high=DEFAULT_XGB_SEARCH_SPACE["min_child_weight"][1],
    )
    subs_low, subs_high = _resolve_bounds(
        space=search_space,
        key="subsample",
        default_low=DEFAULT_XGB_SEARCH_SPACE["subsample"][0],
        default_high=DEFAULT_XGB_SEARCH_SPACE["subsample"][1],
    )
    cols_low, cols_high = _resolve_bounds(
        space=search_space,
        key="colsample_bytree",
        default_low=DEFAULT_XGB_SEARCH_SPACE["colsample_bytree"][0],
        default_high=DEFAULT_XGB_SEARCH_SPACE["colsample_bytree"][1],
    )
    gamma_low, gamma_high = _resolve_bounds(
        space=search_space,
        key="gamma",
        default_low=DEFAULT_XGB_SEARCH_SPACE["gamma"][0],
        default_high=DEFAULT_XGB_SEARCH_SPACE["gamma"][1],
    )
    lr_low = max(float(lr_low), 1e-6)
    lr_high = max(float(lr_high), lr_low)
    subs_low = max(float(subs_low), 1e-6)
    subs_high = max(float(subs_high), subs_low)
    cols_low = max(float(cols_low), 1e-6)
    cols_high = max(float(cols_high), cols_low)
    gamma_low = max(float(gamma_low), 0.0)
    gamma_high = max(float(gamma_high), gamma_low)

    # integer params need valid integer bounds with low <= high.
    n_est_low_i = max(int(round(n_est_low)), 1)
    n_est_high_i = max(int(round(n_est_high)), n_est_low_i)
    depth_low_i = max(int(round(depth_low)), 1)
    depth_high_i = max(int(round(depth_high)), depth_low_i)
    child_low_i = max(int(round(child_low)), 1)
    child_high_i = max(int(round(child_high)), child_low_i)

    def _search_space_fn(trial: optuna.Trial) -> Dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", n_est_low_i, n_est_high_i),
            "learning_rate": trial.suggest_float("learning_rate", lr_low, lr_high, log=True),
            "max_depth": trial.suggest_int("max_depth", depth_low_i, depth_high_i),
            "min_child_weight": trial.suggest_int("min_child_weight", child_low_i, child_high_i),
            "subsample": trial.suggest_float("subsample", subs_low, subs_high),
            "colsample_bytree": trial.suggest_float("colsample_bytree", cols_low, cols_high),
            "gamma": trial.suggest_float("gamma", gamma_low, gamma_high),
        }

    return _search_space_fn


# =====================================================
# HPO Runner (MODELER Stage)
# =====================================================

class HPORunner:
    """
    MODELER-compliant HPO runner.

    - FixedKFoldSplitter
    - XGBoost only (current policy)
    - Robust objective (mean + lambda * std)
    - ResultSaver based persistence
    """

    def __init__(
        self,
        *,
        n_trials: int = 80,
        lambda_std: float = 0.5,
        use_timestamp: bool = True,
        show_optuna_log: bool = False,
        search_space: dict[str, Any] | None = None,
        hpo_mode: str = "default",
    ):
        self.n_trials = n_trials
        self.lambda_std = lambda_std
        self.show_optuna_log = bool(show_optuna_log)
        self.search_space = dict(search_space) if isinstance(search_space, dict) else None
        self.hpo_mode = str(hpo_mode)

    # -------------------------------------------------
    # Main entry
    # -------------------------------------------------

    def run_xgb(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        base_random_seed: int,
        problem_name: str,
        kfold_splits: int = 5,
    ) -> Dict:
        """
        Run HPO for XGBoost.

        Returns
        -------
        dict with best_params and metrics
        """
        if self.show_optuna_log:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(seed=base_random_seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        objective = make_robust_objective(
            X=X,
            y=y,
            base_random_seed=base_random_seed,
            search_space_fn=make_xgb_search_space(search_space=self.search_space),
            lambda_std=self.lambda_std,
            kfold_splits=kfold_splits,
        )

        study.optimize(objective, n_trials=self.n_trials)

        best_trial = study.best_trial

        result = format_hpo_result(
            best_trial=best_trial,
            base_random_seed=base_random_seed,
            n_trials=self.n_trials,
            lambda_std=self.lambda_std,
        )

        return result

    def run(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        base_random_seed: int = 42,
        problem_name: str = "DOE",
    ) -> Dict:
        """
        Adapter for DOE / Orchestrator usage.

        Returns
        -------
        best_params : dict
            XGBoost best hyperparameters
        """

        result = self.run_xgb(
            X=X,
            y=y,
            base_random_seed=base_random_seed,
            problem_name=problem_name,
        )

        # 🔑 Orchestrator가 필요한 건 params만
        return result["best_params"]
