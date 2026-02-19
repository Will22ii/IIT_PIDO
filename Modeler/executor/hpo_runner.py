# MODELER/executor/hpo_runner.py

from typing import Dict

import numpy as np

import optuna

from Modeler.executor.hpo_objective import make_robust_objective
from Modeler.executor.hpo_format import format_hpo_result


# =====================================================
# XGBoost Search Space 
# =====================================================

def xgb_search_space(trial: optuna.Trial) -> Dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.1, log=True
        ),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 0.3),
    }


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
    ):
        self.n_trials = n_trials
        self.lambda_std = lambda_std

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

        study = optuna.create_study(direction="minimize")

        objective = make_robust_objective(
            X=X,
            y=y,
            base_random_seed=base_random_seed,
            search_space_fn=xgb_search_space,
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
