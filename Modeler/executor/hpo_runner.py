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
    "min_child_weight": (0.5, 12.0),
    "subsample": (0.7, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "gamma": (0.0, 0.3),
}
DEFAULT_HPO_SAMPLER = "tpe"


DEFAULT_PRUNING_CONFIG: dict[str, float | int | bool] = {
    "enabled": False,
    "percentile": 85.0,
    "n_startup_trials": 12,
    "n_warmup_steps": 4,
    "interval_steps": 1,
    "min_completed_trials": 8,
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


def _has_space_key(space: dict[str, Any] | None, key: str) -> bool:
    return isinstance(space, dict) and key in space


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
    child_low = max(float(child_low), 1e-6)
    child_high = max(float(child_high), child_low)
    use_reg_alpha = _has_space_key(search_space, "reg_alpha")
    use_reg_lambda = _has_space_key(search_space, "reg_lambda")
    if use_reg_alpha:
        reg_alpha_low, reg_alpha_high = _resolve_bounds(
            space=search_space,
            key="reg_alpha",
            default_low=0.0,
            default_high=1.0,
        )
        reg_alpha_low = max(float(reg_alpha_low), 0.0)
        reg_alpha_high = max(float(reg_alpha_high), reg_alpha_low)
    if use_reg_lambda:
        reg_lambda_low, reg_lambda_high = _resolve_bounds(
            space=search_space,
            key="reg_lambda",
            default_low=1.0,
            default_high=10.0,
        )
        reg_lambda_low = max(float(reg_lambda_low), 1e-6)
        reg_lambda_high = max(float(reg_lambda_high), reg_lambda_low)

    # integer params need valid integer bounds with low <= high.
    n_est_low_i = max(int(round(n_est_low)), 1)
    n_est_high_i = max(int(round(n_est_high)), n_est_low_i)
    depth_low_i = max(int(round(depth_low)), 1)
    depth_high_i = max(int(round(depth_high)), depth_low_i)

    def _search_space_fn(trial: optuna.Trial) -> Dict:
        params: Dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", n_est_low_i, n_est_high_i),
            "learning_rate": trial.suggest_float("learning_rate", lr_low, lr_high, log=True),
            "max_depth": trial.suggest_int("max_depth", depth_low_i, depth_high_i),
            "min_child_weight": trial.suggest_float("min_child_weight", child_low, child_high, log=True),
            "subsample": trial.suggest_float("subsample", subs_low, subs_high),
            "colsample_bytree": trial.suggest_float("colsample_bytree", cols_low, cols_high),
            "gamma": trial.suggest_float("gamma", gamma_low, gamma_high),
        }
        if use_reg_alpha:
            params["reg_alpha"] = trial.suggest_float("reg_alpha", reg_alpha_low, reg_alpha_high, log=True)
        if use_reg_lambda:
            params["reg_lambda"] = trial.suggest_float("reg_lambda", reg_lambda_low, reg_lambda_high, log=True)
        return params

    return _search_space_fn


def _resolve_pruning_config(config: dict[str, Any] | None) -> dict[str, float | int | bool]:
    cfg = dict(DEFAULT_PRUNING_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)

    try:
        percentile = float(cfg.get("percentile", DEFAULT_PRUNING_CONFIG["percentile"]))
    except (TypeError, ValueError):
        percentile = float(DEFAULT_PRUNING_CONFIG["percentile"])
    percentile = float(np.clip(percentile, 50.0, 99.9))

    def _to_int(key: str, default: int, *, min_value: int = 0) -> int:
        try:
            out = int(cfg.get(key, default))
        except (TypeError, ValueError):
            out = int(default)
        return max(out, int(min_value))

    return {
        "enabled": bool(cfg.get("enabled", DEFAULT_PRUNING_CONFIG["enabled"])),
        "percentile": float(percentile),
        "n_startup_trials": int(_to_int("n_startup_trials", int(DEFAULT_PRUNING_CONFIG["n_startup_trials"]), min_value=0)),
        "n_warmup_steps": int(_to_int("n_warmup_steps", int(DEFAULT_PRUNING_CONFIG["n_warmup_steps"]), min_value=1)),
        "interval_steps": int(_to_int("interval_steps", int(DEFAULT_PRUNING_CONFIG["interval_steps"]), min_value=1)),
        "min_completed_trials": int(_to_int("min_completed_trials", int(DEFAULT_PRUNING_CONFIG["min_completed_trials"]), min_value=1)),
    }


def _resolve_sampler_name(value: Any) -> str:
    raw = str(value or DEFAULT_HPO_SAMPLER).strip().lower()
    if raw in {"tpe", "cmaes"}:
        return raw
    return DEFAULT_HPO_SAMPLER


def _build_sampler(*, sampler_name: str, base_random_seed: int):
    resolved = _resolve_sampler_name(sampler_name)
    if resolved == "cmaes":
        try:
            return optuna.samplers.CmaEsSampler(seed=base_random_seed)
        except Exception as exc:
            print(f"[HPO] CmaEsSampler unavailable ({exc}). Fallback to TPESampler.")
    return optuna.samplers.TPESampler(seed=base_random_seed)


# =====================================================
# HPO Runner (MODELER Stage)
# =====================================================

class HPORunner:
    """
    MODELER-compliant HPO runner.

    - FixedKFoldSplitter
    - XGBoost only (current policy)
    - Robust objective (mean + lambda_std * std + lambda_gap * overfit_gap)
    - ResultSaver based persistence
    """

    def __init__(
        self,
        *,
        n_trials: int = 80,
        lambda_std: float = 0.5,
        lambda_gap: float = 0.0,
        use_timestamp: bool = True,
        show_optuna_log: bool = False,
        search_space: dict[str, Any] | None = None,
        hpo_mode: str = "default",
        pruning_config: dict[str, Any] | None = None,
        sampler_name: str = DEFAULT_HPO_SAMPLER,
    ):
        self.n_trials = n_trials
        self.lambda_std = lambda_std
        self.lambda_gap = float(max(float(lambda_gap), 0.0))
        self.show_optuna_log = bool(show_optuna_log)
        self.search_space = dict(search_space) if isinstance(search_space, dict) else None
        self.hpo_mode = str(hpo_mode)
        self.pruning_config = _resolve_pruning_config(pruning_config)
        self.sampler_name = _resolve_sampler_name(sampler_name)

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

        sampler = _build_sampler(
            sampler_name=self.sampler_name,
            base_random_seed=base_random_seed,
        )
        if bool(self.pruning_config["enabled"]):
            pruner = optuna.pruners.PercentilePruner(
                percentile=float(self.pruning_config["percentile"]),
                n_startup_trials=int(self.pruning_config["n_startup_trials"]),
                n_warmup_steps=int(self.pruning_config["n_warmup_steps"]),
                interval_steps=int(self.pruning_config["interval_steps"]),
                n_min_trials=int(self.pruning_config["min_completed_trials"]),
            )
        else:
            pruner = optuna.pruners.NopPruner()

        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

        objective = make_robust_objective(
            X=X,
            y=y,
            base_random_seed=base_random_seed,
            search_space_fn=make_xgb_search_space(search_space=self.search_space),
            lambda_std=self.lambda_std,
            lambda_gap=self.lambda_gap,
            kfold_splits=kfold_splits,
            pruning_enabled=bool(self.pruning_config["enabled"]),
            pruning_warmup_steps=int(self.pruning_config["n_warmup_steps"]),
        )

        study.optimize(objective, n_trials=self.n_trials)

        best_trial = study.best_trial

        result = format_hpo_result(
            best_trial=best_trial,
            base_random_seed=base_random_seed,
            n_trials=self.n_trials,
            lambda_std=self.lambda_std,
            lambda_gap=self.lambda_gap,
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
