from dataclasses import dataclass
from typing import Any

import numpy as np

from Modeler.executor.hpo_runner import HPORunner


# -------------------------------------------------
# XGB HPO param type spec
# -------------------------------------------------
XGB_PARAM_TYPES = {
    "n_estimators": int,
    "max_depth": int,
    "min_child_weight": float,
    "subsample": float,
    "colsample_bytree": float,
    "learning_rate": float,
    "gamma": float,
    "reg_alpha": float,
    "reg_lambda": float,
}

DEFAULT_HPO_N_TRIALS = 10
DEFAULT_HPO_LAMBDA_STD = 0.5
DEFAULT_HPO_LAMBDA_GAP = 0.2
DEFAULT_LOW_DATA_HPO_N_TRIALS = 10
DEFAULT_PRUNING_PERCENTILE = 85.0
DEFAULT_PRUNING_STARTUP_TRIALS = 12
DEFAULT_PRUNING_MIN_COMPLETED_TRIALS = 8
DEFAULT_PRUNING_INTERVAL_STEPS = 1
DEFAULT_MODELER_XGB_SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "n_estimators": (80, 200),
    "learning_rate": (0.01, 0.10),
    "max_depth": (2, 4),
    "min_child_weight": (3.0, 12.0),
    "subsample": (0.7, 0.9),
    "colsample_bytree": (0.6, 0.9),
    "gamma": (0.5, 2.0),
    "reg_alpha": (0.1, 10.0),
    "reg_lambda": (1.0, 30.0),
}
DEFAULT_MODELER_LOW_DATA_XGB_SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "n_estimators": (60, 160),
    "learning_rate": (0.02, 0.10),
    "max_depth": (2, 4),
    "min_child_weight": (3.0, 10.0),
    "subsample": (0.7, 0.9),
    "colsample_bytree": (0.6, 0.9),
    "gamma": (0.5, 2.0),
    "reg_alpha": (0.1, 8.0),
    "reg_lambda": (1.0, 20.0),
}
MODELER_HPO_GUARDRAILS: dict[str, dict[str, float]] = {
    "n_estimators": {"high_max": 200.0},
    "max_depth": {"high_max": 4.0},
    "min_child_weight": {"low_min": 3.0},
    "gamma": {"low_min": 0.5},
    "reg_alpha": {"low_min": 0.1},
    "reg_lambda": {"low_min": 1.0},
}


@dataclass
class HPOResolveResult:
    best_params: dict | None
    hpo_params_used: bool
    hpo_mode: str
    hpo_n_trials_effective: int | None
    hpo_lambda_std_effective: float | None
    hpo_lambda_gap_effective: float | None


def _safe_int(value: Any, *, default: int, min_value: int = 1) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = int(default)
    return max(out, int(min_value))


def _safe_float(value: Any, *, default: float, min_value: float | None = None) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = float(default)
    if min_value is not None:
        out = max(out, float(min_value))
    return out


def _canonical_search_space(space: Any) -> dict[str, list[float]] | None:
    if not isinstance(space, dict):
        return None
    out: dict[str, list[float]] = {}
    for key, value in space.items():
        if isinstance(value, (list, tuple)) and len(value) == 2:
            low_raw, high_raw = value
        elif isinstance(value, dict):
            low_raw = value.get("low")
            high_raw = value.get("high")
        else:
            continue
        try:
            low = float(low_raw)
            high = float(high_raw)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(low) or not np.isfinite(high):
            continue
        if high < low:
            low, high = high, low
        out[str(key)] = [float(low), float(high)]
    return out if out else None


def _merge_search_space(
    *,
    base_space: dict[str, tuple[float, float]],
    override_space: dict[str, list[float]] | None,
) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {
        str(k): [float(v[0]), float(v[1])]
        for k, v in base_space.items()
    }
    if isinstance(override_space, dict):
        for key, value in override_space.items():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                merged[str(key)] = [float(value[0]), float(value[1])]
    return merged


def _apply_modeler_hpo_guardrails(
    *,
    search_space: dict[str, list[float]],
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for key, value in search_space.items():
        if not (isinstance(value, (list, tuple)) and len(value) == 2):
            continue
        low = float(value[0])
        high = float(value[1])
        if not np.isfinite(low) or not np.isfinite(high):
            continue
        if high < low:
            low, high = high, low
        rule = MODELER_HPO_GUARDRAILS.get(str(key), {})
        if "high_max" in rule:
            high = min(high, float(rule["high_max"]))
            low = min(low, high)
        if "low_min" in rule:
            low = max(low, float(rule["low_min"]))
            high = max(high, low)
        out[str(key)] = [float(low), float(high)]
    return out


def _resolve_modeler_search_space(
    *,
    base_space: dict[str, tuple[float, float]],
    override_raw: Any,
) -> dict[str, list[float]]:
    override = _canonical_search_space(override_raw)
    merged = _merge_search_space(base_space=base_space, override_space=override)
    return _apply_modeler_hpo_guardrails(search_space=merged)


def _resolve_hpo_policy(
    *,
    hpo_config: dict | None,
    low_data: bool,
) -> tuple[str, int, float, float, dict[str, list[float]] | None]:
    cfg = hpo_config or {}
    constrained_enabled = bool(cfg.get("low_data_constrained_enabled", True))
    if bool(low_data) and constrained_enabled:
        mode = "low_data_constrained"
        n_trials = _safe_int(
            cfg.get("low_data_n_trials", DEFAULT_LOW_DATA_HPO_N_TRIALS),
            default=DEFAULT_LOW_DATA_HPO_N_TRIALS,
            min_value=1,
        )
        lambda_std = _safe_float(
            cfg.get("low_data_lambda_std", cfg.get("lambda_std", DEFAULT_HPO_LAMBDA_STD)),
            default=DEFAULT_HPO_LAMBDA_STD,
            min_value=0.0,
        )
        lambda_gap = _safe_float(
            cfg.get("low_data_lambda_gap", cfg.get("lambda_gap", DEFAULT_HPO_LAMBDA_GAP)),
            default=DEFAULT_HPO_LAMBDA_GAP,
            min_value=0.0,
        )
        search_space = _resolve_modeler_search_space(
            base_space=DEFAULT_MODELER_LOW_DATA_XGB_SEARCH_SPACE,
            override_raw=cfg.get("low_data_search_space"),
        )
    else:
        mode = "default"
        n_trials = _safe_int(
            cfg.get("n_trials", DEFAULT_HPO_N_TRIALS),
            default=DEFAULT_HPO_N_TRIALS,
            min_value=1,
        )
        lambda_std = _safe_float(
            cfg.get("lambda_std", DEFAULT_HPO_LAMBDA_STD),
            default=DEFAULT_HPO_LAMBDA_STD,
            min_value=0.0,
        )
        lambda_gap = _safe_float(
            cfg.get("lambda_gap", DEFAULT_HPO_LAMBDA_GAP),
            default=DEFAULT_HPO_LAMBDA_GAP,
            min_value=0.0,
        )
        search_space = _resolve_modeler_search_space(
            base_space=DEFAULT_MODELER_XGB_SEARCH_SPACE,
            override_raw=cfg.get("search_space"),
        )
    return mode, int(n_trials), float(lambda_std), float(lambda_gap), search_space


def _resolve_pruning_policy(
    *,
    hpo_config: dict | None,
    low_data: bool,
    n_trials: int,
    kfold_splits: int,
) -> dict[str, int | float | bool]:
    cfg = hpo_config or {}

    if bool(low_data):
        enabled = bool(cfg.get("low_data_pruning_enabled", False))
    else:
        enabled = bool(cfg.get("pruning_enabled", True))

    percentile = _safe_float(
        cfg.get("pruning_percentile", DEFAULT_PRUNING_PERCENTILE),
        default=DEFAULT_PRUNING_PERCENTILE,
        min_value=50.0,
    )
    percentile = float(min(percentile, 99.9))

    startup_default = min(int(DEFAULT_PRUNING_STARTUP_TRIALS), max(int(n_trials) - 1, 0))
    startup_trials = _safe_int(
        cfg.get("pruning_startup_trials", startup_default),
        default=startup_default,
        min_value=0,
    )

    min_completed_default = min(int(DEFAULT_PRUNING_MIN_COMPLETED_TRIALS), max(int(n_trials) - 1, 1))
    min_completed_trials = _safe_int(
        cfg.get("pruning_min_completed_trials", min_completed_default),
        default=min_completed_default,
        min_value=1,
    )

    warmup_default = max(int(kfold_splits) - 1, 1)
    warmup_steps = _safe_int(
        cfg.get("pruning_warmup_steps", warmup_default),
        default=warmup_default,
        min_value=1,
    )

    interval_steps = _safe_int(
        cfg.get("pruning_interval_steps", DEFAULT_PRUNING_INTERVAL_STEPS),
        default=DEFAULT_PRUNING_INTERVAL_STEPS,
        min_value=1,
    )

    return {
        "enabled": bool(enabled),
        "percentile": float(percentile),
        "n_startup_trials": int(startup_trials),
        "n_warmup_steps": int(warmup_steps),
        "interval_steps": int(interval_steps),
        "min_completed_trials": int(min_completed_trials),
    }


def _resolve_sampler_policy(*, hpo_config: dict | None) -> str:
    cfg = hpo_config or {}
    raw = str(cfg.get("sampler", "tpe")).strip().lower()
    if raw in {"tpe", "cmaes"}:
        return raw
    return "tpe"


def resolve_hpo_params(
    *,
    use_hpo: bool,
    model_name: str,
    hpo_config: dict | None,
    use_timestamp: bool,
    project_root: str,
    problem_name: str,
    objective_sense: str,
    target_col: str,
    X: np.ndarray,
    y: np.ndarray,
    base_seed: int,
    kfold_splits: int,
    low_data: bool,
) -> HPOResolveResult:
    best_params = None
    hpo_params_used = False
    hpo_mode = "disabled"
    hpo_n_trials_effective: int | None = None
    hpo_lambda_std_effective: float | None = None
    hpo_lambda_gap_effective: float | None = None

    if model_name == "xgb":
        (
            hpo_mode,
            hpo_n_trials_effective,
            hpo_lambda_std_effective,
            hpo_lambda_gap_effective,
            hpo_search_space,
        ) = _resolve_hpo_policy(
            hpo_config=hpo_config,
            low_data=bool(low_data),
        )
        hpo_pruning_policy = _resolve_pruning_policy(
            hpo_config=hpo_config,
            low_data=bool(low_data),
            n_trials=int(hpo_n_trials_effective),
            kfold_splits=int(kfold_splits),
        )
        hpo_sampler_name = _resolve_sampler_policy(hpo_config=hpo_config)
    else:
        hpo_search_space = None
        hpo_pruning_policy = {"enabled": False}
        hpo_sampler_name = "tpe"

    if use_hpo:
        hpo_runner = HPORunner(
            n_trials=int(hpo_n_trials_effective or DEFAULT_HPO_N_TRIALS),
            lambda_std=(
                float(hpo_lambda_std_effective)
                if hpo_lambda_std_effective is not None
                else float(DEFAULT_HPO_LAMBDA_STD)
            ),
            lambda_gap=(
                float(hpo_lambda_gap_effective)
                if hpo_lambda_gap_effective is not None
                else float(DEFAULT_HPO_LAMBDA_GAP)
            ),
            use_timestamp=use_timestamp,
            search_space=hpo_search_space,
            hpo_mode=hpo_mode,
            pruning_config=hpo_pruning_policy,
            sampler_name=hpo_sampler_name,
        )
        print(
            "- HPO policy: "
            f"mode={hpo_mode} "
            f"n_trials={hpo_n_trials_effective} "
            f"lambda_std={hpo_lambda_std_effective} "
            f"lambda_gap={hpo_lambda_gap_effective} "
            f"sampler={hpo_sampler_name}"
        )

        hpo_result = hpo_runner.run_xgb(
            X=X,
            y=y,
            base_random_seed=base_seed,
            problem_name=problem_name,
            kfold_splits=kfold_splits,
        )

        best_params = hpo_result["best_params"]
        hpo_params_used = True
        print("- HPO executed")

    elif model_name != "xgb":
        hpo_mode = "disabled_non_xgb"

    return HPOResolveResult(
        best_params=best_params,
        hpo_params_used=hpo_params_used,
        hpo_mode=hpo_mode,
        hpo_n_trials_effective=hpo_n_trials_effective,
        hpo_lambda_std_effective=hpo_lambda_std_effective,
        hpo_lambda_gap_effective=hpo_lambda_gap_effective,
    )
