from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np
import pandas as pd

from DOE.doe_algorithm.lhs import latin_hypercube_sampling
from DOE.executor.anchor_refiner import AcquisitionOptimizer, fit_gp_with_fallback
from DOE.executor.constraint_filter import evaluate_constraints_batch
from utils.boundary_sampling import sample_boundary_corners_random
from Optimizer.algorithms.result import OptimizerAlgorithmResult
from Optimizer.config import OptimizerSystemConfig
from Optimizer.algorithms.focus_bo.focus0_sampler import build_focus0_initial_doe_points
from Optimizer.algorithms.focus_bo.focus_pipeline import FocusPipelinePlan, build_focus_pipeline_plan
from Optimizer.executor.archive_core import append_archive_point, update_raw_and_effective_best
from Optimizer.executor.evaluation_core import (
    CaeObjectiveEvaluator,
    build_pre_constraint_debug_fields,
    evaluate_pre_constraints_raw,
    objective_best_index,
    objective_initial_best,
    objective_is_better,
)
from Optimizer.executor.goal_monitor import GoalMonitor
from Optimizer.executor.history_core import update_goal_and_build_evaluation_row
from Optimizer.executor.result_core import build_optimizer_algorithm_result


def _is_better(*, y_new: float, y_best: float, objective_sense: str) -> bool:
    return objective_is_better(
        y_new=y_new,
        y_best=y_best,
        objective_sense=objective_sense,
    )


def _point_in_bounds(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> bool:
    return bool(np.all(x >= lb) and np.all(x <= ub))


def _round_key(x: np.ndarray, decimals: int) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(x, dtype=float).reshape(-1), decimals=decimals).tolist())


class _BoundsScaledGPModel:
    """Raw-X facade for a GP fitted in bounds-normalized X space."""

    def __init__(self, model: object, *, lb: np.ndarray, ub: np.ndarray, span_floor: float):
        self._model = model
        self._lb = np.asarray(lb, dtype=float).reshape(-1)
        raw_span = np.asarray(ub, dtype=float).reshape(-1) - self._lb
        floor = float(max(float(span_floor), 1e-15))
        self._span = np.where(np.abs(raw_span) > floor, raw_span, floor)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        was_1d = arr.ndim == 1
        arr2 = arr.reshape(1, -1) if was_1d else arr
        return (arr2 - self._lb.reshape(1, -1)) / self._span.reshape(1, -1)

    def predict(self, X: np.ndarray, return_std: bool = False, return_cov: bool = False):
        X_scaled = self._scale(X)
        return self._model.predict(X_scaled, return_std=return_std, return_cov=return_cov)

    def __getattr__(self, name: str):
        return getattr(self._model, name)


def _scale_X_by_bounds(
    *,
    X: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    span_floor: float,
) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    raw_span = np.asarray(ub, dtype=float).reshape(-1) - np.asarray(lb, dtype=float).reshape(-1)
    floor = float(max(float(span_floor), 1e-15))
    span = np.where(np.abs(raw_span) > floor, raw_span, floor)
    return (arr - np.asarray(lb, dtype=float).reshape(1, -1)) / span.reshape(1, -1)


def _gp_x_scaling_policy(
    *,
    system: OptimizerSystemConfig,
    lb: np.ndarray,
    ub: np.ndarray,
) -> dict[str, object]:
    raw_span = np.asarray(ub, dtype=float).reshape(-1) - np.asarray(lb, dtype=float).reshape(-1)
    finite = raw_span[np.isfinite(raw_span)]
    positive = np.abs(finite[finite != 0.0])
    if positive.size == 0:
        span_min = float("nan")
        span_max = float("nan")
        span_ratio = float("nan")
    else:
        span_min = float(np.min(positive))
        span_max = float(np.max(positive))
        span_ratio = float(span_max / max(span_min, 1e-15))

    requested_enabled = bool(getattr(system, "gp_x_scaling_enabled", True))
    mode = str(getattr(system, "gp_x_scaling_mode", "auto")).strip().lower()
    threshold = float(max(float(getattr(system, "gp_x_scaling_auto_span_ratio_threshold", 3.0)), 1.0))
    applied = False
    reason = "disabled"
    if requested_enabled:
        if mode in {"bounds", "bound", "minmax", "on", "true"}:
            applied = True
            reason = "forced_bounds"
        elif mode in {"auto", ""}:
            applied = bool(math.isfinite(span_ratio) and span_ratio >= threshold)
            reason = "auto_span_ratio_pass" if applied else "auto_span_ratio_below_threshold"
        elif mode in {"off", "none", "raw", "false"}:
            reason = "raw_mode"
        else:
            reason = "unsupported_mode"

    return {
        "gp_x_scaling_enabled": bool(requested_enabled),
        "gp_x_scaling_applied": bool(applied),
        "gp_x_scaling_mode": str(mode or "auto"),
        "gp_x_scaling_reason": str(reason),
        "gp_x_scaling_auto_span_ratio_threshold": float(threshold),
        "gp_x_scaling_span_min": span_min,
        "gp_x_scaling_span_max": span_max,
        "gp_x_scaling_span_ratio": span_ratio,
    }


def _fit_optimizer_gp_with_fallback(
    *,
    X: np.ndarray,
    y: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    system: OptimizerSystemConfig,
    include_white: bool = False,
    random_state: int = 0,
) -> tuple[object | None, bool]:
    """Fit an Optimizer GP, optionally in active-bounds normalized coordinates."""
    policy = _gp_x_scaling_policy(system=system, lb=lb, ub=ub)
    span_floor = float(max(float(getattr(system, "gp_x_scaling_span_floor", 1e-12)), 1e-15))
    if not bool(policy.get("gp_x_scaling_applied", False)):
        return fit_gp_with_fallback(
            X=X,
            y=y,
            include_white=include_white,
            random_state=random_state,
        )

    X_scaled = _scale_X_by_bounds(X=X, lb=lb, ub=ub, span_floor=span_floor)
    model, fallback = fit_gp_with_fallback(
        X=X_scaled,
        y=y,
        include_white=include_white,
        random_state=random_state,
    )
    if model is None:
        return None, bool(fallback)
    return _BoundsScaledGPModel(model, lb=lb, ub=ub, span_floor=span_floor), bool(fallback)


def _gp_x_scaling_debug(
    *,
    system: OptimizerSystemConfig,
    lb: np.ndarray,
    ub: np.ndarray,
) -> dict[str, object]:
    return _gp_x_scaling_policy(system=system, lb=lb, ub=ub)


def _pre_constraint_debug_fields(
    *,
    x: np.ndarray,
    var_names: list[str],
    constraint_defs: list[dict],
    enforce_pre_constraints: bool,
) -> dict[str, object]:
    return build_pre_constraint_debug_fields(
        x=x,
        var_names=var_names,
        constraint_defs=constraint_defs,
        enforce_pre_constraints=enforce_pre_constraints,
    )


def _resolve_acq_policy(
    *,
    system: OptimizerSystemConfig,
    iteration: int,
    n_iterations: int,
) -> tuple[str, float]:
    progress = float(iteration) / float(max(n_iterations - 1, 1))
    acq_raw = str(system.acq_type or "auto").strip().upper()
    if acq_raw == "AUTO":
        acq_type = "LCB" if progress < 0.6 else "EI"
    elif acq_raw in {"LCB", "EI"}:
        acq_type = acq_raw
    else:
        acq_type = "LCB"
    kappa = float(system.kappa_start) + (float(system.kappa_end) - float(system.kappa_start)) * progress
    return acq_type, float(kappa)


def _build_starts(
    *,
    rng: np.random.Generator,
    archive_X: np.ndarray,
    archive_y: np.ndarray,
    objective_sense: str,
    lb: np.ndarray,
    ub: np.ndarray,
    starts_per_iter: int,
    random_ratio: float,
) -> np.ndarray:
    n_dim = int(lb.shape[0])
    n_total = max(int(starts_per_iter), 1)
    n_rand = int(round(float(n_total) * float(np.clip(random_ratio, 0.0, 1.0))))
    n_rand = min(max(n_rand, 0), n_total)
    n_archive = n_total - n_rand

    starts: list[np.ndarray] = []
    if archive_X.ndim == 2 and archive_X.shape[0] > 0 and n_archive > 0:
        if objective_sense == "max":
            order = np.argsort(-archive_y)
        else:
            order = np.argsort(archive_y)
        top_idx = order[: min(n_archive, order.shape[0])]
        for idx in top_idx:
            starts.append(np.clip(archive_X[idx].reshape(-1), lb, ub))

    for _ in range(n_rand):
        starts.append(rng.uniform(lb, ub, size=(n_dim,)))

    if not starts:
        starts.append(rng.uniform(lb, ub, size=(n_dim,)))

    return np.asarray(starts, dtype=float)


def _normalize_source_probs(*, topk: float, boundary: float, random_p: float) -> tuple[float, float, float]:
    vals = np.asarray([topk, boundary, random_p], dtype=float)
    vals = np.where(np.isfinite(vals), vals, 0.0)
    vals = np.clip(vals, 0.0, None)
    total = float(np.sum(vals))
    if total <= 0.0:
        return 1.0, 0.0, 0.0
    out = vals / total
    return float(out[0]), float(out[1]), float(out[2])


def _normalize_focus3_source_probs4(
    *,
    topk: float,
    boundary: float,
    random_p: float,
    best_local: float,
) -> tuple[float, float, float, float]:
    vals = np.asarray([topk, boundary, random_p, best_local], dtype=float)
    vals = np.where(np.isfinite(vals), vals, 0.0)
    vals = np.clip(vals, 0.0, None)
    total = float(np.sum(vals))
    if total <= 0.0:
        return 1.0, 0.0, 0.0, 0.0
    out = vals / total
    return float(out[0]), float(out[1]), float(out[2]), float(out[3])


def _resolve_focus3_budget_class(
    *,
    n_samples: int,
    p_dim: int,
    system: OptimizerSystemConfig,
) -> tuple[str, float]:
    ratio = float(max(int(n_samples), 0)) / float(max(int(p_dim), 1))
    ultra = float(getattr(system, "focus3_budget_ultra_low_np", 3.0))
    low = float(getattr(system, "focus3_budget_low_np", 8.0))
    normal = float(getattr(system, "focus3_budget_normal_np", 20.0))
    if ratio < ultra:
        return "ultra_low", ratio
    if ratio < low:
        return "low", ratio
    if ratio < normal:
        return "normal", ratio
    return "rich", ratio


def _resolve_focus3_source_probs(
    *,
    system: OptimizerSystemConfig,
    budget_class: str,
) -> tuple[float, float, float]:
    if not bool(getattr(system, "focus3_budget_policy_enabled", True)):
        return _normalize_source_probs(
            topk=float(getattr(system, "source_topk_prob", 0.60)),
            boundary=float(getattr(system, "source_boundary_prob", 0.25)),
            random_p=float(getattr(system, "source_random_prob", 0.15)),
        )

    cls = str(budget_class or "normal").strip().lower()
    prefix = {
        "ultra_low": "focus3_ultra_low",
        "low": "focus3_low",
        "normal": "focus3_normal",
        "rich": "focus3_rich",
    }.get(cls, "focus3_normal")
    return _normalize_source_probs(
        topk=float(getattr(system, f"{prefix}_source_topk_prob")),
        boundary=float(getattr(system, f"{prefix}_source_boundary_prob")),
        random_p=float(getattr(system, f"{prefix}_source_random_prob")),
    )


def _normalize_focus3_acq_type(value: str | None) -> str:
    acq = str(value or "auto").strip().upper()
    if acq not in {"AUTO", "LCB", "EI", "MEAN"}:
        return "AUTO"
    return acq


def _bounds_ratio_summary(
    *,
    lb: np.ndarray,
    ub: np.ndarray,
    full_lb: np.ndarray,
    full_ub: np.ndarray,
) -> tuple[float, float]:
    denom = np.maximum(np.asarray(full_ub, dtype=float) - np.asarray(full_lb, dtype=float), 1e-12)
    widths = np.clip((np.asarray(ub, dtype=float) - np.asarray(lb, dtype=float)) / denom, 0.0, 1.0)
    finite = widths[np.isfinite(widths)]
    if finite.size == 0:
        return float("nan"), float("nan")
    mean_width_ratio = float(np.mean(finite))
    log_volume = float(np.sum(np.log(np.maximum(finite, 1e-300))))
    volume_ratio = float(0.0 if log_volume < -700.0 else math.exp(log_volume))
    return volume_ratio, mean_width_ratio


def _normalized_rms_distance_to_set(
    *,
    x: np.ndarray,
    X_ref: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> float:
    x_arr = np.asarray(x, dtype=float).reshape(1, -1)
    X = np.asarray(X_ref, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        return float("inf")
    span = np.maximum(np.asarray(ub, dtype=float) - np.asarray(lb, dtype=float), 1e-12)
    diff = (X - x_arr) / span.reshape(1, -1)
    dist = np.sqrt(np.mean(diff * diff, axis=1))
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return float("inf")
    return float(np.min(finite))


def _resolve_focus3_effective_acq_type(
    *,
    requested: str,
    system: OptimizerSystemConfig,
    p_dim: int,
    n_train: int,
    gp_fallback_used: bool,
    lb: np.ndarray,
    ub: np.ndarray,
    full_lb: np.ndarray,
    full_ub: np.ndarray,
    best_local_active: bool = False,
) -> tuple[str, dict[str, object]]:
    req = _normalize_focus3_acq_type(requested)
    data_ratio = float(max(int(n_train), 0)) / float(max(int(p_dim), 1))
    volume_ratio, mean_width_ratio = _bounds_ratio_summary(lb=lb, ub=ub, full_lb=full_lb, full_ub=full_ub)
    min_data_ratio = float(max(float(getattr(system, "focus3_auto_ei_min_data_ratio", 15.0)), 0.0))
    max_volume_ratio = float(np.clip(float(getattr(system, "focus3_auto_ei_max_volume_ratio", 0.25)), 0.0, 1.0))
    max_mean_width_ratio = float(np.clip(float(getattr(system, "focus3_auto_ei_max_mean_width_ratio", 0.75)), 0.0, 1.0))
    mean_enabled = bool(getattr(system, "focus3_auto_mean_enabled", True))
    mean_min_data_ratio = float(max(float(getattr(system, "focus3_auto_mean_min_data_ratio", 20.0)), 0.0))
    mean_max_volume_ratio = float(np.clip(float(getattr(system, "focus3_auto_mean_max_volume_ratio", 0.20)), 0.0, 1.0))
    mean_requires_best_local = bool(getattr(system, "focus3_auto_mean_best_local_required", True))

    if req in {"LCB", "EI", "MEAN"}:
        return req, {
            "requested": req,
            "effective": req,
            "reason": "explicit",
            "data_ratio": data_ratio,
            "volume_ratio": volume_ratio,
            "mean_width_ratio": mean_width_ratio,
        }

    if bool(gp_fallback_used):
        reason = "gp_fallback"
        effective = "LCB"
    elif data_ratio < min_data_ratio:
        reason = "low_data_ratio"
        effective = "LCB"
    elif math.isfinite(volume_ratio) and volume_ratio > max_volume_ratio:
        reason = "wide_volume"
        effective = "LCB"
    elif math.isfinite(mean_width_ratio) and mean_width_ratio > max_mean_width_ratio:
        reason = "wide_mean_width"
        effective = "LCB"
    elif (
        mean_enabled
        and data_ratio >= mean_min_data_ratio
        and (not mean_requires_best_local or bool(best_local_active))
        and math.isfinite(volume_ratio)
        and volume_ratio <= mean_max_volume_ratio
    ):
        reason = "best_local_reliable_exploitation"
        effective = "MEAN"
    else:
        reason = "reliable_focus_region"
        effective = "EI"

    return effective, {
        "requested": req,
        "effective": effective,
        "reason": reason,
        "data_ratio": data_ratio,
        "volume_ratio": volume_ratio,
        "mean_width_ratio": mean_width_ratio,
        "min_data_ratio": min_data_ratio,
        "max_volume_ratio": max_volume_ratio,
        "max_mean_width_ratio": max_mean_width_ratio,
        "mean_enabled": bool(mean_enabled),
        "mean_min_data_ratio": float(mean_min_data_ratio),
        "mean_max_volume_ratio": float(mean_max_volume_ratio),
        "best_local_active": bool(best_local_active),
    }


def _adapt_focus3_source_probs(
    *,
    system: OptimizerSystemConfig,
    p_topk: float,
    p_boundary: float,
    p_random: float,
    p_dim: int,
    n_train: int,
    gp_fallback_used: bool,
    best_raw_history: list[float],
    objective_sense: str,
) -> tuple[float, float, float, dict[str, object]]:
    if not bool(getattr(system, "focus3_source_adaptive_enabled", True)):
        p_topk, p_boundary, p_random = _normalize_source_probs(
            topk=p_topk,
            boundary=p_boundary,
            random_p=p_random,
        )
        return p_topk, p_boundary, p_random, {
            "enabled": False,
            "data_ratio": float(max(int(n_train), 0)) / float(max(int(p_dim), 1)),
            "reliability": "disabled",
            "recent_state": "disabled",
            "recent_improvement": float("nan"),
        }

    data_ratio = float(max(int(n_train), 0)) / float(max(int(p_dim), 1))
    low = float(getattr(system, "focus3_data_ratio_low", 5.0))
    good = float(getattr(system, "focus3_data_ratio_good", 15.0))
    reliability = "high"

    if data_ratio < low:
        reliability = "low"
        p_boundary += float(max(float(getattr(system, "focus3_low_reliability_boundary_bonus", 0.05)), 0.0))
        p_random += float(max(float(getattr(system, "focus3_low_reliability_random_bonus", 0.15)), 0.0))
    elif data_ratio < good:
        reliability = "medium"
        p_random += float(max(float(getattr(system, "focus3_mid_reliability_random_bonus", 0.05)), 0.0))

    if bool(gp_fallback_used):
        p_boundary += float(max(float(getattr(system, "focus3_gp_fallback_boundary_bonus", 0.05)), 0.0))
        p_random += float(max(float(getattr(system, "focus3_gp_fallback_random_bonus", 0.10)), 0.0))

    win = int(max(int(getattr(system, "source_stagnation_window", 8)), 2))
    tol_abs = float(max(float(getattr(system, "source_stagnation_tol", 1e-8)), 0.0))
    tol_effective = float(tol_abs)
    recent_improvement = float("nan")
    recent_state = "insufficient"
    if len(best_raw_history) >= win:
        ref_idx = max(len(best_raw_history) - win, 0)
        if objective_sense == "max":
            recent_improvement = float(best_raw_history[-1] - best_raw_history[ref_idx])
        else:
            recent_improvement = float(best_raw_history[ref_idx] - best_raw_history[-1])
        scale = float(max(abs(float(best_raw_history[ref_idx])), abs(float(best_raw_history[-1])), 1.0))
        tol = float(max(
            tol_abs,
            float(max(float(getattr(system, "focus3_recover_relative_tol", 0.0)), 0.0)) * scale,
        ))
        tol_effective = float(tol)
        if recent_improvement > tol:
            recent_state = "improving"
            p_topk += float(max(float(getattr(system, "focus3_recent_improvement_topk_bonus", 0.10)), 0.0))
        else:
            recent_state = "stagnated"
            p_boundary += float(max(float(getattr(system, "source_stagnation_boundary_bonus", 0.10)), 0.0))
            p_random += float(max(float(getattr(system, "source_stagnation_random_bonus", 0.05)), 0.0))

    p_topk, p_boundary, p_random = _normalize_source_probs(
        topk=p_topk,
        boundary=p_boundary,
        random_p=p_random,
    )
    return p_topk, p_boundary, p_random, {
        "enabled": True,
        "data_ratio": float(data_ratio),
        "reliability": str(reliability),
        "gp_fallback_used": bool(gp_fallback_used),
        "recent_state": str(recent_state),
        "recent_improvement": float(recent_improvement),
        "recent_improvement_tol": float(tol_effective),
    }


def _apply_focus3_recover_policy(
    *,
    system: OptimizerSystemConfig,
    p_topk: float,
    p_boundary: float,
    p_random: float,
    kappa: float,
    best_raw_history: list[float],
    objective_sense: str,
    has_constraints: bool,
) -> tuple[float, float, float, float, dict[str, object]]:
    info: dict[str, object] = {
        "enabled": bool(getattr(system, "focus3_recover_enabled", True)),
        "active": False,
        "reason": "disabled",
        "level": "none",
        "window": int(max(int(getattr(system, "focus3_recover_window", 5)), 2)),
        "recent_improvement": float("nan"),
        "no_improve_count": 0,
        "kappa_before": float(kappa),
        "kappa_after": float(kappa),
    }
    if not bool(info["enabled"]):
        return p_topk, p_boundary, p_random, float(kappa), info

    win = int(info["window"])
    min_history = int(max(int(getattr(system, "focus3_recover_min_history", win)), win))
    info["min_history"] = int(min_history)
    if len(best_raw_history) < min_history:
        info["reason"] = "insufficient_history"
        return p_topk, p_boundary, p_random, float(kappa), info

    ref_idx = max(len(best_raw_history) - win, 0)
    if objective_sense == "max":
        recent_improvement = float(best_raw_history[-1] - best_raw_history[ref_idx])
    else:
        recent_improvement = float(best_raw_history[ref_idx] - best_raw_history[-1])
    info["recent_improvement"] = float(recent_improvement)

    tol_abs = float(max(float(getattr(system, "focus3_recover_tol", 1e-8)), 0.0))
    tol_rel = float(max(float(getattr(system, "focus3_recover_relative_tol", 0.0)), 0.0))
    ref_scale = float(max(abs(float(best_raw_history[ref_idx])), abs(float(best_raw_history[-1])), 1.0))
    tol = float(max(tol_abs, tol_rel * ref_scale))
    info["tol_abs"] = float(tol_abs)
    info["tol_rel"] = float(tol_rel)
    info["tol_effective"] = float(tol)
    if recent_improvement > tol:
        info["reason"] = "improving"
        return p_topk, p_boundary, p_random, float(kappa), info

    no_improve_count = 0
    for idx in range(len(best_raw_history) - 1, 0, -1):
        if objective_sense == "max":
            step_improvement = float(best_raw_history[idx] - best_raw_history[idx - 1])
        else:
            step_improvement = float(best_raw_history[idx - 1] - best_raw_history[idx])
        if step_improvement > tol:
            break
        no_improve_count += 1
    strong_after = int(max(int(getattr(system, "focus3_recover_strong_no_improve", 20)), win))
    boundary_bonus = float(max(float(getattr(system, "focus3_recover_boundary_bonus", 0.15)), 0.0))
    random_bonus = float(max(float(getattr(system, "focus3_recover_random_bonus", 0.10)), 0.0))
    mult = float(max(float(getattr(system, "focus3_recover_kappa_multiplier", 1.50)), 1.0))
    level = "strong"
    if no_improve_count < strong_after:
        level = "mild"
        boundary_bonus *= float(np.clip(float(getattr(system, "focus3_recover_mild_boundary_scale", 0.35)), 0.0, 1.0))
        random_bonus *= float(np.clip(float(getattr(system, "focus3_recover_mild_random_scale", 0.50)), 0.0, 1.0))
        mult = float(max(float(getattr(system, "focus3_recover_mild_kappa_multiplier", 1.15)), 1.0))

    topk_bonus = 0.0
    if not bool(has_constraints):
        boundary_scale = float(np.clip(float(getattr(system, "focus3_no_constraint_recover_boundary_scale", 0.0)), 0.0, 1.0))
        random_scale = float(max(float(getattr(system, "focus3_no_constraint_recover_random_scale", 1.20)), 0.0))
        topk_bonus = float(max(float(getattr(system, "focus3_no_constraint_recover_topk_bonus", 0.05)), 0.0))
        boundary_bonus *= boundary_scale
        random_bonus *= random_scale

    p_topk += topk_bonus
    p_boundary += boundary_bonus
    p_random += random_bonus
    p_topk, p_boundary, p_random = _normalize_source_probs(
        topk=p_topk,
        boundary=p_boundary,
        random_p=p_random,
    )
    max_kappa = float(max(float(getattr(system, "focus3_recover_max_kappa", 3.50)), 0.0))
    kappa_after = float(kappa) * mult
    if max_kappa > 0.0:
        kappa_after = float(min(kappa_after, max_kappa))
    info.update({
        "active": True,
        "reason": "stagnated",
        "level": str(level),
        "no_improve_count": int(no_improve_count),
        "has_constraints": bool(has_constraints),
        "topk_bonus": float(topk_bonus),
        "boundary_bonus": float(boundary_bonus),
        "random_bonus": float(random_bonus),
        "kappa_after": float(kappa_after),
    })
    return p_topk, p_boundary, p_random, float(kappa_after), info


def _apply_focus3_no_constraint_source_cap(
    *,
    system: OptimizerSystemConfig,
    p_topk: float,
    p_boundary: float,
    p_random: float,
    constraint_defs: list[dict],
) -> tuple[float, float, float, dict[str, object]]:
    enabled = bool(getattr(system, "focus3_no_constraint_source_cap_enabled", True))
    has_constraints = bool(constraint_defs)
    p_topk, p_boundary, p_random = _normalize_source_probs(
        topk=p_topk,
        boundary=p_boundary,
        random_p=p_random,
    )
    info: dict[str, object] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled" if not enabled else "",
        "has_constraints": bool(has_constraints),
        "before": {
            "topk": float(p_topk),
            "boundary": float(p_boundary),
            "random": float(p_random),
        },
    }
    if not enabled:
        info["after"] = dict(info["before"])
        return p_topk, p_boundary, p_random, info
    if has_constraints:
        info["reason"] = "constraints_present"
        info["after"] = dict(info["before"])
        return p_topk, p_boundary, p_random, info

    boundary_max = float(np.clip(float(getattr(system, "focus3_no_constraint_boundary_max", 0.18)), 0.0, 1.0))
    topk_min = float(np.clip(float(getattr(system, "focus3_no_constraint_topk_min", 0.55)), 0.0, 1.0))
    random_min = float(np.clip(float(getattr(system, "focus3_no_constraint_random_min", 0.10)), 0.0, 1.0))
    if topk_min + random_min > 1.0:
        scale = 1.0 / max(topk_min + random_min, 1e-12)
        topk_min *= scale
        random_min *= scale

    old = (float(p_topk), float(p_boundary), float(p_random))
    p_boundary = min(float(p_boundary), boundary_max)
    remaining = max(1.0 - p_boundary, 0.0)
    non_boundary = max(float(old[0] + old[2]), 1e-12)
    topk_share = float(old[0]) / non_boundary
    p_topk = remaining * topk_share
    p_random = remaining - p_topk

    if p_topk < topk_min:
        need = topk_min - p_topk
        take = min(need, p_boundary)
        p_boundary -= take
        p_topk += take
        need -= take
        if need > 0.0:
            take = min(need, max(p_random - random_min, 0.0))
            p_random -= take
            p_topk += take

    if p_random < random_min:
        need = random_min - p_random
        take = min(need, p_boundary)
        p_boundary -= take
        p_random += take
        need -= take
        if need > 0.0:
            take = min(need, max(p_topk - topk_min, 0.0))
            p_topk -= take
            p_random += take

    p_topk, p_boundary, p_random = _normalize_source_probs(
        topk=p_topk,
        boundary=p_boundary,
        random_p=p_random,
    )
    after = {
        "topk": float(p_topk),
        "boundary": float(p_boundary),
        "random": float(p_random),
    }
    applied = any(abs(after[k] - float(info["before"][k])) > 1e-12 for k in after)
    info.update({
        "applied": bool(applied),
        "reason": "no_constraint_cap" if applied else "already_within_policy",
        "boundary_max": float(boundary_max),
        "topk_min": float(topk_min),
        "random_min": float(random_min),
        "after": after,
    })
    return p_topk, p_boundary, p_random, info


def _apply_focus3_best_local_policy(
    *,
    system: OptimizerSystemConfig,
    p_topk: float,
    p_boundary: float,
    p_random: float,
    p_dim: int,
    n_train: int,
    focus3_eval_count: int,
    has_constraints: bool,
) -> tuple[float, float, float, float, dict[str, object]]:
    enabled = bool(getattr(system, "focus3_best_local_enabled", True))
    p_topk, p_boundary, p_random = _normalize_source_probs(
        topk=p_topk,
        boundary=p_boundary,
        random_p=p_random,
    )
    info: dict[str, object] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled" if not enabled else "",
        "best_local_prob": 0.0,
        "has_constraints": bool(has_constraints),
        "focus3_eval_count": int(focus3_eval_count),
        "data_ratio": float(max(int(n_train), 0)) / float(max(int(p_dim), 1)),
    }
    if not enabled:
        return p_topk, p_boundary, p_random, 0.0, info
    if bool(has_constraints):
        info["reason"] = "constraints_present"
        return p_topk, p_boundary, p_random, 0.0, info

    min_evals = int(max(int(getattr(system, "focus3_best_local_min_focus3_evals", 5)), 0))
    if int(focus3_eval_count) < min_evals:
        info["reason"] = "insufficient_focus3_history"
        return p_topk, p_boundary, p_random, 0.0, info

    data_ratio = float(info["data_ratio"])
    min_data_ratio = float(max(float(getattr(system, "focus3_best_local_min_data_ratio", 5.0)), 0.0))
    if data_ratio < min_data_ratio:
        info["reason"] = "insufficient_data_ratio"
        return p_topk, p_boundary, p_random, 0.0, info

    requested = float(np.clip(float(getattr(system, "focus3_best_local_prob", 0.15)), 0.0, 1.0))
    max_prob = float(np.clip(float(getattr(system, "focus3_best_local_max_prob", 0.25)), 0.0, 1.0))
    p_best = float(min(requested, max_prob, p_topk))
    if p_best <= 0.0:
        info["reason"] = "no_topk_quota_available"
        return p_topk, p_boundary, p_random, 0.0, info
    p_topk -= p_best
    p_topk, p_boundary, p_random, p_best = _normalize_focus3_source_probs4(
        topk=p_topk,
        boundary=p_boundary,
        random_p=p_random,
        best_local=p_best,
    )
    info.update({
        "applied": True,
        "reason": "best_local_from_topk",
        "best_local_prob": float(p_best),
        "requested_prob": float(requested),
        "max_prob": float(max_prob),
        "min_focus3_evals": int(min_evals),
        "min_data_ratio": float(min_data_ratio),
    })
    return p_topk, p_boundary, p_random, p_best, info


def _apply_focus3_fallback_bounds_source_policy(
    *,
    system: OptimizerSystemConfig,
    p_topk: float,
    p_boundary: float,
    p_random: float,
    p_best_local: float,
    fallback_bounds: bool,
) -> tuple[float, float, float, float, dict[str, object]]:
    enabled = bool(getattr(system, "focus3_fallback_bounds_source_policy_enabled", True))
    p_topk, p_boundary, p_random, p_best_local = _normalize_focus3_source_probs4(
        topk=p_topk,
        boundary=p_boundary,
        random_p=p_random,
        best_local=p_best_local,
    )
    info: dict[str, object] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled" if not enabled else "",
        "fallback_bounds": bool(fallback_bounds),
        "before": {
            "topk": float(p_topk),
            "boundary": float(p_boundary),
            "random": float(p_random),
            "best_local": float(p_best_local),
        },
    }
    if not enabled:
        info["after"] = dict(info["before"])
        return p_topk, p_boundary, p_random, p_best_local, info
    if not bool(fallback_bounds):
        info["reason"] = "not_fallback_bounds"
        info["after"] = dict(info["before"])
        return p_topk, p_boundary, p_random, p_best_local, info

    limits = {
        "boundary": float(np.clip(float(getattr(system, "focus3_fallback_bounds_boundary_max", 0.05)), 0.0, 1.0)),
        "topk": float(np.clip(float(getattr(system, "focus3_fallback_bounds_topk_min", 0.45)), 0.0, 1.0)),
        "best_local": float(np.clip(float(getattr(system, "focus3_fallback_bounds_best_local_min", 0.20)), 0.0, 1.0)),
        "random": float(np.clip(float(getattr(system, "focus3_fallback_bounds_random_min", 0.20)), 0.0, 1.0)),
    }
    min_sum = limits["topk"] + limits["best_local"] + limits["random"]
    if min_sum + limits["boundary"] > 1.0:
        scale = max((1.0 - limits["boundary"]) / max(min_sum, 1e-12), 0.0)
        limits["topk"] *= scale
        limits["best_local"] *= scale
        limits["random"] *= scale

    probs = {
        "topk": float(p_topk),
        "boundary": float(min(p_boundary, limits["boundary"])),
        "random": float(p_random),
        "best_local": float(p_best_local),
    }
    # Boundary에서 덜어낸 quota를 우선 부족한 topk/best_local/random floor에 채운다.
    leftover = max(0.0, 1.0 - sum(probs.values()))
    for key in ("topk", "best_local", "random"):
        need = max(0.0, limits[key] - probs[key])
        take = min(need, leftover)
        probs[key] += take
        leftover -= take
    if leftover > 0.0:
        weights = np.asarray([0.45, 0.25, 0.30], dtype=float)
        weights = weights / float(np.sum(weights))
        for key, w in zip(("topk", "best_local", "random"), weights):
            probs[key] += leftover * float(w)

    p_topk, p_boundary, p_random, p_best_local = _normalize_focus3_source_probs4(
        topk=probs["topk"],
        boundary=probs["boundary"],
        random_p=probs["random"],
        best_local=probs["best_local"],
    )
    after = {
        "topk": float(p_topk),
        "boundary": float(p_boundary),
        "random": float(p_random),
        "best_local": float(p_best_local),
    }
    applied = any(abs(after[k] - float(info["before"][k])) > 1e-12 for k in after)
    info.update({
        "applied": bool(applied),
        "reason": "fallback_bounds_policy" if applied else "already_within_policy",
        "limits": limits,
        "after": after,
    })
    return p_topk, p_boundary, p_random, p_best_local, info


def _apply_focus3_recover_best_local_policy(
    *,
    system: OptimizerSystemConfig,
    p_topk: float,
    p_boundary: float,
    p_random: float,
    p_best_local: float,
    recover_info: dict[str, object],
    has_constraints: bool,
) -> tuple[float, float, float, float, dict[str, object]]:
    p_topk, p_boundary, p_random, p_best_local = _normalize_focus3_source_probs4(
        topk=p_topk,
        boundary=p_boundary,
        random_p=p_random,
        best_local=p_best_local,
    )
    info: dict[str, object] = {
        "enabled": True,
        "applied": False,
        "reason": "not_recovering",
        "before": {
            "topk": float(p_topk),
            "boundary": float(p_boundary),
            "random": float(p_random),
            "best_local": float(p_best_local),
        },
    }
    if bool(has_constraints):
        info["reason"] = "constraints_present"
        info["after"] = dict(info["before"])
        return p_topk, p_boundary, p_random, p_best_local, info
    if not bool(recover_info.get("active", False)):
        info["after"] = dict(info["before"])
        return p_topk, p_boundary, p_random, p_best_local, info
    if p_best_local <= 0.0:
        info["reason"] = "best_local_inactive"
        info["after"] = dict(info["before"])
        return p_topk, p_boundary, p_random, p_best_local, info

    level = str(recover_info.get("level", "mild")).strip().lower()
    if level == "strong":
        bonus = float(max(float(getattr(system, "focus3_no_constraint_recover_best_local_strong_bonus", 0.15)), 0.0))
    else:
        bonus = float(max(float(getattr(system, "focus3_no_constraint_recover_best_local_mild_bonus", 0.05)), 0.0))
    max_best = float(np.clip(float(getattr(system, "focus3_no_constraint_recover_best_local_max", 0.45)), 0.0, 1.0))
    target = float(min(max_best, p_best_local + bonus))
    need = float(max(target - p_best_local, 0.0))
    if need <= 0.0:
        info["reason"] = "already_at_or_above_recover_target"
        info["after"] = dict(info["before"])
        return p_topk, p_boundary, p_random, p_best_local, info

    # Recovering in unconstrained Focus3 should intensify locally. Take quota
    # first from boundary/random exploration, then from top-k if needed.
    take_random = min(need, p_random)
    p_random -= take_random
    p_best_local += take_random
    need -= take_random
    take_boundary = min(need, p_boundary)
    p_boundary -= take_boundary
    p_best_local += take_boundary
    need -= take_boundary
    take_topk = min(need, max(p_topk, 0.0))
    p_topk -= take_topk
    p_best_local += take_topk

    p_topk, p_boundary, p_random, p_best_local = _normalize_focus3_source_probs4(
        topk=p_topk,
        boundary=p_boundary,
        random_p=p_random,
        best_local=p_best_local,
    )
    after = {
        "topk": float(p_topk),
        "boundary": float(p_boundary),
        "random": float(p_random),
        "best_local": float(p_best_local),
    }
    info.update({
        "applied": True,
        "reason": f"recover_{level}_best_local",
        "level": str(level),
        "bonus": float(bonus),
        "max_best_local": float(max_best),
        "after": after,
    })
    return p_topk, p_boundary, p_random, p_best_local, info


def _resolve_focus3_best_local_sigma(
    *,
    system: OptimizerSystemConfig,
    focus3_eval_count: int,
    recover_info: dict[str, object] | None,
) -> tuple[float, dict[str, object]]:
    base = float(max(float(getattr(system, "focus3_best_local_sigma", 0.015)), 1e-9))
    enabled = bool(getattr(system, "focus3_best_local_sigma_schedule_enabled", True))
    sigma = float(base)
    reason = "base"
    if enabled:
        mid_eval = int(max(int(getattr(system, "focus3_best_local_sigma_mid_eval", 80)), 0))
        late_eval = int(max(int(getattr(system, "focus3_best_local_sigma_late_eval", 250)), mid_eval))
        mid_sigma = float(max(float(getattr(system, "focus3_best_local_sigma_mid", base)), 1e-9))
        late_sigma = float(max(float(getattr(system, "focus3_best_local_sigma_late", mid_sigma)), 1e-9))
        if int(focus3_eval_count) >= late_eval:
            sigma = late_sigma
            reason = "late_focus3"
        elif int(focus3_eval_count) >= mid_eval:
            sigma = mid_sigma
            reason = "mid_focus3"
    if recover_info and str(recover_info.get("level", "")).strip().lower() == "strong":
        mult = float(np.clip(float(getattr(system, "focus3_best_local_sigma_recover_strong_multiplier", 0.75)), 0.05, 1.0))
        sigma = float(max(sigma * mult, 1e-9))
        reason = str(reason) + "+strong_recover"
    return sigma, {
        "enabled": bool(enabled),
        "base": float(base),
        "effective": float(sigma),
        "reason": str(reason),
        "focus3_eval_count": int(focus3_eval_count),
    }


def _resolve_focus3_refine_cooldown(
    *,
    system: OptimizerSystemConfig,
    history_rows: list[dict],
    focus3_eval_count: int,
) -> dict[str, object]:
    enabled = bool(getattr(system, "focus3_refine_cooldown_enabled", True))
    window = int(max(int(getattr(system, "focus3_refine_cooldown_window", 50)), 1))
    threshold = int(max(int(getattr(system, "focus3_refine_cooldown_worse_count", 12)), 1))
    min_evals = int(max(int(getattr(system, "focus3_refine_cooldown_min_focus3_evals", 80)), 0))
    info: dict[str, object] = {
        "enabled": bool(enabled),
        "active": False,
        "reason": "disabled" if not enabled else "",
        "window": int(window),
        "worse_count_threshold": int(threshold),
        "min_focus3_evals": int(min_evals),
        "recent_worse_count": 0,
    }
    if not enabled:
        return info
    if int(focus3_eval_count) < min_evals:
        info["reason"] = "insufficient_focus3_evals"
        return info
    rows = [r for r in history_rows if str(r.get("segment", "")) == "focus3"]
    recent = rows[-window:]
    worse_count = int(sum(1 for r in recent if str(r.get("focus3_refine_adaptive_reason", "")) == "refine_recently_worse"))
    info["recent_worse_count"] = int(worse_count)
    if worse_count >= threshold:
        info["active"] = True
        info["reason"] = "repeated_refine_recently_worse"
    else:
        info["reason"] = "below_worse_threshold"
    return info


def _resolve_focus3_refine_every_adaptive(
    *,
    system: OptimizerSystemConfig,
    history_rows: list[dict],
    default_every: int,
) -> tuple[int, dict[str, object]]:
    base = int(max(int(default_every), 1))
    info: dict[str, object] = {
        "enabled": bool(getattr(system, "focus3_refine_adaptive_enabled", True)),
        "base_every": int(base),
        "effective_every": int(base),
        "reason": "disabled",
        "window": int(max(int(getattr(system, "focus3_refine_adaptive_window", 80)), 1)),
        "discrete_count": 0,
        "refine_count": 0,
        "discrete_improve_rate": float("nan"),
        "refine_improve_rate": float("nan"),
    }
    if not bool(info["enabled"]):
        return base, info

    focus3_rows = [r for r in history_rows if str(r.get("segment", "")) == "focus3"]
    window = int(info["window"])
    recent = focus3_rows[-window:] if window > 0 else focus3_rows
    min_samples = int(max(int(getattr(system, "focus3_refine_adaptive_min_samples", 8)), 1))
    if len(recent) < min_samples:
        info["reason"] = "insufficient_recent_history"
        return base, info

    discrete_rows = [r for r in recent if str(r.get("focus3_plan_mode", "")).strip().lower() == "discrete"]
    refine_rows = [
        r
        for r in recent
        if str(r.get("focus3_plan_mode", "")).strip().lower() in {"refine", "refine_skipped"}
    ]

    def _rate(rows: list[dict]) -> float:
        if not rows:
            return float("nan")
        return float(sum(1 for r in rows if bool(r.get("raw_improved", False))) / float(len(rows)))

    disc_rate = _rate(discrete_rows)
    ref_rate = _rate(refine_rows)
    info.update({
        "discrete_count": int(len(discrete_rows)),
        "refine_count": int(len(refine_rows)),
        "discrete_improve_rate": float(disc_rate),
        "refine_improve_rate": float(ref_rate),
    })

    if len(discrete_rows) < min_samples or len(refine_rows) < min_samples:
        info["reason"] = "insufficient_mode_samples"
        return base, info

    min_every = int(max(int(getattr(system, "focus3_refine_adaptive_min_every", 1)), 1))
    max_every = int(max(int(getattr(system, "focus3_refine_adaptive_max_every", 4)), min_every))
    better_every = int(np.clip(
        int(getattr(system, "focus3_refine_adaptive_better_every", min_every)),
        min_every,
        max_every,
    ))
    worse_every = int(np.clip(
        int(getattr(system, "focus3_refine_adaptive_worse_every", max_every)),
        min_every,
        max_every,
    ))
    advantage = float(max(float(getattr(system, "focus3_refine_adaptive_advantage_ratio", 1.50)), 1.0))
    disadvantage = float(np.clip(float(getattr(system, "focus3_refine_adaptive_disadvantage_ratio", 0.50)), 0.0, 1.0))

    effective = base
    if math.isfinite(ref_rate) and math.isfinite(disc_rate):
        if ref_rate > 0.0 and ref_rate >= max(disc_rate, 1e-12) * advantage:
            effective = better_every
            info["reason"] = "refine_recently_better"
        elif disc_rate > 0.0 and ref_rate <= disc_rate * disadvantage:
            effective = worse_every
            info["reason"] = "refine_recently_worse"
        else:
            info["reason"] = "balanced"
    else:
        info["reason"] = "nonfinite_rates"
    info["effective_every"] = int(effective)
    info["better_every"] = int(better_every)
    info["worse_every"] = int(worse_every)
    return int(effective), info


def _focus3_no_constraint_boundary_gate(
    *,
    system: OptimizerSystemConfig,
    best_scores: dict[str, float],
    has_constraints: bool,
) -> dict[str, object]:
    enabled = bool(getattr(system, "focus3_no_constraint_boundary_gate_enabled", True))
    info: dict[str, object] = {
        "enabled": bool(enabled),
        "applied": False,
        "allowed": True,
        "reason": "disabled" if not enabled else "",
        "margin": float("nan"),
        "boundary_score": float(best_scores.get("boundary", float("nan"))),
        "non_boundary_best": float("nan"),
    }
    if not enabled:
        return info
    if has_constraints:
        info["reason"] = "constraints_present"
        return info
    b_score = float(best_scores.get("boundary", float("nan")))
    non_boundary = [
        float(best_scores.get(src, float("nan")))
        for src in ("topk", "random", "best_local")
        if math.isfinite(float(best_scores.get(src, float("nan"))))
    ]
    if not math.isfinite(b_score):
        info.update({"allowed": False, "applied": True, "reason": "boundary_score_nan"})
        return info
    if not non_boundary:
        info["reason"] = "no_non_boundary_candidate"
        return info
    nb_best = float(min(non_boundary))
    margin_ratio = float(max(float(getattr(system, "focus3_no_constraint_boundary_gate_margin_ratio", 0.02)), 0.0))
    margin = float(max(abs(nb_best), abs(b_score), 1.0) * margin_ratio)
    allowed = bool(b_score <= nb_best - margin)
    info.update({
        "allowed": bool(allowed),
        "applied": bool(not allowed),
        "reason": "boundary_not_sufficiently_better" if not allowed else "boundary_sufficiently_better",
        "margin": float(margin),
        "boundary_score": float(b_score),
        "non_boundary_best": float(nb_best),
    })
    return info


def _apply_focus3_boundary_score_penalty(
    *,
    system: OptimizerSystemConfig,
    source: str,
    scores: np.ndarray,
    has_constraints: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    enabled = bool(getattr(system, "focus3_no_constraint_boundary_score_penalty_enabled", True))
    arr = np.asarray(scores, dtype=float).copy()
    info: dict[str, object] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled" if not enabled else "",
        "source": str(source),
        "penalty": 0.0,
        "ratio": float(getattr(system, "focus3_no_constraint_boundary_score_penalty_ratio", 0.0)),
        "scale": float("nan"),
    }
    if not enabled:
        return arr, info
    if bool(has_constraints):
        info["reason"] = "constraints_present"
        return arr, info
    if str(source) != "boundary":
        info["reason"] = "not_boundary"
        return arr, info
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        info["reason"] = "no_finite_scores"
        return arr, info
    ratio = float(max(float(getattr(system, "focus3_no_constraint_boundary_score_penalty_ratio", 0.25)), 0.0))
    scale = float(max(abs(float(np.nanmedian(finite))), abs(float(np.nanmin(finite))), 1.0))
    penalty = float(ratio * scale)
    if penalty <= 0.0:
        info["reason"] = "zero_penalty"
        return arr, info
    arr[np.isfinite(arr)] = arr[np.isfinite(arr)] + penalty
    info.update({
        "applied": True,
        "reason": "no_constraint_boundary_penalty",
        "penalty": float(penalty),
        "ratio": float(ratio),
        "scale": float(scale),
    })
    return arr, info


def _select_focus3_dedup_fallback(
    *,
    rng: np.random.Generator,
    candidate_pool: np.ndarray,
    model: object | None,
    y_best: float,
    objective_sense: str,
    acq_type: str,
    kappa: float,
    xi: float,
    lb: np.ndarray,
    ub: np.ndarray,
    X_archive: np.ndarray,
    min_rms_distance: float,
    post_feasible_prob_fn: Callable[[np.ndarray], float] | None,
    post_penalty_lambda: float,
    random_attempts: int,
) -> tuple[np.ndarray | None, float, str]:
    min_dist = float(max(float(min_rms_distance), 0.0))
    pool = np.asarray(candidate_pool, dtype=float)
    if pool.ndim == 2 and pool.shape[0] > 0:
        scores = _acquisition_scores_for_pool(
            pool=pool,
            model=model,
            y_best=y_best,
            objective_sense=objective_sense,
            acq_type=acq_type,
            kappa=kappa,
            xi=xi,
            post_feasible_prob_fn=post_feasible_prob_fn,
            post_penalty_lambda=post_penalty_lambda,
        )
        score_arr = np.asarray(scores, dtype=float).reshape(-1)
        score_arr = np.where(np.isfinite(score_arr), score_arr, float("inf"))
        for idx in np.argsort(score_arr).tolist():
            x = np.clip(pool[int(idx)].reshape(-1), lb, ub)
            dist = _normalized_rms_min_distance(x=x, X_ref=X_archive, lb=lb, ub=ub)
            if dist >= min_dist:
                return x.copy(), float(dist), "refine_pool"

    for _ in range(int(max(random_attempts, 0))):
        x = rng.uniform(lb, ub, size=(lb.shape[0],))
        dist = _normalized_rms_min_distance(x=x, X_ref=X_archive, lb=lb, ub=ub)
        if dist >= min_dist:
            return x.copy(), float(dist), "random"
    return None, float("nan"), "none"


def _select_warmstart_top_diversity(
    *,
    X: np.ndarray,
    y: np.ndarray,
    objective_sense: str,
    max_points: int,
    topk_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] == 0 or y.shape[0] != X.shape[0]:
        return X, y
    n = int(X.shape[0])
    k = int(max(1, min(max_points, n)))
    if n <= k:
        return X, y

    if objective_sense == "max":
        order = np.argsort(-y)
    else:
        order = np.argsort(y)
    n_top = int(max(1, round(float(k) * float(np.clip(topk_fraction, 0.0, 1.0)))))
    n_top = min(n_top, k, n)
    top_idx = order[:n_top].tolist()

    if n_top >= k:
        keep = np.asarray(top_idx[:k], dtype=int)
        return X[keep], y[keep]

    span = np.maximum(np.max(X, axis=0) - np.min(X, axis=0), 1e-12)
    selected: list[int] = list(top_idx)
    remaining: list[int] = [int(i) for i in order.tolist() if int(i) not in set(selected)]
    while len(selected) < k and remaining:
        best_i = remaining[0]
        best_score = -1.0
        for cand in remaining[: min(len(remaining), 200)]:
            d = np.linalg.norm((X[cand].reshape(1, -1) - X[selected]) / span.reshape(1, -1), axis=1)
            score = float(np.min(d))
            if score > best_score:
                best_score = score
                best_i = int(cand)
        selected.append(best_i)
        remaining = [idx for idx in remaining if idx != best_i]

    keep = np.asarray(selected[:k], dtype=int)
    return X[keep], y[keep]


def _resolve_gp_train_cap(
    *,
    system: OptimizerSystemConfig,
    p_dim: int,
    n_available: int,
) -> int:
    if int(n_available) <= 0:
        return 0
    p = int(max(p_dim, 1))
    ratio = float(max(float(getattr(system, "init_train_np_ratio", 20.0)), 1.0))
    ratio_cap = int(max(1, math.ceil(ratio * p)))
    abs_cap = int(max(int(getattr(system, "init_max_points", 500)), 1))
    return int(max(1, min(int(n_available), ratio_cap, abs_cap)))


def _select_gp_train_from_archive(
    *,
    X: np.ndarray,
    y: np.ndarray,
    objective_sense: str,
    system: OptimizerSystemConfig,
    p_dim: int,
    initial_count: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] == 0 or y.shape[0] != X.shape[0]:
        return X, y, int(X.shape[0]) if X.ndim == 2 else 0

    n = int(X.shape[0])
    cap = _resolve_gp_train_cap(system=system, p_dim=p_dim, n_available=n)
    if n <= cap:
        return X.copy(), y.copy(), int(cap)

    if objective_sense == "max":
        order = np.argsort(-y)
    else:
        order = np.argsort(y)

    selected: list[int] = []
    selected_set: set[int] = set()

    def _add(idx: int) -> bool:
        i = int(idx)
        if i in selected_set or len(selected) >= cap:
            return False
        selected.append(i)
        selected_set.add(i)
        return True

    recent_fraction = float(np.clip(float(getattr(system, "gp_train_recent_fraction", 0.25)), 0.0, 1.0))
    recent_quota = int(round(float(cap) * recent_fraction))
    recent_start = int(min(max(initial_count, 0), n))
    if recent_quota > 0 and recent_start < n:
        for idx in list(range(recent_start, n))[-recent_quota:]:
            _add(idx)

    top_fraction = float(np.clip(float(getattr(system, "init_train_top_fraction", 0.70)), 0.0, 1.0))
    top_quota = int(round(float(cap) * top_fraction))
    top_quota = int(max(top_quota, min(int(getattr(system, "init_from_doe_topk", 20)), cap)))
    top_added = 0
    for idx in order.tolist():
        if top_added >= top_quota or len(selected) >= cap:
            break
        if _add(int(idx)):
            top_added += 1

    if not selected:
        _add(int(order[0]))

    if len(selected) < cap:
        span = np.maximum(np.max(X, axis=0) - np.min(X, axis=0), 1e-12)
        remaining = [int(i) for i in order.tolist() if int(i) not in selected_set]
        while len(selected) < cap and remaining:
            selected_arr = np.asarray(selected, dtype=int)
            best_i = remaining[0]
            best_score = -1.0
            for cand in remaining[: min(len(remaining), 300)]:
                d = np.linalg.norm(
                    (X[int(cand)].reshape(1, -1) - X[selected_arr]) / span.reshape(1, -1),
                    axis=1,
                )
                score = float(np.min(d))
                if score > best_score:
                    best_score = score
                    best_i = int(cand)
            _add(best_i)
            remaining = [idx for idx in remaining if idx != best_i]

    keep = np.asarray(selected[:cap], dtype=int)
    return X[keep].copy(), y[keep].copy(), int(cap)


def _bounds_volume_ratio(
    *,
    lb: np.ndarray,
    ub: np.ndarray,
    base_lb: np.ndarray,
    base_ub: np.ndarray,
) -> float:
    width = np.maximum(np.asarray(ub, dtype=float) - np.asarray(lb, dtype=float), 1e-12)
    base_width = np.maximum(np.asarray(base_ub, dtype=float) - np.asarray(base_lb, dtype=float), 1e-12)
    ratios = np.clip(width / base_width, 1e-12, 1.0)
    log_ratio = float(np.sum(np.log(ratios)))
    return float(np.clip(np.exp(log_ratio), 0.0, 1.0))


def _centered_bounds_from_width(
    *,
    center: np.ndarray,
    width: np.ndarray,
    full_lb: np.ndarray,
    full_ub: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    full_width = np.maximum(full_ub - full_lb, 1e-12)
    w = np.minimum(np.maximum(width, 1e-12), full_width)
    lo = np.asarray(center, dtype=float) - 0.5 * w
    hi = np.asarray(center, dtype=float) + 0.5 * w
    below = full_lb - lo
    mask = below > 0.0
    lo[mask] += below[mask]
    hi[mask] += below[mask]
    above = hi - full_ub
    mask = above > 0.0
    lo[mask] -= above[mask]
    hi[mask] -= above[mask]
    lo = np.maximum(lo, full_lb)
    hi = np.minimum(hi, full_ub)
    return lo, hi


def _enforce_min_volume_ratio(
    *,
    focus_lb: np.ndarray,
    focus_ub: np.ndarray,
    full_lb: np.ndarray,
    full_ub: np.ndarray,
    min_volume_ratio: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    floor = float(np.clip(float(min_volume_ratio), 0.0, 1.0))
    if floor <= 0.0:
        ratio = _bounds_volume_ratio(lb=focus_lb, ub=focus_ub, base_lb=full_lb, base_ub=full_ub)
        return focus_lb, focus_ub, ratio

    ratio = _bounds_volume_ratio(lb=focus_lb, ub=focus_ub, base_lb=full_lb, base_ub=full_ub)
    if ratio >= floor:
        return focus_lb, focus_ub, ratio

    p = int(max(focus_lb.shape[0], 1))
    center = 0.5 * (focus_lb + focus_ub)
    width = np.maximum(focus_ub - focus_lb, 1e-12)
    scale = float((floor / max(ratio, 1e-12)) ** (1.0 / float(p)))
    lo, hi = _centered_bounds_from_width(
        center=center,
        width=width * scale,
        full_lb=full_lb,
        full_ub=full_ub,
    )
    ratio2 = _bounds_volume_ratio(lb=lo, ub=hi, base_lb=full_lb, base_ub=full_ub)
    if ratio2 >= floor:
        return lo, hi, ratio2

    axis_floor = float(floor ** (1.0 / float(p)))
    full_width = np.maximum(full_ub - full_lb, 1e-12)
    lo, hi = _centered_bounds_from_width(
        center=center,
        width=np.maximum(hi - lo, full_width * axis_floor),
        full_lb=full_lb,
        full_ub=full_ub,
    )
    ratio3 = _bounds_volume_ratio(lb=lo, ub=hi, base_lb=full_lb, base_ub=full_ub)
    return lo, hi, ratio3


def _resolve_focus2_region_pool_size(
    *,
    system: OptimizerSystemConfig,
    p_dim: int,
) -> int:
    p = int(max(p_dim, 1))
    min_pool = int(max(int(getattr(system, "focus2_region_candidate_pool_min", 4096)), 1))
    per_dim = int(max(int(getattr(system, "focus2_region_candidate_pool_per_dim", 200)), 1))
    max_pool = int(max(int(getattr(system, "focus2_region_candidate_pool_max", 30000)), min_pool))
    return int(min(max(per_dim * p, min_pool), max_pool))


def _build_focus2_regions_from_archive(
    *,
    rng: np.random.Generator,
    model: object | None,
    X_archive: np.ndarray,
    y_archive: np.ndarray,
    objective_sense: str,
    lb: np.ndarray,
    ub: np.ndarray,
    selected_features: list[str],
    system: OptimizerSystemConfig,
    var_names: list[str],
    constraint_defs: list[dict],
    enforce_pre_constraints: bool,
) -> dict[str, object]:
    p = int(lb.shape[0])
    min_archive = int(max(int(getattr(system, "focus2_region_min_archive_points", 5)), 1))
    min_good_candidates = int(max(int(getattr(system, "focus2_region_min_good_candidates", 10)), 1))
    min_volume = float(np.clip(float(getattr(system, "focus2_region_min_volume_ratio", 0.05)), 0.0, 1.0))
    if model is None:
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "missing_gp_model",
            "regions": [],
            "min_archive_points": int(min_archive),
            "min_volume_ratio": float(min_volume),
        }
    if X_archive.ndim != 2 or y_archive.ndim != 1 or X_archive.shape[0] < min_archive or y_archive.shape[0] != X_archive.shape[0]:
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "insufficient_archive",
            "regions": [],
            "archive_count": int(X_archive.shape[0]) if X_archive.ndim == 2 else 0,
            "min_archive_points": int(min_archive),
            "min_volume_ratio": float(min_volume),
        }

    pool_size = _resolve_focus2_region_pool_size(system=system, p_dim=p)
    pool = latin_hypercube_sampling(
        n_samples=pool_size,
        bounds=[(float(lo), float(hi)) for lo, hi in zip(lb, ub)],
        rng=rng,
        n_divisions=max(pool_size, 1),
    )
    if enforce_pre_constraints and constraint_defs:
        mask, _payloads, _margins = evaluate_constraints_batch(
            X=pool,
            var_names=var_names,
            constraint_defs=constraint_defs,
            scope="pre",
        )
        pool = pool[mask]
    if pool.shape[0] == 0:
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "empty_candidate_pool",
            "regions": [],
            "archive_count": int(X_archive.shape[0]),
            "min_archive_points": int(min_archive),
            "min_volume_ratio": float(min_volume),
        }

    mu, sigma = model.predict(pool, return_std=True)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    sigma = np.maximum(np.asarray(sigma, dtype=float).reshape(-1), 0.0)
    beta = float(max(float(getattr(system, "focus1_beta", 1.0)), 0.0))
    tau = _focus1_tau(
        y=y_archive,
        objective_sense=objective_sense,
        quantile=float(getattr(system, "focus1_tau_quantile", 0.20)),
    )
    lower = mu - beta * sigma
    upper = mu + beta * sigma
    if objective_sense == "max":
        good_mask = lower >= tau
        bad_mask = upper < tau
        uncertain_mask = ~(good_mask | bad_mask)
        quality_score = -upper
        straddle_score = beta * sigma - np.abs(mu - tau)
    else:
        good_mask = upper <= tau
        bad_mask = lower > tau
        uncertain_mask = ~(good_mask | bad_mask)
        quality_score = lower
        straddle_score = beta * sigma - np.abs(mu - tau)

    span = np.maximum(ub - lb, 1e-12)
    pool_n = (pool - lb.reshape(1, -1)) / span.reshape(1, -1)
    max_regions = int(max(int(getattr(system, "focus2_region_max_regions", 5)), 1))
    qlo = float(np.clip(float(getattr(system, "focus2_region_quantile_low", 0.10)), 0.0, 0.49))
    qhi = float(np.clip(float(getattr(system, "focus2_region_quantile_high", 0.90)), qlo + 0.01, 1.0))

    def _collect_regions(
        *,
        uncertain_seed_fraction: float,
        radius: float,
        min_good: int,
        expand_ratio: float,
        pass_name: str,
    ) -> list[dict[str, object]]:
        good_idx = np.where(good_mask)[0]
        uncertain_idx = np.where(uncertain_mask)[0]
        n_uncertain_seed = int(round(float(uncertain_idx.size) * float(np.clip(uncertain_seed_fraction, 0.0, 1.0))))
        if n_uncertain_seed > 0 and uncertain_idx.size > 0:
            uncertain_order = uncertain_idx[np.argsort(-straddle_score[uncertain_idx])]
            uncertain_seed_idx = uncertain_order[:n_uncertain_seed]
        else:
            uncertain_seed_idx = np.empty((0,), dtype=int)
        seed_idx = np.unique(np.concatenate([good_idx, uncertain_seed_idx])).astype(int)
        if seed_idx.size == 0:
            return []

        seed_score = quality_score[seed_idx]
        seed_order = seed_idx[np.argsort(seed_score)]
        assigned = np.zeros((pool.shape[0],), dtype=bool)
        out: list[dict[str, object]] = []
        for center_idx in seed_order.tolist():
            if len(out) >= max_regions:
                break
            if assigned[int(center_idx)]:
                continue
            d_seed = np.linalg.norm(pool_n[seed_idx] - pool_n[int(center_idx)].reshape(1, -1), axis=1) / math.sqrt(float(p))
            cluster_seed_idx = seed_idx[d_seed <= radius]
            if cluster_seed_idx.size < min_good:
                continue

            raw_lb = np.quantile(pool[cluster_seed_idx], qlo, axis=0)
            raw_ub = np.quantile(pool[cluster_seed_idx], qhi, axis=0)
            center = 0.5 * (raw_lb + raw_ub)
            width = np.maximum(raw_ub - raw_lb, 1e-12) * expand_ratio
            region_lb, region_ub = _centered_bounds_from_width(
                center=center,
                width=width,
                full_lb=lb,
                full_ub=ub,
            )
            region_lb, region_ub, volume_ratio = _enforce_min_volume_ratio(
                focus_lb=region_lb,
                focus_ub=region_ub,
                full_lb=lb,
                full_ub=ub,
                min_volume_ratio=min_volume,
            )

            cand_inside = np.all((pool >= region_lb.reshape(1, -1)) & (pool <= region_ub.reshape(1, -1)), axis=1)
            archive_inside = np.all((X_archive >= region_lb.reshape(1, -1)) & (X_archive <= region_ub.reshape(1, -1)), axis=1)
            archive_count = int(np.sum(archive_inside))
            good_count = int(np.sum(cand_inside & good_mask))
            uncertain_count = int(np.sum(cand_inside & uncertain_mask))
            bad_count = int(np.sum(cand_inside & bad_mask))
            if archive_count < min_archive:
                assigned[cluster_seed_idx] = True
                continue
            if good_count < min_good:
                assigned[cluster_seed_idx] = True
                continue
            if volume_ratio < min_volume:
                assigned[cluster_seed_idx] = True
                continue

            y_inside = y_archive[archive_inside]
            archive_best = float(np.max(y_inside) if objective_sense == "max" else np.min(y_inside)) if y_inside.size else float("nan")
            total_cand = max(int(np.sum(cand_inside)), 1)
            good_density = float(good_count) / float(total_cand)
            bad_density = float(bad_count) / float(total_cand)
            archive_support = min(float(archive_count) / float(max(min_archive * 2, 1)), 1.0)
            score = float(good_density + 0.25 * archive_support - 0.5 * bad_density)
            confidence = float(np.clip(0.5 * good_density + 0.3 * archive_support + 0.2 * (1.0 - bad_density), 0.0, 1.0))

            region_id = f"region_{len(out) + 1:03d}"
            out.append(
                {
                    "region_id": region_id,
                    "rank": 0,
                    "generation_pass": str(pass_name),
                    "bounds": {
                        str(name): [float(lo), float(hi)]
                        for name, lo, hi in zip(selected_features, region_lb, region_ub)
                    },
                    "center": {
                        str(name): float(v)
                        for name, v in zip(selected_features, 0.5 * (region_lb + region_ub))
                    },
                    "score": float(score),
                    "confidence": float(confidence),
                    "size_ratio": float(volume_ratio),
                    "candidate_counts": {
                        "good": int(good_count),
                        "uncertain": int(uncertain_count),
                        "bad": int(bad_count),
                        "total": int(np.sum(cand_inside)),
                    },
                    "archive_counts": {
                        "total": int(archive_count),
                        "min_required": int(min_archive),
                    },
                    "archive_best_objective": float(archive_best),
                }
            )
            assigned[cand_inside] = True
        return out

    base_uncertain = float(np.clip(float(getattr(system, "focus2_region_uncertain_seed_fraction", 0.30)), 0.0, 1.0))
    base_radius = float(max(float(getattr(system, "focus2_region_cluster_radius", 0.20)), 1e-6))
    base_expand = float(max(float(getattr(system, "focus2_region_expand_ratio", 1.15)), 1.0))
    regions = _collect_regions(
        uncertain_seed_fraction=base_uncertain,
        radius=base_radius,
        min_good=min_good_candidates,
        expand_ratio=base_expand,
        pass_name="primary",
    )
    retry_used = False
    if not regions and bool(getattr(system, "focus2_region_retry_enabled", True)):
        retry_used = True
        regions = _collect_regions(
            uncertain_seed_fraction=float(getattr(system, "focus2_region_retry_uncertain_seed_fraction", 0.60)),
            radius=base_radius * float(max(float(getattr(system, "focus2_region_retry_radius_scale", 1.50)), 1.0)),
            min_good=max(1, int(math.ceil(float(min_good_candidates) * float(np.clip(float(getattr(system, "focus2_region_retry_min_good_scale", 0.50)), 0.1, 1.0))))),
            expand_ratio=float(max(float(getattr(system, "focus2_region_retry_expand_ratio", 1.30)), base_expand)),
            pass_name="retry_relaxed",
        )

    regions.sort(key=lambda r: (float(r.get("score", 0.0)), float(r.get("confidence", 0.0))), reverse=True)
    for rank, region in enumerate(regions, start=1):
        region["rank"] = int(rank)

    return {
        "enabled": True,
        "source_focus": "focus1",
        "status": "ok" if regions else "empty",
        "reason": "" if regions else "no_region_passed_filters",
        "retry_used": bool(retry_used),
        "tau": float(tau),
        "beta": float(beta),
        "candidate_pool_count": int(pool.shape[0]),
        "candidate_counts": {
            "good": int(np.sum(good_mask)),
            "uncertain": int(np.sum(uncertain_mask)),
            "bad": int(np.sum(bad_mask)),
        },
        "region_filters": {
            "min_archive_points": int(min_archive),
            "min_good_candidates": int(min_good_candidates),
            "min_volume_ratio": float(min_volume),
        },
        "regions": regions,
    }


def _build_focus2_fallback_region_from_archive(
    *,
    X_archive: np.ndarray,
    y_archive: np.ndarray,
    objective_sense: str,
    lb: np.ndarray,
    ub: np.ndarray,
    selected_features: list[str],
    system: OptimizerSystemConfig,
) -> dict[str, object]:
    """Create one conservative Focus2 region directly from evaluated archive points.

    This is used when the global-GP candidate classifier cannot form a region,
    but the optimizer still has enough real CAE/objective data to seed Focus3.
    """
    p = int(lb.shape[0])
    min_archive = int(max(int(getattr(system, "focus2_region_min_archive_points", 5)), 1))
    min_volume = float(np.clip(float(getattr(system, "focus2_region_min_volume_ratio", 0.05)), 0.0, 1.0))
    max_volume = float(np.clip(float(getattr(system, "focus2_active_bounds_max_volume_ratio", 0.25)), 1e-12, 1.0))
    if X_archive.ndim != 2 or y_archive.ndim != 1 or X_archive.shape[1] != p:
        return {
            "enabled": True,
            "source_focus": "focus1",
            "status": "skipped",
            "reason": "fallback_invalid_archive_shape",
            "regions": [],
            "min_archive_points": int(min_archive),
        }
    finite_mask = np.isfinite(y_archive) & np.all(np.isfinite(X_archive), axis=1)
    if int(np.sum(finite_mask)) < min_archive:
        return {
            "enabled": True,
            "source_focus": "focus1",
            "status": "skipped",
            "reason": "fallback_insufficient_finite_archive",
            "archive_count": int(np.sum(finite_mask)),
            "regions": [],
            "min_archive_points": int(min_archive),
        }

    Xf = np.asarray(X_archive[finite_mask], dtype=float)
    yf = np.asarray(y_archive[finite_mask], dtype=float).reshape(-1)
    order = np.argsort(yf)
    if objective_sense == "max":
        order = order[::-1]
    n_top = int(min(Xf.shape[0], max(min_archive, int(math.ceil(0.25 * float(Xf.shape[0]))))))
    top_idx = order[:n_top]
    best_idx = int(order[0])
    span = np.maximum(ub - lb, 1e-12)

    def _region_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
        qlo = float(np.clip(float(getattr(system, "focus2_region_quantile_low", 0.10)), 0.0, 0.49))
        qhi = float(np.clip(float(getattr(system, "focus2_region_quantile_high", 0.90)), qlo + 0.01, 1.0))
        expand_ratio = float(max(float(getattr(system, "focus2_region_expand_ratio", 1.15)), 1.0))
        raw_lb = np.quantile(points, qlo, axis=0)
        raw_ub = np.quantile(points, qhi, axis=0)
        center = 0.5 * (raw_lb + raw_ub)
        width = np.maximum(raw_ub - raw_lb, span * 1e-6) * expand_ratio
        region_lb, region_ub = _centered_bounds_from_width(
            center=center,
            width=width,
            full_lb=lb,
            full_ub=ub,
        )
        region_lb, region_ub, _ = _enforce_min_volume_ratio(
            focus_lb=region_lb,
            focus_ub=region_ub,
            full_lb=lb,
            full_ub=ub,
            min_volume_ratio=min_volume,
        )
        region_lb, region_ub, volume_ratio = _enforce_max_volume_ratio(
            focus_lb=region_lb,
            focus_ub=region_ub,
            full_lb=lb,
            full_ub=ub,
            max_volume_ratio=max_volume,
        )
        inside = np.all((Xf >= region_lb.reshape(1, -1)) & (Xf <= region_ub.reshape(1, -1)), axis=1)
        return region_lb, region_ub, float(volume_ratio), int(np.sum(inside))

    region_lb, region_ub, volume_ratio, archive_count = _region_from_points(Xf[top_idx])
    if archive_count < min_archive:
        Xn = (Xf - lb.reshape(1, -1)) / span.reshape(1, -1)
        best_n = Xn[best_idx].reshape(1, -1)
        dist = np.linalg.norm(Xn - best_n, axis=1) / math.sqrt(float(max(p, 1)))
        near_idx = np.argsort(dist)[:min_archive]
        combined_idx = np.unique(np.concatenate([top_idx, near_idx])).astype(int)
        region_lb, region_ub, volume_ratio, archive_count = _region_from_points(Xf[combined_idx])

    if archive_count < min_archive:
        axis_ratio = float(max_volume ** (1.0 / float(max(p, 1))))
        box_width = np.maximum(span * axis_ratio, span * 1e-6)
        Xn = (Xf - lb.reshape(1, -1)) / span.reshape(1, -1)
        best_order_rank = np.empty_like(order)
        best_order_rank[order] = np.arange(order.size)
        best_candidate: tuple[float, int, np.ndarray, np.ndarray, float] | None = None
        for center_idx in order.tolist():
            center = Xf[int(center_idx)]
            cand_lb, cand_ub = _centered_bounds_from_width(
                center=center,
                width=box_width,
                full_lb=lb,
                full_ub=ub,
            )
            cand_ratio = _bounds_volume_ratio(lb=cand_lb, ub=cand_ub, base_lb=lb, base_ub=ub)
            inside = np.all((Xf >= cand_lb.reshape(1, -1)) & (Xf <= cand_ub.reshape(1, -1)), axis=1)
            count = int(np.sum(inside))
            inside_idx = np.where(inside)[0]
            if inside_idx.size:
                best_inside_rank = int(np.min(best_order_rank[inside_idx]))
            else:
                best_inside_rank = int(10**9)
            # Higher count first, then better objective-rank support.
            score = float(count) - 1e-3 * float(best_inside_rank)
            if best_candidate is None or score > best_candidate[0]:
                best_candidate = (score, count, cand_lb, cand_ub, float(cand_ratio))
        if best_candidate is not None and int(best_candidate[1]) >= min_archive:
            _score, archive_count, region_lb, region_ub, volume_ratio = best_candidate

    if archive_count < min_archive:
        return {
            "enabled": True,
            "source_focus": "focus1",
            "status": "skipped",
            "reason": "fallback_region_lacks_archive_support",
            "archive_count": int(archive_count),
            "regions": [],
            "min_archive_points": int(min_archive),
            "max_volume_ratio": float(max_volume),
        }

    inside = np.all((Xf >= region_lb.reshape(1, -1)) & (Xf <= region_ub.reshape(1, -1)), axis=1)
    y_inside = yf[inside]
    archive_best = float(np.max(y_inside) if objective_sense == "max" else np.min(y_inside))
    region = {
        "region_id": "region_fallback_001",
        "rank": 1,
        "bounds": {
            str(name): [float(lo), float(hi)]
            for name, lo, hi in zip(selected_features, region_lb, region_ub)
        },
        "center": {
            str(name): float(v)
            for name, v in zip(selected_features, 0.5 * (region_lb + region_ub))
        },
        "score": 0.0,
        "confidence": float(np.clip(float(archive_count) / float(max(min_archive * 2, 1)), 0.0, 1.0)),
        "size_ratio": float(volume_ratio),
        "candidate_counts": {
            "good": int(n_top),
            "uncertain": 0,
            "bad": 0,
            "total": int(n_top),
        },
        "archive_counts": {
            "total": int(archive_count),
            "min_required": int(min_archive),
        },
        "archive_best_objective": float(archive_best),
    }
    return {
        "enabled": True,
        "source_focus": "focus1",
        "status": "fallback",
        "reason": "archive_quantile_region",
        "archive_count": int(Xf.shape[0]),
        "top_archive_count": int(n_top),
        "region_filters": {
            "min_archive_points": int(min_archive),
            "min_volume_ratio": float(min_volume),
            "max_volume_ratio": float(max_volume),
        },
        "regions": [region],
    }


def _region_bounds_to_arrays(
    *,
    region: dict[str, object],
    selected_features: list[str],
    default_lb: np.ndarray,
    default_ub: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    bounds = region.get("bounds", {}) if isinstance(region, dict) else {}
    lb_vals: list[float] = []
    ub_vals: list[float] = []
    for idx, name in enumerate(selected_features):
        item = bounds.get(str(name)) if isinstance(bounds, dict) else None
        lo = float(default_lb[idx])
        hi = float(default_ub[idx])
        if isinstance(item, dict):
            try:
                lo = float(item.get("lb", lo))
                hi = float(item.get("ub", hi))
            except Exception:
                pass
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                lo = float(item[0])
                hi = float(item[1])
            except Exception:
                pass
        lo, hi = float(min(lo, hi)), float(max(lo, hi))
        lb_vals.append(float(max(lo, float(default_lb[idx]))))
        ub_vals.append(float(min(hi, float(default_ub[idx]))))
    lb_arr = np.asarray(lb_vals, dtype=float)
    ub_arr = np.asarray(ub_vals, dtype=float)
    bad = ~np.isfinite(lb_arr) | ~np.isfinite(ub_arr) | (ub_arr <= lb_arr)
    if np.any(bad):
        lb_arr = default_lb.copy()
        ub_arr = default_ub.copy()
    return lb_arr, ub_arr


def _build_focus2_region_states(
    *,
    regions_payload: dict[str, object],
    selected_features: list[str],
    default_lb: np.ndarray,
    default_ub: np.ndarray,
) -> list[dict[str, object]]:
    regions = regions_payload.get("regions", []) if isinstance(regions_payload, dict) else []
    if not isinstance(regions, list):
        return []
    states: list[dict[str, object]] = []
    for idx, region in enumerate(regions):
        if not isinstance(region, dict):
            continue
        rlb, rub = _region_bounds_to_arrays(
            region=region,
            selected_features=selected_features,
            default_lb=default_lb,
            default_ub=default_ub,
        )
        rid = str(region.get("region_id", f"region_{idx + 1:03d}"))
        states.append(
            {
                "region_id": rid,
                "rank": int(region.get("rank", idx + 1)),
                "parent_lb": rlb.copy(),
                "parent_ub": rub.copy(),
                "bounds_lb": rlb.copy(),
                "bounds_ub": rub.copy(),
                "status": "active",
                "success_count": 0,
                "no_improve_count": 0,
                "skip_duplicate_count": 0,
                "skip_budget_count": 0,
                "shared_eval_count": 0,
                "shared_success_count": 0,
                "merged_into": "",
                "merge_count": 0,
                "active_bounds_update_count": 0,
                "active_bounds_volume_ratio": _bounds_volume_ratio(lb=rlb, ub=rub, base_lb=default_lb, base_ub=default_ub),
                "active_bounds_last_reason": "initial_parent_bounds",
                "last_skip_reason": "",
                "eval_count": 0,
                "best_objective": float(region.get("archive_best_objective", float("nan"))),
                "best_x": None,
                "last_candidate_x": None,
                "last_objective": float("nan"),
                "source_region": dict(region),
            }
        )
    states.sort(key=lambda s: int(s.get("rank", 999999)))
    return states


def _rank_focus2_region_states_for_batch(
    *,
    region_states: list[dict[str, object]],
    X_archive: np.ndarray,
    objective_sense: str,
    system: OptimizerSystemConfig,
    max_regions: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    active = [s for s in region_states if str(s.get("status", "active")) == "active"]
    if not active:
        return [], []
    max_regions = int(max(int(max_regions), 0))
    if max_regions <= 0:
        return [], []

    if not bool(getattr(system, "focus2_scheduler_enabled", True)):
        selected = active[:max_regions]
        rows = [
            {
                "region_id": str(s.get("region_id", "")),
                "priority": float("nan"),
                "selected": bool(s in selected),
                "reason": "scheduler_disabled",
            }
            for s in active
        ]
        return selected, rows

    if bool(getattr(system, "focus2_scheduler_initial_full_round", True)):
        never_eval = [s for s in active if int(s.get("eval_count", 0)) <= 0]
        if never_eval:
            selected = never_eval[:max_regions]
            rows = [
                {
                    "region_id": str(s.get("region_id", "")),
                    "priority": 1.0 if s in selected else 0.0,
                    "selected": bool(s in selected),
                    "reason": "initial_full_round",
                    "eval_count": int(s.get("eval_count", 0)),
                }
                for s in active
            ]
            return selected, rows

    best_values: list[float] = []
    support_values: list[float] = []
    success_values: list[float] = []
    exploration_values: list[float] = []
    active_bounds_values: list[float] = []
    no_improve_values: list[float] = []
    duplicate_values: list[float] = []

    max_eval = max([int(s.get("eval_count", 0)) for s in active] + [1])
    for state in active:
        best_values.append(float(state.get("best_objective", float("nan"))))
        rlb = np.asarray(state.get("bounds_lb"), dtype=float)
        rub = np.asarray(state.get("bounds_ub"), dtype=float)
        if X_archive.ndim == 2 and X_archive.shape[0] > 0 and rlb.shape[0] == X_archive.shape[1]:
            inside = np.all((X_archive >= rlb.reshape(1, -1)) & (X_archive <= rub.reshape(1, -1)), axis=1)
            archive_count = int(np.sum(inside))
        else:
            archive_count = 0
        min_required = max(int(getattr(system, "focus2_region_min_archive_points", 5)), 1)
        support_values.append(float(np.clip(archive_count / float(2 * min_required), 0.0, 1.0)))
        success_values.append(float(np.clip(int(state.get("success_count", 0)) / 3.0, 0.0, 1.0)))
        exploration_values.append(float(1.0 - np.clip(int(state.get("eval_count", 0)) / float(max_eval + 1), 0.0, 1.0)))
        volume = float(state.get("active_bounds_volume_ratio", float("nan")))
        if math.isfinite(volume):
            active_bounds_values.append(float(1.0 - np.clip(volume / 0.25, 0.0, 1.0)))
        else:
            active_bounds_values.append(0.0)
        no_improve_values.append(float(max(int(state.get("no_improve_count", state.get("failure_count", 0))), 0)))
        duplicate_values.append(float(max(int(state.get("skip_duplicate_count", 0)), 0)))

    archive_scores = _normalize_rank_scores(best_values, objective_sense=objective_sense)
    no_improve_max = max(no_improve_values + [1.0])
    duplicate_max = max(duplicate_values + [1.0])

    w_archive = float(max(float(getattr(system, "focus2_scheduler_archive_weight", 0.50)), 0.0))
    w_support = float(max(float(getattr(system, "focus2_scheduler_support_weight", 0.15)), 0.0))
    w_success = float(max(float(getattr(system, "focus2_scheduler_success_weight", 0.15)), 0.0))
    w_explore = float(max(float(getattr(system, "focus2_scheduler_exploration_weight", 0.10)), 0.0))
    w_bounds = float(max(float(getattr(system, "focus2_scheduler_active_bounds_weight", 0.10)), 0.0))
    no_improve_penalty = float(max(float(getattr(system, "focus2_scheduler_no_improve_penalty", 0.10)), 0.0))
    duplicate_penalty = float(max(float(getattr(system, "focus2_scheduler_duplicate_penalty", 0.03)), 0.0))
    w_sum = max(w_archive + w_support + w_success + w_explore + w_bounds, 1e-12)
    w_archive, w_support, w_success, w_explore, w_bounds = (
        w_archive / w_sum,
        w_support / w_sum,
        w_success / w_sum,
        w_explore / w_sum,
        w_bounds / w_sum,
    )

    scored: list[tuple[float, int, dict[str, object], dict[str, object]]] = []
    for idx, state in enumerate(active):
        no_improve_norm = float(no_improve_values[idx] / max(no_improve_max, 1.0))
        duplicate_norm = float(duplicate_values[idx] / max(duplicate_max, 1.0))
        priority = (
            w_archive * float(archive_scores[idx])
            + w_support * float(support_values[idx])
            + w_success * float(success_values[idx])
            + w_explore * float(exploration_values[idx])
            + w_bounds * float(active_bounds_values[idx])
            - no_improve_penalty * no_improve_norm
            - duplicate_penalty * duplicate_norm
        )
        row = {
            "region_id": str(state.get("region_id", "")),
            "priority": float(priority),
            "archive_score": float(archive_scores[idx]),
            "support_score": float(support_values[idx]),
            "success_score": float(success_values[idx]),
            "exploration_score": float(exploration_values[idx]),
            "active_bounds_score": float(active_bounds_values[idx]),
            "no_improve_norm": float(no_improve_norm),
            "duplicate_norm": float(duplicate_norm),
            "eval_count": int(state.get("eval_count", 0)),
            "selected": False,
            "reason": "priority",
        }
        scored.append((float(priority), int(state.get("rank", 999999)), state, row))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected_ids = {str(item[2].get("region_id", "")) for item in scored[:max_regions]}
    rows: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    for priority, _rank, state, row in scored:
        row = dict(row)
        row["selected"] = str(state.get("region_id", "")) in selected_ids
        rows.append(row)
        if row["selected"]:
            selected.append(state)
    return selected, rows


def _build_focus2_lite_batch(
    *,
    rng: np.random.Generator,
    region_states: list[dict[str, object]],
    X_archive: np.ndarray,
    y_archive: np.ndarray,
    objective_sense: str,
    system: OptimizerSystemConfig,
    selected_features: list[str],
    full_lb: np.ndarray,
    full_ub: np.ndarray,
    kappa: float,
    xi: float,
    post_feasible_prob_fn: Callable[[np.ndarray], float] | None,
    post_penalty_lambda: float,
    max_candidates: int,
    var_names: list[str],
    constraint_defs: list[dict],
    enforce_pre_constraints: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    active = [s for s in region_states if str(s.get("status", "active")) == "active"]
    if not active:
        return [], {"reason": "no_active_region", "candidate_count": 0, "plan_pool_count": 0}

    min_archive = int(max(int(getattr(system, "focus2_region_min_archive_points", 5)), 1))
    pool_size = int(max(int(getattr(system, "focus2_local_pool_size", 512)), 8))
    max_candidates = int(max(int(max_candidates), 0))
    if max_candidates <= 0:
        return [], {"reason": "no_remaining_budget", "candidate_count": 0, "plan_pool_count": 0}
    scheduled_active, scheduler_rows = _rank_focus2_region_states_for_batch(
        region_states=region_states,
        X_archive=X_archive,
        objective_sense=objective_sense,
        system=system,
        max_regions=max_candidates,
    )
    if not scheduled_active:
        return [], {
            "reason": "no_scheduled_region",
            "candidate_count": 0,
            "plan_pool_count": 0,
            "scheduler": scheduler_rows,
        }
    p_topk, p_boundary, p_random = _normalize_source_probs(
        topk=float(getattr(system, "focus2_local_topk_ratio", 0.30)),
        boundary=float(getattr(system, "focus2_local_boundary_ratio", 0.35)),
        random_p=float(getattr(system, "focus2_local_random_ratio", 0.35)),
    )
    min_rms = float(max(float(getattr(system, "focus2_min_candidate_rms_distance", 0.02)), 0.0))
    pool_quotas = _allocate_source_quotas(
        total=pool_size,
        probs={"topk": p_topk, "boundary": p_boundary, "random": p_random},
    )
    candidates: list[tuple[int, float, np.ndarray, dict[str, object], int, str]] = []
    skipped: list[dict[str, object]] = []
    plan_pool_count = 0
    pre_raw_count = 0
    pre_feasible_count = 0
    pre_margin_values: list[float] = []
    for state in scheduled_active:
        rlb = np.asarray(state.get("bounds_lb"), dtype=float)
        rub = np.asarray(state.get("bounds_ub"), dtype=float)
        inside = np.all((X_archive >= rlb.reshape(1, -1)) & (X_archive <= rub.reshape(1, -1)), axis=1)
        idx = np.where(inside)[0]
        if idx.size < min_archive:
            skipped.append({"region_id": str(state.get("region_id")), "reason": "insufficient_archive", "count": int(idx.size)})
            continue
        X_local = X_archive[idx]
        y_local = y_archive[idx]
        try:
            local_model, _local_fallback = _fit_optimizer_gp_with_fallback(
                X=X_local,
                y=y_local,
                lb=rlb,
                ub=rub,
                system=system,
                include_white=False,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
        except Exception:
            skipped.append({"region_id": str(state.get("region_id")), "reason": "local_gp_failed", "count": int(idx.size)})
            continue
        pool_parts: list[np.ndarray] = []
        pool_labels: list[str] = []
        for source in ("topk", "boundary", "random"):
            n_source = int(pool_quotas.get(source, 0))
            if n_source <= 0:
                continue
            part = _build_source_pool_starts(
                rng=rng,
                source=source,
                X_train=X_local,
                y_train=y_local,
                objective_sense=objective_sense,
                lb=rlb,
                ub=rub,
                pool_size=n_source,
                topk_fraction=float(getattr(system, "source_topk_fraction", 0.20)),
                topk_sigma=float(getattr(system, "source_topk_perturb_sigma", 0.08)),
                boundary_near_ratio=float(getattr(system, "source_boundary_near_ratio", 0.03)),
            )
            if part.shape[0] > 0:
                pool_parts.append(part)
                pool_labels.extend([source] * int(part.shape[0]))
        if pool_parts:
            pool = np.vstack(pool_parts)
        else:
            pool = rng.uniform(rlb, rub, size=(pool_size, len(selected_features)))
            pool_labels = ["random"] * int(pool.shape[0])
        plan_pool_count += int(pool.shape[0])
        pool, pool_labels, pool_margins, pool_constraint_stats = _filter_pool_by_pre_constraints(
            pool=pool,
            labels=pool_labels,
            var_names=var_names,
            constraint_defs=constraint_defs,
            enforce_pre_constraints=enforce_pre_constraints,
        )
        pre_raw_count += int(pool_constraint_stats.get("raw_count", 0))
        pre_feasible_count += int(pool_constraint_stats.get("feasible_count", 0))
        if pool_margins.size:
            pre_margin_values.extend(float(v) for v in pool_margins[np.isfinite(pool_margins)].tolist())
        if pool.shape[0] == 0:
            skipped.append({
                "region_id": str(state.get("region_id")),
                "reason": "no_pre_feasible_candidate",
                "count": int(idx.size),
            })
            continue
        scores = _acquisition_scores_for_pool(
            pool=pool,
            model=local_model,
            y_best=float(np.max(y_local) if objective_sense == "max" else np.min(y_local)),
            objective_sense=objective_sense,
            acq_type="LCB",
            kappa=float(kappa),
            xi=float(xi),
            post_feasible_prob_fn=post_feasible_prob_fn,
            post_penalty_lambda=post_penalty_lambda,
        )
        finite = np.where(np.isfinite(scores))[0]
        if finite.size == 0:
            skipped.append({"region_id": str(state.get("region_id")), "reason": "no_finite_score", "count": int(idx.size)})
            continue
        best_i = int(finite[np.argmin(scores[finite])])
        if (
            enforce_pre_constraints
            and constraint_defs
            and bool(getattr(system, "constraint_boundary_source_enabled", True))
            and pool_margins.size == scores.size
            and rng.uniform(0.0, 1.0) < float(np.clip(float(getattr(system, "focus2_constraint_boundary_select_prob", 0.25)), 0.0, 1.0))
        ):
            boundary_i = _select_constraint_boundary_index(
                scores=scores,
                margins=pool_margins,
                acq_quantile=float(getattr(system, "constraint_boundary_acq_quantile", 0.75)),
            )
            if boundary_i is not None:
                best_i = int(boundary_i)
        best_source = str(pool_labels[best_i]) if best_i < len(pool_labels) else "unknown"
        if enforce_pre_constraints and constraint_defs and pool_margins.size == scores.size:
            boundary_i = _select_constraint_boundary_index(
                scores=scores,
                margins=pool_margins,
                acq_quantile=float(getattr(system, "constraint_boundary_acq_quantile", 0.75)),
            )
            if boundary_i is not None and int(boundary_i) == int(best_i):
                best_source = "constraint_boundary"
        candidates.append((
            int(state.get("rank", 999999)),
            float(scores[best_i]),
            pool[best_i].reshape(-1),
            state,
            int(idx.size),
            best_source,
        ))

    if not candidates:
        return [], {
            "reason": "no_region_candidate",
            "candidate_count": 0,
            "plan_pool_count": int(plan_pool_count),
            "pre_raw_count": int(pre_raw_count),
            "pre_feasible_count": int(pre_feasible_count),
            "pre_feasible_ratio": float(pre_feasible_count) / float(max(pre_raw_count, 1)) if enforce_pre_constraints and constraint_defs else 1.0,
            "pre_margin_min": float(np.min(pre_margin_values)) if pre_margin_values else float("nan"),
            "pre_margin_median": float(np.median(pre_margin_values)) if pre_margin_values else float("nan"),
            "pre_margin_max": float(np.max(pre_margin_values)) if pre_margin_values else float("nan"),
            "skipped": skipped,
        }
    candidates.sort(key=lambda item: item[1])
    batch: list[dict[str, object]] = []
    selected_X: list[np.ndarray] = []
    selected_region_ids: list[str] = []
    duplicate_skips: list[dict[str, object]] = []
    budget_skips: list[dict[str, object]] = []
    scheduler_by_region = {
        str(row.get("region_id", "")): row
        for row in scheduler_rows
        if isinstance(row, dict)
    }
    for rank, score, x, state, archive_count, selected_source in candidates:
        rid = str(state.get("region_id", ""))
        archive_dist = _normalized_rms_distance_to_set(x=x, X_ref=X_archive, lb=full_lb, ub=full_ub)
        batch_ref = np.asarray(selected_X, dtype=float) if selected_X else np.empty((0, x.shape[0]), dtype=float)
        batch_dist = _normalized_rms_distance_to_set(x=x, X_ref=batch_ref, lb=full_lb, ub=full_ub)
        if min(archive_dist, batch_dist) < min_rms:
            duplicate_with = ""
            if selected_X:
                span = np.maximum(full_ub - full_lb, 1e-12)
                sel = np.asarray(selected_X, dtype=float)
                d = np.sqrt(np.mean(((sel - x.reshape(1, -1)) / span.reshape(1, -1)) ** 2, axis=1))
                finite_d = np.where(np.isfinite(d))[0]
                if finite_d.size > 0:
                    nearest_i = int(finite_d[np.argmin(d[finite_d])])
                    if nearest_i < len(selected_region_ids):
                        duplicate_with = str(selected_region_ids[nearest_i])
            state["skip_duplicate_count"] = int(state.get("skip_duplicate_count", 0)) + 1
            state["last_skip_reason"] = "duplicate_candidate"
            duplicate_skips.append(
                {
                    "region_id": rid,
                    "duplicate_with_region_id": duplicate_with,
                    "score": float(score),
                    "archive_distance": float(archive_dist),
                    "batch_distance": float(batch_dist),
                }
            )
            continue
        if len(batch) >= max_candidates:
            state["skip_budget_count"] = int(state.get("skip_budget_count", 0)) + 1
            state["last_skip_reason"] = "budget_trimmed"
            budget_skips.append({"region_id": rid, "score": float(score)})
            continue
        selected_X.append(np.asarray(x, dtype=float).reshape(-1))
        selected_region_ids.append(rid)
        batch.append(
            {
                "x": np.clip(x, full_lb, full_ub),
                "region_id": rid,
                "region_state": state,
                "score": float(score),
                "selected_source": str(selected_source),
                "selected_region_archive_count": int(archive_count),
                "rank": int(rank),
                "archive_distance": float(archive_dist),
                "batch_distance": float(batch_dist),
                "scheduler_priority": float(scheduler_by_region.get(rid, {}).get("priority", float("nan"))),
            }
        )
    return batch, {
        "reason": "ok",
        "candidate_count": int(len(batch)),
        "available_candidate_count": int(len(candidates)),
        "plan_pool_count": int(plan_pool_count),
        "pre_raw_count": int(pre_raw_count),
        "pre_feasible_count": int(pre_feasible_count),
        "pre_feasible_ratio": float(pre_feasible_count) / float(max(pre_raw_count, 1)) if enforce_pre_constraints and constraint_defs else 1.0,
        "pre_margin_min": float(np.min(pre_margin_values)) if pre_margin_values else float("nan"),
        "pre_margin_median": float(np.median(pre_margin_values)) if pre_margin_values else float("nan"),
        "pre_margin_max": float(np.max(pre_margin_values)) if pre_margin_values else float("nan"),
        "local_pool_size": int(pool_size),
        "local_pool_quotas": {k: int(v) for k, v in pool_quotas.items()},
        "min_candidate_rms_distance": float(min_rms),
        "duplicate_skip_count": int(len(duplicate_skips)),
        "budget_skip_count": int(len(budget_skips)),
        "scheduler": scheduler_rows,
        "scheduled_region_count": int(len(scheduled_active)),
        "duplicate_skips": duplicate_skips,
        "budget_skips": budget_skips,
        "skipped": skipped,
    }


def _update_focus2_region_state(
    *,
    state: dict[str, object] | None,
    x_next: np.ndarray | None,
    y_next: float,
    objective_sense: str,
    system: OptimizerSystemConfig,
) -> None:
    if not isinstance(state, dict):
        return
    prev_best = state.get("best_objective", float("nan"))
    try:
        prev_best_f = float(prev_best)
    except Exception:
        prev_best_f = float("nan")
    improved = (
        not math.isfinite(prev_best_f)
        or _is_better(y_new=float(y_next), y_best=prev_best_f, objective_sense=objective_sense)
    )
    state["eval_count"] = int(state.get("eval_count", 0)) + 1
    state["last_objective"] = float(y_next)
    if improved:
        state["best_objective"] = float(y_next)
        state["best_x"] = np.asarray(x_next, dtype=float).reshape(-1).copy() if x_next is not None else state.get("best_x")
        state["success_count"] = int(state.get("success_count", 0)) + 1
        state["no_improve_count"] = 0
    else:
        state["no_improve_count"] = int(state.get("no_improve_count", state.get("failure_count", 0))) + 1
        state["success_count"] = 0
    no_improve_tol = int(max(int(getattr(system, "focus2_region_no_improve_tolerance", getattr(system, "focus2_region_failure_tolerance", 3))), 1))
    min_evals_before_drop = int(max(int(getattr(system, "focus2_region_min_evals_before_drop", 1)), 1))
    if int(state.get("eval_count", 0)) >= min_evals_before_drop and int(state.get("no_improve_count", 0)) >= no_improve_tol:
        state["status"] = "dropped"


def _update_focus2_shared_region_states(
    *,
    region_states: list[dict[str, object]],
    selected_state: dict[str, object] | None,
    x_next: np.ndarray,
    y_next: float,
    objective_sense: str,
    system: OptimizerSystemConfig,
) -> list[str]:
    min_rms = float(max(float(getattr(system, "focus2_min_candidate_rms_distance", 0.02)), 0.0))
    if min_rms <= 0.0:
        return []
    selected_id = str(selected_state.get("region_id", "")) if isinstance(selected_state, dict) else ""
    x = np.asarray(x_next, dtype=float).reshape(-1)
    shared_ids: list[str] = []
    for state in region_states:
        if not isinstance(state, dict):
            continue
        rid = str(state.get("region_id", ""))
        if rid == selected_id:
            continue
        rlb = np.asarray(state.get("bounds_lb"), dtype=float)
        rub = np.asarray(state.get("bounds_ub"), dtype=float)
        if rlb.shape != x.shape or rub.shape != x.shape:
            continue
        refs: list[np.ndarray] = []
        last_candidate = state.get("last_candidate_x")
        best_x = state.get("best_x")
        if last_candidate is not None:
            refs.append(np.asarray(last_candidate, dtype=float).reshape(-1))
        if best_x is not None:
            refs.append(np.asarray(best_x, dtype=float).reshape(-1))
        if not refs:
            continue
        ref_arr = np.asarray(refs, dtype=float)
        near_dist = _normalized_rms_distance_to_set(x=x, X_ref=ref_arr, lb=rlb, ub=rub)
        if near_dist >= min_rms:
            continue
        prev_best = state.get("best_objective", float("nan"))
        try:
            prev_best_f = float(prev_best)
        except Exception:
            prev_best_f = float("nan")
        improved = (
            not math.isfinite(prev_best_f)
            or _is_better(y_new=float(y_next), y_best=prev_best_f, objective_sense=objective_sense)
        )
        state["shared_eval_count"] = int(state.get("shared_eval_count", 0)) + 1
        shared_ids.append(rid)
        if improved:
            state["best_objective"] = float(y_next)
            state["best_x"] = x.copy()
            state["shared_success_count"] = int(state.get("shared_success_count", 0)) + 1
    return shared_ids


def _focus2_pair_key(a: str, b: str) -> tuple[str, str]:
    aa, bb = str(a), str(b)
    return (aa, bb) if aa <= bb else (bb, aa)


def _increment_focus2_pair_count(counts: dict[tuple[str, str], int], a: str, b: str) -> None:
    if not a or not b or str(a) == str(b):
        return
    key = _focus2_pair_key(str(a), str(b))
    counts[key] = int(counts.get(key, 0)) + 1


def _normalize_rank_scores(values: list[float], *, objective_sense: str) -> list[float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return [0.0 for _ in values]
    lo = min(finite)
    hi = max(finite)
    if abs(hi - lo) <= 1e-12:
        return [1.0 if math.isfinite(float(v)) else 0.0 for v in values]
    out: list[float] = []
    for v in values:
        vf = float(v)
        if not math.isfinite(vf):
            out.append(0.0)
        elif objective_sense == "max":
            out.append(float((vf - lo) / (hi - lo)))
        else:
            out.append(float((hi - vf) / (hi - lo)))
    return out


def _focus2_state_best_x(state: dict[str, object]) -> np.ndarray | None:
    for key in ("best_x", "last_candidate_x"):
        value = state.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size > 0 and np.isfinite(arr).all():
            return arr
    return None


def _focus2_same_basin_score(
    *,
    a: dict[str, object],
    b: dict[str, object],
    y_archive: np.ndarray,
    full_lb: np.ndarray,
    full_ub: np.ndarray,
    system: OptimizerSystemConfig,
) -> dict[str, object]:
    ax = _focus2_state_best_x(a)
    bx = _focus2_state_best_x(b)
    point_threshold = float(max(float(getattr(system, "focus2_merge_best_point_rms", 0.05)), 1e-12))
    objective_gap_ratio = float(max(float(getattr(system, "focus2_merge_objective_gap_ratio", 0.10)), 1e-12))
    if ax is None or bx is None or ax.shape != bx.shape:
        point_dist = float("inf")
        point_score = 0.0
    else:
        point_dist = _normalized_rms_distance_to_set(
            x=ax,
            X_ref=bx.reshape(1, -1),
            lb=full_lb,
            ub=full_ub,
        )
        point_score = float(np.clip(1.0 - point_dist / point_threshold, 0.0, 1.0))

    ay = float(a.get("best_objective", float("nan")))
    by = float(b.get("best_objective", float("nan")))
    finite_y = np.asarray(y_archive, dtype=float)
    finite_y = finite_y[np.isfinite(finite_y)]
    if finite_y.size >= 2:
        y_scale = float(np.quantile(finite_y, 0.90) - np.quantile(finite_y, 0.10))
    else:
        y_scale = float("nan")
    if not math.isfinite(y_scale) or y_scale <= 1e-12:
        y_scale = max(abs(ay), abs(by), 1.0)
    if math.isfinite(ay) and math.isfinite(by):
        objective_gap = float(abs(ay - by) / max(y_scale, 1e-12))
        objective_score = float(np.clip(1.0 - objective_gap / objective_gap_ratio, 0.0, 1.0))
    else:
        objective_gap = float("inf")
        objective_score = 0.0
    score = float(0.70 * point_score + 0.30 * objective_score)
    return {
        "score": score,
        "point_distance": float(point_dist),
        "point_score": float(point_score),
        "objective_gap": float(objective_gap),
        "objective_score": float(objective_score),
    }


def _enforce_max_volume_ratio(
    *,
    focus_lb: np.ndarray,
    focus_ub: np.ndarray,
    full_lb: np.ndarray,
    full_ub: np.ndarray,
    max_volume_ratio: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    ceiling = float(np.clip(float(max_volume_ratio), 1e-12, 1.0))
    ratio = _bounds_volume_ratio(lb=focus_lb, ub=focus_ub, base_lb=full_lb, base_ub=full_ub)
    if ratio <= ceiling:
        return focus_lb, focus_ub, ratio
    p = int(max(focus_lb.shape[0], 1))
    center = 0.5 * (focus_lb + focus_ub)
    width = np.maximum(focus_ub - focus_lb, 1e-12)
    scale = float((ceiling / max(ratio, 1e-12)) ** (1.0 / float(p)))
    lo, hi = _centered_bounds_from_width(
        center=center,
        width=width * scale,
        full_lb=full_lb,
        full_ub=full_ub,
    )
    ratio2 = _bounds_volume_ratio(lb=lo, ub=hi, base_lb=full_lb, base_ub=full_ub)
    return lo, hi, ratio2


def _merge_focus2_region_states(
    *,
    region_states: list[dict[str, object]],
    duplicate_pair_counts: dict[tuple[str, str], int],
    shared_pair_counts: dict[tuple[str, str], int],
    X_archive: np.ndarray,
    y_archive: np.ndarray,
    objective_sense: str,
    full_lb: np.ndarray,
    full_ub: np.ndarray,
    selected_features: list[str],
    system: OptimizerSystemConfig,
) -> list[dict[str, object]]:
    if not bool(getattr(system, "focus2_merge_enabled", True)):
        return []
    active = {
        str(s.get("region_id", "")): s
        for s in region_states
        if isinstance(s, dict) and str(s.get("status", "active")) == "active"
    }
    if len(active) < 2:
        return []

    duplicate_threshold = int(max(int(getattr(system, "focus2_merge_duplicate_pair_threshold", 2)), 1))
    shared_threshold = int(max(int(getattr(system, "focus2_merge_shared_pair_threshold", 2)), 1))
    same_basin_min = float(np.clip(float(getattr(system, "focus2_merge_same_basin_min_score", 0.75)), 0.0, 1.0))
    max_volume = float(np.clip(float(getattr(system, "focus2_merge_max_volume_ratio", 0.25)), 1e-12, 1.0))
    expand_ratio = float(max(float(getattr(system, "focus2_merge_expand_ratio", 1.15)), 1.0))
    top_fraction = float(np.clip(float(getattr(system, "focus2_merge_top_fraction", 0.50)), 0.05, 1.0))
    min_volume = float(np.clip(float(getattr(system, "focus2_region_min_volume_ratio", 0.05)), 0.0, max_volume))
    qlo = float(np.clip(float(getattr(system, "focus2_region_quantile_low", 0.10)), 0.0, 0.49))
    qhi = float(np.clip(float(getattr(system, "focus2_region_quantile_high", 0.90)), qlo + 0.01, 1.0))

    merge_events: list[dict[str, object]] = []
    state_by_id = active
    used: set[str] = set()
    pair_keys = set(duplicate_pair_counts.keys()) | set(shared_pair_counts.keys())
    scored_pairs: list[tuple[float, tuple[str, str], dict[str, object]]] = []
    for key in pair_keys:
        a_id, b_id = key
        if a_id not in state_by_id or b_id not in state_by_id:
            continue
        dup_count = int(duplicate_pair_counts.get(key, 0))
        shared_count = int(shared_pair_counts.get(key, 0))
        basin = _focus2_same_basin_score(
            a=state_by_id[a_id],
            b=state_by_id[b_id],
            y_archive=y_archive,
            full_lb=full_lb,
            full_ub=full_ub,
            system=system,
        )
        duplicate_trigger = dup_count >= duplicate_threshold
        shared_trigger = shared_count >= shared_threshold and float(basin["score"]) >= same_basin_min
        if not duplicate_trigger and not shared_trigger:
            continue
        priority = float(dup_count * 10 + shared_count + float(basin["score"]))
        scored_pairs.append((
            priority,
            key,
            {
                "duplicate_count": int(dup_count),
                "shared_count": int(shared_count),
                "same_basin": basin,
                "trigger": "duplicate" if duplicate_trigger else "shared_same_basin",
            },
        ))
    scored_pairs.sort(key=lambda item: item[0], reverse=True)

    for _priority, (a_id, b_id), info in scored_pairs:
        if a_id in used or b_id in used:
            continue
        a = state_by_id.get(a_id)
        b = state_by_id.get(b_id)
        if a is None or b is None:
            continue
        a_best = float(a.get("best_objective", float("nan")))
        b_best = float(b.get("best_objective", float("nan")))
        if not math.isfinite(a_best):
            survivor, absorbed = b, a
        elif not math.isfinite(b_best):
            survivor, absorbed = a, b
        elif _is_better(y_new=a_best, y_best=b_best, objective_sense=objective_sense):
            survivor, absorbed = a, b
        else:
            survivor, absorbed = b, a

        parent_lb = np.minimum(np.asarray(a.get("bounds_lb"), dtype=float), np.asarray(b.get("bounds_lb"), dtype=float))
        parent_ub = np.maximum(np.asarray(a.get("bounds_ub"), dtype=float), np.asarray(b.get("bounds_ub"), dtype=float))
        inside = np.all((X_archive >= parent_lb.reshape(1, -1)) & (X_archive <= parent_ub.reshape(1, -1)), axis=1)
        X_inside = X_archive[inside]
        y_inside = y_archive[inside]
        if X_inside.shape[0] >= 2:
            order = np.argsort(-y_inside) if objective_sense == "max" else np.argsort(y_inside)
            n_keep = int(max(2, math.ceil(float(order.shape[0]) * top_fraction)))
            keep = order[: min(n_keep, order.shape[0])]
            X_sel = X_inside[keep]
            raw_lb = np.quantile(X_sel, qlo, axis=0)
            raw_ub = np.quantile(X_sel, qhi, axis=0)
            center = 0.5 * (raw_lb + raw_ub)
            width = np.maximum(raw_ub - raw_lb, 1e-12) * expand_ratio
            merged_lb, merged_ub = _centered_bounds_from_width(
                center=center,
                width=width,
                full_lb=parent_lb,
                full_ub=parent_ub,
            )
        else:
            merged_lb, merged_ub = parent_lb.copy(), parent_ub.copy()

        merged_lb, merged_ub, merged_volume = _enforce_min_volume_ratio(
            focus_lb=merged_lb,
            focus_ub=merged_ub,
            full_lb=full_lb,
            full_ub=full_ub,
            min_volume_ratio=min_volume,
        )
        merged_lb, merged_ub, merged_volume = _enforce_max_volume_ratio(
            focus_lb=merged_lb,
            focus_ub=merged_ub,
            full_lb=full_lb,
            full_ub=full_ub,
            max_volume_ratio=max_volume,
        )

        old_survivor_id = str(survivor.get("region_id", ""))
        absorbed_id = str(absorbed.get("region_id", ""))
        survivor["bounds_lb"] = merged_lb
        survivor["bounds_ub"] = merged_ub
        survivor["parent_lb"] = np.minimum(np.asarray(a.get("parent_lb", a.get("bounds_lb")), dtype=float), np.asarray(b.get("parent_lb", b.get("bounds_lb")), dtype=float))
        survivor["parent_ub"] = np.maximum(np.asarray(a.get("parent_ub", a.get("bounds_ub")), dtype=float), np.asarray(b.get("parent_ub", b.get("bounds_ub")), dtype=float))
        survivor["active_bounds_update_count"] = int(survivor.get("active_bounds_update_count", 0))
        survivor["active_bounds_volume_ratio"] = float(merged_volume)
        survivor["active_bounds_last_reason"] = "merge"
        survivor["no_improve_count"] = min(
            int(a.get("no_improve_count", a.get("failure_count", 0))),
            int(b.get("no_improve_count", b.get("failure_count", 0))),
        )
        survivor["success_count"] = max(int(a.get("success_count", 0)), int(b.get("success_count", 0)))
        survivor["eval_count"] = int(a.get("eval_count", 0)) + int(b.get("eval_count", 0))
        survivor["shared_eval_count"] = int(a.get("shared_eval_count", 0)) + int(b.get("shared_eval_count", 0))
        survivor["shared_success_count"] = int(a.get("shared_success_count", 0)) + int(b.get("shared_success_count", 0))
        survivor["skip_duplicate_count"] = int(a.get("skip_duplicate_count", 0)) + int(b.get("skip_duplicate_count", 0))
        survivor["skip_budget_count"] = int(a.get("skip_budget_count", 0)) + int(b.get("skip_budget_count", 0))
        survivor["merge_count"] = int(survivor.get("merge_count", 0)) + 1
        survivor["last_skip_reason"] = ""
        absorbed["status"] = "merged"
        absorbed["merged_into"] = old_survivor_id
        used.add(old_survivor_id)
        used.add(absorbed_id)
        bounds_payload = {
            str(name): [float(lo), float(hi)]
            for name, lo, hi in zip(selected_features, merged_lb, merged_ub)
        }
        merge_events.append({
            "survivor_region_id": old_survivor_id,
            "absorbed_region_id": absorbed_id,
            "trigger": str(info.get("trigger", "")),
            "duplicate_count": int(info.get("duplicate_count", 0)),
            "shared_count": int(info.get("shared_count", 0)),
            "same_basin": info.get("same_basin", {}),
            "merged_volume_ratio": float(merged_volume),
            "bounds": bounds_payload,
        })
    return merge_events


def _update_focus2_active_bounds_from_archive(
    *,
    region_states: list[dict[str, object]],
    X_archive: np.ndarray,
    y_archive: np.ndarray,
    objective_sense: str,
    full_lb: np.ndarray,
    full_ub: np.ndarray,
    selected_features: list[str],
    system: OptimizerSystemConfig,
) -> list[dict[str, object]]:
    if not bool(getattr(system, "focus2_active_bounds_enabled", True)):
        return []
    if X_archive.ndim != 2 or y_archive.ndim != 1 or X_archive.shape[0] != y_archive.shape[0]:
        return []
    min_archive = int(max(int(getattr(system, "focus2_active_bounds_min_archive_points", 5)), 1))
    min_region_evals = int(max(int(getattr(system, "focus2_active_bounds_min_region_evals", 3)), 0))
    top_fraction = float(np.clip(float(getattr(system, "focus2_active_bounds_top_fraction", 0.50)), 0.05, 1.0))
    min_top = int(max(int(getattr(system, "focus2_active_bounds_min_top_points", 3)), 1))
    qlo = float(np.clip(float(getattr(system, "focus2_active_bounds_quantile_low", 0.10)), 0.0, 0.49))
    qhi = float(np.clip(float(getattr(system, "focus2_active_bounds_quantile_high", 0.90)), qlo + 0.01, 1.0))
    expand_ratio = float(max(float(getattr(system, "focus2_active_bounds_expand_ratio", 1.20)), 1.0))
    min_volume = float(np.clip(float(getattr(system, "focus2_active_bounds_min_volume_ratio", 0.05)), 0.0, 1.0))
    max_volume = float(np.clip(float(getattr(system, "focus2_active_bounds_max_volume_ratio", 0.25)), 1e-12, 1.0))
    update_rate = float(np.clip(float(getattr(system, "focus2_active_bounds_update_rate", 0.50)), 0.0, 1.0))
    if update_rate <= 0.0:
        return []

    events: list[dict[str, object]] = []
    for state in region_states:
        if not isinstance(state, dict) or str(state.get("status", "active")) != "active":
            continue
        parent_lb = np.asarray(state.get("parent_lb", state.get("bounds_lb")), dtype=float)
        parent_ub = np.asarray(state.get("parent_ub", state.get("bounds_ub")), dtype=float)
        old_lb = np.asarray(state.get("bounds_lb"), dtype=float)
        old_ub = np.asarray(state.get("bounds_ub"), dtype=float)
        if parent_lb.shape != full_lb.shape or parent_ub.shape != full_ub.shape:
            continue
        region_eval_count = int(state.get("eval_count", 0)) + int(state.get("shared_eval_count", 0))
        if region_eval_count < min_region_evals:
            state["active_bounds_last_reason"] = "insufficient_region_evals"
            continue
        inside = np.all((X_archive >= parent_lb.reshape(1, -1)) & (X_archive <= parent_ub.reshape(1, -1)), axis=1)
        X_inside = X_archive[inside]
        y_inside = y_archive[inside]
        if X_inside.shape[0] < min_archive:
            state["active_bounds_last_reason"] = "insufficient_archive"
            continue
        order = np.argsort(-y_inside) if objective_sense == "max" else np.argsort(y_inside)
        n_keep = int(max(min_top, math.ceil(float(order.shape[0]) * top_fraction)))
        keep = order[: min(n_keep, order.shape[0])]
        if keep.size < min_top:
            state["active_bounds_last_reason"] = "insufficient_top_points"
            continue
        X_sel = X_inside[keep]
        raw_lb = np.quantile(X_sel, qlo, axis=0)
        raw_ub = np.quantile(X_sel, qhi, axis=0)
        center = 0.5 * (raw_lb + raw_ub)
        width = np.maximum(raw_ub - raw_lb, 1e-12) * expand_ratio
        proposed_lb, proposed_ub = _centered_bounds_from_width(
            center=center,
            width=width,
            full_lb=parent_lb,
            full_ub=parent_ub,
        )
        region_max_volume = min(
            max_volume,
            _bounds_volume_ratio(lb=parent_lb, ub=parent_ub, base_lb=full_lb, base_ub=full_ub),
        )
        proposed_lb, proposed_ub, _min_ratio = _enforce_min_volume_ratio(
            focus_lb=proposed_lb,
            focus_ub=proposed_ub,
            full_lb=full_lb,
            full_ub=full_ub,
            min_volume_ratio=min(min_volume, region_max_volume),
        )
        proposed_lb, proposed_ub, _max_ratio = _enforce_max_volume_ratio(
            focus_lb=proposed_lb,
            focus_ub=proposed_ub,
            full_lb=full_lb,
            full_ub=full_ub,
            max_volume_ratio=region_max_volume,
        )
        smoothed_lb = (1.0 - update_rate) * old_lb + update_rate * proposed_lb
        smoothed_ub = (1.0 - update_rate) * old_ub + update_rate * proposed_ub
        smoothed_lb = np.maximum(smoothed_lb, parent_lb)
        smoothed_ub = np.minimum(smoothed_ub, parent_ub)
        bad = ~np.isfinite(smoothed_lb) | ~np.isfinite(smoothed_ub) | (smoothed_ub <= smoothed_lb)
        if np.any(bad):
            state["active_bounds_last_reason"] = "invalid_smoothed_bounds"
            continue
        smoothed_lb, smoothed_ub, final_ratio = _enforce_min_volume_ratio(
            focus_lb=smoothed_lb,
            focus_ub=smoothed_ub,
            full_lb=full_lb,
            full_ub=full_ub,
            min_volume_ratio=min(min_volume, region_max_volume),
        )
        smoothed_lb, smoothed_ub, final_ratio = _enforce_max_volume_ratio(
            focus_lb=smoothed_lb,
            focus_ub=smoothed_ub,
            full_lb=full_lb,
            full_ub=full_ub,
            max_volume_ratio=region_max_volume,
        )
        state["bounds_lb"] = smoothed_lb
        state["bounds_ub"] = smoothed_ub
        state["active_bounds_update_count"] = int(state.get("active_bounds_update_count", 0)) + 1
        state["active_bounds_volume_ratio"] = float(final_ratio)
        state["active_bounds_last_reason"] = "updated_from_archive_top_points"
        events.append({
            "region_id": str(state.get("region_id", "")),
            "archive_count": int(X_inside.shape[0]),
            "region_eval_count": int(region_eval_count),
            "top_count": int(keep.size),
            "update_rate": float(update_rate),
            "volume_ratio": float(final_ratio),
            "bounds": {
                str(name): [float(lo), float(hi)]
                for name, lo, hi in zip(selected_features, smoothed_lb, smoothed_ub)
            },
        })
    return events


def _select_focus2_final_region(
    *,
    region_states: list[dict[str, object]],
    objective_sense: str,
    system: OptimizerSystemConfig,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    if not region_states:
        return None, {
            "reason": "no_regions",
            "candidates": [],
        }
    active = [s for s in region_states if str(s.get("status", "active")) == "active"]
    non_merged = [s for s in region_states if str(s.get("status", "active")) != "merged"]
    candidates = active if active else non_merged
    if not candidates:
        candidates = list(region_states)
    best_values: list[float] = []
    support_scores: list[float] = []
    success_counts: list[float] = []
    for state in candidates:
        src = state.get("source_region", {}) if isinstance(state.get("source_region", {}), dict) else {}
        try:
            best_values.append(float(state.get("best_objective", src.get("archive_best_objective", float("nan")))))
        except Exception:
            best_values.append(float("nan"))
        archive_counts = src.get("archive_counts", {}) if isinstance(src.get("archive_counts", {}), dict) else {}
        try:
            total = float(archive_counts.get("total", 0.0))
            required = max(float(archive_counts.get("min_required", 1.0)), 1.0)
            support_scores.append(float(np.clip(total / (2.0 * required), 0.0, 1.0)))
        except Exception:
            support_scores.append(0.0)
        success_counts.append(float(max(int(state.get("success_count", 0)), 0)))

    archive_norm = _normalize_rank_scores(best_values, objective_sense=objective_sense)
    support_norm = [float(np.clip(v, 0.0, 1.0)) for v in support_scores]
    success_norm = [float(np.clip(v / 3.0, 0.0, 1.0)) for v in success_counts]
    w_archive = float(max(float(getattr(system, "focus2_final_archive_weight", 0.70)), 0.0))
    w_support = float(max(float(getattr(system, "focus2_final_support_weight", 0.15)), 0.0))
    w_success = float(max(float(getattr(system, "focus2_final_success_weight", 0.10)), 0.0))
    no_improve_penalty_unit = float(max(float(getattr(system, "focus2_final_no_improve_penalty", getattr(system, "focus2_final_failure_penalty", 0.08))), 0.0))
    dropped_penalty = float(max(float(getattr(system, "focus2_final_dropped_penalty", 0.25)), 0.0))
    w_sum = max(w_archive + w_support + w_success, 1e-12)
    w_archive, w_support, w_success = w_archive / w_sum, w_support / w_sum, w_success / w_sum

    candidate_rows: list[dict[str, object]] = []
    best_idx = 0
    best_final_score = float("-inf")
    for idx, state in enumerate(candidates):
        no_improve_count = int(state.get("no_improve_count", state.get("failure_count", 0)))
        success_count = int(state.get("success_count", 0))
        status = str(state.get("status", "active"))
        drop_penalty = dropped_penalty if status == "dropped" else 0.0
        no_improve_penalty = no_improve_penalty_unit * float(max(no_improve_count, 0))
        final_score = (
            w_archive * archive_norm[idx]
            + w_support * support_norm[idx]
            + w_success * success_norm[idx]
            - no_improve_penalty
            - drop_penalty
        )
        row = {
            "region_id": str(state.get("region_id", "")),
            "status": status,
            "best_objective": float(best_values[idx]),
            "archive_score": float(archive_norm[idx]),
            "support_score": float(support_norm[idx]),
            "success_score": float(success_norm[idx]),
            "success_count": int(success_count),
            "no_improve_count": int(no_improve_count),
            "eval_count": int(state.get("eval_count", 0)),
            "final_score": float(final_score),
        }
        candidate_rows.append(row)
        if final_score > best_final_score:
            best_final_score = float(final_score)
            best_idx = idx

    selected = candidates[int(best_idx)]
    return selected, {
        "reason": "best_final_score",
        "weights": {
            "archive": float(w_archive),
            "support": float(w_support),
            "success": float(w_success),
            "no_improve_penalty": float(no_improve_penalty_unit),
            "dropped_penalty": float(dropped_penalty),
        },
        "selected_region_id": str(selected.get("region_id", "")),
        "selected_score": float(best_final_score),
        "used_active_only": bool(len(active) > 0),
        "candidates": candidate_rows,
    }


def _build_focus2_final_bounds(
    *,
    region_states: list[dict[str, object]],
    X_archive: np.ndarray | None = None,
    y_archive: np.ndarray | None = None,
    objective_sense: str,
    system: OptimizerSystemConfig,
    full_lb: np.ndarray,
    full_ub: np.ndarray,
) -> tuple[dict[str, object] | None, dict[str, object], np.ndarray | None, np.ndarray | None, float]:
    selected, selection = _select_focus2_final_region(
        region_states=region_states,
        objective_sense=objective_sense,
        system=system,
    )
    if selected is None:
        return None, selection, None, None, float("nan")

    union_enabled = bool(getattr(system, "focus2_final_union_enabled", True))
    top_k = int(max(int(getattr(system, "focus2_final_union_top_k", 2)), 1))
    include_dropped_competitive = bool(getattr(system, "focus2_final_include_dropped_competitive", True))
    use_parent = bool(getattr(system, "focus2_final_union_use_parent_bounds", False))
    archive_trim_enabled = bool(getattr(system, "focus2_final_archive_trim_enabled", True))
    trim_top_fraction = float(np.clip(float(getattr(system, "focus2_final_archive_trim_top_fraction", 0.50)), 0.0, 1.0))
    trim_min_points = int(max(int(getattr(system, "focus2_final_archive_trim_min_points", 20)), 1))
    trim_min_per_dim = float(max(float(getattr(system, "focus2_final_archive_trim_min_per_dim", 4.0)), 0.0))
    trim_keep_min_points = int(max(int(getattr(system, "focus2_final_archive_trim_keep_min_points", 10)), 1))
    trim_keep_min_per_dim = float(max(float(getattr(system, "focus2_final_archive_trim_keep_min_per_dim", 2.0)), 0.0))
    trim_qlo = float(np.clip(float(getattr(system, "focus2_final_archive_trim_quantile_low", 0.10)), 0.0, 0.49))
    trim_qhi = float(np.clip(float(getattr(system, "focus2_final_archive_trim_quantile_high", 0.90)), trim_qlo + 0.01, 1.0))
    trim_blend = float(np.clip(float(getattr(system, "focus2_final_archive_trim_blend", 0.30)), 0.0, 1.0))
    expand_ratio = float(max(float(getattr(system, "focus2_final_bounds_expand_ratio", 1.20)), 1.0))
    min_volume = float(np.clip(float(getattr(system, "focus2_final_bounds_min_volume_ratio", 0.25)), 0.0, 1.0))
    max_volume = float(np.clip(float(getattr(system, "focus2_final_bounds_max_volume_ratio", 1.0)), 1e-12, 1.0))

    state_by_id = {
        str(s.get("region_id", "")): s
        for s in region_states
        if isinstance(s, dict) and str(s.get("region_id", ""))
    }
    rows = selection.get("candidates", []) if isinstance(selection, dict) else []
    ranked_rows = [r for r in rows if isinstance(r, dict) and str(r.get("region_id", "")) in state_by_id]
    ranked_rows.sort(key=lambda r: float(r.get("final_score", float("-inf"))), reverse=True)

    selected_rows: list[dict[str, object]] = []
    if union_enabled and ranked_rows:
        if include_dropped_competitive:
            eligible_rows = [r for r in ranked_rows if str(r.get("status", "")) != "merged"]
        else:
            active_rows = [r for r in ranked_rows if str(r.get("status", "active")) == "active"]
            eligible_rows = active_rows if active_rows else [r for r in ranked_rows if str(r.get("status", "")) != "merged"]
        if not eligible_rows:
            eligible_rows = ranked_rows
        selected_rows = eligible_rows[: min(top_k, len(eligible_rows))]

    if not selected_rows:
        selected_rows = [
            {
                "region_id": str(selected.get("region_id", "")),
                "status": str(selected.get("status", "active")),
                "final_score": float(selection.get("selected_score", float("nan"))) if isinstance(selection, dict) else float("nan"),
            }
        ]

    states: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for row in selected_rows:
        rid = str(row.get("region_id", ""))
        state = state_by_id.get(rid)
        if not isinstance(state, dict) or rid in seen_ids:
            continue
        states.append(state)
        seen_ids.add(rid)
    if not states:
        states = [selected]

    fallback_selected = any(
        str(s.get("status", "")).lower() == "fallback"
        or str(s.get("region_id", "")).startswith("region_fallback")
        for s in states
        if isinstance(s, dict)
    )
    if bool(fallback_selected):
        trim_blend = float(min(
            trim_blend,
            np.clip(float(getattr(system, "focus2_final_fallback_archive_trim_blend", 0.0)), 0.0, 1.0),
        ))
        expand_ratio = float(max(
            expand_ratio,
            float(getattr(system, "focus2_final_fallback_bounds_expand_ratio", 1.35)),
        ))
        min_volume = float(max(
            min_volume,
            np.clip(float(getattr(system, "focus2_final_fallback_min_volume_ratio", 0.40)), 0.0, 1.0),
        ))

    lb_key = "parent_lb" if use_parent else "bounds_lb"
    ub_key = "parent_ub" if use_parent else "bounds_ub"
    lbs = [np.asarray(s.get(lb_key, s.get("bounds_lb")), dtype=float).reshape(-1) for s in states]
    ubs = [np.asarray(s.get(ub_key, s.get("bounds_ub")), dtype=float).reshape(-1) for s in states]
    valid = [
        idx
        for idx, (lo, hi) in enumerate(zip(lbs, ubs))
        if lo.shape == full_lb.shape and hi.shape == full_ub.shape and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(hi > lo)
    ]
    if not valid:
        final_lb = np.asarray(selected.get("bounds_lb"), dtype=float).copy()
        final_ub = np.asarray(selected.get("bounds_ub"), dtype=float).copy()
    else:
        final_lb = np.min(np.asarray([lbs[idx] for idx in valid], dtype=float), axis=0)
        final_ub = np.max(np.asarray([ubs[idx] for idx in valid], dtype=float), axis=0)
    union_lb = final_lb.copy()
    union_ub = final_ub.copy()

    p_dim = int(max(full_lb.shape[0], 1))
    trim_required_points = int(max(trim_min_points, math.ceil(trim_min_per_dim * float(p_dim))))
    trim_keep_min = int(max(trim_keep_min_points, math.ceil(trim_keep_min_per_dim * float(p_dim))))
    trim_info: dict[str, object] = {
        "enabled": bool(archive_trim_enabled),
        "applied": False,
        "reason": "disabled" if not archive_trim_enabled else "",
        "top_fraction": float(trim_top_fraction),
        "min_points": int(trim_min_points),
        "min_per_dim": float(trim_min_per_dim),
        "required_points": int(trim_required_points),
        "keep_min_points": int(trim_keep_min_points),
        "keep_min_per_dim": float(trim_keep_min_per_dim),
        "keep_required_points": int(trim_keep_min),
        "quantile_low": float(trim_qlo),
        "quantile_high": float(trim_qhi),
        "blend": float(trim_blend),
        "fallback_selected": bool(fallback_selected),
    }
    if archive_trim_enabled:
        X_arr = np.asarray(X_archive, dtype=float) if X_archive is not None else np.empty((0, full_lb.shape[0]), dtype=float)
        y_arr = np.asarray(y_archive, dtype=float).reshape(-1) if y_archive is not None else np.empty((0,), dtype=float)
        if X_arr.ndim != 2 or X_arr.shape[1] != full_lb.shape[0] or y_arr.shape[0] != X_arr.shape[0]:
            trim_info["reason"] = "invalid_archive_shape"
        else:
            finite = np.isfinite(y_arr) & np.all(np.isfinite(X_arr), axis=1)
            inside = np.all((X_arr >= final_lb.reshape(1, -1)) & (X_arr <= final_ub.reshape(1, -1)), axis=1)
            mask = finite & inside
            inside_count = int(np.sum(mask))
            trim_info["archive_inside_count"] = int(inside_count)
            if trim_blend <= 0.0:
                trim_info["reason"] = "blend_zero"
            elif inside_count < max(trim_required_points, trim_keep_min):
                trim_info["reason"] = "insufficient_archive_inside"
            else:
                X_inside = X_arr[mask]
                y_inside = y_arr[mask]
                if objective_sense == "max":
                    order = np.argsort(-y_inside)
                else:
                    order = np.argsort(y_inside)
                keep_count = int(max(trim_keep_min, math.ceil(float(inside_count) * max(trim_top_fraction, 1e-12))))
                keep_count = int(min(keep_count, inside_count))
                keep = order[:keep_count]
                X_top = X_inside[keep]
                raw_lb = np.quantile(X_top, trim_qlo, axis=0)
                raw_ub = np.quantile(X_top, trim_qhi, axis=0)
                bad = ~np.isfinite(raw_lb) | ~np.isfinite(raw_ub) | (raw_ub <= raw_lb)
                if np.any(bad):
                    trim_info["reason"] = "invalid_trim_quantile_bounds"
                else:
                    trim_volume_before_policy = _bounds_volume_ratio(
                        lb=raw_lb,
                        ub=raw_ub,
                        base_lb=full_lb,
                        base_ub=full_ub,
                    )
                    union_center = 0.5 * (union_lb + union_ub)
                    union_width = np.maximum(union_ub - union_lb, 1e-12)
                    trim_center = 0.5 * (raw_lb + raw_ub)
                    trim_width = np.maximum(raw_ub - raw_lb, 1e-12)
                    blended_center = (1.0 - trim_blend) * union_center + trim_blend * trim_center
                    blended_width = (1.0 - trim_blend) * union_width + trim_blend * trim_width
                    final_lb, final_ub = _centered_bounds_from_width(
                        center=blended_center,
                        width=blended_width,
                        full_lb=full_lb,
                        full_ub=full_ub,
                    )
                    volume_after_blend = _bounds_volume_ratio(
                        lb=final_lb,
                        ub=final_ub,
                        base_lb=full_lb,
                        base_ub=full_ub,
                    )
                    trim_info.update({
                        "applied": True,
                        "reason": "archive_top_quantile_blend",
                        "kept_points": int(keep_count),
                        "volume_before_policy": float(trim_volume_before_policy),
                        "volume_after_blend": float(volume_after_blend),
                    })

    if expand_ratio > 1.0:
        center = 0.5 * (final_lb + final_ub)
        width = np.maximum(final_ub - final_lb, 1e-12) * expand_ratio
        final_lb, final_ub = _centered_bounds_from_width(
            center=center,
            width=width,
            full_lb=full_lb,
            full_ub=full_ub,
        )

    final_lb, final_ub, final_volume = _enforce_min_volume_ratio(
        focus_lb=final_lb,
        focus_ub=final_ub,
        full_lb=full_lb,
        full_ub=full_ub,
        min_volume_ratio=min_volume,
    )
    if max_volume < 1.0:
        final_lb, final_ub, final_volume = _enforce_max_volume_ratio(
            focus_lb=final_lb,
            focus_ub=final_ub,
            full_lb=full_lb,
            full_ub=full_ub,
            max_volume_ratio=max_volume,
        )

    selection["final_bounds_policy"] = {
        "union_enabled": bool(union_enabled),
        "union_top_k": int(top_k),
        "include_dropped_competitive": bool(include_dropped_competitive),
        "use_parent_bounds": bool(use_parent),
        "archive_trim": trim_info,
        "expand_ratio": float(expand_ratio),
        "min_volume_ratio": float(min_volume),
        "max_volume_ratio": float(max_volume),
        "fallback_selected": bool(fallback_selected),
        "selected_region_ids": [str(s.get("region_id", "")) for s in states],
    }
    selection["selected_region_ids"] = [str(s.get("region_id", "")) for s in states]
    return selected, selection, final_lb, final_ub, float(final_volume)


def _focus2_should_early_stop(
    *,
    region_states: list[dict[str, object]],
    focus2_eval_count: int,
    objective_sense: str,
    system: OptimizerSystemConfig,
) -> tuple[bool, dict[str, object]]:
    if not bool(getattr(system, "focus2_early_stop_enabled", True)):
        return False, {"enabled": False, "reason": "disabled"}
    min_evals = int(max(int(getattr(system, "focus2_early_stop_min_evals", 1)), 0))
    if int(focus2_eval_count) < min_evals:
        return False, {
            "enabled": True,
            "reason": "insufficient_focus2_evals",
            "focus2_eval_count": int(focus2_eval_count),
            "min_evals": int(min_evals),
        }
    active = [s for s in region_states if isinstance(s, dict) and str(s.get("status", "active")) == "active"]
    if not active:
        return True, {"enabled": True, "reason": "no_active_regions"}

    min_evals_per_region = int(max(int(getattr(system, "focus2_early_stop_min_evals_per_region", 3)), 0))
    min_selected_evals = int(max(int(getattr(system, "focus2_early_stop_min_selected_region_evals", 3)), 0))
    min_active_updates = int(max(int(getattr(system, "focus2_early_stop_min_active_bounds_updates", 2)), 0))
    require_all_active_sampled = bool(getattr(system, "focus2_early_stop_require_all_active_sampled", True))

    selected, selection = _select_focus2_final_region(
        region_states=region_states,
        objective_sense=objective_sense,
        system=system,
    )
    if selected is None:
        return True, {"enabled": True, "reason": "no_selectable_region", "selection": selection}

    max_active = int(max(int(getattr(system, "focus2_early_stop_max_active_regions", 1)), 1))
    max_volume = float(np.clip(float(getattr(system, "focus2_early_stop_max_volume_ratio", 0.25)), 1e-12, 1.0))
    score_gap_threshold = float(max(float(getattr(system, "focus2_early_stop_score_gap", 0.20)), 0.0))
    min_archive = int(max(int(getattr(system, "focus2_early_stop_min_archive_points", 5)), 1))
    selected_volume = float(selected.get("active_bounds_volume_ratio", float("nan")))
    if not math.isfinite(selected_volume):
        src = selected.get("source_region", {}) if isinstance(selected.get("source_region", {}), dict) else {}
        selected_volume = float(src.get("size_ratio", float("inf")))
    src = selected.get("source_region", {}) if isinstance(selected.get("source_region", {}), dict) else {}
    archive_counts = src.get("archive_counts", {}) if isinstance(src.get("archive_counts", {}), dict) else {}
    archive_count = int(archive_counts.get("total", 0) or 0)
    selected_eval_count = int(selected.get("eval_count", 0)) + int(selected.get("shared_eval_count", 0))
    selected_evidence_count = int(archive_count) + int(selected_eval_count)
    active_eval_counts = [
        int(s.get("eval_count", 0)) + int(s.get("shared_eval_count", 0))
        for s in active
    ]
    all_active_sampled_ok = (
        not require_all_active_sampled
        or all(count >= min_evals_per_region for count in active_eval_counts)
    )
    selected_eval_ok = selected_eval_count >= min_selected_evals
    active_bounds_update_count = int(selected.get("active_bounds_update_count", 0))
    active_bounds_update_ok = active_bounds_update_count >= min_active_updates
    support_ok = selected_evidence_count >= min_archive
    volume_ok = math.isfinite(selected_volume) and selected_volume <= max_volume

    candidates = selection.get("candidates", []) if isinstance(selection, dict) else []
    score_gap = 0.0
    if isinstance(candidates, list) and len(candidates) >= 2:
        scores = sorted(
            [float(c.get("final_score", float("-inf"))) for c in candidates if isinstance(c, dict)],
            reverse=True,
        )
        if len(scores) >= 2 and math.isfinite(scores[0]) and math.isfinite(scores[1]):
            score_gap = float(scores[0] - scores[1])

    reason = ""
    should_stop = False
    evidence_ok = bool(all_active_sampled_ok and selected_eval_ok and active_bounds_update_ok)
    if evidence_ok and support_ok and volume_ok and len(active) <= max_active:
        should_stop = True
        reason = "active_region_count_target_reached"
    elif evidence_ok and support_ok and volume_ok and score_gap >= score_gap_threshold:
        should_stop = True
        reason = "selected_region_score_gap_reached"
    else:
        reason = "conditions_not_met"

    return bool(should_stop), {
        "enabled": True,
        "reason": reason,
        "active_region_count": int(len(active)),
        "max_active_regions": int(max_active),
        "selected_region_id": str(selected.get("region_id", "")),
        "selected_volume_ratio": float(selected_volume),
        "max_volume_ratio": float(max_volume),
        "archive_count": int(archive_count),
        "selected_eval_count": int(selected_eval_count),
        "selected_evidence_count": int(selected_evidence_count),
        "min_selected_region_evals": int(min_selected_evals),
        "active_eval_counts": [int(v) for v in active_eval_counts],
        "min_evals_per_region": int(min_evals_per_region),
        "all_active_sampled_ok": bool(all_active_sampled_ok),
        "active_bounds_update_count": int(active_bounds_update_count),
        "min_active_bounds_updates": int(min_active_updates),
        "active_bounds_update_ok": bool(active_bounds_update_ok),
        "min_archive_points": int(min_archive),
        "score_gap": float(score_gap),
        "score_gap_threshold": float(score_gap_threshold),
        "support_ok": bool(support_ok),
        "volume_ok": bool(volume_ok),
        "evidence_ok": bool(evidence_ok),
        "selection": selection,
    }


def _build_focus2_boundary_starts(
    *,
    rng: np.random.Generator,
    lb: np.ndarray,
    ub: np.ndarray,
    pool_size: int,
    boundary_near_ratio: float,
) -> np.ndarray:
    n_dim = int(lb.shape[0])
    n = int(max(pool_size, 1))
    span = np.maximum(ub - lb, 1e-12)
    near = float(np.clip(boundary_near_ratio, 0.0, 0.5))
    starts: list[np.ndarray] = []

    for j in range(n_dim):
        for side in ("lower", "upper"):
            x = rng.uniform(lb, ub, size=(n_dim,))
            if side == "lower":
                x[j] = lb[j] + near * span[j] * rng.uniform(0.0, 1.0)
            else:
                x[j] = ub[j] - near * span[j] * rng.uniform(0.0, 1.0)
            starts.append(np.clip(x, lb, ub))
            if len(starts) >= n:
                return np.asarray(starts, dtype=float)

    while len(starts) < n:
        x = rng.uniform(lb, ub, size=(n_dim,))
        j = int(rng.integers(0, n_dim))
        if rng.uniform(0.0, 1.0) < 0.5:
            x[j] = lb[j] + near * span[j] * rng.uniform(0.0, 1.0)
        else:
            x[j] = ub[j] - near * span[j] * rng.uniform(0.0, 1.0)
        starts.append(np.clip(x, lb, ub))
    return np.asarray(starts, dtype=float)


def _acquisition_scores_for_pool(
    *,
    pool: np.ndarray,
    model: object | None,
    y_best: float,
    objective_sense: str,
    acq_type: str,
    kappa: float,
    xi: float,
    post_feasible_prob_fn: Callable[[np.ndarray], float] | None,
    post_penalty_lambda: float,
) -> np.ndarray:
    if pool.ndim != 2 or pool.shape[0] == 0:
        return np.empty((0,), dtype=float)
    if model is None:
        return np.zeros((pool.shape[0],), dtype=float)

    acq = str(acq_type or "LCB").strip().upper()
    try:
        mu, sigma = model.predict(pool, return_std=True)
        mu = np.asarray(mu, dtype=float).reshape(-1)
        sigma = np.asarray(sigma, dtype=float).reshape(-1)
        if acq == "EI":
            safe_sigma = np.maximum(sigma, 1e-9)
            if objective_sense == "max":
                imp = mu - float(y_best) - float(xi)
            else:
                imp = float(y_best) - mu - float(xi)
            z = imp / safe_sigma
            from scipy.stats import norm

            scores = -(imp * norm.cdf(z) + safe_sigma * norm.pdf(z))
        elif acq == "MEAN":
            if objective_sense == "max":
                scores = -mu
            else:
                scores = mu
        else:
            if objective_sense == "max":
                scores = -(mu + float(kappa) * sigma)
            else:
                scores = mu - float(kappa) * sigma
        scores = np.asarray(scores, dtype=float).reshape(-1)
        scores[~np.isfinite(scores)] = float("inf")
    except Exception:
        scores = np.full((pool.shape[0],), float("inf"), dtype=float)

    if post_feasible_prob_fn is not None and post_penalty_lambda > 0.0:
        penalties: list[float] = []
        for row in pool:
            try:
                p_post = float(np.clip(post_feasible_prob_fn(np.asarray(row, dtype=float).reshape(-1)), 0.0, 1.0))
            except Exception:
                p_post = 1.0
            penalties.append(float(post_penalty_lambda) * (1.0 - p_post))
        scores = scores + np.asarray(penalties, dtype=float)
    return scores


def _allocate_source_quotas(
    *,
    total: int,
    probs: dict[str, float],
) -> dict[str, int]:
    sources = [str(s) for s in probs.keys() if str(s)]
    if not sources:
        sources = ["topk"]
    n_total = int(max(total, 1))
    vals = np.asarray([max(float(probs.get(s, 0.0)), 0.0) for s in sources], dtype=float)
    if float(np.sum(vals)) <= 0.0:
        vals = np.zeros((len(sources),), dtype=float)
        vals[0] = 1.0
    vals = vals / float(np.sum(vals))

    quotas = {s: int(math.floor(float(p) * n_total)) for s, p in zip(sources, vals)}
    for s, p in zip(sources, vals):
        if p > 0.0 and quotas[s] == 0 and sum(quotas.values()) < n_total:
            quotas[s] = 1

    while sum(quotas.values()) > n_total:
        candidates = [s for s in sources if quotas[s] > (1 if vals[sources.index(s)] > 0.0 else 0)]
        if not candidates:
            candidates = [s for s in sources if quotas[s] > 0]
        s_drop = max(candidates, key=lambda s: quotas[s])
        quotas[s_drop] -= 1

    remainders = {
        s: float(vals[idx] * n_total - math.floor(float(vals[idx]) * n_total))
        for idx, s in enumerate(sources)
    }
    while sum(quotas.values()) < n_total:
        s_add = max(sources, key=lambda s: remainders[s])
        quotas[s_add] += 1
        remainders[s_add] = 0.0

    return {s: int(quotas[s]) for s in sources}


def _select_top_scored_rows(
    *,
    pool: np.ndarray,
    scores: np.ndarray,
    n: int,
) -> np.ndarray:
    if pool.ndim != 2 or pool.shape[0] == 0 or int(n) <= 0:
        return np.empty((0, pool.shape[1] if pool.ndim == 2 else 0), dtype=float)
    score_arr = np.asarray(scores, dtype=float).reshape(-1)
    score_arr = np.where(np.isfinite(score_arr), score_arr, float("inf"))
    order = np.argsort(score_arr)
    keep = order[: min(int(n), pool.shape[0])]
    return np.asarray(pool[keep], dtype=float)


def _select_cover_scored_rows(
    *,
    rng: np.random.Generator,
    pool: np.ndarray,
    scores: np.ndarray,
    n: int,
    X_ref: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    acq_quantile: float,
    reference_max: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if pool.ndim != 2 or pool.shape[0] == 0 or int(n) <= 0:
        width = pool.shape[1] if pool.ndim == 2 else 0
        return np.empty((0, width), dtype=float), {
            "enabled": True,
            "candidate_count": 0,
            "acq_threshold": float("nan"),
            "mean_distance": float("nan"),
            "min_distance": float("nan"),
        }

    score_arr = np.asarray(scores, dtype=float).reshape(-1)
    finite = np.isfinite(score_arr)
    if not np.any(finite):
        part = _select_top_scored_rows(pool=pool, scores=scores, n=n)
        return part, {
            "enabled": True,
            "candidate_count": int(pool.shape[0]),
            "acq_threshold": float("nan"),
            "mean_distance": float("nan"),
            "min_distance": float("nan"),
        }

    q = float(np.clip(float(acq_quantile), 0.01, 1.0))
    threshold = float(np.quantile(score_arr[finite], q))
    candidates = np.where(finite & (score_arr <= threshold))[0]
    if candidates.size < int(n):
        candidates = np.where(finite)[0]
    if candidates.size == 0:
        candidates = np.arange(pool.shape[0])

    ref = np.asarray(X_ref, dtype=float)
    if ref.ndim != 2 or ref.shape[0] == 0 or ref.shape[1] != pool.shape[1]:
        part = _select_top_scored_rows(pool=pool[candidates], scores=score_arr[candidates], n=n)
        return part, {
            "enabled": True,
            "candidate_count": int(candidates.size),
            "acq_threshold": float(threshold),
            "mean_distance": float("nan"),
            "min_distance": float("nan"),
        }

    ref_max = int(max(int(reference_max), 1))
    if ref.shape[0] > ref_max:
        keep = rng.choice(ref.shape[0], size=ref_max, replace=False)
        ref = ref[np.asarray(keep, dtype=int)]

    span = np.maximum(np.asarray(ub, dtype=float) - np.asarray(lb, dtype=float), 1e-12)
    ref_norm = (ref - np.asarray(lb, dtype=float).reshape(1, -1)) / span.reshape(1, -1)
    pool_norm = (np.asarray(pool, dtype=float) - np.asarray(lb, dtype=float).reshape(1, -1)) / span.reshape(1, -1)

    selected: list[int] = []
    remaining = [int(v) for v in candidates.tolist()]
    distance_values: list[float] = []
    while remaining and len(selected) < int(n):
        cand_norm = pool_norm[np.asarray(remaining, dtype=int)]
        d = np.linalg.norm(cand_norm[:, None, :] - ref_norm[None, :, :], axis=2)
        min_d = np.min(d, axis=1)
        pos = int(np.argmax(min_d))
        idx = int(remaining[pos])
        selected.append(idx)
        distance_values.append(float(min_d[pos]))
        ref_norm = np.vstack([ref_norm, pool_norm[idx].reshape(1, -1)])
        remaining.pop(pos)

    if len(selected) < int(n):
        selected_set = set(selected)
        fill_order = candidates[np.argsort(score_arr[candidates])]
        for idx in fill_order.tolist():
            if int(idx) in selected_set:
                continue
            selected.append(int(idx))
            selected_set.add(int(idx))
            if len(selected) >= int(n):
                break

    keep = np.asarray(selected[: int(n)], dtype=int)
    part = np.asarray(pool[keep], dtype=float)
    return part, {
        "enabled": True,
        "candidate_count": int(candidates.size),
        "acq_threshold": float(threshold),
        "mean_distance": float(np.mean(distance_values)) if distance_values else float("nan"),
        "min_distance": float(np.min(distance_values)) if distance_values else float("nan"),
    }


def _normalized_rms_min_distance(
    *,
    x: np.ndarray,
    X_ref: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> float:
    ref = np.asarray(X_ref, dtype=float)
    if ref.ndim != 2 or ref.shape[0] == 0:
        return float("inf")
    span = np.maximum(np.asarray(ub, dtype=float) - np.asarray(lb, dtype=float), 1e-12)
    x_norm = (np.asarray(x, dtype=float).reshape(1, -1) - lb.reshape(1, -1)) / span.reshape(1, -1)
    ref_norm = (ref - lb.reshape(1, -1)) / span.reshape(1, -1)
    d = np.sqrt(np.mean((ref_norm - x_norm) ** 2, axis=1))
    return float(np.min(d)) if d.size else float("inf")


def _focus1_tau(
    *,
    y: np.ndarray,
    objective_sense: str,
    quantile: float,
) -> float:
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    y_arr = y_arr[np.isfinite(y_arr)]
    if y_arr.size == 0:
        return float("nan")
    q = float(np.clip(float(quantile), 0.01, 0.99))
    if objective_sense == "max":
        q = 1.0 - q
    return float(np.quantile(y_arr, q))


def _focus1_target_count(
    *,
    system: OptimizerSystemConfig,
    p_dim: int,
) -> int:
    ratio = float(max(float(getattr(system, "focus1_target_np_ratio", 10.0)), 0.0))
    return int(max(math.ceil(ratio * max(int(p_dim), 1)), 0))


def _focus0_target_count(
    *,
    system: OptimizerSystemConfig,
    p_dim: int,
) -> int:
    ratio = float(max(float(getattr(system, "focus0_min_np_ratio", 1.0)), 0.0))
    by_ratio = int(math.ceil(ratio * max(int(p_dim), 1)))
    by_abs = int(max(int(getattr(system, "focus0_min_points", 3)), 0))
    return int(max(by_ratio, by_abs, 2))


def _resolve_focus1_pool_size(
    *,
    system: OptimizerSystemConfig,
    p_dim: int,
) -> int:
    p = int(max(p_dim, 1))
    min_pool = int(max(int(getattr(system, "focus1_candidate_pool_min", 2048)), 1))
    per_dim = int(max(int(getattr(system, "focus1_candidate_pool_per_dim", 100)), 1))
    max_pool = int(max(int(getattr(system, "focus1_candidate_pool_max", 20000)), min_pool))
    return int(min(max(per_dim * p, min_pool), max_pool))


def _normalize_focus1_quotas(
    *,
    n: int,
    system: OptimizerSystemConfig,
) -> dict[str, int]:
    sources = ["good", "uncertain", "bad"]
    vals = np.asarray(
        [
            max(float(getattr(system, "focus1_good_ratio", 0.20)), 0.0),
            max(float(getattr(system, "focus1_uncertain_ratio", 0.60)), 0.0),
            max(float(getattr(system, "focus1_bad_ratio", 0.20)), 0.0),
        ],
        dtype=float,
    )
    if float(np.sum(vals)) <= 0.0:
        vals = np.asarray([0.2, 0.6, 0.2], dtype=float)
    vals = vals / float(np.sum(vals))
    total = int(max(n, 0))
    quotas = {s: int(math.floor(float(v) * total)) for s, v in zip(sources, vals)}
    remainders = {s: float(v * total - math.floor(float(v) * total)) for s, v in zip(sources, vals)}
    while sum(quotas.values()) < total:
        s_add = max(sources, key=lambda s: remainders[s])
        quotas[s_add] += 1
        remainders[s_add] = -1.0
    while sum(quotas.values()) > total:
        s_drop = max(sources, key=lambda s: quotas[s])
        quotas[s_drop] -= 1
    return {s: int(max(v, 0)) for s, v in quotas.items()}


def _dedup_pool_against_reference(
    *,
    pool: np.ndarray,
    X_ref: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    min_rms_distance: float,
    decimals: int,
) -> np.ndarray:
    if pool.ndim != 2 or pool.shape[0] == 0:
        return np.empty((0, pool.shape[1] if pool.ndim == 2 else 0), dtype=float)
    min_dist = float(max(float(min_rms_distance), 0.0))
    seen: set[tuple[float, ...]] = set()
    if X_ref.ndim == 2 and X_ref.shape[0] > 0:
        seen.update(_round_key(row, decimals=decimals) for row in X_ref)
    keep: list[np.ndarray] = []
    ref = np.asarray(X_ref, dtype=float) if X_ref.ndim == 2 else np.empty((0, pool.shape[1]), dtype=float)
    for row in np.asarray(pool, dtype=float):
        key = _round_key(row, decimals=decimals)
        if key in seen:
            continue
        if min_dist > 0.0 and _normalized_rms_min_distance(x=row, X_ref=ref, lb=lb, ub=ub) < min_dist:
            continue
        keep.append(np.asarray(row, dtype=float).reshape(-1))
        seen.add(key)
        ref = np.vstack([ref, np.asarray(row, dtype=float).reshape(1, -1)])
    if not keep:
        return np.empty((0, pool.shape[1]), dtype=float)
    return np.asarray(keep, dtype=float)


def _build_focus0_candidate_pool(
    *,
    rng: np.random.Generator,
    lb: np.ndarray,
    ub: np.ndarray,
    n: int,
    system: OptimizerSystemConfig,
    var_names: list[str],
    constraint_defs: list[dict],
    enforce_pre_constraints: bool,
) -> np.ndarray:
    p = int(lb.shape[0])
    n_target = int(max(n, 0))
    if n_target <= 0:
        return np.empty((0, p), dtype=float)

    bounds = [(float(lo), float(hi)) for lo, hi in zip(lb, ub)]
    boundary_ratio = float(np.clip(float(getattr(system, "focus0_initial_corner_ratio", 0.10)), 0.0, 1.0))
    n_boundary = int(round(float(n_target) * boundary_ratio))
    n_regular = int(max(n_target - n_boundary, 0))

    parts: list[np.ndarray] = []
    if n_boundary > 0:
        n_boundary_pool = n_boundary
        if enforce_pre_constraints and constraint_defs:
            n_boundary_pool = int(max(n_boundary_pool, math.ceil(n_boundary * float(getattr(system, "focus0_probe_multiplier", 2.0)))))
        X_boundary = sample_boundary_corners_random(
            bounds,
            offset=np.zeros((p,), dtype=float),
            n_samples=n_boundary_pool,
            rng=rng,
        )
        parts.append(X_boundary)

    if n_regular > 0:
        n_regular_pool = n_regular
        if enforce_pre_constraints and constraint_defs:
            n_regular_pool = int(max(n_regular_pool, math.ceil(n_regular * float(getattr(system, "focus0_probe_multiplier", 2.0)))))
        X_regular = latin_hypercube_sampling(
            n_samples=n_regular_pool,
            bounds=bounds,
            rng=rng,
            n_divisions=max(n_regular_pool, 1),
        )
        parts.append(X_regular)

    pool = np.vstack([p for p in parts if p.size > 0]) if parts else np.empty((0, p), dtype=float)
    if pool.shape[0] == 0:
        return pool
    pool = np.unique(np.asarray(pool, dtype=float), axis=0)

    if enforce_pre_constraints and constraint_defs:
        mask, _payloads, _margins = evaluate_constraints_batch(
            X=pool,
            var_names=var_names,
            constraint_defs=constraint_defs,
            scope="pre",
        )
        pool = pool[mask]
    if pool.shape[0] > n_target:
        idx = rng.choice(pool.shape[0], size=n_target, replace=False)
        pool = pool[np.asarray(idx, dtype=int)]
    return np.asarray(pool, dtype=float)


def _filter_pool_by_pre_constraints(
    *,
    pool: np.ndarray,
    labels: list[str] | None,
    var_names: list[str],
    constraint_defs: list[dict],
    enforce_pre_constraints: bool,
) -> tuple[np.ndarray, list[str], np.ndarray, dict[str, object]]:
    pool = np.asarray(pool, dtype=float)
    n_raw = int(pool.shape[0]) if pool.ndim == 2 else 0
    if pool.ndim != 2 or n_raw == 0:
        return (
            np.empty((0, 0), dtype=float) if pool.ndim != 2 else pool,
            [],
            np.empty((0,), dtype=float),
            {
                "active": bool(enforce_pre_constraints and constraint_defs),
                "raw_count": 0,
                "feasible_count": 0,
                "feasible_ratio": float("nan"),
                "margin_min": float("nan"),
                "margin_median": float("nan"),
                "margin_max": float("nan"),
            },
        )
    label_list = list(labels or ["unknown"] * n_raw)
    if len(label_list) < n_raw:
        label_list.extend(["unknown"] * (n_raw - len(label_list)))
    label_list = label_list[:n_raw]
    if not enforce_pre_constraints or not constraint_defs:
        return (
            pool,
            label_list,
            np.full((n_raw,), float("inf"), dtype=float),
            {
                "active": False,
                "raw_count": n_raw,
                "feasible_count": n_raw,
                "feasible_ratio": 1.0,
                "margin_min": float("inf"),
                "margin_median": float("inf"),
                "margin_max": float("inf"),
            },
        )

    mask, _payloads, margins = evaluate_constraints_batch(
        X=pool,
        var_names=var_names,
        constraint_defs=constraint_defs,
        scope="pre",
    )
    keep_idx = np.where(mask)[0]
    finite_margins = margins[np.isfinite(margins)]
    stats = {
        "active": True,
        "raw_count": n_raw,
        "feasible_count": int(keep_idx.size),
        "feasible_ratio": float(keep_idx.size) / float(max(n_raw, 1)),
        "margin_min": float(np.min(finite_margins)) if finite_margins.size else float("nan"),
        "margin_median": float(np.median(finite_margins)) if finite_margins.size else float("nan"),
        "margin_max": float(np.max(finite_margins)) if finite_margins.size else float("nan"),
    }
    return (
        pool[keep_idx],
        [label_list[int(i)] for i in keep_idx.tolist()],
        margins[keep_idx],
        stats,
    )


def _select_constraint_boundary_index(
    *,
    scores: np.ndarray,
    margins: np.ndarray,
    acq_quantile: float,
) -> int | None:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    margins = np.asarray(margins, dtype=float).reshape(-1)
    if scores.size == 0 or margins.size != scores.size:
        return None
    finite = np.where(np.isfinite(scores) & np.isfinite(margins) & (margins >= 0.0))[0]
    if finite.size == 0:
        return None
    q = float(np.clip(acq_quantile, 0.0, 1.0))
    threshold = float(np.quantile(scores[finite], q))
    eligible = finite[scores[finite] <= threshold]
    if eligible.size == 0:
        eligible = finite
    return int(eligible[np.argmin(margins[eligible])])


def _build_focus0_batch(
    *,
    rng: np.random.Generator,
    X_archive: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    n: int,
    system: OptimizerSystemConfig,
    var_names: list[str],
    constraint_defs: list[dict],
    enforce_pre_constraints: bool,
    decimals: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    n_batch = int(max(n, 0))
    p = int(lb.shape[0])
    if n_batch <= 0:
        return [], {"mode": "empty", "pool_count": 0}
    min_dist = float(max(float(getattr(system, "focus1_min_rms_distance", 0.03)), 0.0))
    result = build_focus0_initial_doe_points(
        n_samples=n_batch,
        bounds=[(float(lo), float(hi)) for lo, hi in zip(lb, ub)],
        var_names=var_names,
        constraint_defs=constraint_defs if enforce_pre_constraints else [],
        rng=rng,
        system=system,
    )
    pool = _dedup_pool_against_reference(
        pool=result.X,
        X_ref=X_archive,
        lb=lb,
        ub=ub,
        min_rms_distance=min_dist,
        decimals=decimals,
    )
    labels = list(result.source_labels)
    if result.X.shape[0] == len(labels) and pool.shape[0] != result.X.shape[0]:
        kept_labels: list[str] = []
        for row in pool:
            d = np.linalg.norm(result.X - np.asarray(row, dtype=float).reshape(1, -1), axis=1)
            kept_labels.append(labels[int(np.argmin(d))] if d.size else "focus0_lhs")
        labels = kept_labels
    if pool.shape[0] < n_batch:
        fill = rng.uniform(lb, ub, size=(max(n_batch * 4, n_batch), p))
        fill = _dedup_pool_against_reference(
            pool=fill,
            X_ref=np.vstack([X_archive, pool]) if pool.shape[0] > 0 else X_archive,
            lb=lb,
            ub=ub,
            min_rms_distance=0.0,
            decimals=decimals,
        )
        labels.extend(["focus0_lhs_fallback"] * int(fill.shape[0]))
        pool = np.vstack([pool, fill]) if pool.shape[0] > 0 else fill
    rows = [
        {
            "x": np.asarray(row, dtype=float).reshape(-1),
            "class": str(labels[idx]) if idx < len(labels) else "focus0_lhs",
            "score": float("nan"),
        }
        for idx, row in enumerate(pool[:n_batch])
    ]
    # Focus0 diagnostics are intentionally not carried into public optimizer rows.
    # Keep the sampler fields internal unless a dedicated debug summary is added later.
    return rows, {
        "mode": "focus0",
        "pool_count": int(pool.shape[0]),
    }


def _select_focus1_rows_from_order(
    *,
    pool: np.ndarray,
    order: np.ndarray,
    n: int,
    X_ref: np.ndarray,
    selected: list[np.ndarray],
    lb: np.ndarray,
    ub: np.ndarray,
    min_rms_distance: float,
    decimals: int,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    ref = np.asarray(X_ref, dtype=float) if X_ref.ndim == 2 else np.empty((0, pool.shape[1]), dtype=float)
    if selected:
        ref = np.vstack([ref, np.asarray(selected, dtype=float)])
    seen = {_round_key(row, decimals=decimals) for row in ref} if ref.shape[0] > 0 else set()
    for idx in np.asarray(order, dtype=int).tolist():
        if len(out) >= int(n):
            break
        row = np.asarray(pool[int(idx)], dtype=float).reshape(-1)
        key = _round_key(row, decimals=decimals)
        if key in seen:
            continue
        if float(min_rms_distance) > 0.0 and _normalized_rms_min_distance(x=row, X_ref=ref, lb=lb, ub=ub) < float(min_rms_distance):
            continue
        out.append(row)
        selected.append(row)
        seen.add(key)
        ref = np.vstack([ref, row.reshape(1, -1)])
    return out


def _build_focus1_batch(
    *,
    rng: np.random.Generator,
    model: object | None,
    X_archive: np.ndarray,
    y_archive: np.ndarray,
    objective_sense: str,
    lb: np.ndarray,
    ub: np.ndarray,
    n: int,
    system: OptimizerSystemConfig,
    var_names: list[str],
    constraint_defs: list[dict],
    enforce_pre_constraints: bool,
    decimals: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    n_batch = int(max(n, 0))
    p = int(lb.shape[0])
    if n_batch <= 0:
        return [], {"mode": "empty", "tau": float("nan"), "pool_count": 0}

    min_dist = float(max(float(getattr(system, "focus1_min_rms_distance", 0.03)), 0.0))
    if model is None or y_archive.shape[0] < int(max(int(getattr(system, "focus1_min_gp_points", 3)), 2)):
        return [], {
            "mode": "insufficient_gp",
            "tau": float("nan"),
            "pool_count": 0,
            "good_count": 0,
            "uncertain_count": 0,
            "bad_count": 0,
        }

    pool_size = _resolve_focus1_pool_size(system=system, p_dim=p)
    raw_pool = _build_focus0_candidate_pool(
        rng=rng,
        lb=lb,
        ub=ub,
        n=pool_size,
        system=system,
        var_names=var_names,
        constraint_defs=constraint_defs,
        enforce_pre_constraints=False,
    )
    pool, _pool_labels, pool_margins, constraint_stats = _filter_pool_by_pre_constraints(
        pool=raw_pool,
        labels=None,
        var_names=var_names,
        constraint_defs=constraint_defs,
        enforce_pre_constraints=enforce_pre_constraints,
    )
    pool = _dedup_pool_against_reference(
        pool=pool,
        X_ref=X_archive,
        lb=lb,
        ub=ub,
        min_rms_distance=min_dist,
        decimals=decimals,
    )
    if pool.shape[0] == 0:
        return _build_focus1_batch(
            rng=rng,
            model=None,
            X_archive=X_archive,
            y_archive=y_archive,
            objective_sense=objective_sense,
            lb=lb,
            ub=ub,
            n=n_batch,
            system=system,
            var_names=var_names,
            constraint_defs=constraint_defs,
            enforce_pre_constraints=enforce_pre_constraints,
            decimals=decimals,
        )

    mu, sigma = model.predict(pool, return_std=True)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    sigma = np.maximum(np.asarray(sigma, dtype=float).reshape(-1), 0.0)
    beta = float(max(float(getattr(system, "focus1_beta", 1.0)), 0.0))
    tau = _focus1_tau(
        y=y_archive,
        objective_sense=objective_sense,
        quantile=float(getattr(system, "focus1_tau_quantile", 0.20)),
    )
    lower = mu - beta * sigma
    upper = mu + beta * sigma
    if objective_sense == "max":
        good_mask = lower >= tau
        bad_mask = upper < tau
        uncertain_mask = ~(good_mask | bad_mask)
        good_score = -upper
        uncertain_score = -(beta * sigma - np.abs(mu - tau))
        bad_score = tau - upper
    else:
        good_mask = upper <= tau
        bad_mask = lower > tau
        uncertain_mask = ~(good_mask | bad_mask)
        good_score = lower
        uncertain_score = -(beta * sigma - np.abs(mu - tau))
        bad_score = lower - tau

    quotas = _normalize_focus1_quotas(n=n_batch, system=system)
    selected: list[np.ndarray] = []
    selected_rows: list[dict[str, object]] = []

    class_specs = [
        ("good", np.where(good_mask)[0], good_score, quotas.get("good", 0)),
        ("uncertain", np.where(uncertain_mask)[0], uncertain_score, quotas.get("uncertain", 0)),
        ("bad", np.where(bad_mask)[0], bad_score, quotas.get("bad", 0)),
    ]
    for class_name, idxs, score_arr, quota in class_specs:
        if int(quota) <= 0 or idxs.size == 0:
            continue
        order = idxs[np.argsort(score_arr[idxs])]
        before = len(selected)
        rows = _select_focus1_rows_from_order(
            pool=pool,
            order=order,
            n=int(quota),
            X_ref=X_archive,
            selected=selected,
            lb=lb,
            ub=ub,
            min_rms_distance=min_dist,
            decimals=decimals,
        )
        for row in rows:
            idx = int(np.argmin(np.linalg.norm(pool - row.reshape(1, -1), axis=1)))
            selected_rows.append({"x": row, "class": class_name, "score": float(score_arr[idx])})
        if len(selected) == before and min_dist > 0.0:
            rows = _select_focus1_rows_from_order(
                pool=pool,
                order=order,
                n=int(quota),
                X_ref=X_archive,
                selected=selected,
                lb=lb,
                ub=ub,
                min_rms_distance=0.0,
                decimals=decimals,
            )
            for row in rows:
                idx = int(np.argmin(np.linalg.norm(pool - row.reshape(1, -1), axis=1)))
                selected_rows.append({"x": row, "class": class_name, "score": float(score_arr[idx])})

    if len(selected_rows) < n_batch:
        combined_score = np.where(uncertain_mask, uncertain_score, np.where(good_mask, good_score, bad_score))
        order = np.argsort(combined_score)
        rows = _select_focus1_rows_from_order(
            pool=pool,
            order=order,
            n=n_batch - len(selected_rows),
            X_ref=X_archive,
            selected=selected,
            lb=lb,
            ub=ub,
            min_rms_distance=0.0,
            decimals=decimals,
        )
        for row in rows:
            idx = int(np.argmin(np.linalg.norm(pool - row.reshape(1, -1), axis=1)))
            cls = "uncertain" if bool(uncertain_mask[idx]) else ("good" if bool(good_mask[idx]) else "bad")
            selected_rows.append({"x": row, "class": cls, "score": float(combined_score[idx])})

    return selected_rows[:n_batch], {
        "mode": "classification",
        "tau": float(tau),
        "pool_count": int(pool.shape[0]),
        "raw_pool_count": int(constraint_stats.get("raw_count", pool.shape[0])),
        "pre_feasible_count": int(constraint_stats.get("feasible_count", pool.shape[0])),
        "pre_feasible_ratio": float(constraint_stats.get("feasible_ratio", 1.0)),
        "pre_margin_min": float(constraint_stats.get("margin_min", float("nan"))),
        "pre_margin_median": float(constraint_stats.get("margin_median", float("nan"))),
        "pre_margin_max": float(constraint_stats.get("margin_max", float("nan"))),
        "good_count": int(np.sum(good_mask)),
        "uncertain_count": int(np.sum(uncertain_mask)),
        "bad_count": int(np.sum(bad_mask)),
    }


def _resolve_focus3_plan_pool_per_source(
    *,
    system: OptimizerSystemConfig,
    p_dim: int,
) -> int:
    legacy = getattr(system, "focus3_plan_pool_per_source", None)
    if legacy is not None:
        try:
            legacy_int = int(legacy)
            if legacy_int > 0:
                return legacy_int
        except Exception:
            pass

    p = int(max(p_dim, 1))
    min_pool = int(max(int(getattr(system, "focus3_plan_pool_min_per_source", 1024)), 1))
    per_dim = int(max(int(getattr(system, "focus3_plan_pool_per_dim", 80)), 1))
    max_pool = int(max(int(getattr(system, "focus3_plan_pool_max_per_source", 4096)), min_pool))
    return int(min(max(per_dim * p, min_pool), max_pool))


def _build_focus3_plan_refine_starts(
    *,
    rng: np.random.Generator,
    system: OptimizerSystemConfig,
    model: object | None,
    X_train: np.ndarray,
    X_cover_ref: np.ndarray,
    y_train: np.ndarray,
    objective_sense: str,
    lb: np.ndarray,
    ub: np.ndarray,
    y_best: float,
    p_topk: float,
    p_boundary: float,
    p_random: float,
    p_best_local: float,
    plan_pool_per_source: int,
    refine_starts: int,
    topk_fraction: float,
    topk_sigma: float,
    boundary_near_ratio: float,
    best_local_sigma: float,
    best_local_top_count: int,
    acq_type: str,
    kappa: float,
    xi: float,
    post_feasible_prob_fn: Callable[[np.ndarray], float] | None,
    post_penalty_lambda: float,
    has_constraints: bool,
) -> tuple[np.ndarray, dict[str, int], dict[str, int], dict[str, object]]:
    n_pool = int(max(plan_pool_per_source, 1))
    n_refine = int(max(refine_starts, 1))
    source_probs = {
        "topk": float(p_topk),
        "best_local": float(p_best_local),
        "boundary": float(p_boundary),
        "random": float(p_random),
    }
    quotas = _allocate_source_quotas(total=n_refine, probs=source_probs)

    selected_parts: list[np.ndarray] = []
    selected_labels: list[str] = []
    candidate_parts: dict[str, np.ndarray] = {}
    candidate_scores: dict[str, np.ndarray] = {}
    pool_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    best_scores: dict[str, float] = {}
    boundary_score_penalty_info: dict[str, object] = {}
    random_cover_info: dict[str, object] = {
        "enabled": False,
        "candidate_count": 0,
        "acq_threshold": float("nan"),
        "mean_distance": float("nan"),
        "min_distance": float("nan"),
    }
    plan_acq = _normalize_focus3_acq_type(acq_type)
    source_order = ("topk", "best_local", "boundary", "random") if float(p_best_local) > 0.0 else ("topk", "boundary", "random")
    for source in source_order:
        source_pool_size = n_pool
        if source == "best_local":
            source_pool_size = int(max(1, round(n_pool * float(np.clip(float(getattr(system, "focus3_best_local_pool_ratio", 0.50)), 0.05, 1.0)))))
        pool = _build_source_pool_starts(
            rng=rng,
            source=source,
            X_train=X_train,
            y_train=y_train,
            objective_sense=objective_sense,
            lb=lb,
            ub=ub,
            pool_size=source_pool_size,
            topk_fraction=topk_fraction,
            topk_sigma=topk_sigma,
            boundary_near_ratio=boundary_near_ratio,
            best_local_sigma=best_local_sigma,
            best_local_top_count=best_local_top_count,
        )
        pool_counts[source] = int(pool.shape[0])
        scores = _acquisition_scores_for_pool(
            pool=pool,
            model=model,
            y_best=y_best,
            objective_sense=objective_sense,
            acq_type=plan_acq,
            kappa=kappa,
            xi=xi,
            post_feasible_prob_fn=post_feasible_prob_fn,
            post_penalty_lambda=post_penalty_lambda,
        )
        scores, penalty_info = _apply_focus3_boundary_score_penalty(
            system=system,
            source=source,
            scores=scores,
            has_constraints=bool(has_constraints),
        )
        if str(source) == "boundary":
            boundary_score_penalty_info = dict(penalty_info)
        finite_scores = scores[np.isfinite(scores)]
        best_scores[source] = float(np.min(finite_scores)) if finite_scores.size else float("nan")
        candidate_parts[source] = pool
        candidate_scores[source] = scores
    boundary_gate_info = _focus3_no_constraint_boundary_gate(
        system=system,
        best_scores=best_scores,
        has_constraints=bool(has_constraints),
    )
    if bool(boundary_gate_info.get("applied", False)):
        boundary_quota = int(quotas.get("boundary", 0))
        quotas["boundary"] = 0
        if boundary_quota > 0:
            # Boundary가 충분히 우수하지 않으면 exploitation/coverage 쪽으로 재분배한다.
            if float(p_best_local) > 0.0:
                n_best = int(math.floor(boundary_quota * 0.4))
                n_topk = int(math.floor(boundary_quota * 0.4))
                if boundary_quota > 0 and n_best == 0:
                    n_best = 1
                n_topk = int(min(n_topk, max(boundary_quota - n_best, 0)))
                quotas["best_local"] = int(quotas.get("best_local", 0)) + n_best
                quotas["topk"] = int(quotas.get("topk", 0)) + n_topk
                quotas["random"] = int(quotas.get("random", 0)) + int(max(boundary_quota - n_best - n_topk, 0))
            else:
                quotas["topk"] = int(quotas.get("topk", 0)) + int(math.ceil(boundary_quota * 0.7))
                quotas["random"] = int(quotas.get("random", 0)) + int(boundary_quota - math.ceil(boundary_quota * 0.7))

    for source in source_order:
        pool = candidate_parts.get(source, np.empty((0, lb.shape[0]), dtype=float))
        scores = candidate_scores.get(source, np.empty((0,), dtype=float))
        if source == "random" and bool(getattr(system, "focus3_random_cover_enabled", True)):
            part, random_cover_info = _select_cover_scored_rows(
                rng=rng,
                pool=pool,
                scores=scores,
                n=quotas[source],
                X_ref=X_cover_ref if X_cover_ref.ndim == 2 and X_cover_ref.shape[0] > 0 else X_train,
                lb=lb,
                ub=ub,
                acq_quantile=float(getattr(system, "focus3_random_cover_acq_quantile", 0.70)),
                reference_max=int(getattr(system, "focus3_random_cover_reference_max", 1000)),
            )
        else:
            part = _select_top_scored_rows(pool=pool, scores=scores, n=quotas[source])
        selected_counts[source] = int(part.shape[0])
        if part.shape[0] > 0:
            selected_parts.append(part)
            selected_labels.extend([source] * int(part.shape[0]))

    if selected_parts:
        starts = np.vstack(selected_parts)
    else:
        starts = rng.uniform(lb, ub, size=(n_refine, lb.shape[0]))
        selected_labels = ["fallback"] * int(starts.shape[0])
    if starts.shape[0] > n_refine:
        starts = starts[:n_refine]
        selected_labels = selected_labels[:n_refine]
    diag = {
        "quotas": {k: int(v) for k, v in quotas.items()},
        "best_scores": best_scores,
        "selected_labels": list(selected_labels),
        "acq_type": str(plan_acq),
        "random_cover": dict(random_cover_info),
        "boundary_gate": dict(boundary_gate_info),
        "boundary_score_penalty": dict(boundary_score_penalty_info),
        "best_plan_source": min(
            [s for s in best_scores.keys() if not (s == "boundary" and bool(boundary_gate_info.get("applied", False)))],
            key=lambda s: best_scores[s] if math.isfinite(float(best_scores[s])) else float("inf"),
        )
        if best_scores and [s for s in best_scores.keys() if not (s == "boundary" and bool(boundary_gate_info.get("applied", False)))]
        else "",
    }
    return np.asarray(starts, dtype=float), pool_counts, selected_counts, diag


def _build_focus3_plan_discrete_candidate(
    *,
    rng: np.random.Generator,
    system: OptimizerSystemConfig,
    model: object | None,
    X_train: np.ndarray,
    X_cover_ref: np.ndarray,
    y_train: np.ndarray,
    objective_sense: str,
    lb: np.ndarray,
    ub: np.ndarray,
    y_best: float,
    p_best_local: float,
    plan_pool_per_source: int,
    topk_fraction: float,
    topk_sigma: float,
    boundary_near_ratio: float,
    best_local_sigma: float,
    best_local_top_count: int,
    acq_type: str,
    kappa: float,
    xi: float,
    post_feasible_prob_fn: Callable[[np.ndarray], float] | None,
    post_penalty_lambda: float,
    has_constraints: bool,
) -> tuple[np.ndarray | None, dict[str, int], dict[str, int], dict[str, object]]:
    n_pool = int(max(plan_pool_per_source, 1))
    pool_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {"topk": 0, "best_local": 0, "boundary": 0, "random": 0}
    best_scores: dict[str, float] = {}
    best_rows: list[tuple[str, np.ndarray, float]] = []
    boundary_score_penalty_info: dict[str, object] = {}
    plan_acq = _normalize_focus3_acq_type(acq_type)

    source_order = ("topk", "best_local", "boundary", "random") if float(p_best_local) > 0.0 else ("topk", "boundary", "random")
    for source in source_order:
        source_pool_size = n_pool
        if source == "best_local":
            source_pool_size = int(max(1, round(n_pool * float(np.clip(float(getattr(system, "focus3_best_local_pool_ratio", 0.50)), 0.05, 1.0)))))
        pool = _build_source_pool_starts(
            rng=rng,
            source=source,
            X_train=X_train,
            y_train=y_train,
            objective_sense=objective_sense,
            lb=lb,
            ub=ub,
            pool_size=source_pool_size,
            topk_fraction=topk_fraction,
            topk_sigma=topk_sigma,
            boundary_near_ratio=boundary_near_ratio,
            best_local_sigma=best_local_sigma,
            best_local_top_count=best_local_top_count,
        )
        pool_counts[source] = int(pool.shape[0])
        scores = _acquisition_scores_for_pool(
            pool=pool,
            model=model,
            y_best=y_best,
            objective_sense=objective_sense,
            acq_type=plan_acq,
            kappa=kappa,
            xi=xi,
            post_feasible_prob_fn=post_feasible_prob_fn,
            post_penalty_lambda=post_penalty_lambda,
        )
        scores, penalty_info = _apply_focus3_boundary_score_penalty(
            system=system,
            source=source,
            scores=scores,
            has_constraints=bool(has_constraints),
        )
        if str(source) == "boundary":
            boundary_score_penalty_info = dict(penalty_info)
        finite = np.where(np.isfinite(scores))[0]
        if finite.size == 0:
            best_scores[source] = float("nan")
            continue
        local_pos = int(finite[np.argmin(scores[finite])])
        local_score = float(scores[local_pos])
        best_scores[source] = float(local_score)
        best_rows.append((source, np.asarray(pool[local_pos], dtype=float).reshape(-1), local_score))

    boundary_gate_info = _focus3_no_constraint_boundary_gate(
        system=system,
        best_scores=best_scores,
        has_constraints=bool(has_constraints),
    )
    if bool(boundary_gate_info.get("applied", False)):
        best_rows = [row for row in best_rows if str(row[0]) != "boundary"]

    if not best_rows:
        return None, pool_counts, selected_counts, {
            "mode": "discrete",
            "acq_type": str(plan_acq),
            "best_scores": best_scores,
            "selected_labels": [],
            "best_plan_source": "",
            "boundary_gate": dict(boundary_gate_info),
            "boundary_score_penalty": dict(boundary_score_penalty_info),
            "random_cover": {
                "enabled": False,
                "candidate_count": 0,
                "acq_threshold": float("nan"),
                "mean_distance": float("nan"),
                "min_distance": float("nan"),
            },
        }

    source, x_best, score_best = min(best_rows, key=lambda item: item[2])
    selected_counts[source] = 1
    random_cover_info = {
        "enabled": False,
        "candidate_count": int(pool_counts.get("random", 0)),
        "acq_threshold": float("nan"),
        "mean_distance": float("nan"),
        "min_distance": float("nan"),
    }
    if source == "random":
        ref = X_cover_ref if X_cover_ref.ndim == 2 and X_cover_ref.shape[0] > 0 else X_train
        if ref.ndim == 2 and ref.shape[0] > 0:
            dist = _normalized_rms_distance_to_set(x=x_best, X_ref=ref, lb=lb, ub=ub)
            random_cover_info["mean_distance"] = float(dist)
            random_cover_info["min_distance"] = float(dist)
    return x_best.copy(), pool_counts, selected_counts, {
        "mode": "discrete",
        "acq_type": str(plan_acq),
        "best_scores": best_scores,
        "selected_labels": [str(source)],
        "best_plan_source": str(source),
        "best_plan_score": float(score_best),
        "boundary_gate": dict(boundary_gate_info),
        "boundary_score_penalty": dict(boundary_score_penalty_info),
        "random_cover": random_cover_info,
    }


def _nearest_start_source(
    *,
    x: np.ndarray,
    starts: np.ndarray,
    labels: list[str],
    lb: np.ndarray,
    ub: np.ndarray,
) -> str:
    if starts.ndim != 2 or starts.shape[0] == 0 or not labels:
        return ""
    span = np.maximum(ub - lb, 1e-12)
    x_arr = np.asarray(x, dtype=float).reshape(1, -1)
    d = np.linalg.norm((starts - x_arr) / span.reshape(1, -1), axis=1)
    idx = int(np.argmin(d))
    if idx >= len(labels):
        return ""
    return str(labels[idx])


def _select_best_scored_candidate(
    *,
    pool: np.ndarray,
    model: object | None,
    y_best: float,
    objective_sense: str,
    acq_type: str,
    kappa: float,
    xi: float,
    post_feasible_prob_fn: Callable[[np.ndarray], float] | None,
    post_penalty_lambda: float,
) -> np.ndarray | None:
    arr = np.asarray(pool, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    scores = _acquisition_scores_for_pool(
        pool=arr,
        model=model,
        y_best=y_best,
        objective_sense=objective_sense,
        acq_type=acq_type,
        kappa=kappa,
        xi=xi,
        post_feasible_prob_fn=post_feasible_prob_fn,
        post_penalty_lambda=post_penalty_lambda,
    )
    finite = np.where(np.isfinite(scores))[0]
    if finite.size == 0:
        return arr[0].copy()
    idx = int(finite[np.argmin(scores[finite])])
    return arr[idx].copy()


def _build_source_pool_starts(
    *,
    rng: np.random.Generator,
    source: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    objective_sense: str,
    lb: np.ndarray,
    ub: np.ndarray,
    pool_size: int,
    topk_fraction: float,
    topk_sigma: float,
    boundary_near_ratio: float,
    best_local_sigma: float = 0.025,
    best_local_top_count: int = 3,
) -> np.ndarray:
    n_dim = int(lb.shape[0])
    n = int(max(pool_size, 1))
    span = np.maximum(ub - lb, 1e-12)
    starts: list[np.ndarray] = []

    if source in {"topk", "best_local"} and X_train.shape[0] > 0:
        if objective_sense == "max":
            order = np.argsort(-y_train)
        else:
            order = np.argsort(y_train)
        if source == "best_local":
            n_top = int(max(1, min(int(best_local_top_count), X_train.shape[0])))
            sigma = float(max(best_local_sigma, 1e-5))
        else:
            n_top = int(max(1, round(float(X_train.shape[0]) * float(np.clip(topk_fraction, 0.01, 1.0)))))
            sigma = float(max(topk_sigma, 1e-4))
        top_idx = order[:n_top]
        for _ in range(n):
            base = X_train[int(rng.choice(top_idx))].reshape(-1)
            noise = rng.normal(0.0, sigma, size=n_dim) * span
            starts.append(np.clip(base + noise, lb, ub))
    elif source == "boundary":
        near = float(np.clip(boundary_near_ratio, 0.0, 0.5))
        for _ in range(n):
            x = rng.uniform(lb, ub, size=(n_dim,))
            j = int(rng.integers(0, n_dim))
            if rng.uniform(0.0, 1.0) < 0.5:
                x[j] = lb[j] + near * span[j] * rng.uniform(0.0, 1.0)
            else:
                x[j] = ub[j] - near * span[j] * rng.uniform(0.0, 1.0)
            starts.append(np.clip(x, lb, ub))
    else:
        for _ in range(n):
            starts.append(rng.uniform(lb, ub, size=(n_dim,)))

    return np.asarray(starts, dtype=float)


def _sample_feasible_candidate(
    *,
    x: np.ndarray,
    rng: np.random.Generator,
    lb: np.ndarray,
    ub: np.ndarray,
    var_names: list[str],
    constraint_defs: list[dict],
    enforce_pre_constraints: bool,
    max_attempts: int = 128,
) -> np.ndarray:
    if not enforce_pre_constraints or not constraint_defs:
        return x

    x_cur = np.clip(np.asarray(x, dtype=float).reshape(-1), lb, ub)
    best_x = x_cur.copy()
    best_margin = float("-inf")
    for _ in range(max(int(max_attempts), 1)):
        _payload, feasible, _margin = evaluate_pre_constraints_raw(
            x=x_cur,
            var_names=var_names,
            constraint_defs=constraint_defs,
            fail_fast_output_missing=False,
        )
        if bool(feasible):
            return x_cur
        if float(_margin) > float(best_margin):
            best_margin = float(_margin)
            best_x = x_cur.copy()
        x_cur = rng.uniform(lb, ub, size=(lb.shape[0],))
    return best_x


def _evaluate_pre_constraint(
    *,
    x: np.ndarray,
    var_names: list[str],
    constraint_defs: list[dict],
    enforce_pre_constraints: bool,
) -> tuple[bool, float]:
    if not enforce_pre_constraints or not constraint_defs:
        return True, float("inf")
    _payload, feasible, margin = evaluate_pre_constraints_raw(
        x=x,
        var_names=var_names,
        constraint_defs=constraint_defs,
        fail_fast_output_missing=False,
    )
    return bool(feasible), float(margin)


def _build_preconstrained_source_starts(
    *,
    rng: np.random.Generator,
    source: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    objective_sense: str,
    lb: np.ndarray,
    ub: np.ndarray,
    base_pool_size: int,
    topk_fraction: float,
    topk_sigma: float,
    boundary_near_ratio: float,
    var_names: list[str],
    constraint_defs: list[dict],
    enforce_pre_constraints: bool,
    feasible_multiplier: int,
    feasible_retry: int,
    min_starts: int,
) -> tuple[np.ndarray, np.ndarray | None, float, int, int]:
    raw_pool_size = int(max(base_pool_size, 1))
    n_target = int(max(raw_pool_size, max(min_starts, 1)))
    if not enforce_pre_constraints or not constraint_defs:
        starts = _build_source_pool_starts(
            rng=rng,
            source=source,
            X_train=X_train,
            y_train=y_train,
            objective_sense=objective_sense,
            lb=lb,
            ub=ub,
            pool_size=n_target,
            topk_fraction=topk_fraction,
            topk_sigma=topk_sigma,
            boundary_near_ratio=boundary_near_ratio,
        )
        return starts, None, float("inf"), 0, int(starts.shape[0])

    multiplier = int(max(feasible_multiplier, 1))
    max_retry = int(max(feasible_retry, 0))
    n_generate = int(max(n_target * multiplier, n_target))

    feasible_accum: list[np.ndarray] = []
    feasible_keys: set[tuple[float, ...]] = set()
    fallback_x: np.ndarray | None = None
    fallback_margin = float("-inf")
    generated_total = 0
    retry_used = 0

    for retry_idx in range(max_retry + 1):
        retry_used = int(retry_idx)
        if str(source) == "constraint_boundary":
            n_a = int(max(n_generate // 2, 1))
            n_b = int(max(n_generate - n_a, 1))
            starts = np.vstack([
                _build_source_pool_starts(
                    rng=rng,
                    source="random",
                    X_train=X_train,
                    y_train=y_train,
                    objective_sense=objective_sense,
                    lb=lb,
                    ub=ub,
                    pool_size=n_a,
                    topk_fraction=topk_fraction,
                    topk_sigma=topk_sigma,
                    boundary_near_ratio=boundary_near_ratio,
                ),
                _build_source_pool_starts(
                    rng=rng,
                    source="boundary",
                    X_train=X_train,
                    y_train=y_train,
                    objective_sense=objective_sense,
                    lb=lb,
                    ub=ub,
                    pool_size=n_b,
                    topk_fraction=topk_fraction,
                    topk_sigma=topk_sigma,
                    boundary_near_ratio=boundary_near_ratio,
                ),
            ])
        else:
            starts = _build_source_pool_starts(
                rng=rng,
                source=source,
                X_train=X_train,
                y_train=y_train,
                objective_sense=objective_sense,
                lb=lb,
                ub=ub,
                pool_size=n_generate,
                topk_fraction=topk_fraction,
                topk_sigma=topk_sigma,
                boundary_near_ratio=boundary_near_ratio,
            )
        generated_total += int(starts.shape[0])
        mask, _payloads, margins = evaluate_constraints_batch(
            X=starts,
            var_names=var_names,
            constraint_defs=constraint_defs,
            scope="pre",
        )
        if starts.shape[0] > 0:
            best_i = int(np.argmax(margins))
            if float(margins[best_i]) > float(fallback_margin):
                fallback_margin = float(margins[best_i])
                fallback_x = starts[best_i].reshape(-1).copy()

        feasible_idx = np.where(mask)[0]
        if str(source) == "constraint_boundary" and feasible_idx.size > 0:
            feasible_idx = feasible_idx[np.argsort(margins[feasible_idx])]
        feasible_rows = starts[feasible_idx] if starts.shape[0] > 0 else np.empty((0, lb.shape[0]), dtype=float)
        for row in feasible_rows:
            key = _round_key(row, decimals=12)
            if key in feasible_keys:
                continue
            feasible_keys.add(key)
            feasible_accum.append(np.asarray(row, dtype=float).reshape(-1))
            if len(feasible_accum) >= n_target:
                break
        if len(feasible_accum) >= n_target:
            break
        if retry_idx >= max_retry:
            break

    if feasible_accum:
        starts_out = np.asarray(feasible_accum, dtype=float)
        if starts_out.shape[0] > n_target:
            idx = rng.choice(starts_out.shape[0], size=n_target, replace=False)
            starts_out = starts_out[np.asarray(idx, dtype=int)]
        return starts_out, fallback_x, float(fallback_margin), int(retry_used), int(generated_total)

    return (
        np.empty((0, lb.shape[0]), dtype=float),
        fallback_x,
        float(fallback_margin),
        int(retry_used),
        int(generated_total),
    )


def _segment_to_focus_level(*, segment: str) -> int:
    seg = str(segment or "").strip().lower()
    if seg in {"focus0"}:
        return 0
    if seg in {"focus1"}:
        return 1
    if seg in {"focus2"}:
        return 2
    if seg in {"focus3"}:
        return 3
    if seg in {"focus4"}:
        return 4
    return 3


def _normalize_post_score_mode(mode: str | None) -> str:
    value = str(mode or "add_penalty").strip().lower()
    if value not in {"add_penalty"}:
        return "add_penalty"
    return value


def _build_post_feasible_prob_fn(
    *,
    feasibility_payload: dict | None,
    selected_features: list[str],
) -> tuple[Callable[[np.ndarray], float] | None, str, str]:
    if not isinstance(feasibility_payload, dict):
        return None, "none", "no_payload"

    kind = str(feasibility_payload.get("kind", "unknown")).strip().lower() or "unknown"
    model_feature_cols = feasibility_payload.get("feature_cols", [])
    if isinstance(model_feature_cols, list) and model_feature_cols:
        feature_cols = [str(f) for f in model_feature_cols]
    else:
        feature_cols = list(selected_features)

    feature_to_idx = {name: idx for idx, name in enumerate(selected_features)}
    missing = [f for f in feature_cols if f not in feature_to_idx]
    if missing:
        return None, kind, "missing_selected_features:" + ",".join(missing)
    col_idx = np.asarray([feature_to_idx[f] for f in feature_cols], dtype=int)

    if kind == "constant":
        p0 = float(np.clip(float(feasibility_payload.get("constant_prob", 0.5)), 0.0, 1.0))

        def _const_prob(_x: np.ndarray) -> float:
            return float(p0)

        return _const_prob, "constant", "ok"

    model = feasibility_payload.get("model")
    if model is None:
        return None, kind, "missing_model_object"

    def _model_prob(x_row: np.ndarray) -> float:
        try:
            x = np.asarray(x_row, dtype=float).reshape(-1)
            if x.shape[0] < len(selected_features):
                return 1.0
            x_in = x[col_idx].reshape(1, -1)
            if hasattr(model, "predict_proba"):
                prob = np.asarray(model.predict_proba(x_in), dtype=float)
                if prob.ndim == 2 and prob.shape[1] >= 2:
                    p = float(prob[0, 1])
                else:
                    p = float(prob.reshape(-1)[0])
            elif hasattr(model, "predict"):
                pred = np.asarray(model.predict(x_in), dtype=float).reshape(-1)
                p = float(pred[0])
            else:
                return 1.0
            if not np.isfinite(p):
                return 1.0
            return float(np.clip(p, 0.0, 1.0))
        except Exception:
            return 1.0

    return _model_prob, kind, "ok"


def _apply_post_penalty_to_objective(
    *,
    y_raw: float,
    p_feasible: float,
    objective_sense: str,
    penalty_lambda: float,
    score_mode: str,
    p_feasible_min: float,
    hard_penalty: float,
) -> float:
    y = float(y_raw)
    if score_mode != "add_penalty" or penalty_lambda <= 0.0:
        return y
    p = float(np.clip(p_feasible, 0.0, 1.0))
    penalty = float(penalty_lambda) * (1.0 - p)
    p_min = float(np.clip(p_feasible_min, 0.0, 1.0))
    hard = float(max(hard_penalty, 0.0))
    if hard > 0.0 and p < p_min:
        penalty += hard
    if objective_sense == "max":
        return y - penalty
    return y + penalty


def _best_index(y: np.ndarray, objective_sense: str) -> int:
    return objective_best_index(y, objective_sense)


def _build_cae_objective_evaluator(
    *,
    problem_name: str,
    variables: list[dict],
    selected_features: list[str],
) -> Callable[[np.ndarray], float]:
    evaluator = CaeObjectiveEvaluator(
        problem_name=problem_name,
        variables=variables,
        selected_features=selected_features,
        error_context="optimizer iteration",
    )
    return evaluator.evaluate_selected


def run_bo_engine(
    *,
    problem_name: str,
    variables: list[dict],
    doe_df: pd.DataFrame | None,
    selected_features: list[str],
    selected_bounds: dict[str, tuple[float, float]],
    bounds_source: str,
    objective_col: str,
    objective_sense: str,
    n_samples: int,
    goal: float | None = None,
    system: OptimizerSystemConfig,
    seed: int,
    constraint_defs: list[dict],
    post_feasibility_payload: dict | None = None,
) -> OptimizerAlgorithmResult:
    if int(n_samples) < 0:
        raise ValueError("Optimizer user.n_samples must be >= 0.")
    if len(selected_features) == 0:
        raise RuntimeError("selected_features is empty.")

    rng = np.random.default_rng(int(seed))
    p_dim = len(selected_features)
    lb = np.asarray([selected_bounds[f][0] for f in selected_features], dtype=float)
    ub = np.asarray([selected_bounds[f][1] for f in selected_features], dtype=float)
    if np.any(~np.isfinite(lb)) or np.any(~np.isfinite(ub)) or np.any(ub <= lb):
        raise RuntimeError("Invalid optimization bounds.")
    evaluate_objective = _build_cae_objective_evaluator(
        problem_name=problem_name,
        variables=variables,
        selected_features=selected_features,
    )

    post_score_mode = _normalize_post_score_mode(getattr(system, "post_score_mode", "add_penalty"))
    post_penalty_lambda = max(float(getattr(system, "post_penalty_lambda", 0.0)), 0.0)
    post_p_feasible_min = float(np.clip(float(getattr(system, "post_p_feasible_min", 0.0)), 0.0, 1.0))
    post_p_feasible_hard_penalty = float(max(float(getattr(system, "post_p_feasible_hard_penalty", 0.0)), 0.0))
    post_constraint_enabled = bool(getattr(system, "post_constraint_enabled", False))
    post_prob_fn: Callable[[np.ndarray], float] | None = None
    feasibility_model_kind = "none"
    feasibility_status = "disabled"
    if post_constraint_enabled and post_penalty_lambda > 0.0:
        post_prob_fn, feasibility_model_kind, feasibility_status = _build_post_feasible_prob_fn(
            feasibility_payload=post_feasibility_payload,
            selected_features=selected_features,
        )
        if post_prob_fn is None:
            feasibility_status = (
                feasibility_status
                if feasibility_status != "ok"
                else "disabled_no_probability_model"
            )
    else:
        feasibility_status = "disabled_by_config"
    post_penalty_active = bool(post_prob_fn is not None and post_penalty_lambda > 0.0)
    if post_constraint_enabled and not post_penalty_active:
        print(f"[Optimizer] post penalty disabled: {feasibility_status}")

    X_archive = np.empty((0, len(selected_features)), dtype=float)
    y_archive = np.empty((0,), dtype=float)
    X_seed_from_doe = np.empty((0, len(selected_features)), dtype=float)
    doe_seed_scope = str(getattr(system, "doe_seed_scope", "in_bounds") or "in_bounds").strip().lower()
    if doe_seed_scope not in {"in_bounds", "all"}:
        doe_seed_scope = "in_bounds"

    has_doe_objective = False
    if isinstance(doe_df, pd.DataFrame) and not doe_df.empty:
        if all(f in doe_df.columns for f in selected_features):
            X_doe = doe_df[selected_features].to_numpy(dtype=float)
            mask_finite = np.isfinite(X_doe).all(axis=1)
            X_doe = X_doe[mask_finite]
            if X_doe.shape[0] > 0:
                if doe_seed_scope == "all":
                    X_seed_from_doe = X_doe.copy()
                else:
                    in_bounds = np.all((X_doe >= lb.reshape(1, -1)) & (X_doe <= ub.reshape(1, -1)), axis=1)
                    X_seed_from_doe = X_doe[in_bounds]
                if X_seed_from_doe.shape[0] > 0:
                    X_seed_from_doe = np.unique(X_seed_from_doe, axis=0)

            if objective_col in doe_df.columns:
                y_doe_raw = pd.to_numeric(doe_df[objective_col], errors="coerce").to_numpy(dtype=float)
                if y_doe_raw.shape[0] == mask_finite.shape[0]:
                    y_doe_raw = y_doe_raw[mask_finite]
                    if X_doe.shape[0] == y_doe_raw.shape[0]:
                        if doe_seed_scope == "all":
                            X_obj = X_doe.copy()
                            y_obj = y_doe_raw.copy()
                        else:
                            in_bounds = np.all((X_doe >= lb.reshape(1, -1)) & (X_doe <= ub.reshape(1, -1)), axis=1)
                            X_obj = X_doe[in_bounds]
                            y_obj = y_doe_raw[in_bounds]
                        finite_y = np.isfinite(y_obj)
                        X_obj = X_obj[finite_y]
                        y_obj = y_obj[finite_y]
                        if X_obj.shape[0] > 0:
                            X_archive = X_obj.copy()
                            y_archive = y_obj.copy()
                            has_doe_objective = True

    gp_train_cap = 0
    initial_archive_count = 0

    # DOE objective 기반 warm-start. objective가 없으면 Focus1이 n_samples 안에서 archive를 만든다.
    if has_doe_objective:
        initial_archive_count = int(X_archive.shape[0])
        X_train, y_train, gp_train_cap = _select_gp_train_from_archive(
            X=X_archive,
            y=y_archive,
            objective_sense=objective_sense,
            system=system,
            p_dim=p_dim,
            initial_count=initial_archive_count,
        )
        init_source = "doe_objective"
    else:
        X_train = np.empty((0, len(selected_features)), dtype=float)
        y_train = np.empty((0,), dtype=float)
        init_source = "focus1_empty"

    if X_archive.shape[0] > 0:
        best_i_raw = _best_index(y_archive, objective_sense)
        best_x_raw = X_archive[best_i_raw].copy()
        best_y_raw = float(y_archive[best_i_raw])
    else:
        best_x_raw = 0.5 * (lb + ub)
        best_y_raw = objective_initial_best(objective_sense)

    if post_penalty_active:
        if X_archive.shape[0] > 0:
            p_archive = np.array([float(np.clip(post_prob_fn(x), 0.0, 1.0)) for x in X_archive], dtype=float)
            y_archive_eff = np.array(
                [
                    _apply_post_penalty_to_objective(
                        y_raw=float(y),
                        p_feasible=float(p),
                        objective_sense=objective_sense,
                        penalty_lambda=post_penalty_lambda,
                        score_mode=post_score_mode,
                        p_feasible_min=post_p_feasible_min,
                        hard_penalty=post_p_feasible_hard_penalty,
                    )
                    for y, p in zip(y_archive, p_archive)
                ],
                dtype=float,
            )
        else:
            y_archive_eff = np.empty((0,), dtype=float)
    else:
        y_archive_eff = y_archive.copy()

    if y_archive_eff.shape[0] > 0:
        best_i_eff = _best_index(y_archive_eff, objective_sense)
        best_x_eff = X_archive[best_i_eff].copy()
        best_y_eff = float(y_archive_eff[best_i_eff])
    else:
        best_x_eff = best_x_raw.copy()
        best_y_eff = objective_initial_best(objective_sense)

    seen = {_round_key(x, decimals=int(system.dedup_decimals)) for x in X_archive}
    acq = AcquisitionOptimizer()
    history_rows: list[dict] = []
    gp_model = None
    gp_fallback_used = False
    var_names = list(selected_features)
    best_raw_history: list[float] = [float(best_y_raw)] if math.isfinite(float(best_y_raw)) else []
    goal_monitor = GoalMonitor.from_goal(
        goal=goal,
        system_config=system,
        objective_sense=objective_sense,
    )
    goal_monitor.initialize(best_objective=float(best_y_raw))
    full_lb = lb.copy()
    full_ub = ub.copy()
    external_selected_bounds = str(bounds_source or "").strip().lower() in {"explorer", "path", "override"}
    focus_pipeline_plan: FocusPipelinePlan = build_focus_pipeline_plan(
        system=system,
        bounds_source=bounds_source,
        initial_objective_count=int(initial_archive_count),
        p_dim=p_dim,
        n_samples=int(n_samples),
    )
    focus2_counts = {"local": 0, "dropped": 0, "skipped": 0}
    focus2_initial_info: dict[str, object] = {}
    focus2_final_info: dict[str, object] = {}
    focus0_budget = int(focus_pipeline_plan.budget("focus0"))
    focus1_budget = int(focus_pipeline_plan.budget("focus1"))
    focus2_budget = int(focus_pipeline_plan.budget("focus2"))
    focus3_budget = int(focus_pipeline_plan.budget("focus3"))
    focus2_active = (
        focus_pipeline_plan.has("focus2")
        and
        bool(getattr(system, "focus2_enabled", True))
        and int(focus2_budget) > 0
    )
    if not focus2_active:
        focus2_budget = 0
    focus_lb = lb.copy()
    focus_ub = ub.copy()
    focus2_finalized = not bool(focus2_active)
    focus2_regions_payload: dict[str, object] = {}
    focus2_region_states: list[dict[str, object]] = []
    focus2_regions_initialized = False
    focus2_eval_count = 0
    focus2_batch_id = 0
    focus2_pending: list[dict[str, object]] = []
    focus2_duplicate_pair_counts: dict[tuple[str, str], int] = {}
    focus2_shared_pair_counts: dict[tuple[str, str], int] = {}
    focus2_merge_history: list[dict[str, object]] = []
    focus2_active_bounds_history: list[dict[str, object]] = []
    focus2_scheduler_history: list[dict[str, object]] = []
    focus2_bounds_applied = False
    focus2_final_region_selection: dict[str, object] = {}
    focus2_early_stop_info: dict[str, object] = {}
    focus2_selected_region_id = ""
    focus2_selected_region_state: dict[str, object] | None = None
    focus2_candidate_info: dict[str, object] = {}
    focus3_budget_class = str(focus_pipeline_plan.focus3_budget_class)
    focus3_budget_ratio = float(focus_pipeline_plan.focus3_budget_ratio)
    focus0_target_count = int(focus_pipeline_plan.target("focus0_objective_count", _focus0_target_count(system=system, p_dim=p_dim)))
    focus1_target_count = int(focus_pipeline_plan.target("focus1_objective_count", _focus1_target_count(system=system, p_dim=p_dim)))
    focus0_eval_count = 0
    focus0_pending: list[dict[str, object]] = []
    focus0_batch_id = 0
    focus1_eval_count = 0
    focus1_pending: list[dict[str, object]] = []
    focus1_batch_id = 0
    focus1_early_stopped = False
    focus1_early_stop_reason = ""
    focus3_eval_count = 0
    progress_log_every = int(max(int(getattr(system, "optimizer_progress_log_every", 25)), 0))
    timing_enabled = bool(getattr(system, "optimizer_timing_log_enabled", True))
    timing_keys = [
        "gp_fit",
        "starts_build",
        "focus0_batch",
        "focus1_batch",
        "focus2_region_init",
        "focus2_batch",
        "focus2_update",
        "focus3_plan",
        "focus3_refine",
        "focus3_dedup",
        "gp_predict_next",
        "objective_eval",
        "archive_update",
        "iteration",
    ]
    timing_totals: dict[str, float] = {key: 0.0 for key in timing_keys}
    timing_iter: dict[str, float] = {key: 0.0 for key in timing_keys}
    optimizer_t0 = time.perf_counter()

    def _timer_start() -> float:
        return time.perf_counter()

    def _timer_add(name: str, started_at: float) -> None:
        if not timing_enabled:
            return
        elapsed = float(time.perf_counter() - float(started_at))
        timing_totals[name] = float(timing_totals.get(name, 0.0) + elapsed)
        timing_iter[name] = float(timing_iter.get(name, 0.0) + elapsed)

    def _timing_brief() -> str:
        if not timing_enabled:
            return ""
        elapsed_total = float(time.perf_counter() - optimizer_t0)
        heavy = [
            ("gp_fit", timing_totals.get("gp_fit", 0.0)),
            ("focus3_plan", timing_totals.get("focus3_plan", 0.0)),
            ("focus3_refine", timing_totals.get("focus3_refine", 0.0)),
            ("focus2_batch", timing_totals.get("focus2_batch", 0.0)),
            ("objective_eval", timing_totals.get("objective_eval", 0.0)),
        ]
        heavy_txt = ",".join([f"{k}:{v:.1f}s" for k, v in heavy])
        return f" elapsed={elapsed_total:.1f}s timing={{" + heavy_txt + "}"
    print(
        "[Optimizer][Plan] "
        f"problem={problem_name} p={p_dim} n_samples={int(n_samples)} "
        f"bounds_source={bounds_source} stages={list(focus_pipeline_plan.stages)} "
        f"budgets={{focus0:{focus0_budget}, focus1:{focus1_budget}, "
        f"focus2:{focus2_budget}, focus3:{focus3_budget}}} "
        f"targets={{focus0:{focus0_target_count}, focus1:{focus1_target_count}}} "
        f"initial_archive={initial_archive_count}",
        flush=True,
    )

    def _apply_focus2_final_bounds_if_ready() -> bool:
        nonlocal lb, ub, focus_lb, focus_ub, focus2_bounds_applied
        nonlocal focus2_final_region_selection, focus2_final_info
        if not (focus2_active and focus2_finalized and not focus2_bounds_applied and focus2_region_states):
            return False
        chosen_state, focus2_final_region_selection, selected_lb, selected_ub, final_volume_ratio = _build_focus2_final_bounds(
            region_states=focus2_region_states,
            X_archive=X_archive,
            y_archive=y_archive,
            objective_sense=objective_sense,
            system=system,
            full_lb=full_lb,
            full_ub=full_ub,
        )
        if chosen_state is not None and selected_lb is not None and selected_ub is not None:
            lb = selected_lb.copy()
            ub = selected_ub.copy()
            focus_lb = selected_lb.copy()
            focus_ub = selected_ub.copy()
            selected_region_ids = list(focus2_final_region_selection.get("selected_region_ids", []))
            focus2_final_info = {
                "method": "turbo_lite_final_region_union" if bool(getattr(system, "focus2_final_union_enabled", True)) else "turbo_lite_final_region_selector",
                "selected_region_id": str(chosen_state.get("region_id", "")),
                "selected_region_ids": selected_region_ids,
                "selection": dict(focus2_final_region_selection),
                "active_region_count": int(sum(1 for s in focus2_region_states if str(s.get("status", "active")) == "active")),
                "dropped_region_count": int(sum(1 for s in focus2_region_states if str(s.get("status", "active")) == "dropped")),
                "merged_region_count": int(sum(1 for s in focus2_region_states if str(s.get("status", "active")) == "merged")),
                "volume_ratio": float(final_volume_ratio),
                "bounds_policy": dict(focus2_final_region_selection.get("final_bounds_policy", {})),
            }
            print(
                "[Optimizer][Focus2] final bounds applied "
                f"regions={','.join(selected_region_ids) if selected_region_ids else focus2_final_info['selected_region_id']} "
                f"volume_ratio={float(final_volume_ratio):.4f} "
                f"active_regions={focus2_final_info['active_region_count']} "
                f"dropped_regions={focus2_final_info['dropped_region_count']}",
                flush=True,
            )
        focus2_bounds_applied = True
        return True

    for i in range(int(n_samples)):
        timing_iter = {key: 0.0 for key in timing_keys}
        iter_t0 = _timer_start()
        _apply_focus2_final_bounds_if_ready()

        if i == 0 or (int(system.gp_refit_every) > 0 and i % int(system.gp_refit_every) == 0):
            if X_train.shape[0] >= 2 and y_train.shape[0] >= 2:
                t0 = _timer_start()
                gp_model, gp_fallback_used = _fit_optimizer_gp_with_fallback(
                    X=X_train,
                    y=y_train,
                    lb=lb,
                    ub=ub,
                    system=system,
                    include_white=False,
                    random_state=int(seed + i),
                )
                _timer_add("gp_fit", t0)
            else:
                gp_model = None
                gp_fallback_used = False

        acq_type, kappa = _resolve_acq_policy(
            system=system,
            iteration=i,
            n_iterations=max(int(n_samples), 1),
        )
        t0 = _timer_start()
        starts = _build_starts(
            rng=rng,
            archive_X=X_train,
            archive_y=y_train,
            objective_sense=objective_sense,
            lb=lb,
            ub=ub,
            starts_per_iter=int(system.starts_per_iter),
            random_ratio=float(system.random_starts_ratio),
        )
        _timer_add("starts_build", t0)

        segment = "default"
        source_mode = "default"
        x_next = None
        pre_feasible_next = True
        pre_margin_next = float("inf")
        pre_violation_next = 0.0
        pre_retry_used = 0
        pre_generated_count = 0
        pre_fallback_used = False
        pre_constraint_debug_fields: dict[str, object] = {}
        source_prob_topk = float("nan")
        source_prob_boundary = float("nan")
        source_prob_random = float("nan")
        source_prob_best_local = float("nan")
        focus3_source_adapt_info: dict[str, object] = {}
        focus3_acq_policy_info: dict[str, object] = {}
        focus3_plan_pool_count = 0
        focus3_refine_start_count = 0
        focus3_selected_topk = 0
        focus3_selected_best_local = 0
        focus3_selected_boundary = 0
        focus3_selected_random = 0
        focus3_best_plan_source = ""
        focus3_nearest_refine_source = ""
        focus3_plan_mode = ""
        focus3_best_score_topk = float("nan")
        focus3_best_score_best_local = float("nan")
        focus3_best_score_boundary = float("nan")
        focus3_best_score_random = float("nan")
        focus3_random_cover_info: dict[str, object] = {
            "enabled": False,
            "candidate_count": 0,
            "acq_threshold": float("nan"),
            "mean_distance": float("nan"),
            "min_distance": float("nan"),
        }
        focus3_boundary_gate_info: dict[str, object] = {}
        focus3_boundary_score_penalty_info: dict[str, object] = {}
        focus3_recover_info: dict[str, object] = {}
        focus3_source_cap_info: dict[str, object] = {}
        focus3_best_local_info: dict[str, object] = {}
        focus3_recover_best_local_info: dict[str, object] = {}
        focus3_best_local_sigma_info: dict[str, object] = {
            "enabled": bool(getattr(system, "focus3_best_local_sigma_schedule_enabled", True)),
            "base": float(getattr(system, "focus3_best_local_sigma", 0.015)),
            "effective": float(getattr(system, "focus3_best_local_sigma", 0.015)),
            "reason": "",
        }
        focus3_fallback_bounds_source_info: dict[str, object] = {}
        focus3_refine_filter_info: dict[str, object] = {
            "enabled": bool(getattr(system, "focus3_refine_source_filter_enabled", True)),
            "skipped": False,
            "reason": "",
            "allowed_sources": str(getattr(system, "focus3_refine_allowed_sources", "topk,best_local")),
        }
        focus3_refine_cooldown_info: dict[str, object] = {
            "enabled": bool(getattr(system, "focus3_refine_cooldown_enabled", True)),
            "active": False,
            "reason": "",
        }
        focus3_refine_adaptive_info: dict[str, object] = {
            "enabled": bool(getattr(system, "focus3_refine_adaptive_enabled", True)),
            "base_every": int(getattr(system, "focus3_refine_every", 2)),
            "effective_every": int(getattr(system, "focus3_refine_every", 2)),
            "reason": "",
        }
        focus3_dedup_info: dict[str, object] = {
            "enabled": bool(getattr(system, "focus3_dedup_enabled", True)),
            "applied": False,
            "archive_distance_before": float("nan"),
            "archive_distance_after": float("nan"),
            "min_rms_distance": float(getattr(system, "focus3_min_candidate_rms_distance", 0.01)),
            "fallback": "",
        }
        focus3_dedup_pool = np.empty((0, p_dim), dtype=float)
        focus3_effective_kappa = float(kappa)
        focus3_dedup_acq_type = str(acq_type)
        focus0_active = False
        focus0_target_count_row = int(focus0_target_count)
        focus0_batch_size_planned = 0
        focus0_batch_remaining = 0
        focus0_pool_count = 0
        focus1_active = False
        focus1_class = ""
        focus1_batch_mode = ""
        focus1_batch_size_planned = 0
        focus1_batch_remaining = 0
        focus1_tau = float("nan")
        focus1_pool_count = 0
        focus1_raw_pool_count = 0
        focus1_pre_feasible_count = 0
        focus1_pre_feasible_ratio = float("nan")
        focus1_pre_margin_min = float("nan")
        focus1_pre_margin_median = float("nan")
        focus1_pre_margin_max = float("nan")
        focus1_good_count = 0
        focus1_uncertain_count = 0
        focus1_bad_count = 0
        focus1_score = float("nan")
        focus1_data_ratio = float(X_archive.shape[0]) / float(max(p_dim, 1))
        focus2_selected_region_id = ""
        focus2_selected_region_state = None
        focus2_candidate_info = {}
        focus0_needed = (
            focus_pipeline_plan.has("focus0")
            and
            bool(getattr(system, "focus0_enabled", True))
            and int(X_archive.shape[0]) < int(focus0_target_count)
            and int(focus0_eval_count) < int(focus0_budget)
        )
        if focus0_needed:
            if not focus0_pending:
                remaining_budget = min(int(n_samples) - int(i), int(focus0_budget) - int(focus0_eval_count))
                needed = int(max(int(focus0_target_count) - int(X_archive.shape[0]), 0))
                planned = int(min(
                    max(int(getattr(system, "focus1_batch_size", 10)), 1),
                    max(remaining_budget, 0),
                    max(needed, 0),
                ))
                if planned > 0:
                    focus0_batch_id += 1
                    t0 = _timer_start()
                    batch_rows, batch_info = _build_focus0_batch(
                        rng=rng,
                        X_archive=X_archive,
                        lb=lb,
                        ub=ub,
                        n=planned,
                        system=system,
                        var_names=var_names,
                        constraint_defs=constraint_defs,
                        enforce_pre_constraints=bool(system.enforce_pre_constraints),
                        decimals=int(system.dedup_decimals),
                    )
                    _timer_add("focus0_batch", t0)
                    if not batch_rows:
                        raise RuntimeError("Focus0 could not generate candidate points.")
                    batch_size_actual = int(len(batch_rows))
                    for item in batch_rows:
                        item["batch_id"] = int(focus0_batch_id)
                        item["batch_size"] = int(batch_size_actual)
                        item["batch_mode"] = str(batch_info.get("mode", "focus0"))
                        item["pool_count"] = int(batch_info.get("pool_count", 0))
                    focus0_pending.extend(batch_rows)
                    print(
                        "[Optimizer][Focus0] batch "
                        f"id={focus0_batch_id} size={batch_size_actual} "
                        f"archive={int(X_archive.shape[0])}/{focus0_target_count} "
                        f"budget={focus0_eval_count}/{focus0_budget} "
                        f"pool={int(batch_info.get('pool_count', 0))}",
                        flush=True,
                    )
            if focus0_pending:
                item = focus0_pending.pop(0)
                x_next = np.asarray(item.get("x"), dtype=float).reshape(-1)
                segment = "focus0"
                source_mode = str(item.get("class", "focus0_initial_doe"))
                acq_type = "FOCUS0"
                focus0_active = True
                focus0_batch_size_planned = int(item.get("batch_size", len(focus0_pending) + 1))
                focus0_batch_remaining = int(len(focus0_pending))
                focus0_pool_count = int(item.get("pool_count", 0))
                pre_generated_count = int(focus0_pool_count)
                focus0_eval_count += 1

        focus1_needed = (
            focus_pipeline_plan.has("focus1")
            and
            bool(getattr(system, "focus1_enabled", True))
            and (
                bool(focus1_pending)
                or (
                    not bool(focus1_early_stopped)
                    and int(X_archive.shape[0]) < int(focus1_target_count)
                    and int(focus1_eval_count) < int(focus1_budget)
                )
            )
        )
        if x_next is None and focus1_needed:
            if not focus1_pending:
                remaining_budget = min(int(n_samples) - int(i), int(focus1_budget) - int(focus1_eval_count))
                needed = int(max(int(focus1_target_count) - int(X_archive.shape[0]), 0))
                planned = int(min(
                    max(int(getattr(system, "focus1_batch_size", 10)), 1),
                    max(remaining_budget, 0),
                    max(needed, 0),
                ))
                if planned > 0:
                    focus1_batch_id += 1
                    t0 = _timer_start()
                    batch_rows, batch_info = _build_focus1_batch(
                        rng=rng,
                        model=gp_model,
                        X_archive=X_archive,
                        y_archive=y_archive,
                        objective_sense=objective_sense,
                        lb=lb,
                        ub=ub,
                        n=planned,
                        system=system,
                        var_names=var_names,
                        constraint_defs=constraint_defs,
                        enforce_pre_constraints=bool(system.enforce_pre_constraints),
                        decimals=int(system.dedup_decimals),
                    )
                    _timer_add("focus1_batch", t0)
                    if not batch_rows:
                        raise RuntimeError(
                            "Focus1 could not generate classification candidates. "
                            "Provide more objective data or include focus0 before focus1."
                        )
                    if bool(getattr(system, "focus1_early_stop_enabled", True)):
                        pool_count_for_stop = int(max(int(batch_info.get("pool_count", 0)), 1))
                        uncertain_ratio_for_stop = float(batch_info.get("uncertain_count", 0)) / float(pool_count_for_stop)
                        good_count_for_stop = int(batch_info.get("good_count", 0))
                        expected_archive_count_for_stop = int(X_archive.shape[0]) + int(len(batch_rows))
                        min_data_ratio_for_stop = float(max(float(getattr(system, "focus1_early_stop_min_data_ratio", 10.0)), 0.0))
                        data_ratio_for_stop = float(expected_archive_count_for_stop) / float(max(p_dim, 1))
                        if (
                            data_ratio_for_stop >= min_data_ratio_for_stop
                            and uncertain_ratio_for_stop <= float(getattr(system, "focus1_early_stop_uncertain_ratio", 0.25))
                            and good_count_for_stop >= int(getattr(system, "focus1_early_stop_min_good_candidates", 10))
                        ):
                            focus1_early_stopped = True
                            focus1_early_stop_reason = (
                                "classification_stable:"
                                f"data_ratio={data_ratio_for_stop:.4f},"
                                f"uncertain_ratio={uncertain_ratio_for_stop:.4f},"
                                f"good_count={good_count_for_stop}"
                            )
                    print(
                        "[Optimizer][Focus1] batch "
                        f"id={focus1_batch_id} size={len(batch_rows)} "
                        f"archive_after_batch={int(X_archive.shape[0]) + int(len(batch_rows))}/{focus1_target_count} "
                        f"data_ratio={(float(int(X_archive.shape[0]) + int(len(batch_rows))) / float(max(p_dim, 1))):.2f} "
                        f"good={int(batch_info.get('good_count', 0))} "
                        f"uncertain={int(batch_info.get('uncertain_count', 0))} "
                        f"bad={int(batch_info.get('bad_count', 0))} "
                        f"early_stop={bool(focus1_early_stopped)} "
                        f"reason={focus1_early_stop_reason or 'none'}",
                        flush=True,
                    )
                    batch_size_actual = int(len(batch_rows))
                    for item in batch_rows:
                        item["batch_id"] = int(focus1_batch_id)
                        item["batch_size"] = int(batch_size_actual)
                        item["batch_mode"] = str(batch_info.get("mode", ""))
                        item["tau"] = float(batch_info.get("tau", float("nan")))
                        item["pool_count"] = int(batch_info.get("pool_count", 0))
                        item["raw_pool_count"] = int(batch_info.get("raw_pool_count", batch_info.get("pool_count", 0)))
                        item["pre_feasible_count"] = int(batch_info.get("pre_feasible_count", batch_info.get("pool_count", 0)))
                        item["pre_feasible_ratio"] = float(batch_info.get("pre_feasible_ratio", 1.0))
                        item["pre_margin_min"] = float(batch_info.get("pre_margin_min", float("nan")))
                        item["pre_margin_median"] = float(batch_info.get("pre_margin_median", float("nan")))
                        item["pre_margin_max"] = float(batch_info.get("pre_margin_max", float("nan")))
                        item["good_count"] = int(batch_info.get("good_count", 0))
                        item["uncertain_count"] = int(batch_info.get("uncertain_count", 0))
                        item["bad_count"] = int(batch_info.get("bad_count", 0))
                    focus1_pending.extend(batch_rows)
            if focus1_pending:
                item = focus1_pending.pop(0)
                x_next = np.asarray(item.get("x"), dtype=float).reshape(-1)
                segment = "focus1"
                source_mode = "focus1_" + str(item.get("class", "unknown"))
                acq_type = "FOCUS1"
                focus1_active = True
                focus1_class = str(item.get("class", ""))
                focus1_batch_mode = str(item.get("batch_mode", ""))
                focus1_batch_size_planned = int(item.get("batch_size", len(focus1_pending) + 1))
                focus1_batch_remaining = int(len(focus1_pending))
                focus1_tau = float(item.get("tau", float("nan")))
                focus1_pool_count = int(item.get("pool_count", 0))
                focus1_raw_pool_count = int(item.get("raw_pool_count", focus1_pool_count))
                focus1_pre_feasible_count = int(item.get("pre_feasible_count", focus1_pool_count))
                focus1_pre_feasible_ratio = float(item.get("pre_feasible_ratio", float("nan")))
                focus1_pre_margin_min = float(item.get("pre_margin_min", float("nan")))
                focus1_pre_margin_median = float(item.get("pre_margin_median", float("nan")))
                focus1_pre_margin_max = float(item.get("pre_margin_max", float("nan")))
                focus1_good_count = int(item.get("good_count", 0))
                focus1_uncertain_count = int(item.get("uncertain_count", 0))
                focus1_bad_count = int(item.get("bad_count", 0))
                focus1_score = float(item.get("score", float("nan")))
                pre_generated_count = int(focus1_pool_count)
                focus1_eval_count += 1
        focus2_needed = (
            x_next is None
            and focus2_active
            and not focus2_finalized
            and not focus1_needed
            and int(focus2_eval_count) < int(focus2_budget)
        )
        if focus2_needed:
            if not focus2_regions_initialized:
                t0 = _timer_start()
                focus2_regions_payload = _build_focus2_regions_from_archive(
                    rng=rng,
                    model=gp_model,
                    X_archive=X_archive,
                    y_archive=y_archive,
                    objective_sense=objective_sense,
                    lb=full_lb,
                    ub=full_ub,
                    selected_features=selected_features,
                    system=system,
                    var_names=var_names,
                    constraint_defs=constraint_defs,
                    enforce_pre_constraints=bool(system.enforce_pre_constraints),
                )
                if not focus2_regions_payload.get("regions"):
                    primary_focus2_regions_payload = dict(focus2_regions_payload)
                    fallback_focus2_regions_payload = _build_focus2_fallback_region_from_archive(
                        X_archive=X_archive,
                        y_archive=y_archive,
                        objective_sense=objective_sense,
                        lb=full_lb,
                        ub=full_ub,
                        selected_features=selected_features,
                        system=system,
                    )
                    if fallback_focus2_regions_payload.get("regions"):
                        fallback_focus2_regions_payload["primary_region_output"] = primary_focus2_regions_payload
                        focus2_regions_payload = fallback_focus2_regions_payload
                focus2_region_states = _build_focus2_region_states(
                    regions_payload=focus2_regions_payload,
                    selected_features=selected_features,
                    default_lb=full_lb,
                    default_ub=full_ub,
                )
                _timer_add("focus2_region_init", t0)
                focus2_initial_info = {
                    "region_count": int(len(focus2_region_states)),
                    "status": str(focus2_regions_payload.get("status", "")),
                    "reason": str(focus2_regions_payload.get("reason", "")),
                    "retry_used": bool(focus2_regions_payload.get("retry_used", False)),
                    "source_focus": str(focus2_regions_payload.get("source_focus", "")),
                    "candidate_pool_count": int(focus2_regions_payload.get("candidate_pool_count", 0)),
                    "candidate_good_count": int((focus2_regions_payload.get("candidate_counts", {}) or {}).get("good", 0))
                    if isinstance(focus2_regions_payload.get("candidate_counts", {}), dict) else 0,
                    "candidate_uncertain_count": int((focus2_regions_payload.get("candidate_counts", {}) or {}).get("uncertain", 0))
                    if isinstance(focus2_regions_payload.get("candidate_counts", {}), dict) else 0,
                    "candidate_bad_count": int((focus2_regions_payload.get("candidate_counts", {}) or {}).get("bad", 0))
                    if isinstance(focus2_regions_payload.get("candidate_counts", {}), dict) else 0,
                    "primary_status": str(
                        (focus2_regions_payload.get("primary_region_output", {}) or {}).get("status", "")
                    )
                    if isinstance(focus2_regions_payload.get("primary_region_output", {}), dict) else "",
                    "primary_reason": str(
                        (focus2_regions_payload.get("primary_region_output", {}) or {}).get("reason", "")
                    )
                    if isinstance(focus2_regions_payload.get("primary_region_output", {}), dict) else "",
                    "used_fallback": bool(str(focus2_regions_payload.get("status", "")).lower() == "fallback"),
                }
                focus2_regions_initialized = True
                print(
                    "[Optimizer][Focus2] regions initialized "
                    f"count={len(focus2_region_states)} "
                    f"status={focus2_initial_info['status']} "
                    f"reason={focus2_initial_info['reason']} "
                    f"archive={int(X_archive.shape[0])} "
                    f"budget={focus2_budget}",
                    flush=True,
                )
                if not focus2_region_states:
                    focus2_finalized = True
                    focus2_counts["skipped"] = int(focus2_counts.get("skipped", 0)) + 1

            if not focus2_finalized:
                focus2_progress = float(focus2_eval_count) / float(max(int(focus2_budget) - 1, 1))
                focus2_kappa = (
                    float(getattr(system, "focus2_kappa_start", 2.50))
                    + (
                        float(getattr(system, "focus2_kappa_end", 1.20))
                        - float(getattr(system, "focus2_kappa_start", 2.50))
                    )
                    * float(np.clip(focus2_progress, 0.0, 1.0))
                )
                focus2_kappa = float(max(focus2_kappa, float(getattr(system, "focus2_kappa_min", 1.50))))
                if not focus2_pending:
                    remaining_budget = int(max(int(focus2_budget) - int(focus2_eval_count), 0))
                    t0 = _timer_start()
                    focus2_batch, focus2_batch_info = _build_focus2_lite_batch(
                        rng=rng,
                        region_states=focus2_region_states,
                        X_archive=X_archive,
                        y_archive=y_archive,
                        objective_sense=objective_sense,
                        system=system,
                        selected_features=selected_features,
                        full_lb=full_lb,
                        full_ub=full_ub,
                        kappa=float(focus2_kappa),
                        xi=float(system.ei_xi),
                        post_feasible_prob_fn=post_prob_fn if post_penalty_active else None,
                        post_penalty_lambda=post_penalty_lambda if post_penalty_active else 0.0,
                        max_candidates=remaining_budget,
                        var_names=var_names,
                        constraint_defs=constraint_defs,
                        enforce_pre_constraints=bool(system.enforce_pre_constraints),
                    )
                    _timer_add("focus2_batch", t0)
                    if focus2_batch:
                        for dup in list(focus2_batch_info.get("duplicate_skips", []) or []):
                            if not isinstance(dup, dict):
                                continue
                            _increment_focus2_pair_count(
                                focus2_duplicate_pair_counts,
                                str(dup.get("region_id", "")),
                                str(dup.get("duplicate_with_region_id", "")),
                            )
                        focus2_batch_id += 1
                        scheduler_rows = list(focus2_batch_info.get("scheduler", []) or [])
                        if scheduler_rows:
                            focus2_scheduler_history.append({
                                "batch_id": int(focus2_batch_id),
                                "remaining_budget": int(remaining_budget),
                                "scheduled_region_count": int(focus2_batch_info.get("scheduled_region_count", 0)),
                                "rows": scheduler_rows,
                            })
                        batch_size = int(len(focus2_batch))
                        for item in focus2_batch:
                            state_for_item = item.get("region_state")
                            if isinstance(state_for_item, dict):
                                state_for_item["last_candidate_x"] = np.asarray(item.get("x"), dtype=float).reshape(-1).copy()
                            item["batch_id"] = int(focus2_batch_id)
                            item["batch_size"] = int(batch_size)
                            item["kappa"] = float(focus2_kappa)
                            item["batch_info"] = dict(focus2_batch_info)
                        focus2_pending.extend(focus2_batch)
                        print(
                            "[Optimizer][Focus2] batch "
                            f"id={focus2_batch_id} size={batch_size} "
                            f"eval={focus2_eval_count}/{focus2_budget} "
                            f"active_regions={sum(1 for s in focus2_region_states if str(s.get('status', 'active')) == 'active')} "
                            f"duplicate_skips={int(focus2_batch_info.get('duplicate_skip_count', 0))} "
                            f"budget_skips={int(focus2_batch_info.get('budget_skip_count', 0))}",
                            flush=True,
                        )
                    else:
                        focus2_candidate_info = dict(focus2_batch_info)
                        focus2_finalized = True
                        focus2_counts["skipped"] = int(focus2_counts.get("skipped", 0)) + 1
                        print(
                            "[Optimizer][Focus2] finalized without candidate "
                            f"reason={focus2_candidate_info.get('reason', 'unknown')} "
                            f"eval={focus2_eval_count}/{focus2_budget}",
                            flush=True,
                        )

                if focus2_pending:
                    item = focus2_pending.pop(0)
                    x_next = np.asarray(item.get("x"), dtype=float).reshape(-1)
                    focus2_selected_region_id = str(item.get("region_id", ""))
                    focus2_selected_region_state = item.get("region_state") if isinstance(item.get("region_state"), dict) else None
                    focus2_candidate_info = {
                        "reason": "ok",
                        "candidate_count": int(item.get("batch_size", 1)),
                        "available_candidate_count": int(
                            (item.get("batch_info") if isinstance(item.get("batch_info"), dict) else {}).get("available_candidate_count", item.get("batch_size", 1))
                        ),
                        "local_pool_size": int(
                            (item.get("batch_info") if isinstance(item.get("batch_info"), dict) else {}).get("local_pool_size", getattr(system, "focus2_local_pool_size", 512))
                        ),
                        "local_pool_quotas": dict(
                            (item.get("batch_info") if isinstance(item.get("batch_info"), dict) else {}).get("local_pool_quotas", {})
                        ),
                        "plan_pool_count": int(
                            (item.get("batch_info") if isinstance(item.get("batch_info"), dict) else {}).get("plan_pool_count", 0)
                        ),
                        "pre_raw_count": int(
                            (item.get("batch_info") if isinstance(item.get("batch_info"), dict) else {}).get("pre_raw_count", 0)
                        ),
                        "pre_feasible_count": int(
                            (item.get("batch_info") if isinstance(item.get("batch_info"), dict) else {}).get("pre_feasible_count", 0)
                        ),
                        "pre_feasible_ratio": float(
                            (item.get("batch_info") if isinstance(item.get("batch_info"), dict) else {}).get("pre_feasible_ratio", float("nan"))
                        ),
                        "pre_margin_min": float(
                            (item.get("batch_info") if isinstance(item.get("batch_info"), dict) else {}).get("pre_margin_min", float("nan"))
                        ),
                        "pre_margin_median": float(
                            (item.get("batch_info") if isinstance(item.get("batch_info"), dict) else {}).get("pre_margin_median", float("nan"))
                        ),
                        "pre_margin_max": float(
                            (item.get("batch_info") if isinstance(item.get("batch_info"), dict) else {}).get("pre_margin_max", float("nan"))
                        ),
                        "duplicate_skip_count": int(
                            (item.get("batch_info") if isinstance(item.get("batch_info"), dict) else {}).get("duplicate_skip_count", 0)
                        ),
                        "budget_skip_count": int(
                            (item.get("batch_info") if isinstance(item.get("batch_info"), dict) else {}).get("budget_skip_count", 0)
                        ),
                        "min_candidate_rms_distance": float(
                            (item.get("batch_info") if isinstance(item.get("batch_info"), dict) else {}).get("min_candidate_rms_distance", float("nan"))
                        ),
                        "archive_distance": float(item.get("archive_distance", float("nan"))),
                        "batch_distance": float(item.get("batch_distance", float("nan"))),
                        "scheduler_priority": float(item.get("scheduler_priority", float("nan"))),
                        "selected_score": float(item.get("score", float("nan"))),
                        "selected_source": str(item.get("selected_source", "local")),
                        "selected_region_archive_count": int(item.get("selected_region_archive_count", 0)),
                        "batch_id": int(item.get("batch_id", 0)),
                        "batch_size": int(item.get("batch_size", 1)),
                        "batch_remaining": int(len(focus2_pending)),
                        "kappa": float(item.get("kappa", focus2_kappa)),
                    }
                    focus2_eval_count += 1
                    focus2_counts["local"] = int(focus2_counts.get("local", 0)) + 1
                    segment = "focus2"
                    focus2_local_source = str(focus2_candidate_info.get("selected_source", "local"))
                    source_mode = "focus2_" + focus2_local_source
                    acq_type = "FOCUS2_LCB"
                    pre_generated_count = int(focus2_candidate_info.get("plan_pool_count", 0))
                else:
                    focus2_finalized = True
        if x_next is None:
            _apply_focus2_final_bounds_if_ready()
        focus3_available_budget = int(focus3_budget)
        if (
            focus_pipeline_plan.has("focus0")
            and not focus0_pending
            and (int(X_archive.shape[0]) >= int(focus0_target_count) or int(focus0_eval_count) >= int(focus0_budget))
        ):
            focus3_available_budget += int(max(int(focus0_budget) - int(focus0_eval_count), 0))
        if (
            focus_pipeline_plan.has("focus1")
            and not focus1_pending
            and (
                bool(focus1_early_stopped)
                or int(X_archive.shape[0]) >= int(focus1_target_count)
                or int(focus1_eval_count) >= int(focus1_budget)
            )
        ):
            focus3_available_budget += int(max(int(focus1_budget) - int(focus1_eval_count), 0))
        if bool(focus2_finalized):
            focus3_available_budget += int(max(int(focus2_budget) - int(focus2_eval_count), 0))
        focus3_bounds_ready = bool(external_selected_bounds or focus2_bounds_applied)
        if (
            x_next is None
            and focus_pipeline_plan.has("focus3")
            and bool(focus3_bounds_ready)
            and int(focus3_eval_count) < int(focus3_available_budget)
        ):
            if X_archive.shape[0] < 2:
                raise RuntimeError(
                    "Focus3 requires objective archive data. "
                    "Run DOE/provide objective CSV, or include focus0/focus1 before focus3."
                )
            if bool(getattr(system, "source_mixture_enabled", True)) and X_archive.shape[0] > 0:
                p_topk, p_bnd, p_rand = _resolve_focus3_source_probs(
                    system=system,
                    budget_class=focus3_budget_class,
                )
                p_topk, p_bnd, p_rand, focus3_source_adapt_info = _adapt_focus3_source_probs(
                    system=system,
                    p_topk=p_topk,
                    p_boundary=p_bnd,
                    p_random=p_rand,
                    p_dim=p_dim,
                    n_train=int(X_train.shape[0]),
                    gp_fallback_used=bool(gp_fallback_used),
                    best_raw_history=best_raw_history,
                    objective_sense=objective_sense,
                )
                p_topk, p_bnd, p_rand, focus3_effective_kappa, focus3_recover_info = _apply_focus3_recover_policy(
                    system=system,
                    p_topk=p_topk,
                    p_boundary=p_bnd,
                    p_random=p_rand,
                    kappa=float(kappa),
                    best_raw_history=best_raw_history,
                    objective_sense=objective_sense,
                    has_constraints=bool(constraint_defs),
                )
                p_topk, p_bnd, p_rand, focus3_source_cap_info = _apply_focus3_no_constraint_source_cap(
                    system=system,
                    p_topk=p_topk,
                    p_boundary=p_bnd,
                    p_random=p_rand,
                    constraint_defs=constraint_defs,
                )
                p_topk, p_bnd, p_rand, p_best_local, focus3_best_local_info = _apply_focus3_best_local_policy(
                    system=system,
                    p_topk=p_topk,
                    p_boundary=p_bnd,
                    p_random=p_rand,
                    p_dim=p_dim,
                    n_train=int(X_train.shape[0]),
                    focus3_eval_count=int(focus3_eval_count),
                    has_constraints=bool(constraint_defs),
                )
                focus3_fallback_bounds = bool(
                    focus2_final_info.get("bounds_policy", {}).get("fallback_selected", False)
                    if isinstance(focus2_final_info.get("bounds_policy", {}), dict)
                    else False
                ) or bool(focus2_initial_info.get("used_fallback", False))
                p_topk, p_bnd, p_rand, p_best_local, focus3_fallback_bounds_source_info = _apply_focus3_fallback_bounds_source_policy(
                    system=system,
                    p_topk=p_topk,
                    p_boundary=p_bnd,
                    p_random=p_rand,
                    p_best_local=p_best_local,
                    fallback_bounds=bool(focus3_fallback_bounds),
                )
                p_topk, p_bnd, p_rand, p_best_local, focus3_recover_best_local_info = _apply_focus3_recover_best_local_policy(
                    system=system,
                    p_topk=p_topk,
                    p_boundary=p_bnd,
                    p_random=p_rand,
                    p_best_local=p_best_local,
                    recover_info=focus3_recover_info,
                    has_constraints=bool(constraint_defs),
                )
                source_prob_topk = float(p_topk)
                source_prob_boundary = float(p_bnd)
                source_prob_random = float(p_rand)
                source_prob_best_local = float(p_best_local)
                focus3_best_local_sigma, focus3_best_local_sigma_info = _resolve_focus3_best_local_sigma(
                    system=system,
                    focus3_eval_count=int(focus3_eval_count),
                    recover_info=focus3_recover_info,
                )
                pre_active = bool(system.enforce_pre_constraints) and bool(constraint_defs)
                segment = "focus3"
                focus3_requested_acq_type = _normalize_focus3_acq_type(getattr(system, "focus3_acq_type", "auto"))
                focus3_acq_type, focus3_acq_policy_info = _resolve_focus3_effective_acq_type(
                    requested=focus3_requested_acq_type,
                    system=system,
                    p_dim=p_dim,
                    n_train=int(X_train.shape[0]),
                    gp_fallback_used=bool(gp_fallback_used),
                    lb=lb,
                    ub=ub,
                    full_lb=full_lb,
                    full_ub=full_ub,
                    best_local_active=bool(focus3_best_local_info.get("applied", False)),
                )
                if (
                    bool(focus3_recover_info.get("active", False))
                    and focus3_requested_acq_type == "AUTO"
                    and str(focus3_acq_type).upper() in {"EI", "MEAN"}
                ):
                    focus3_acq_type = "LCB"
                    focus3_acq_policy_info["effective"] = "LCB"
                    focus3_acq_policy_info["reason"] = str(focus3_acq_policy_info.get("reason", "")) + "+recover"
                focus3_dedup_acq_type = str(focus3_acq_type)
                if pre_active:
                    if bool(getattr(system, "constraint_boundary_source_enabled", True)):
                        p_constraint = float(np.clip(float(getattr(system, "focus3_constraint_boundary_prob", 0.20)), 0.0, 0.8))
                        scaled = float(max(1.0 - p_constraint, 0.0))
                        source_names = np.asarray(["topk", "boundary", "random", "constraint_boundary"])
                        source_probs = np.asarray([
                            float(p_topk) * scaled,
                            float(p_bnd) * scaled,
                            float(p_rand) * scaled,
                            p_constraint,
                        ], dtype=float)
                        source_probs = source_probs / float(np.sum(source_probs))
                    else:
                        source_names = np.asarray(["topk", "boundary", "random"])
                        source_probs = np.asarray([p_topk, p_bnd, p_rand], dtype=float)
                    source = str(rng.choice(source_names, p=source_probs))
                    source_mode = str(source)
                    pool_starts, pre_fallback_x, pre_fallback_margin, pre_retry_used, pre_generated_count = _build_preconstrained_source_starts(
                        rng=rng,
                        source=source,
                        X_train=X_train,
                        y_train=y_train,
                        objective_sense=objective_sense,
                        lb=lb,
                        ub=ub,
                        base_pool_size=int(getattr(system, "source_pool_size", 24)),
                        topk_fraction=float(getattr(system, "source_topk_fraction", 0.20)),
                        topk_sigma=float(getattr(system, "source_topk_perturb_sigma", 0.08)),
                        boundary_near_ratio=float(getattr(system, "source_boundary_near_ratio", 0.03)),
                        var_names=var_names,
                        constraint_defs=constraint_defs,
                        enforce_pre_constraints=pre_active,
                        feasible_multiplier=int(getattr(system, "source_feasible_multiplier", 3)),
                        feasible_retry=int(getattr(system, "source_feasible_retry", 3)),
                        min_starts=int(getattr(system, "source_feasible_min_starts", 1)),
                    )
                    acq_type_source = focus3_acq_type
                    focus3_dedup_pool = pool_starts.copy()
                    if gp_model is not None and pool_starts.shape[0] > 0:
                        def _pre_fn(x_arr: np.ndarray) -> bool:
                            ok, _m = _evaluate_pre_constraint(
                                x=x_arr,
                                var_names=var_names,
                                constraint_defs=constraint_defs,
                                enforce_pre_constraints=pre_active,
                            )
                            return bool(ok)

                        t0 = _timer_start()
                        x_next = acq.optimize(
                            model=gp_model,
                            y_best=best_y_raw,
                            lb=lb,
                            ub=ub,
                            starts=pool_starts,
                            objective_sense=objective_sense,
                            acq_type=acq_type_source,
                            kappa=float(focus3_effective_kappa),
                            xi=float(system.ei_xi),
                            pre_feasible_fn=_pre_fn,
                            post_feasible_prob_fn=post_prob_fn if post_penalty_active else None,
                            post_penalty_lambda=post_penalty_lambda if post_penalty_active else 0.0,
                            maxiter=int(getattr(system, "focus3_refine_maxiter", 35)),
                            maxfun=int(getattr(system, "focus3_refine_maxfun", 90)),
                        )
                        _timer_add("focus3_refine", t0)
                    if x_next is None and pool_starts.shape[0] > 0:
                        x_next = pool_starts[int(rng.integers(0, pool_starts.shape[0]))]
                    if x_next is None and pre_fallback_x is not None and float(pre_fallback_margin) >= 0.0:
                        x_next = pre_fallback_x.copy()
                        pre_fallback_used = True
                        pre_feasible_next = True
                        pre_margin_next = float(pre_fallback_margin)
                        pre_violation_next = float(max(0.0, -pre_margin_next))
                    acq_type = acq_type_source
                else:
                    base_refine_every = int(max(int(getattr(system, "focus3_refine_every", 3)), 1))
                    refine_every, focus3_refine_adaptive_info = _resolve_focus3_refine_every_adaptive(
                        system=system,
                        history_rows=history_rows,
                        default_every=base_refine_every,
                    )
                    discrete_enabled = bool(getattr(system, "focus3_discrete_enabled", True))
                    focus3_eval_index = int(focus3_eval_count) + 1
                    refine_this_iter = (not discrete_enabled) or refine_every <= 1 or (focus3_eval_index % refine_every == 0)
                    focus3_refine_cooldown_info = _resolve_focus3_refine_cooldown(
                        system=system,
                        history_rows=history_rows,
                        focus3_eval_count=int(focus3_eval_count),
                    )
                    if (
                        bool(discrete_enabled)
                        and bool(focus3_refine_cooldown_info.get("active", False))
                    ):
                        refine_this_iter = False
                    t0 = _timer_start()
                    if refine_this_iter:
                        refine_starts, pool_counts, selected_counts, focus3_plan_diag = _build_focus3_plan_refine_starts(
                            rng=rng,
                            system=system,
                            model=gp_model,
                            X_train=X_train,
                            X_cover_ref=X_archive,
                            y_train=y_train,
                            objective_sense=objective_sense,
                            lb=lb,
                            ub=ub,
                            y_best=best_y_raw,
                            p_topk=p_topk,
                            p_boundary=p_bnd,
                            p_random=p_rand,
                            p_best_local=p_best_local,
                            plan_pool_per_source=_resolve_focus3_plan_pool_per_source(
                                system=system,
                                p_dim=p_dim,
                            ),
                            refine_starts=int(getattr(system, "focus3_refine_starts", 30)),
                            topk_fraction=float(getattr(system, "source_topk_fraction", 0.20)),
                            topk_sigma=float(getattr(system, "source_topk_perturb_sigma", 0.08)),
                            boundary_near_ratio=float(getattr(system, "source_boundary_near_ratio", 0.03)),
                            best_local_sigma=float(focus3_best_local_sigma),
                            best_local_top_count=int(getattr(system, "focus3_best_local_top_count", 3)),
                            acq_type=focus3_acq_type,
                            kappa=float(focus3_effective_kappa),
                            xi=float(system.ei_xi),
                            post_feasible_prob_fn=post_prob_fn if post_penalty_active else None,
                            post_penalty_lambda=post_penalty_lambda if post_penalty_active else 0.0,
                            has_constraints=bool(constraint_defs),
                        )
                        focus3_plan_mode = "refine"
                    else:
                        x_discrete, pool_counts, selected_counts, focus3_plan_diag = _build_focus3_plan_discrete_candidate(
                            rng=rng,
                            system=system,
                            model=gp_model,
                            X_train=X_train,
                            X_cover_ref=X_archive,
                            y_train=y_train,
                            objective_sense=objective_sense,
                            lb=lb,
                            ub=ub,
                            y_best=best_y_raw,
                            p_best_local=p_best_local,
                            plan_pool_per_source=_resolve_focus3_plan_pool_per_source(
                                system=system,
                                p_dim=p_dim,
                            ),
                            topk_fraction=float(getattr(system, "source_topk_fraction", 0.20)),
                            topk_sigma=float(getattr(system, "source_topk_perturb_sigma", 0.08)),
                            boundary_near_ratio=float(getattr(system, "source_boundary_near_ratio", 0.03)),
                            best_local_sigma=float(focus3_best_local_sigma),
                            best_local_top_count=int(getattr(system, "focus3_best_local_top_count", 3)),
                            acq_type=focus3_acq_type,
                            kappa=float(focus3_effective_kappa),
                            xi=float(system.ei_xi),
                            post_feasible_prob_fn=post_prob_fn if post_penalty_active else None,
                            post_penalty_lambda=post_penalty_lambda if post_penalty_active else 0.0,
                            has_constraints=bool(constraint_defs),
                        )
                        refine_starts = x_discrete.reshape(1, -1) if x_discrete is not None else np.empty((0, p_dim), dtype=float)
                        focus3_plan_mode = "discrete"
                    _timer_add("focus3_plan", t0)
                    focus3_plan_pool_count = int(sum(pool_counts.values()))
                    focus3_refine_start_count = 0 if focus3_plan_mode == "discrete" else int(refine_starts.shape[0])
                    focus3_selected_topk = int(selected_counts.get("topk", 0))
                    focus3_selected_best_local = int(selected_counts.get("best_local", 0))
                    focus3_selected_boundary = int(selected_counts.get("boundary", 0))
                    focus3_selected_random = int(selected_counts.get("random", 0))
                    focus3_best_plan_source = str(focus3_plan_diag.get("best_plan_source", ""))
                    best_scores = focus3_plan_diag.get("best_scores", {})
                    if isinstance(best_scores, dict):
                        focus3_best_score_topk = float(best_scores.get("topk", float("nan")))
                        focus3_best_score_best_local = float(best_scores.get("best_local", float("nan")))
                        focus3_best_score_boundary = float(best_scores.get("boundary", float("nan")))
                        focus3_best_score_random = float(best_scores.get("random", float("nan")))
                    random_cover = focus3_plan_diag.get("random_cover", {})
                    if isinstance(random_cover, dict):
                        focus3_random_cover_info = dict(random_cover)
                    boundary_gate = focus3_plan_diag.get("boundary_gate", {})
                    if isinstance(boundary_gate, dict):
                        focus3_boundary_gate_info = dict(boundary_gate)
                    boundary_score_penalty = focus3_plan_diag.get("boundary_score_penalty", {})
                    if isinstance(boundary_score_penalty, dict):
                        focus3_boundary_score_penalty_info = dict(boundary_score_penalty)
                    pre_generated_count = int(focus3_plan_pool_count)
                    source_mode = "plan_" + str(focus3_plan_mode or "refine")
                    focus3_dedup_pool = refine_starts.copy()
                    labels = focus3_plan_diag.get("selected_labels", [])
                    best_plan_source_for_filter = str(focus3_plan_diag.get("best_plan_source", ""))
                    refine_allowed = True
                    if (
                        focus3_plan_mode == "refine"
                        and bool(getattr(system, "focus3_refine_source_filter_enabled", True))
                        and not bool(constraint_defs)
                    ):
                        allowed_sources = {
                            item.strip()
                            for item in str(getattr(system, "focus3_refine_allowed_sources", "topk,best_local")).split(",")
                            if item.strip()
                        }
                        refine_allowed = best_plan_source_for_filter in allowed_sources
                        focus3_refine_filter_info = {
                            "enabled": True,
                            "skipped": bool(not refine_allowed),
                            "reason": "source_not_allowed" if not refine_allowed else "source_allowed",
                            "best_plan_source": str(best_plan_source_for_filter),
                            "allowed_sources": ",".join(sorted(allowed_sources)),
                        }
                    else:
                        focus3_refine_filter_info = {
                            "enabled": bool(getattr(system, "focus3_refine_source_filter_enabled", True)),
                            "skipped": False,
                            "reason": "not_applicable",
                            "best_plan_source": str(best_plan_source_for_filter),
                            "allowed_sources": str(getattr(system, "focus3_refine_allowed_sources", "topk,best_local")),
                        }
                    if focus3_plan_mode == "discrete" and refine_starts.shape[0] > 0:
                        x_next = refine_starts[0].copy()
                    elif focus3_plan_mode == "refine" and not refine_allowed and refine_starts.shape[0] > 0:
                        x_next = _select_best_scored_candidate(
                            pool=refine_starts,
                            model=gp_model,
                            y_best=best_y_raw,
                            objective_sense=objective_sense,
                            acq_type=focus3_acq_type,
                            kappa=float(focus3_effective_kappa),
                            xi=float(system.ei_xi),
                            post_feasible_prob_fn=post_prob_fn if post_penalty_active else None,
                            post_penalty_lambda=post_penalty_lambda if post_penalty_active else 0.0,
                        )
                        focus3_plan_mode = "refine_skipped"
                    elif gp_model is not None and refine_starts.shape[0] > 0:
                        t0 = _timer_start()
                        x_next = acq.optimize(
                            model=gp_model,
                            y_best=best_y_raw,
                            lb=lb,
                            ub=ub,
                            starts=refine_starts,
                            objective_sense=objective_sense,
                            acq_type=focus3_acq_type,
                            kappa=float(focus3_effective_kappa),
                            xi=float(system.ei_xi),
                            post_feasible_prob_fn=post_prob_fn if post_penalty_active else None,
                            post_penalty_lambda=post_penalty_lambda if post_penalty_active else 0.0,
                            maxiter=int(getattr(system, "focus3_refine_maxiter", 35)),
                            maxfun=int(getattr(system, "focus3_refine_maxfun", 90)),
                        )
                        _timer_add("focus3_refine", t0)
                    if x_next is None and refine_starts.shape[0] > 0:
                        x_next = refine_starts[int(rng.integers(0, refine_starts.shape[0]))]
                    if isinstance(labels, list) and x_next is not None:
                        focus3_nearest_refine_source = _nearest_start_source(
                            x=np.asarray(x_next, dtype=float),
                            starts=refine_starts,
                            labels=[str(v) for v in labels],
                            lb=lb,
                            ub=ub,
                        )
                    acq_type = focus3_acq_type
            elif gp_model is not None:
                t0 = _timer_start()
                x_next = acq.optimize(
                    model=gp_model,
                    y_best=best_y_raw,
                    lb=lb,
                    ub=ub,
                    starts=starts,
                    objective_sense=objective_sense,
                    acq_type=acq_type,
                    kappa=float(kappa),
                    xi=float(system.ei_xi),
                    post_feasible_prob_fn=post_prob_fn if post_penalty_active else None,
                    post_penalty_lambda=post_penalty_lambda if post_penalty_active else 0.0,
                    maxiter=int(getattr(system, "focus3_refine_maxiter", 35)),
                    maxfun=int(getattr(system, "focus3_refine_maxfun", 90)),
                )
                _timer_add("focus3_refine", t0)
                focus3_dedup_pool = starts.copy()
                focus3_dedup_acq_type = str(acq_type)
            if x_next is None:
                x_next = rng.uniform(lb, ub, size=(lb.shape[0],))
            if x_next is not None:
                if bool(getattr(system, "focus3_dedup_enabled", True)):
                    min_dist = float(max(float(getattr(system, "focus3_min_candidate_rms_distance", 0.01)), 0.0))
                    before_dist = _normalized_rms_min_distance(
                        x=np.asarray(x_next, dtype=float),
                        X_ref=X_archive,
                        lb=lb,
                        ub=ub,
                    )
                    focus3_dedup_info["archive_distance_before"] = float(before_dist)
                    focus3_dedup_info["archive_distance_after"] = float(before_dist)
                    if min_dist > 0.0 and before_dist < min_dist:
                        t0 = _timer_start()
                        fallback_x, fallback_dist, fallback_kind = _select_focus3_dedup_fallback(
                            rng=rng,
                            candidate_pool=focus3_dedup_pool,
                            model=gp_model,
                            y_best=best_y_raw,
                            objective_sense=objective_sense,
                            acq_type=focus3_dedup_acq_type,
                            kappa=float(focus3_effective_kappa),
                            xi=float(system.ei_xi),
                            lb=lb,
                            ub=ub,
                            X_archive=X_archive,
                            min_rms_distance=min_dist,
                            post_feasible_prob_fn=post_prob_fn if post_penalty_active else None,
                            post_penalty_lambda=post_penalty_lambda if post_penalty_active else 0.0,
                            random_attempts=int(getattr(system, "focus3_dedup_random_attempts", 64)),
                        )
                        _timer_add("focus3_dedup", t0)
                        focus3_dedup_info["fallback"] = str(fallback_kind)
                        if fallback_x is not None:
                            x_next = fallback_x.copy()
                            focus3_dedup_info["applied"] = True
                            focus3_dedup_info["archive_distance_after"] = float(fallback_dist)
                focus3_eval_count += 1
        if x_next is None:
            break

        x_next = np.clip(np.asarray(x_next, dtype=float).reshape(-1), lb, ub)
        if np.any(~np.isfinite(x_next)):
            x_next = rng.uniform(lb, ub, size=(lb.shape[0],))
        if not pre_fallback_used:
            x_next = _sample_feasible_candidate(
                x=x_next,
                rng=rng,
                lb=lb,
                ub=ub,
                var_names=var_names,
                constraint_defs=constraint_defs,
                enforce_pre_constraints=bool(system.enforce_pre_constraints),
                max_attempts=int(getattr(system, "pre_final_feasible_attempts", 256)),
            )

        pre_feasible_next, pre_margin_next = _evaluate_pre_constraint(
            x=x_next,
            var_names=var_names,
            constraint_defs=constraint_defs,
            enforce_pre_constraints=bool(system.enforce_pre_constraints),
        )
        pre_violation_next = float(max(0.0, -float(pre_margin_next)))

        key = _round_key(x_next, decimals=int(system.dedup_decimals))
        if key in seen:
            for _ in range(12):
                trial = rng.uniform(lb, ub, size=(lb.shape[0],))
                tkey = _round_key(trial, decimals=int(system.dedup_decimals))
                if tkey not in seen and _point_in_bounds(trial, lb, ub):
                    x_next = trial
                    key = tkey
                    break
        if bool(system.enforce_pre_constraints) and bool(constraint_defs):
            x_next = _sample_feasible_candidate(
                x=x_next,
                rng=rng,
                lb=lb,
                ub=ub,
                var_names=var_names,
                constraint_defs=constraint_defs,
                enforce_pre_constraints=True,
                max_attempts=int(getattr(system, "pre_final_feasible_attempts", 256)),
            )
            pre_feasible_next, pre_margin_next = _evaluate_pre_constraint(
                x=x_next,
                var_names=var_names,
                constraint_defs=constraint_defs,
                enforce_pre_constraints=True,
            )
            pre_violation_next = float(max(0.0, -float(pre_margin_next)))
            if not pre_feasible_next:
                raise RuntimeError(
                    "Pre-constraint hard filter could not find a feasible optimizer candidate. "
                    f"segment={segment} margin={pre_margin_next:.6g}"
                )
            key = _round_key(x_next, decimals=int(system.dedup_decimals))
        pre_constraint_debug_fields = _pre_constraint_debug_fields(
            x=x_next,
            var_names=var_names,
            constraint_defs=constraint_defs,
            enforce_pre_constraints=bool(system.enforce_pre_constraints),
        )
        seen.add(key)

        if gp_model is not None:
            t0 = _timer_start()
            mu_next, std_next = gp_model.predict(x_next.reshape(1, -1), return_std=True)
            _timer_add("gp_predict_next", t0)
            pred_mean = float(mu_next.reshape(-1)[0])
            pred_std = float(std_next.reshape(-1)[0])
        else:
            pred_mean = float(np.mean(y_train)) if y_train.size > 0 else float("nan")
            pred_std = float(np.std(y_train)) if y_train.size > 1 else float("nan")

        prev_best_y_raw = float(best_y_raw)
        prev_best_y_eff = float(best_y_eff)
        t0 = _timer_start()
        y_next = float(evaluate_objective(x_next))
        _timer_add("objective_eval", t0)
        objective_source = "cae_eval"

        p_feasible = 1.0
        if post_penalty_active and post_prob_fn is not None:
            p_feasible = float(np.clip(post_prob_fn(x_next), 0.0, 1.0))
        y_next_effective = _apply_post_penalty_to_objective(
            y_raw=y_next,
            p_feasible=p_feasible,
            objective_sense=objective_sense,
            penalty_lambda=post_penalty_lambda if post_penalty_active else 0.0,
            score_mode=post_score_mode,
            p_feasible_min=post_p_feasible_min,
            hard_penalty=post_p_feasible_hard_penalty,
        )
        pred_error_raw = float(y_next - pred_mean) if math.isfinite(float(pred_mean)) else float("nan")
        pred_abs_error_raw = float(abs(pred_error_raw)) if math.isfinite(pred_error_raw) else float("nan")
        if objective_sense == "max":
            raw_improvement = float(y_next - prev_best_y_raw)
            effective_improvement = float(y_next_effective - prev_best_y_eff)
        else:
            raw_improvement = float(prev_best_y_raw - y_next)
            effective_improvement = float(prev_best_y_eff - y_next_effective)
        raw_improved = bool(raw_improvement > 0.0)
        effective_improved = bool(effective_improvement > 0.0)
        row_kappa = float(focus3_effective_kappa) if segment == "focus3" else float(kappa)

        acq_base = float("nan")
        acq_effective = float("nan")
        if gp_model is not None:
            try:
                if str(acq_type).upper() == "EI":
                    acq_base = float(
                        acq.acquisition_ei(
                            x_next,
                            gp_model,
                            y_best=best_y_raw,
                            objective_sense=objective_sense,
                            xi=float(system.ei_xi),
                        )
                    )
                else:
                    acq_base = float(
                        acq.acquisition_lcb(
                            x_next,
                            gp_model,
                            kappa=float(row_kappa),
                            objective_sense=objective_sense,
                        )
                    )
                acq_effective = float(acq_base)
                if post_penalty_active:
                    acq_effective = float(
                        _apply_post_penalty_to_objective(
                            y_raw=acq_effective,
                            p_feasible=p_feasible,
                            objective_sense="min",
                            penalty_lambda=post_penalty_lambda,
                            score_mode=post_score_mode,
                            p_feasible_min=post_p_feasible_min,
                            hard_penalty=post_p_feasible_hard_penalty,
                        )
                    )
            except Exception:
                pass

        t0 = _timer_start()
        X_archive, y_archive = append_archive_point(
            X_archive=X_archive,
            y_archive=y_archive,
            x_next=x_next,
            y_next=float(y_next),
        )
        X_train, y_train, gp_train_cap = _select_gp_train_from_archive(
            X=X_archive,
            y=y_archive,
            objective_sense=objective_sense,
            system=system,
            p_dim=p_dim,
            initial_count=initial_archive_count,
        )
        _timer_add("archive_update", t0)

        best_update = update_raw_and_effective_best(
            best_x_raw=best_x_raw,
            best_y_raw=float(best_y_raw),
            best_x_effective=best_x_eff,
            best_y_effective=float(best_y_eff),
            x_next=x_next,
            y_raw_next=float(y_next),
            y_effective_next=float(y_next_effective),
            objective_sense=objective_sense,
        )
        best_x_raw = best_update.best_x_raw.copy()
        best_y_raw = float(best_update.best_y_raw)
        best_x_eff = best_update.best_x_effective.copy()
        best_y_eff = float(best_update.best_y_effective)
        raw_improved = bool(best_update.raw_improved)
        effective_improved = bool(best_update.effective_improved)
        best_raw_history.append(float(best_y_raw))
        opt_focus_level = _segment_to_focus_level(segment=segment)
        goal_early_stop_min_focus = int(max(int(getattr(system, "goal_early_stop_min_focus", 2)), 0))
        goal_early_stop_allowed = bool(opt_focus_level >= goal_early_stop_min_focus)
        row, goal_state = update_goal_and_build_evaluation_row(
            goal_monitor=goal_monitor,
            iteration=int(i + 1),
            best_objective=float(best_y_raw),
            improved=bool(raw_improved),
            allow_early_stop=goal_early_stop_allowed,
            segment=str(segment),
            opt_focus_level=int(opt_focus_level),
            source_mode=str(source_mode),
            x_values={fname: float(value) for fname, value in zip(selected_features, x_next)},
            objective_raw=float(y_next),
            objective_effective=float(y_next_effective),
            previous_best_objective_raw=float(prev_best_y_raw),
            objective_sense=objective_sense,
            pre_feasible=bool(pre_feasible_next),
            pre_margin=float(pre_margin_next),
            algorithm_id="focus_bo",
        )
        if segment == "focus2":
            t0 = _timer_start()
            _update_focus2_region_state(
                state=focus2_selected_region_state,
                x_next=x_next,
                y_next=float(y_next),
                objective_sense=objective_sense,
                system=system,
            )
            focus2_shared_region_ids = _update_focus2_shared_region_states(
                region_states=focus2_region_states,
                selected_state=focus2_selected_region_state,
                x_next=x_next,
                y_next=float(y_next),
                objective_sense=objective_sense,
                system=system,
            )
            if focus2_selected_region_id:
                for shared_id in focus2_shared_region_ids:
                    _increment_focus2_pair_count(
                        focus2_shared_pair_counts,
                        focus2_selected_region_id,
                        str(shared_id),
                    )
            focus2_counts["dropped"] = int(
                sum(1 for s in focus2_region_states if str(s.get("status", "active")) == "dropped")
            )
            if not focus2_pending:
                active_bounds_events = _update_focus2_active_bounds_from_archive(
                    region_states=focus2_region_states,
                    X_archive=X_archive,
                    y_archive=y_archive,
                    objective_sense=objective_sense,
                    full_lb=full_lb,
                    full_ub=full_ub,
                    selected_features=selected_features,
                    system=system,
                )
                if active_bounds_events:
                    for event in active_bounds_events:
                        if isinstance(event, dict):
                            event["iter"] = int(i + 1)
                            event["batch_id"] = int(focus2_candidate_info.get("batch_id", focus2_batch_id)) if focus2_candidate_info else int(focus2_batch_id)
                    focus2_active_bounds_history.extend(active_bounds_events)
                merge_events = _merge_focus2_region_states(
                    region_states=focus2_region_states,
                    duplicate_pair_counts=focus2_duplicate_pair_counts,
                    shared_pair_counts=focus2_shared_pair_counts,
                    X_archive=X_archive,
                    y_archive=y_archive,
                    objective_sense=objective_sense,
                    full_lb=full_lb,
                    full_ub=full_ub,
                    selected_features=selected_features,
                    system=system,
                )
                if merge_events:
                    focus2_merge_history.extend(merge_events)
                should_stop_focus2, focus2_early_stop_info = _focus2_should_early_stop(
                    region_states=focus2_region_states,
                    focus2_eval_count=int(focus2_eval_count),
                    objective_sense=objective_sense,
                    system=system,
                )
                if should_stop_focus2:
                    focus2_finalized = True
            if int(focus2_eval_count) >= int(focus2_budget):
                focus2_finalized = True
            _timer_add("focus2_update", t0)

        _timer_add("iteration", iter_t0)
        row.update({
            "acq_type": str(acq_type),
            "kappa": float(row_kappa),
            "acq_base": float(acq_base),
            "acq_effective": float(acq_effective),
            "pred_mean": float(pred_mean),
            "pred_std": float(pred_std),
            "pred_error_raw": float(pred_error_raw),
            "pred_abs_error_raw": float(pred_abs_error_raw),
            "effective_improvement": float(effective_improvement),
            "effective_improved": bool(effective_improved),
            "objective_source": str(objective_source),
            "init_source": str(init_source),
            "goal_early_stop_min_focus": int(goal_early_stop_min_focus),
            "goal_early_stop_allowed": bool(goal_early_stop_allowed),
            "focus_pipeline_mode": str(focus_pipeline_plan.mode),
            "focus_pipeline_stages": ",".join([str(s) for s in focus_pipeline_plan.stages]),
            "focus_pipeline_bounds_source": str(focus_pipeline_plan.bounds_source),
            "focus_budget_adaptive_enabled": bool(getattr(system, "focus_budget_adaptive_enabled", True)),
            "focus0_budget": int(focus0_budget),
            "focus1_budget": int(focus1_budget),
            "focus2_budget": int(focus2_budget),
            "focus3_budget": int(focus3_budget),
            "focus0_budget_fraction": float(focus0_budget) / float(max(int(n_samples), 1)),
            "focus1_budget_fraction": float(focus1_budget) / float(max(int(n_samples), 1)),
            "focus2_budget_fraction": float(focus2_budget) / float(max(int(n_samples), 1)),
            "focus3_budget_fraction": float(focus3_budget) / float(max(int(n_samples), 1)),
            "focus3_available_budget": int(focus3_available_budget),
            "focus0_eval_count": int(focus0_eval_count),
            "focus1_eval_count": int(focus1_eval_count),
            "focus2_eval_count": int(focus2_eval_count),
            "focus3_eval_count": int(focus3_eval_count),
            "focus0_active": bool(focus0_active),
            "focus0_target_count": int(focus0_target_count_row),
            "focus0_batch_size_planned": int(focus0_batch_size_planned),
            "focus0_batch_remaining": int(focus0_batch_remaining),
            "focus0_pool_count": int(focus0_pool_count),
            "focus1_active": bool(focus1_active),
            "focus1_target_count": int(focus1_target_count),
            "focus1_data_ratio": float(focus1_data_ratio),
            "focus1_class": str(focus1_class),
            "focus1_batch_mode": str(focus1_batch_mode),
            "focus1_batch_size_planned": int(focus1_batch_size_planned),
            "focus1_batch_remaining": int(focus1_batch_remaining),
            "focus1_tau": float(focus1_tau),
            "focus1_pool_count": int(focus1_pool_count),
            "focus1_raw_pool_count": int(focus1_raw_pool_count),
            "focus1_pre_feasible_count": int(focus1_pre_feasible_count),
            "focus1_pre_feasible_ratio": float(focus1_pre_feasible_ratio),
            "focus1_pre_margin_min": float(focus1_pre_margin_min),
            "focus1_pre_margin_median": float(focus1_pre_margin_median),
            "focus1_pre_margin_max": float(focus1_pre_margin_max),
            "focus1_good_count": int(focus1_good_count),
            "focus1_uncertain_count": int(focus1_uncertain_count),
            "focus1_bad_count": int(focus1_bad_count),
            "focus1_score": float(focus1_score),
            "focus1_early_stopped": bool(focus1_early_stopped),
            "focus1_early_stop_reason": str(focus1_early_stop_reason),
            "focus3_budget_class": str(focus3_budget_class),
            "focus3_budget_ratio": float(focus3_budget_ratio),
            "focus2_active": bool(focus2_active),
            "focus2_counts_local": int(focus2_counts.get("local", 0)),
            "focus2_counts_dropped": int(focus2_counts.get("dropped", 0)),
            "focus2_selected_region_id": str(focus2_selected_region_id),
            "focus2_region_init_status": str(focus2_initial_info.get("status", "")) if focus2_initial_info else "",
            "focus2_region_init_reason": str(focus2_initial_info.get("reason", "")) if focus2_initial_info else "",
            "focus2_region_init_retry_used": bool(focus2_initial_info.get("retry_used", False)) if focus2_initial_info else False,
            "focus2_region_init_used_fallback": bool(focus2_initial_info.get("used_fallback", False)) if focus2_initial_info else False,
            "focus2_region_init_primary_status": str(focus2_initial_info.get("primary_status", "")) if focus2_initial_info else "",
            "focus2_region_init_primary_reason": str(focus2_initial_info.get("primary_reason", "")) if focus2_initial_info else "",
            "focus2_region_init_candidate_pool_count": int(focus2_initial_info.get("candidate_pool_count", 0)) if focus2_initial_info else 0,
            "focus2_region_init_good_count": int(focus2_initial_info.get("candidate_good_count", 0)) if focus2_initial_info else 0,
            "focus2_region_init_uncertain_count": int(focus2_initial_info.get("candidate_uncertain_count", 0)) if focus2_initial_info else 0,
            "focus2_region_init_bad_count": int(focus2_initial_info.get("candidate_bad_count", 0)) if focus2_initial_info else 0,
            "focus2_batch_id": int(focus2_candidate_info.get("batch_id", 0)) if focus2_candidate_info else 0,
            "focus2_batch_size": int(focus2_candidate_info.get("batch_size", 0)) if focus2_candidate_info else 0,
            "focus2_batch_remaining": int(focus2_candidate_info.get("batch_remaining", 0)) if focus2_candidate_info else 0,
            "focus2_local_candidate_count": int(focus2_candidate_info.get("candidate_count", 0)) if focus2_candidate_info else 0,
            "focus2_pre_raw_count": int(focus2_candidate_info.get("pre_raw_count", 0)) if focus2_candidate_info else 0,
            "focus2_pre_feasible_count": int(focus2_candidate_info.get("pre_feasible_count", 0)) if focus2_candidate_info else 0,
            "focus2_pre_feasible_ratio": float(focus2_candidate_info.get("pre_feasible_ratio", float("nan"))) if focus2_candidate_info else float("nan"),
            "focus2_pre_margin_min": float(focus2_candidate_info.get("pre_margin_min", float("nan"))) if focus2_candidate_info else float("nan"),
            "focus2_pre_margin_median": float(focus2_candidate_info.get("pre_margin_median", float("nan"))) if focus2_candidate_info else float("nan"),
            "focus2_pre_margin_max": float(focus2_candidate_info.get("pre_margin_max", float("nan"))) if focus2_candidate_info else float("nan"),
            "focus2_duplicate_skip_count": int(focus2_candidate_info.get("duplicate_skip_count", 0)) if focus2_candidate_info else 0,
            "focus2_budget_skip_count": int(focus2_candidate_info.get("budget_skip_count", 0)) if focus2_candidate_info else 0,
            "focus2_min_candidate_rms_distance": float(focus2_candidate_info.get("min_candidate_rms_distance", float("nan"))) if focus2_candidate_info else float("nan"),
            "focus2_archive_distance": float(focus2_candidate_info.get("archive_distance", float("nan"))) if focus2_candidate_info else float("nan"),
            "focus2_batch_distance": float(focus2_candidate_info.get("batch_distance", float("nan"))) if focus2_candidate_info else float("nan"),
            "focus2_scheduler_priority": float(focus2_candidate_info.get("scheduler_priority", float("nan"))) if focus2_candidate_info else float("nan"),
            "focus2_local_selected_source": str(focus2_candidate_info.get("selected_source", "")) if focus2_candidate_info else "",
            "focus2_local_kappa": float(focus2_candidate_info.get("kappa", float("nan"))) if focus2_candidate_info else float("nan"),
            "focus2_initial_volume_ratio": float(focus2_initial_info.get("volume_ratio", float("nan"))) if focus2_initial_info else float("nan"),
            "focus2_final_volume_ratio": float(focus2_final_info.get("volume_ratio", focus2_initial_info.get("volume_ratio", float("nan")))) if (focus2_final_info or focus2_initial_info) else float("nan"),
            "focus2_early_stop_active": bool(focus2_early_stop_info.get("reason", "") not in {"", "conditions_not_met", "insufficient_focus2_evals", "disabled"}),
            "focus2_early_stop_reason": str(focus2_early_stop_info.get("reason", "")),
            "focus3_plan_pool_count": int(focus3_plan_pool_count),
            "focus3_refine_start_count": int(focus3_refine_start_count),
            "focus3_plan_mode": str(focus3_plan_mode),
            "focus3_selected_topk": int(focus3_selected_topk),
            "focus3_selected_best_local": int(focus3_selected_best_local),
            "focus3_selected_boundary": int(focus3_selected_boundary),
            "focus3_selected_random": int(focus3_selected_random),
            "focus3_best_plan_source": str(focus3_best_plan_source),
            "focus3_nearest_refine_source": str(focus3_nearest_refine_source),
            "focus3_best_score_topk": float(focus3_best_score_topk),
            "focus3_best_score_best_local": float(focus3_best_score_best_local),
            "focus3_best_score_boundary": float(focus3_best_score_boundary),
            "focus3_best_score_random": float(focus3_best_score_random),
            "focus3_random_cover_enabled": bool(focus3_random_cover_info.get("enabled", False)),
            "focus3_random_cover_candidate_count": int(focus3_random_cover_info.get("candidate_count", 0)),
            "focus3_random_cover_acq_threshold": float(focus3_random_cover_info.get("acq_threshold", float("nan"))),
            "focus3_random_cover_mean_distance": float(focus3_random_cover_info.get("mean_distance", float("nan"))),
            "focus3_random_cover_min_distance": float(focus3_random_cover_info.get("min_distance", float("nan"))),
            "focus3_boundary_gate_enabled": bool(focus3_boundary_gate_info.get("enabled", False)) if focus3_boundary_gate_info else False,
            "focus3_boundary_gate_applied": bool(focus3_boundary_gate_info.get("applied", False)) if focus3_boundary_gate_info else False,
            "focus3_boundary_gate_allowed": bool(focus3_boundary_gate_info.get("allowed", True)) if focus3_boundary_gate_info else True,
            "focus3_boundary_gate_reason": str(focus3_boundary_gate_info.get("reason", "")) if focus3_boundary_gate_info else "",
            "focus3_boundary_gate_margin": float(focus3_boundary_gate_info.get("margin", float("nan"))) if focus3_boundary_gate_info else float("nan"),
            "focus3_boundary_gate_boundary_score": float(focus3_boundary_gate_info.get("boundary_score", float("nan"))) if focus3_boundary_gate_info else float("nan"),
            "focus3_boundary_gate_non_boundary_best": float(focus3_boundary_gate_info.get("non_boundary_best", float("nan"))) if focus3_boundary_gate_info else float("nan"),
            "focus3_boundary_score_penalty_enabled": bool(focus3_boundary_score_penalty_info.get("enabled", False)) if focus3_boundary_score_penalty_info else False,
            "focus3_boundary_score_penalty_applied": bool(focus3_boundary_score_penalty_info.get("applied", False)) if focus3_boundary_score_penalty_info else False,
            "focus3_boundary_score_penalty_reason": str(focus3_boundary_score_penalty_info.get("reason", "")) if focus3_boundary_score_penalty_info else "",
            "focus3_boundary_score_penalty_value": float(focus3_boundary_score_penalty_info.get("penalty", 0.0)) if focus3_boundary_score_penalty_info else 0.0,
            "focus3_boundary_score_penalty_scale": float(focus3_boundary_score_penalty_info.get("scale", float("nan"))) if focus3_boundary_score_penalty_info else float("nan"),
            "focus3_source_adaptive_enabled": bool(focus3_source_adapt_info.get("enabled", False)) if focus3_source_adapt_info else False,
            "focus3_data_ratio": float(focus3_source_adapt_info.get("data_ratio", float("nan"))) if focus3_source_adapt_info else float("nan"),
            "focus3_data_reliability": str(focus3_source_adapt_info.get("reliability", "")) if focus3_source_adapt_info else "",
            "focus3_recent_state": str(focus3_source_adapt_info.get("recent_state", "")) if focus3_source_adapt_info else "",
            "focus3_recent_improvement": float(focus3_source_adapt_info.get("recent_improvement", float("nan"))) if focus3_source_adapt_info else float("nan"),
            "focus3_acq_requested": str(focus3_acq_policy_info.get("requested", "")) if focus3_acq_policy_info else "",
            "focus3_acq_effective": str(focus3_acq_policy_info.get("effective", "")) if focus3_acq_policy_info else "",
            "focus3_acq_reason": str(focus3_acq_policy_info.get("reason", "")) if focus3_acq_policy_info else "",
            "focus3_acq_data_ratio": float(focus3_acq_policy_info.get("data_ratio", float("nan"))) if focus3_acq_policy_info else float("nan"),
            "focus3_bounds_volume_ratio": float(focus3_acq_policy_info.get("volume_ratio", float("nan"))) if focus3_acq_policy_info else float("nan"),
            "focus3_bounds_mean_width_ratio": float(focus3_acq_policy_info.get("mean_width_ratio", float("nan"))) if focus3_acq_policy_info else float("nan"),
            "focus3_acq_mean_enabled": bool(focus3_acq_policy_info.get("mean_enabled", False)) if focus3_acq_policy_info else False,
            "focus3_acq_mean_min_data_ratio": float(focus3_acq_policy_info.get("mean_min_data_ratio", float("nan"))) if focus3_acq_policy_info else float("nan"),
            "focus3_acq_mean_max_volume_ratio": float(focus3_acq_policy_info.get("mean_max_volume_ratio", float("nan"))) if focus3_acq_policy_info else float("nan"),
            "focus3_acq_best_local_active": bool(focus3_acq_policy_info.get("best_local_active", False)) if focus3_acq_policy_info else False,
            "focus3_recover_enabled": bool(focus3_recover_info.get("enabled", False)) if focus3_recover_info else False,
            "focus3_recover_active": bool(focus3_recover_info.get("active", False)) if focus3_recover_info else False,
            "focus3_recover_reason": str(focus3_recover_info.get("reason", "")) if focus3_recover_info else "",
            "focus3_recover_level": str(focus3_recover_info.get("level", "")) if focus3_recover_info else "",
            "focus3_recover_no_improve_count": int(focus3_recover_info.get("no_improve_count", 0)) if focus3_recover_info else 0,
            "focus3_recover_recent_improvement": float(focus3_recover_info.get("recent_improvement", float("nan"))) if focus3_recover_info else float("nan"),
            "focus3_recover_topk_bonus": float(focus3_recover_info.get("topk_bonus", 0.0)) if focus3_recover_info else 0.0,
            "focus3_recover_boundary_bonus": float(focus3_recover_info.get("boundary_bonus", 0.0)) if focus3_recover_info else 0.0,
            "focus3_recover_random_bonus": float(focus3_recover_info.get("random_bonus", 0.0)) if focus3_recover_info else 0.0,
            "focus3_kappa_before_recover": float(focus3_recover_info.get("kappa_before", float("nan"))) if focus3_recover_info else float("nan"),
            "focus3_kappa_after_recover": float(focus3_recover_info.get("kappa_after", float("nan"))) if focus3_recover_info else float("nan"),
            "focus3_source_cap_enabled": bool(focus3_source_cap_info.get("enabled", False)) if focus3_source_cap_info else False,
            "focus3_source_cap_applied": bool(focus3_source_cap_info.get("applied", False)) if focus3_source_cap_info else False,
            "focus3_source_cap_reason": str(focus3_source_cap_info.get("reason", "")) if focus3_source_cap_info else "",
            "focus3_source_cap_boundary_max": float(focus3_source_cap_info.get("boundary_max", float("nan"))) if focus3_source_cap_info else float("nan"),
            "focus3_source_cap_topk_min": float(focus3_source_cap_info.get("topk_min", float("nan"))) if focus3_source_cap_info else float("nan"),
            "focus3_source_cap_random_min": float(focus3_source_cap_info.get("random_min", float("nan"))) if focus3_source_cap_info else float("nan"),
            "focus3_best_local_enabled": bool(focus3_best_local_info.get("enabled", False)) if focus3_best_local_info else False,
            "focus3_best_local_applied": bool(focus3_best_local_info.get("applied", False)) if focus3_best_local_info else False,
            "focus3_best_local_reason": str(focus3_best_local_info.get("reason", "")) if focus3_best_local_info else "",
            "focus3_best_local_prob": float(focus3_best_local_info.get("best_local_prob", 0.0)) if focus3_best_local_info else 0.0,
            "focus3_best_local_data_ratio": float(focus3_best_local_info.get("data_ratio", float("nan"))) if focus3_best_local_info else float("nan"),
            "focus3_best_local_min_focus3_evals": int(focus3_best_local_info.get("min_focus3_evals", 0)) if focus3_best_local_info else 0,
            "focus3_best_local_min_data_ratio": float(focus3_best_local_info.get("min_data_ratio", float("nan"))) if focus3_best_local_info else float("nan"),
            "focus3_recover_best_local_applied": bool(focus3_recover_best_local_info.get("applied", False)) if focus3_recover_best_local_info else False,
            "focus3_recover_best_local_reason": str(focus3_recover_best_local_info.get("reason", "")) if focus3_recover_best_local_info else "",
            "focus3_recover_best_local_bonus": float(focus3_recover_best_local_info.get("bonus", 0.0)) if focus3_recover_best_local_info else 0.0,
            "focus3_best_local_sigma_enabled": bool(focus3_best_local_sigma_info.get("enabled", False)) if focus3_best_local_sigma_info else False,
            "focus3_best_local_sigma_base": float(focus3_best_local_sigma_info.get("base", float("nan"))) if focus3_best_local_sigma_info else float("nan"),
            "focus3_best_local_sigma_effective": float(focus3_best_local_sigma_info.get("effective", float("nan"))) if focus3_best_local_sigma_info else float("nan"),
            "focus3_best_local_sigma_reason": str(focus3_best_local_sigma_info.get("reason", "")) if focus3_best_local_sigma_info else "",
            "focus3_fallback_bounds_source_enabled": bool(focus3_fallback_bounds_source_info.get("enabled", False)) if focus3_fallback_bounds_source_info else False,
            "focus3_fallback_bounds_source_applied": bool(focus3_fallback_bounds_source_info.get("applied", False)) if focus3_fallback_bounds_source_info else False,
            "focus3_fallback_bounds_source_reason": str(focus3_fallback_bounds_source_info.get("reason", "")) if focus3_fallback_bounds_source_info else "",
            "focus3_fallback_bounds_source_flag": bool(focus3_fallback_bounds_source_info.get("fallback_bounds", False)) if focus3_fallback_bounds_source_info else False,
            "focus3_refine_filter_enabled": bool(focus3_refine_filter_info.get("enabled", False)) if focus3_refine_filter_info else False,
            "focus3_refine_filter_skipped": bool(focus3_refine_filter_info.get("skipped", False)) if focus3_refine_filter_info else False,
            "focus3_refine_filter_reason": str(focus3_refine_filter_info.get("reason", "")) if focus3_refine_filter_info else "",
            "focus3_refine_filter_best_plan_source": str(focus3_refine_filter_info.get("best_plan_source", "")) if focus3_refine_filter_info else "",
            "focus3_refine_filter_allowed_sources": str(focus3_refine_filter_info.get("allowed_sources", "")) if focus3_refine_filter_info else "",
            "focus3_refine_cooldown_enabled": bool(focus3_refine_cooldown_info.get("enabled", False)) if focus3_refine_cooldown_info else False,
            "focus3_refine_cooldown_active": bool(focus3_refine_cooldown_info.get("active", False)) if focus3_refine_cooldown_info else False,
            "focus3_refine_cooldown_reason": str(focus3_refine_cooldown_info.get("reason", "")) if focus3_refine_cooldown_info else "",
            "focus3_refine_cooldown_recent_worse_count": int(focus3_refine_cooldown_info.get("recent_worse_count", 0)) if focus3_refine_cooldown_info else 0,
            "focus3_refine_cooldown_worse_count_threshold": int(focus3_refine_cooldown_info.get("worse_count_threshold", 0)) if focus3_refine_cooldown_info else 0,
            "focus3_refine_adaptive_enabled": bool(focus3_refine_adaptive_info.get("enabled", False)) if focus3_refine_adaptive_info else False,
            "focus3_refine_adaptive_reason": str(focus3_refine_adaptive_info.get("reason", "")) if focus3_refine_adaptive_info else "",
            "focus3_refine_adaptive_base_every": int(focus3_refine_adaptive_info.get("base_every", 0)) if focus3_refine_adaptive_info else 0,
            "focus3_refine_adaptive_effective_every": int(focus3_refine_adaptive_info.get("effective_every", 0)) if focus3_refine_adaptive_info else 0,
            "focus3_refine_adaptive_discrete_count": int(focus3_refine_adaptive_info.get("discrete_count", 0)) if focus3_refine_adaptive_info else 0,
            "focus3_refine_adaptive_refine_count": int(focus3_refine_adaptive_info.get("refine_count", 0)) if focus3_refine_adaptive_info else 0,
            "focus3_refine_adaptive_discrete_improve_rate": float(focus3_refine_adaptive_info.get("discrete_improve_rate", float("nan"))) if focus3_refine_adaptive_info else float("nan"),
            "focus3_refine_adaptive_refine_improve_rate": float(focus3_refine_adaptive_info.get("refine_improve_rate", float("nan"))) if focus3_refine_adaptive_info else float("nan"),
            "focus3_refine_adaptive_better_every": int(focus3_refine_adaptive_info.get("better_every", 0)) if focus3_refine_adaptive_info else 0,
            "focus3_refine_adaptive_worse_every": int(focus3_refine_adaptive_info.get("worse_every", 0)) if focus3_refine_adaptive_info else 0,
            "focus3_dedup_enabled": bool(focus3_dedup_info.get("enabled", False)),
            "focus3_dedup_applied": bool(focus3_dedup_info.get("applied", False)),
            "focus3_dedup_fallback": str(focus3_dedup_info.get("fallback", "")),
            "focus3_dedup_min_rms_distance": float(focus3_dedup_info.get("min_rms_distance", float("nan"))),
            "focus3_archive_distance_before_dedup": float(focus3_dedup_info.get("archive_distance_before", float("nan"))),
            "focus3_archive_distance_after_dedup": float(focus3_dedup_info.get("archive_distance_after", float("nan"))),
            "source_prob_topk": float(source_prob_topk),
            "source_prob_best_local": float(source_prob_best_local),
            "source_prob_boundary": float(source_prob_boundary),
            "source_prob_random": float(source_prob_random),
            "gp_fallback_used": bool(gp_fallback_used),
            **_gp_x_scaling_debug(system=system, lb=lb, ub=ub),
            "gp_train_count": int(X_train.shape[0]),
            "gp_train_cap": int(gp_train_cap),
            "archive_count": int(X_archive.shape[0]),
            "initial_archive_count": int(initial_archive_count),
            "post_penalty_active": bool(post_penalty_active),
            "post_penalty_lambda": float(post_penalty_lambda if post_penalty_active else 0.0),
            "post_score_mode": str(post_score_mode),
            "p_feasible": float(p_feasible),
            "pre_retry_used": int(pre_retry_used),
            "pre_generated_count": int(pre_generated_count),
            "pre_fallback_used": bool(pre_fallback_used),
            **pre_constraint_debug_fields,
            "timing_iter_sec": float(timing_iter.get("iteration", 0.0)),
            "timing_gp_fit_sec": float(timing_iter.get("gp_fit", 0.0)),
            "timing_starts_build_sec": float(timing_iter.get("starts_build", 0.0)),
            "timing_focus0_batch_sec": float(timing_iter.get("focus0_batch", 0.0)),
            "timing_focus1_batch_sec": float(timing_iter.get("focus1_batch", 0.0)),
            "timing_focus2_region_init_sec": float(timing_iter.get("focus2_region_init", 0.0)),
            "timing_focus2_batch_sec": float(timing_iter.get("focus2_batch", 0.0)),
            "timing_focus2_update_sec": float(timing_iter.get("focus2_update", 0.0)),
            "timing_focus3_plan_sec": float(timing_iter.get("focus3_plan", 0.0)),
            "timing_focus3_refine_sec": float(timing_iter.get("focus3_refine", 0.0)),
            "timing_focus3_dedup_sec": float(timing_iter.get("focus3_dedup", 0.0)),
            "timing_gp_predict_next_sec": float(timing_iter.get("gp_predict_next", 0.0)),
            "timing_objective_eval_sec": float(timing_iter.get("objective_eval", 0.0)),
            "timing_archive_update_sec": float(timing_iter.get("archive_update", 0.0)),
            "timing_elapsed_total_sec": float(time.perf_counter() - optimizer_t0) if timing_enabled else 0.0,
        })
        history_rows.append(row)
        if progress_log_every > 0:
            should_log_progress = (
                int(i + 1) == 1
                or int(i + 1) == int(n_samples)
                or str(segment) in {"focus0", "focus1", "focus2"}
                or (
                    str(segment) == "focus3"
                    and (
                        int(focus3_eval_count) <= 3
                        or int(focus3_eval_count) % progress_log_every == 0
                    )
                )
            )
            if should_log_progress:
                print(
                    "[Optimizer][Progress] "
                    f"iter={int(i + 1)}/{int(n_samples)} "
                    f"segment={segment} "
                    f"archive={int(X_archive.shape[0])} "
                    f"evals={{f0:{focus0_eval_count}/{focus0_budget}, "
                    f"f1:{focus1_eval_count}/{focus1_budget}, "
                    f"f2:{focus2_eval_count}/{focus2_budget}, "
                    f"f3:{focus3_eval_count}/{focus3_available_budget}}} "
                    f"y={float(y_next):.6g} best={float(best_y_raw):.6g}"
                    f"{_timing_brief()}",
                    flush=True,
                )
        if bool(goal_monitor.should_stop):
            print(
                "[Optimizer][GoalStop] "
                f"iter={int(i + 1)}/{int(n_samples)} "
                f"goal={float(goal_monitor.goal):.6g} "
                f"best={float(best_y_raw):.6g} "
                f"first_hit_iter={int(goal_monitor.first_hit_iter or 0)} "
                f"patience={int(goal_monitor.patience)}",
                flush=True,
            )
            break

    if timing_enabled:
        elapsed_total = float(time.perf_counter() - optimizer_t0)
        print(
            "[Optimizer][Timing] "
            f"elapsed={elapsed_total:.3f}s "
            f"gp_fit={timing_totals.get('gp_fit', 0.0):.3f}s "
            f"starts_build={timing_totals.get('starts_build', 0.0):.3f}s "
            f"focus0_batch={timing_totals.get('focus0_batch', 0.0):.3f}s "
            f"focus1_batch={timing_totals.get('focus1_batch', 0.0):.3f}s "
            f"focus2_region_init={timing_totals.get('focus2_region_init', 0.0):.3f}s "
            f"focus2_batch={timing_totals.get('focus2_batch', 0.0):.3f}s "
            f"focus2_update={timing_totals.get('focus2_update', 0.0):.3f}s "
            f"focus3_plan={timing_totals.get('focus3_plan', 0.0):.3f}s "
            f"focus3_refine={timing_totals.get('focus3_refine', 0.0):.3f}s "
            f"focus3_dedup={timing_totals.get('focus3_dedup', 0.0):.3f}s "
            f"objective_eval={timing_totals.get('objective_eval', 0.0):.3f}s "
            f"archive_update={timing_totals.get('archive_update', 0.0):.3f}s",
            flush=True,
        )

    if focus2_active and focus2_region_states and not focus2_final_region_selection:
        chosen_state, focus2_final_region_selection, final_lb, final_ub, final_volume_ratio = _build_focus2_final_bounds(
            region_states=focus2_region_states,
            X_archive=X_archive,
            y_archive=y_archive,
            objective_sense=objective_sense,
            system=system,
            full_lb=full_lb,
            full_ub=full_ub,
        )
        if chosen_state is not None and final_lb is not None and final_ub is not None:
            focus_lb = final_lb.copy()
            focus_ub = final_ub.copy()
            selected_region_ids = list(focus2_final_region_selection.get("selected_region_ids", []))
            focus2_final_info = {
                "method": "turbo_lite_final_region_union" if bool(getattr(system, "focus2_final_union_enabled", True)) else "turbo_lite_final_region_selector",
                "selected_region_id": str(chosen_state.get("region_id", "")),
                "selected_region_ids": selected_region_ids,
                "selection": dict(focus2_final_region_selection),
                "active_region_count": int(sum(1 for s in focus2_region_states if str(s.get("status", "active")) == "active")),
                "dropped_region_count": int(sum(1 for s in focus2_region_states if str(s.get("status", "active")) == "dropped")),
                "merged_region_count": int(sum(1 for s in focus2_region_states if str(s.get("status", "active")) == "merged")),
                "volume_ratio": float(final_volume_ratio),
                "bounds_policy": dict(focus2_final_region_selection.get("final_bounds_policy", {})),
            }

    history_df = pd.DataFrame(history_rows)
    archive_rows = []
    for x, y in zip(X_archive, y_archive):
        p = 1.0
        if post_penalty_active and post_prob_fn is not None:
            p = float(np.clip(post_prob_fn(x), 0.0, 1.0))
        y_eff = _apply_post_penalty_to_objective(
            y_raw=float(y),
            p_feasible=p,
            objective_sense=objective_sense,
            penalty_lambda=post_penalty_lambda if post_penalty_active else 0.0,
            score_mode=post_score_mode,
            p_feasible_min=post_p_feasible_min,
            hard_penalty=post_p_feasible_hard_penalty,
        )
        item = {
            "objective": float(y_eff),
            "objective_raw": float(y),
            "objective_effective": float(y_eff),
            "p_feasible": float(p),
        }
        for fname, value in zip(selected_features, x):
            item[fname] = float(value)
        archive_rows.append(item)
    archive_df = pd.DataFrame(archive_rows)

    best_point = {f: float(v) for f, v in zip(selected_features, best_x_eff)}
    best_point_raw = {f: float(v) for f, v in zip(selected_features, best_x_raw)}
    if not math.isfinite(float(best_y_eff)) or not math.isfinite(float(best_y_raw)):
        raise RuntimeError("Optimizer finished with non-finite best objective.")

    if focus2_regions_payload:
        focus2_regions_summary = dict(focus2_regions_payload)
    elif focus_pipeline_plan.has("focus2"):
        focus2_region_model = gp_model
        if X_train.shape[0] >= 2 and y_train.shape[0] >= 2:
            try:
                focus2_region_model, _focus2_region_gp_fallback = _fit_optimizer_gp_with_fallback(
                    X=X_train,
                    y=y_train,
                    lb=lb,
                    ub=ub,
                    system=system,
                    include_white=False,
                    random_state=int(seed + int(n_samples) + 7919),
                )
            except Exception:
                focus2_region_model = gp_model
        focus2_regions_summary = _build_focus2_regions_from_archive(
            rng=rng,
            model=focus2_region_model,
            X_archive=X_archive,
            y_archive=y_archive,
            objective_sense=objective_sense,
            lb=full_lb,
            ub=full_ub,
            selected_features=selected_features,
            system=system,
            var_names=var_names,
            constraint_defs=constraint_defs,
            enforce_pre_constraints=bool(system.enforce_pre_constraints),
        )
    else:
        focus2_regions_summary = {
            "enabled": False,
            "status": "skipped",
            "reason": "focus2_not_in_pipeline",
            "regions": [],
        }
    if focus2_region_states:
        state_by_id = {str(s.get("region_id")): s for s in focus2_region_states}
        regions_for_output = []
        for region in list(focus2_regions_summary.get("regions", []) or []):
            if not isinstance(region, dict):
                continue
            out_region = dict(region)
            state = state_by_id.get(str(out_region.get("region_id")))
            if isinstance(state, dict):
                out_region["turbo_lite_state"] = {
                    "status": str(state.get("status", "active")),
                    "success_count": int(state.get("success_count", 0)),
                    "no_improve_count": int(state.get("no_improve_count", state.get("failure_count", 0))),
                    "skip_duplicate_count": int(state.get("skip_duplicate_count", 0)),
                    "skip_budget_count": int(state.get("skip_budget_count", 0)),
                    "shared_eval_count": int(state.get("shared_eval_count", 0)),
                    "shared_success_count": int(state.get("shared_success_count", 0)),
                    "merge_count": int(state.get("merge_count", 0)),
                    "merged_into": str(state.get("merged_into", "")),
                    "active_bounds_update_count": int(state.get("active_bounds_update_count", 0)),
                    "active_bounds_volume_ratio": float(state.get("active_bounds_volume_ratio", float("nan"))),
                    "active_bounds_last_reason": str(state.get("active_bounds_last_reason", "")),
                    "active_bounds": {
                        str(name): [float(lo), float(hi)]
                        for name, lo, hi in zip(
                            selected_features,
                            np.asarray(state.get("bounds_lb"), dtype=float),
                            np.asarray(state.get("bounds_ub"), dtype=float),
                        )
                    },
                    "parent_bounds": {
                        str(name): [float(lo), float(hi)]
                        for name, lo, hi in zip(
                            selected_features,
                            np.asarray(state.get("parent_lb", state.get("bounds_lb")), dtype=float),
                            np.asarray(state.get("parent_ub", state.get("bounds_ub")), dtype=float),
                        )
                    },
                    "last_skip_reason": str(state.get("last_skip_reason", "")),
                    "eval_count": int(state.get("eval_count", 0)),
                    "best_objective": float(state.get("best_objective", float("nan"))),
                }
            regions_for_output.append(out_region)
        focus2_regions_summary["regions"] = regions_for_output
        focus2_regions_summary["turbo_lite"] = {
            "enabled": bool(focus2_active),
            "budget": int(focus2_budget),
            "eval_count": int(focus2_eval_count),
            "active_region_count": int(sum(1 for s in focus2_region_states if str(s.get("status", "active")) == "active")),
            "dropped_region_count": int(sum(1 for s in focus2_region_states if str(s.get("status", "active")) == "dropped")),
            "merged_region_count": int(sum(1 for s in focus2_region_states if str(s.get("status", "active")) == "merged")),
            "merge_history": list(focus2_merge_history),
            "active_bounds_history": list(focus2_active_bounds_history),
            "scheduler_history": list(focus2_scheduler_history),
            "duplicate_pair_counts": {
                f"{a}|{b}": int(v)
                for (a, b), v in sorted(focus2_duplicate_pair_counts.items())
            },
            "shared_pair_counts": {
                f"{a}|{b}": int(v)
                for (a, b), v in sorted(focus2_shared_pair_counts.items())
            },
            "final_region_selection": dict(focus2_final_region_selection),
        }

    focus2_summary: dict[str, object] = {
        "active": bool(focus2_active),
        "bounds_source": str(bounds_source or "cae"),
        "budget": int(focus2_budget),
        "counts": {k: int(v) for k, v in focus2_counts.items()},
        "initial": dict(focus2_initial_info),
        "final": dict(focus2_final_info or focus2_initial_info),
        "early_stop": dict(focus2_early_stop_info),
        "region_output": dict(focus2_regions_summary),
    }
    if focus2_active:
        focus2_summary["final_bounds"] = {
            str(name): {"lb": float(lo), "ub": float(hi)}
            for name, lo, hi in zip(selected_features, focus_lb, focus_ub)
        }

    return build_optimizer_algorithm_result(
        history_df=history_df,
        archive_df=archive_df,
        best_point=best_point,
        best_objective=float(best_y_eff),
        best_point_raw=best_point_raw,
        best_objective_raw=float(best_y_raw),
        post_penalty_active=bool(post_penalty_active),
        post_penalty_lambda=float(post_penalty_lambda if post_penalty_active else 0.0),
        post_score_mode=str(post_score_mode),
        feasibility_model_kind=str(feasibility_model_kind),
        feasibility_status=str(feasibility_status),
        gp_train_cap=int(gp_train_cap),
        initial_archive_count=int(initial_archive_count),
        final_train_count=int(X_train.shape[0]),
        algorithm_id="focus_bo",
        engine="focus_bo",
        goal_summary=goal_monitor.summary(),
        algorithm_summary_extra={
            "goal_early_stop_min_focus": int(max(int(getattr(system, "goal_early_stop_min_focus", 2)), 0)),
        },
        focus2_summary=focus2_summary,
        focus_pipeline_summary=focus_pipeline_plan.as_dict(),
    )
