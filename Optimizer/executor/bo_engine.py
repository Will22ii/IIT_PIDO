from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from CAE_tool_interface.executor.configurator import select_cae_by_name
from DOE.executor.anchor_refiner import AcquisitionOptimizer, fit_gp_with_fallback
from DOE.executor.constraint_filter import evaluate_constraints_batch, evaluate_constraints_point
from DOE.executor.eval_sanitizer import sanitize_evaluate_output
from Optimizer.config import OptimizerSystemConfig


@dataclass
class BOEngineResult:
    history_df: pd.DataFrame
    archive_df: pd.DataFrame
    best_point: dict[str, float]
    best_objective: float
    best_point_raw: dict[str, float]
    best_objective_raw: float
    post_penalty_active: bool
    post_penalty_lambda: float
    post_score_mode: str
    feasibility_model_kind: str
    feasibility_status: str
    n_iterations: int


def _is_better(*, y_new: float, y_best: float, objective_sense: str) -> bool:
    if objective_sense == "max":
        return float(y_new) > float(y_best)
    return float(y_new) < float(y_best)


def _point_in_bounds(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> bool:
    return bool(np.all(x >= lb) and np.all(x <= ub))


def _round_key(x: np.ndarray, decimals: int) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(x, dtype=float).reshape(-1), decimals=decimals).tolist())


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
) -> np.ndarray:
    n_dim = int(lb.shape[0])
    n = int(max(pool_size, 1))
    span = np.maximum(ub - lb, 1e-12)
    starts: list[np.ndarray] = []

    if source == "topk" and X_train.shape[0] > 0:
        if objective_sense == "max":
            order = np.argsort(-y_train)
        else:
            order = np.argsort(y_train)
        n_top = int(max(1, round(float(X_train.shape[0]) * float(np.clip(topk_fraction, 0.01, 1.0)))))
        top_idx = order[:n_top]
        sigma = float(max(topk_sigma, 1e-4))
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
) -> np.ndarray:
    if not enforce_pre_constraints or not constraint_defs:
        return x

    x_cur = np.clip(np.asarray(x, dtype=float).reshape(-1), lb, ub)
    for _ in range(24):
        _payload, feasible, _margin = evaluate_constraints_point(
            x=x_cur,
            var_names=var_names,
            constraint_defs=constraint_defs,
            scope="pre",
            fail_fast_output_missing=False,
        )
        if bool(feasible):
            return x_cur
        x_cur = rng.uniform(lb, ub, size=(lb.shape[0],))
    return x_cur


def _evaluate_pre_constraint(
    *,
    x: np.ndarray,
    var_names: list[str],
    constraint_defs: list[dict],
    enforce_pre_constraints: bool,
) -> tuple[bool, float]:
    if not enforce_pre_constraints or not constraint_defs:
        return True, float("inf")
    _payload, feasible, margin = evaluate_constraints_point(
        x=np.asarray(x, dtype=float).reshape(-1),
        var_names=var_names,
        constraint_defs=constraint_defs,
        scope="pre",
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

        feasible_rows = starts[mask] if starts.shape[0] > 0 else np.empty((0, lb.shape[0]), dtype=float)
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


def _sample_local(
    *,
    rng: np.random.Generator,
    best_x: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    radius_ratio: float,
) -> np.ndarray:
    span = np.maximum(ub - lb, 1e-12)
    direction = rng.normal(0.0, 1.0, size=best_x.shape[0])
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        direction = np.ones_like(direction) / np.sqrt(float(direction.shape[0]))
    else:
        direction = direction / norm
    x = np.asarray(best_x, dtype=float) + float(radius_ratio) * span * direction
    return np.clip(x, lb, ub)


def _resolve_no_doe_mode(
    *,
    n_samples: int,
    threshold: int,
) -> str:
    return "three_phase" if int(n_samples) >= int(threshold) else "two_stage"


def _resolve_no_doe_segment(
    *,
    i: int,
    n_samples: int,
    mode: str,
    system: OptimizerSystemConfig,
) -> str:
    n = max(int(n_samples), 1)
    if mode == "three_phase":
        n1 = int(max(1, round(float(system.no_doe_phase1_ratio) * n)))
        n2 = int(max(1, round(float(system.no_doe_phase2_ratio) * n)))
        if i < n1:
            return "phase1"
        if i < min(n1 + n2, n):
            return "phase2"
        return "phase3"
    n1 = int(max(1, round(float(system.no_doe_stage1_ratio) * n)))
    if i < n1:
        return "stage1"
    return "stage2"


def _segment_to_phase_label(*, segment: str, mode_no_doe: bool) -> str:
    seg = str(segment or "").strip().lower()
    if not mode_no_doe:
        return "phase3"
    if seg in {"stage1", "phase1"}:
        return "phase1"
    if seg in {"stage2", "phase2"}:
        return "phase2"
    return "phase3"


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
    if objective_sense == "max":
        return int(np.argmax(y))
    return int(np.argmin(y))


def _build_cae_objective_evaluator(
    *,
    problem_name: str,
    variables: list[dict],
    selected_features: list[str],
) -> Callable[[np.ndarray], float]:
    _, evaluate_func = select_cae_by_name(str(problem_name).strip())

    var_names: list[str] = []
    var_baseline: list[float] = []
    for v in variables:
        if not isinstance(v, dict):
            continue
        name = str(v.get("name", "")).strip()
        if not name:
            continue
        if "baseline" in v:
            base = float(v["baseline"])
        elif "lb" in v and "ub" in v:
            base = 0.5 * (float(v["lb"]) + float(v["ub"]))
        else:
            base = 0.0
        if not np.isfinite(base):
            base = 0.0
        var_names.append(name)
        var_baseline.append(float(base))

    if len(var_names) == 0:
        raise RuntimeError("CAE variable list is empty for objective evaluation.")

    name_to_idx = {name: idx for idx, name in enumerate(var_names)}
    missing_selected = [f for f in selected_features if f not in name_to_idx]
    if missing_selected:
        raise RuntimeError(
            "Selected features are not present in CAE variable list: "
            + ", ".join(missing_selected)
        )
    selected_idx = np.asarray([name_to_idx[f] for f in selected_features], dtype=int)
    x_base = np.asarray(var_baseline, dtype=float)

    def _evaluate_selected(x_selected: np.ndarray) -> float:
        x_sel = np.asarray(x_selected, dtype=float).reshape(-1)
        if x_sel.shape[0] != len(selected_features):
            raise RuntimeError(
                f"CAE objective evaluation dimension mismatch: got {x_sel.shape[0]}, "
                f"expected {len(selected_features)}."
            )

        x_full = x_base.copy()
        x_full[selected_idx] = x_sel
        out = evaluate_func(x_full)
        success, objective, _outputs, invalid_reason, raw_repr = sanitize_evaluate_output(out)
        if not success:
            reason = invalid_reason or "success_false"
            raise RuntimeError(
                "CAE evaluation failed during optimizer iteration: "
                f"reason={reason}, raw={raw_repr}"
            )
        return float(objective)

    return _evaluate_selected


def run_bo_engine(
    *,
    problem_name: str,
    variables: list[dict],
    doe_df: pd.DataFrame | None,
    selected_features: list[str],
    selected_bounds: dict[str, tuple[float, float]],
    objective_col: str,
    objective_sense: str,
    n_samples: int,
    system: OptimizerSystemConfig,
    seed: int,
    constraint_defs: list[dict],
    post_feasibility_payload: dict | None = None,
) -> BOEngineResult:
    if int(n_samples) < 0:
        raise ValueError("Optimizer user.n_samples must be >= 0.")
    if len(selected_features) == 0:
        raise RuntimeError("selected_features is empty.")

    rng = np.random.default_rng(int(seed))
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

    # DOE objective 기반 warm-start 또는 no-DOE bootstrap
    if has_doe_objective:
        if objective_sense == "max":
            order = np.argsort(-y_archive)
        else:
            order = np.argsort(y_archive)
        n_topk = max(1, min(int(system.init_from_doe_topk), order.shape[0]))
        top_idx = order[:n_topk]
        X_train = X_archive[top_idx].copy()
        y_train = y_archive[top_idx].copy()
        init_cap = int(max(int(getattr(system, "init_max_points", X_train.shape[0])), 1))
        X_train, y_train = _select_warmstart_top_diversity(
            X=X_train,
            y=y_train,
            objective_sense=objective_sense,
            max_points=init_cap,
            topk_fraction=float(getattr(system, "source_topk_fraction", 0.2)),
        )
        mode_no_doe = False
        no_doe_mode_name = "none"
        init_source = "doe_objective"
    else:
        # DOE objective가 없으면(완전 무입력 포함) 주어진 영역에서 자체 시작
        n_boot = max(int(system.no_doe_bootstrap_size), 2)
        n_boot = min(n_boot, max(int(system.no_doe_bootstrap_size), 2) + max(int(n_samples), 0))
        X_boot = X_seed_from_doe.copy()
        if X_boot.shape[0] < n_boot:
            n_more = n_boot - X_boot.shape[0]
            X_more = rng.uniform(lb, ub, size=(n_more, len(selected_features)))
            X_boot = np.vstack([X_boot, X_more]) if X_boot.size else X_more
        X_boot = np.unique(np.asarray(X_boot, dtype=float), axis=0)
        if X_boot.shape[0] < 2:
            X_boot = np.vstack([X_boot, rng.uniform(lb, ub, size=(2 - X_boot.shape[0], len(selected_features)))])

        y_boot = np.array([evaluate_objective(x) for x in X_boot], dtype=float)
        X_train = X_boot.copy()
        y_train = y_boot.copy()
        X_archive = X_boot.copy()
        y_archive = y_boot.copy()
        mode_no_doe = True
        no_doe_mode_name = _resolve_no_doe_mode(
            n_samples=int(n_samples),
            threshold=int(system.no_doe_mode_threshold),
        )
        init_source = "no_doe_bootstrap"

    best_i_raw = _best_index(y_train, objective_sense)
    best_x_raw = X_train[best_i_raw].copy()
    best_y_raw = float(y_train[best_i_raw])

    if post_penalty_active:
        p_train = np.array([float(np.clip(post_prob_fn(x), 0.0, 1.0)) for x in X_train], dtype=float)
        y_train_eff = np.array(
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
                for y, p in zip(y_train, p_train)
            ],
            dtype=float,
        )
    else:
        y_train_eff = y_train.copy()

    best_i_eff = _best_index(y_train_eff, objective_sense)
    best_x_eff = X_train[best_i_eff].copy()
    best_y_eff = float(y_train_eff[best_i_eff])

    seen = {_round_key(x, decimals=int(system.dedup_decimals)) for x in X_archive}
    acq = AcquisitionOptimizer()
    history_rows: list[dict] = []
    gp_model = None
    gp_fallback_used = False
    var_names = list(selected_features)
    best_raw_history: list[float] = [float(best_y_raw)]

    for i in range(int(n_samples)):
        if i == 0 or (int(system.gp_refit_every) > 0 and i % int(system.gp_refit_every) == 0):
            gp_model, gp_fallback_used = fit_gp_with_fallback(
                X=X_train,
                y=y_train,
                include_white=False,
                random_state=int(seed + i),
            )

        acq_type, kappa = _resolve_acq_policy(
            system=system,
            iteration=i,
            n_iterations=max(int(n_samples), 1),
        )
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

        segment = "default"
        source_mode = "default"
        x_next = None
        pre_feasible_next = True
        pre_margin_next = float("inf")
        pre_violation_next = 0.0
        pre_retry_used = 0
        pre_generated_count = 0
        pre_fallback_used = False
        if mode_no_doe:
            segment = _resolve_no_doe_segment(
                i=i,
                n_samples=int(n_samples),
                mode=no_doe_mode_name,
                system=system,
            )
            if segment in {"stage1", "phase1"}:
                # 초기 대역 탐색
                x_next = rng.uniform(lb, ub, size=(lb.shape[0],))
            elif segment in {"stage2", "phase2"}:
                # 중간 단계: acq + local 혼합
                if gp_model is not None and rng.uniform(0.0, 1.0) < 0.45:
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
                    )
                if x_next is None:
                    x_next = _sample_local(
                        rng=rng,
                        best_x=best_x_eff,
                        lb=lb,
                        ub=ub,
                        radius_ratio=0.10,
                    )
            else:
                # phase3: exploitation 강화
                if gp_model is not None and rng.uniform(0.0, 1.0) < 0.80:
                    x_next = acq.optimize(
                        model=gp_model,
                        y_best=best_y_raw,
                        lb=lb,
                        ub=ub,
                        starts=starts,
                        objective_sense=objective_sense,
                        acq_type=acq_type,
                        kappa=float(min(kappa, 0.7)),
                        xi=float(system.ei_xi),
                        post_feasible_prob_fn=post_prob_fn if post_penalty_active else None,
                        post_penalty_lambda=post_penalty_lambda if post_penalty_active else 0.0,
                    )
                if x_next is None:
                    x_next = _sample_local(
                        rng=rng,
                        best_x=best_x_eff,
                        lb=lb,
                        ub=ub,
                        radius_ratio=0.04,
                    )
        else:
            if bool(getattr(system, "source_mixture_enabled", True)) and has_doe_objective:
                p_topk, p_bnd, p_rand = _normalize_source_probs(
                    topk=float(getattr(system, "source_topk_prob", 0.60)),
                    boundary=float(getattr(system, "source_boundary_prob", 0.25)),
                    random_p=float(getattr(system, "source_random_prob", 0.15)),
                )
                win = int(max(int(getattr(system, "source_stagnation_window", 8)), 2))
                tol = float(max(float(getattr(system, "source_stagnation_tol", 1e-8)), 0.0))
                if len(best_raw_history) >= win:
                    ref_idx = max(len(best_raw_history) - win, 0)
                    if objective_sense == "max":
                        improved = float(best_raw_history[-1] - best_raw_history[ref_idx])
                    else:
                        improved = float(best_raw_history[ref_idx] - best_raw_history[-1])
                    if improved <= tol:
                        p_bnd += float(max(float(getattr(system, "source_stagnation_boundary_bonus", 0.10)), 0.0))
                        p_rand += float(max(float(getattr(system, "source_stagnation_random_bonus", 0.05)), 0.0))
                        p_topk, p_bnd, p_rand = _normalize_source_probs(
                            topk=p_topk,
                            boundary=p_bnd,
                            random_p=p_rand,
                        )

                source = str(rng.choice(np.asarray(["topk", "boundary", "random"]), p=[p_topk, p_bnd, p_rand]))
                source_mode = str(source)
                pre_active = bool(system.enforce_pre_constraints) and bool(constraint_defs)
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
                acq_type_source = "EI" if source == "topk" else "LCB"
                segment = "phase3"
                if gp_model is not None and pool_starts.shape[0] > 0:
                    def _pre_fn(x_arr: np.ndarray) -> bool:
                        ok, _m = _evaluate_pre_constraint(
                            x=x_arr,
                            var_names=var_names,
                            constraint_defs=constraint_defs,
                            enforce_pre_constraints=pre_active,
                        )
                        return bool(ok)

                    x_next = acq.optimize(
                        model=gp_model,
                        y_best=best_y_raw,
                        lb=lb,
                        ub=ub,
                        starts=pool_starts,
                        objective_sense=objective_sense,
                        acq_type=acq_type_source,
                        kappa=float(kappa),
                        xi=float(system.ei_xi),
                        pre_feasible_fn=_pre_fn if pre_active else None,
                        post_feasible_prob_fn=post_prob_fn if post_penalty_active else None,
                        post_penalty_lambda=post_penalty_lambda if post_penalty_active else 0.0,
                    )
                if x_next is None and pool_starts.shape[0] > 0:
                    x_next = pool_starts[int(rng.integers(0, pool_starts.shape[0]))]
                if x_next is None and pre_fallback_x is not None:
                    x_next = pre_fallback_x.copy()
                    pre_fallback_used = True
                    pre_feasible_next = False
                    pre_margin_next = float(pre_fallback_margin)
                    pre_violation_next = float(max(0.0, -pre_margin_next))
                acq_type = acq_type_source
            elif gp_model is not None:
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
                )
            if x_next is None:
                x_next = rng.uniform(lb, ub, size=(lb.shape[0],))

        x_next = np.clip(np.asarray(x_next, dtype=float).reshape(-1), lb, ub)
        if not pre_fallback_used:
            x_next = _sample_feasible_candidate(
                x=x_next,
                rng=rng,
                lb=lb,
                ub=ub,
                var_names=var_names,
                constraint_defs=constraint_defs,
                enforce_pre_constraints=bool(system.enforce_pre_constraints),
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
        seen.add(key)

        if gp_model is not None:
            mu_next, std_next = gp_model.predict(x_next.reshape(1, -1), return_std=True)
            pred_mean = float(mu_next.reshape(-1)[0])
            pred_std = float(std_next.reshape(-1)[0])
        else:
            pred_mean = float(np.mean(y_train))
            pred_std = float(np.std(y_train)) if y_train.size > 1 else 0.0

        y_next = float(evaluate_objective(x_next))
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
                            kappa=float(kappa),
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

        X_train = np.vstack([X_train, x_next.reshape(1, -1)])
        y_train = np.append(y_train, y_next)
        X_archive = np.vstack([X_archive, x_next.reshape(1, -1)])
        y_archive = np.append(y_archive, y_next)

        if _is_better(y_new=y_next, y_best=best_y_raw, objective_sense=objective_sense):
            best_y_raw = float(y_next)
            best_x_raw = x_next.copy()
        if _is_better(y_new=y_next_effective, y_best=best_y_eff, objective_sense=objective_sense):
            best_y_eff = float(y_next_effective)
            best_x_eff = x_next.copy()
        best_raw_history.append(float(best_y_raw))

        row = {
            "iter": int(i + 1),
            "acq_type": str(acq_type),
            "kappa": float(kappa),
            "acq_base": float(acq_base),
            "acq_effective": float(acq_effective),
            "pred_mean": float(pred_mean),
            "pred_std": float(pred_std),
            "objective": float(y_next_effective),
            "objective_raw": float(y_next),
            "objective_effective": float(y_next_effective),
            "objective_source": str(objective_source),
            "init_source": str(init_source),
            "no_doe_mode": str(no_doe_mode_name),
            "segment": str(segment),
            "phase": _segment_to_phase_label(segment=segment, mode_no_doe=mode_no_doe),
            "source_mode": str(source_mode),
            "gp_fallback_used": bool(gp_fallback_used),
            "surrogate_only_mode": bool(system.surrogate_only_mode),
            "post_penalty_active": bool(post_penalty_active),
            "post_penalty_lambda": float(post_penalty_lambda if post_penalty_active else 0.0),
            "post_score_mode": str(post_score_mode),
            "p_feasible": float(p_feasible),
            "pre_feasible": bool(pre_feasible_next),
            "pre_margin": float(pre_margin_next),
            "pre_violation": float(pre_violation_next),
            "pre_retry_used": int(pre_retry_used),
            "pre_generated_count": int(pre_generated_count),
            "pre_fallback_used": bool(pre_fallback_used),
        }
        for fname, value in zip(selected_features, x_next):
            row[fname] = float(value)
        history_rows.append(row)

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

    return BOEngineResult(
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
        n_iterations=int(n_samples),
    )
