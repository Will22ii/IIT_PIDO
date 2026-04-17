from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from DOE.executor.anchor_refiner import AcquisitionOptimizer, fit_gp_with_fallback
from DOE.executor.constraint_filter import evaluate_constraints_point
from Optimizer.config import OptimizerSystemConfig


@dataclass
class BOEngineResult:
    history_df: pd.DataFrame
    archive_df: pd.DataFrame
    best_point: dict[str, float]
    best_objective: float
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


def _proxy_objective(
    *,
    x: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    objective_sense: str,
    mode: str,
    rng: np.random.Generator,
) -> float:
    mode_norm = str(mode or "center_distance").strip().lower()
    if mode_norm == "random":
        val = float(rng.uniform(0.0, 1.0))
        return val if objective_sense == "min" else -val

    # center_distance (default)
    center = 0.5 * (lb + ub)
    span = np.maximum(ub - lb, 1e-12)
    z = (np.asarray(x, dtype=float) - center) / span
    dist2 = float(np.sum(z * z))
    if objective_sense == "max":
        return -dist2
    return dist2


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


def run_bo_engine(
    *,
    doe_df: pd.DataFrame | None,
    selected_features: list[str],
    selected_bounds: dict[str, tuple[float, float]],
    objective_col: str,
    objective_sense: str,
    n_samples: int,
    system: OptimizerSystemConfig,
    seed: int,
    constraint_defs: list[dict],
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

        y_boot = np.array(
            [
                _proxy_objective(
                    x=x,
                    lb=lb,
                    ub=ub,
                    objective_sense=objective_sense,
                    mode=str(system.no_doe_objective_proxy),
                    rng=rng,
                )
                for x in X_boot
            ],
            dtype=float,
        )
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

    best_i = int(np.argmax(y_train) if objective_sense == "max" else np.argmin(y_train))
    best_x = X_train[best_i].copy()
    best_y = float(y_train[best_i])

    seen = {_round_key(x, decimals=int(system.dedup_decimals)) for x in X_archive}
    acq = AcquisitionOptimizer()
    history_rows: list[dict] = []
    gp_model = None
    gp_fallback_used = False
    var_names = list(selected_features)

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
        x_next = None
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
                        y_best=best_y,
                        lb=lb,
                        ub=ub,
                        starts=starts,
                        objective_sense=objective_sense,
                        acq_type=acq_type,
                        kappa=float(kappa),
                        xi=float(system.ei_xi),
                    )
                if x_next is None:
                    x_next = _sample_local(
                        rng=rng,
                        best_x=best_x,
                        lb=lb,
                        ub=ub,
                        radius_ratio=0.10,
                    )
            else:
                # phase3: exploitation 강화
                if gp_model is not None and rng.uniform(0.0, 1.0) < 0.80:
                    x_next = acq.optimize(
                        model=gp_model,
                        y_best=best_y,
                        lb=lb,
                        ub=ub,
                        starts=starts,
                        objective_sense=objective_sense,
                        acq_type=acq_type,
                        kappa=float(min(kappa, 0.7)),
                        xi=float(system.ei_xi),
                    )
                if x_next is None:
                    x_next = _sample_local(
                        rng=rng,
                        best_x=best_x,
                        lb=lb,
                        ub=ub,
                        radius_ratio=0.04,
                    )
        else:
            if gp_model is not None:
                x_next = acq.optimize(
                    model=gp_model,
                    y_best=best_y,
                    lb=lb,
                    ub=ub,
                    starts=starts,
                    objective_sense=objective_sense,
                    acq_type=acq_type,
                    kappa=float(kappa),
                    xi=float(system.ei_xi),
                )
            if x_next is None:
                x_next = rng.uniform(lb, ub, size=(lb.shape[0],))

        x_next = np.clip(np.asarray(x_next, dtype=float).reshape(-1), lb, ub)
        x_next = _sample_feasible_candidate(
            x=x_next,
            rng=rng,
            lb=lb,
            ub=ub,
            var_names=var_names,
            constraint_defs=constraint_defs,
            enforce_pre_constraints=bool(system.enforce_pre_constraints),
        )

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

        if mode_no_doe:
            y_next = _proxy_objective(
                x=x_next,
                lb=lb,
                ub=ub,
                objective_sense=objective_sense,
                mode=str(system.no_doe_objective_proxy),
                rng=rng,
            )
            objective_source = f"proxy_{str(system.no_doe_objective_proxy)}"
        else:
            # benchmark 미연결 단계: surrogate 예측값을 pseudo objective로 사용
            y_next = pred_mean if bool(system.surrogate_only_mode) else pred_mean
            objective_source = "surrogate_pred"

        y_next = float(y_next)
        if not np.isfinite(y_next):
            y_next = float(np.mean(y_train))

        X_train = np.vstack([X_train, x_next.reshape(1, -1)])
        y_train = np.append(y_train, y_next)
        X_archive = np.vstack([X_archive, x_next.reshape(1, -1)])
        y_archive = np.append(y_archive, y_next)

        if _is_better(y_new=y_next, y_best=best_y, objective_sense=objective_sense):
            best_y = float(y_next)
            best_x = x_next.copy()

        row = {
            "iter": int(i + 1),
            "acq_type": str(acq_type),
            "kappa": float(kappa),
            "pred_mean": float(pred_mean),
            "pred_std": float(pred_std),
            "objective": float(y_next),
            "objective_source": str(objective_source),
            "init_source": str(init_source),
            "no_doe_mode": str(no_doe_mode_name),
            "segment": str(segment),
            "gp_fallback_used": bool(gp_fallback_used),
            "surrogate_only_mode": bool(system.surrogate_only_mode),
        }
        for fname, value in zip(selected_features, x_next):
            row[fname] = float(value)
        history_rows.append(row)

    history_df = pd.DataFrame(history_rows)
    archive_rows = []
    for x, y in zip(X_archive, y_archive):
        item = {"objective": float(y)}
        for fname, value in zip(selected_features, x):
            item[fname] = float(value)
        archive_rows.append(item)
    archive_df = pd.DataFrame(archive_rows)

    best_point = {f: float(v) for f, v in zip(selected_features, best_x)}
    if not math.isfinite(float(best_y)):
        raise RuntimeError("Optimizer finished with non-finite best objective.")

    return BOEngineResult(
        history_df=history_df,
        archive_df=archive_df,
        best_point=best_point,
        best_objective=float(best_y),
        n_iterations=int(n_samples),
    )
