import ast
import json
import os
import pickle
import warnings
from math import erf, sqrt
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor

from utils.result_loader import ResultLoader
from utils.result_saver import ResultSaver
from utils.bool_mask import to_bool_mask
from DOE.doe_algorithm.lhs import latin_hypercube_sampling
from DOE.executor.anchor_refiner import (
    AcquisitionOptimizer,
    fit_gp_with_fallback,
    kernel_common_best,
    kernel_stable_conservative,
)
from Explorer.config import ExplorerConfig
from utils.boundary_sampling import sample_boundary_corners, sample_boundary_partial
from utils.cluster_selection import select_top_clusters
from utils.bounds_utils import compute_spans_lbs
from DOE.executor.constraint_filter import (
    evaluate_constraints_batch,
    evaluate_constraints_point,
    validate_constraint_defs,
)
from Explorer.executor.explorer_utils import (
    apply_bounds_margin,
    compute_fi_importance_weights,
    compute_gp_boundary_uncertainty,
    compute_selected_bounds,
    format_span_rows,
    resolve_bounds,
    resolve_selected_features,
)
from Explorer.executor.routing import (
    RouterInput,
    RouterOutput,
    route_v2,
)
from Explorer.visualization.explorer_plots import (
    plot_dual_cluster_pair,
    plot_bounds_pair,
)
from Explorer.visualization.plot_doe_vs_optimum import plot_doe_vs_optimum
from pipeline.run_context import (
    RunContext,
    create_run_context,
    get_task_metadata_path,
    update_run_index,
)


def _load_models(pkl_path: str) -> tuple[list, list[str]]:
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    models = payload.get("models", [])
    feature_cols = payload.get("feature_cols", [])
    return models, feature_cols


def _load_feasibility_model(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid feasibility model payload format.")
    return payload


def _artifact_ref(metadata: dict | None, key: str) -> str | None:
    if not isinstance(metadata, dict):
        return None
    artifacts = metadata.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return None
    if key in artifacts:
        return artifacts[key]
    for layer in ("public", "meta", "debug"):
        layer_map = artifacts.get(layer, {})
        if isinstance(layer_map, dict) and key in layer_map:
            return layer_map[key]
    return None


def _extract_expr_vars(expr: str) -> set[str]:
    tree = ast.parse(expr, mode="eval")
    return {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name)
    }


def _predict_ensemble(models: list, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    preds = []
    for model in models:
        preds.append(np.asarray(model.predict(X), dtype=float).reshape(-1))
    if not preds:
        raise RuntimeError("No models available for prediction.")
    stacked = np.vstack(preds)
    return stacked.mean(axis=0), stacked.std(axis=0)


def _normalize_debug_level(value: str | None) -> str:
    level = str(value or "off").strip().lower()
    if level not in {"off", "full"}:
        raise ValueError("Explorer debug_level must be one of: off, full")
    return level


def _normalize_strategy_id(value: str | None) -> str:
    raw = str(value or "S4_dual").strip()
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw)
    safe = safe.strip("_")
    return safe or "S4_dual"


def _safe_stack(points: list[np.ndarray], n_dim: int) -> np.ndarray:
    valid: list[np.ndarray] = []
    for arr in points:
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float)
        if a.ndim != 2 or a.shape[1] != n_dim or a.shape[0] == 0:
            if a.shape[0] > 0:
                print(f"[Explorer] _safe_stack: skipped array shape {a.shape} (expected n_dim={n_dim})")
            continue
        valid.append(a)
    if not valid:
        return np.empty((0, n_dim), dtype=float)
    return np.vstack(valid)


def _bounds_from_points(
    X: np.ndarray,
    *,
    q_low: float | None = None,
    q_high: float | None = None,
) -> list[tuple[float, float]] | None:
    if X.ndim != 2 or X.shape[0] == 0:
        return None
    X = X[np.isfinite(X).all(axis=1)]
    if X.shape[0] == 0:
        return None
    out: list[tuple[float, float]] = []
    use_q = (
        q_low is not None
        and q_high is not None
        and X.shape[0] >= 8
        and 0.0 <= float(q_low) < float(q_high) <= 1.0
    )
    for j in range(X.shape[1]):
        col = X[:, j]
        if use_q:
            lb = float(np.quantile(col, float(q_low)))
            ub = float(np.quantile(col, float(q_high)))
        else:
            lb = float(np.min(col))
            ub = float(np.max(col))
        if not np.isfinite(lb) or not np.isfinite(ub):
            return None
        if ub < lb:
            lb, ub = ub, lb
        out.append((lb, ub))
    return out


def _volume_ratio_for_bounds(
    selected_bounds: list[tuple[float, float]] | None,
    global_bounds: list[tuple[float, float]],
) -> float | None:
    if selected_bounds is None or len(selected_bounds) != len(global_bounds):
        return None
    ratios = []
    for (lb, ub), (s_lb, s_ub) in zip(global_bounds, selected_bounds):
        denom = float(ub - lb)
        if denom <= 0.0:
            ratios.append(0.0)
        else:
            ratios.append(max(0.0, float(s_ub - s_lb) / denom))
    return float(np.prod(ratios)) if ratios else 0.0


def _bounds_center(bounds_like: list[tuple[float, float]] | None) -> np.ndarray | None:
    if bounds_like is None or len(bounds_like) == 0:
        return None
    out = []
    for lb, ub in bounds_like:
        lo = float(min(lb, ub))
        hi = float(max(lb, ub))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return None
        out.append(0.5 * (lo + hi))
    return np.asarray(out, dtype=float)


def _bounds_iou_ratio(
    bounds_a: list[tuple[float, float]] | None,
    bounds_b: list[tuple[float, float]] | None,
) -> float | None:
    if (
        bounds_a is None
        or bounds_b is None
        or len(bounds_a) == 0
        or len(bounds_b) == 0
        or len(bounds_a) != len(bounds_b)
    ):
        return None
    inter_ratio = 1.0
    union_ratio = 1.0
    for (a_lb, a_ub), (b_lb, b_ub) in zip(bounds_a, bounds_b):
        a_lo = float(min(a_lb, a_ub))
        a_hi = float(max(a_lb, a_ub))
        b_lo = float(min(b_lb, b_ub))
        b_hi = float(max(b_lb, b_ub))
        if (not np.isfinite(a_lo)) or (not np.isfinite(a_hi)) or (not np.isfinite(b_lo)) or (not np.isfinite(b_hi)):
            return None
        inter_w = max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))
        union_w = max(0.0, max(a_hi, b_hi) - min(a_lo, b_lo))
        inter_ratio *= inter_w
        union_ratio *= union_w
    if union_ratio <= 0.0:
        return 0.0
    return float(np.clip(inter_ratio / union_ratio, 0.0, 1.0))


def _recenter_bounds_keep_width(
    *,
    base_bounds: list[tuple[float, float]] | None,
    global_bounds: list[tuple[float, float]],
    target_center: np.ndarray | None,
) -> list[tuple[float, float]] | None:
    if (
        base_bounds is None
        or target_center is None
        or len(base_bounds) != len(global_bounds)
    ):
        return base_bounds
    c = np.asarray(target_center, dtype=float).reshape(-1)
    if c.size != len(global_bounds):
        return base_bounds
    out: list[tuple[float, float]] = []
    for j, ((b_lb, b_ub), (g_lb, g_ub)) in enumerate(zip(base_bounds, global_bounds)):
        gl = float(min(g_lb, g_ub))
        gu = float(max(g_lb, g_ub))
        lo = float(np.clip(min(b_lb, b_ub), gl, gu))
        hi = float(np.clip(max(b_lb, b_ub), gl, gu))
        width = max(0.0, hi - lo)
        width = min(width, max(gu - gl, 0.0))
        cc = float(np.clip(c[j], gl, gu))
        if width <= 0.0 or gu <= gl:
            out.append((cc, cc))
            continue
        lo_new = float(np.clip(cc - 0.5 * width, gl, gu - width))
        hi_new = float(lo_new + width)
        if hi_new < lo_new:
            lo_new, hi_new = hi_new, lo_new
        out.append((lo_new, hi_new))
    return out


def _infer_boundary_touch_sides(
    *,
    selected_points: np.ndarray | None,
    global_bounds: list[tuple[float, float]],
    eps_ratio: float = 0.03,
) -> tuple[list[bool], list[bool]]:
    d = len(global_bounds)
    touch_lb = [False] * d
    touch_ub = [False] * d
    if (
        selected_points is None
        or d == 0
    ):
        return touch_lb, touch_ub
    X = np.asarray(selected_points, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != d:
        return touch_lb, touch_ub
    eps_r = float(max(float(eps_ratio), 0.0))
    for j, (g_lb, g_ub) in enumerate(global_bounds):
        gl = float(min(g_lb, g_ub))
        gu = float(max(g_lb, g_ub))
        span = max(gu - gl, 1e-12)
        eps = eps_r * span
        col = np.asarray(X[:, j], dtype=float)
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        cmin = float(np.min(col))
        cmax = float(np.max(col))
        touch_lb[j] = (cmin - gl) <= eps
        touch_ub[j] = (gu - cmax) <= eps
    return touch_lb, touch_ub


def _shrink_bounds_to_target_volume(
    *,
    selected_bounds: list[tuple[float, float]] | None,
    global_bounds: list[tuple[float, float]],
    target_volume_ratio: float,
    center: np.ndarray | None = None,
    boundary_pin_tol_ratio: float = 0.02,
    force_pin_lo: list[bool] | None = None,
    force_pin_hi: list[bool] | None = None,
    preferred_shrink_side: list[str | None] | None = None,
) -> list[tuple[float, float]] | None:
    if selected_bounds is None or len(selected_bounds) != len(global_bounds):
        return selected_bounds
    if not np.isfinite(float(target_volume_ratio)) or float(target_volume_ratio) <= 0.0:
        return selected_bounds
    tol = 1e-12
    d = len(global_bounds)
    if d <= 0:
        return selected_bounds

    c = None
    if center is not None:
        arr = np.asarray(center, dtype=float).reshape(-1)
        if arr.size == d:
            c = arr

    glb: list[tuple[float, float]] = []
    out: list[tuple[float, float]] = []
    spans: list[float] = []
    ratios: list[float] = []
    for (s_lb, s_ub), (g_lb, g_ub) in zip(selected_bounds, global_bounds):
        gl = float(min(g_lb, g_ub))
        gu = float(max(g_lb, g_ub))
        lo = float(np.clip(min(s_lb, s_ub), gl, gu))
        hi = float(np.clip(max(s_lb, s_ub), gl, gu))
        if hi < lo:
            lo, hi = hi, lo
        glb.append((gl, gu))
        out.append((lo, hi))
        span = max(gu - gl, 0.0)
        spans.append(span)
        if span <= 0.0:
            ratios.append(0.0)
        else:
            ratios.append(float(np.clip((hi - lo) / span, 0.0, 1.0)))

    current_volume = float(np.prod(ratios)) if ratios else 0.0
    if (not np.isfinite(current_volume)) or current_volume <= float(target_volume_ratio) + tol:
        return out

    active_dims = max(sum(1 for s in spans if s > 0.0), 1)
    alpha = float((float(target_volume_ratio) / float(current_volume)) ** (1.0 / float(active_dims)))
    alpha = float(np.clip(alpha, 0.0, 1.0))

    # 경계 인접 감지 + 외부 pin 힌트
    _bnd_tol = float(np.clip(float(boundary_pin_tol_ratio), 0.0, 0.5))
    _pref: list[str | None] = []
    if isinstance(preferred_shrink_side, list) and len(preferred_shrink_side) == d:
        for item in preferred_shrink_side:
            sval = str(item).strip().lower() if item is not None else ""
            _pref.append(sval if sval in {"lb", "ub"} else None)
    else:
        _pref = [None] * d
    pin_lo: list[bool] = []
    pin_hi: list[bool] = []
    for j, ((lo, hi), (gl, gu), span) in enumerate(zip(out, glb, spans)):
        ext_lo = bool(force_pin_lo[j]) if isinstance(force_pin_lo, list) and len(force_pin_lo) == d else False
        ext_hi = bool(force_pin_hi[j]) if isinstance(force_pin_hi, list) and len(force_pin_hi) == d else False
        if span <= 0.0:
            pin_lo.append(ext_lo)
            pin_hi.append(ext_hi)
        else:
            pin_lo.append(((lo - gl) < _bnd_tol * span) or ext_lo)
            pin_hi.append(((gu - hi) < _bnd_tol * span) or ext_hi)
        if pin_lo[-1] and pin_hi[-1]:
            _pref[j] = None

    # Pass 1: boundary-aware uniform shrink.
    pass1: list[tuple[float, float]] = []
    for j, ((lo, hi), (gl, gu), span) in enumerate(zip(out, glb, spans)):
        if span <= 0.0:
            pass1.append((gl, gu))
            continue
        cur_w = max(hi - lo, 0.0)
        tgt_w = min(cur_w * alpha, span)
        if pin_lo[j] and not pin_hi[j]:
            lo_new = gl
            hi_new = float(np.clip(gl + tgt_w, gl, gu))
        elif pin_hi[j] and not pin_lo[j]:
            hi_new = gu
            lo_new = float(np.clip(gu - tgt_w, gl, gu))
        elif _pref[j] == "lb":
            lo_new = float(np.clip(hi - tgt_w, gl, gu))
            hi_new = float(np.clip(hi, gl, gu))
        elif _pref[j] == "ub":
            lo_new = float(np.clip(lo, gl, gu))
            hi_new = float(np.clip(lo + tgt_w, gl, gu))
        else:
            cc = float(c[j]) if c is not None else 0.5 * (lo + hi)
            cc = float(np.clip(cc, gl, gu))
            lo_new = float(np.clip(cc - 0.5 * tgt_w, gl, gu - tgt_w))
            hi_new = float(lo_new + tgt_w)
        if hi_new < lo_new:
            lo_new, hi_new = hi_new, lo_new
        pass1.append((lo_new, hi_new))

    out = pass1
    vol = _volume_ratio_for_bounds(out, glb)
    if vol is None:
        return out
    if float(vol) <= float(target_volume_ratio) + tol:
        return out

    # Pass 2: boundary-aware greedy shrink (widest dim first).
    for _ in range(max(2 * d, 1)):
        vol = _volume_ratio_for_bounds(out, glb)
        if vol is None or float(vol) <= float(target_volume_ratio) + tol:
            break

        cur_ratios: list[float] = []
        for (lo, hi), (gl, gu), span in zip(out, glb, spans):
            if span <= 0.0:
                cur_ratios.append(0.0)
            else:
                lo_c = float(np.clip(lo, gl, gu))
                hi_c = float(np.clip(hi, gl, gu))
                cur_ratios.append(float(np.clip((hi_c - lo_c) / span, 0.0, 1.0)))

        # Shrink the widest ratio dimension first.
        j = int(np.argmax(np.asarray(cur_ratios, dtype=float)))
        rj = float(cur_ratios[j])
        if rj <= tol:
            break
        other = float(vol) / max(rj, tol)
        req_rj = float(target_volume_ratio) / max(other, tol)
        req_rj = float(np.clip(req_rj, 0.0, rj))
        if req_rj >= rj - tol:
            req_rj = max(0.0, rj * 0.95)

        gl, gu = glb[j]
        span = spans[j]
        lo, hi = out[j]
        cur_w = max(hi - lo, 0.0)
        tgt_w = float(np.clip(req_rj * span, 0.0, span))
        if tgt_w >= cur_w - tol:
            continue

        cut = cur_w - tgt_w
        lo_new, hi_new = lo, hi

        # Boundary-aware: 경계 고정 면에서는 절단하지 않음
        if pin_lo[j] and not pin_hi[j]:
            hi_new -= min(cut, max(hi_new - lo_new, 0.0))
        elif pin_hi[j] and not pin_lo[j]:
            lo_new += min(cut, max(hi_new - lo_new, 0.0))
        elif _pref[j] == "lb":
            lo_new += min(cut, max(hi_new - lo_new, 0.0))
        elif _pref[j] == "ub":
            hi_new -= min(cut, max(hi_new - lo_new, 0.0))
        else:
            cc = float(c[j]) if c is not None else 0.5 * (lo + hi)
            cc = float(np.clip(cc, lo, hi))
            left_room = cc - lo
            right_room = hi - cc
            # Keep side near center, shrink the opposite side first.
            if left_room <= right_room:
                cut_r = min(cut, right_room)
                hi_new -= cut_r
                cut -= cut_r
                if cut > 0.0:
                    lo_new += cut
            else:
                cut_l = min(cut, left_room)
                lo_new += cut_l
                cut -= cut_l
                if cut > 0.0:
                    hi_new -= cut

        lo_new = float(np.clip(lo_new, gl, gu))
        hi_new = float(np.clip(hi_new, gl, gu))
        if hi_new < lo_new:
            lo_new, hi_new = hi_new, lo_new
        out[j] = (lo_new, hi_new)

    # Final guard: boundary-aware enforcement.
    vol = _volume_ratio_for_bounds(out, glb)
    if vol is not None and float(vol) > float(target_volume_ratio) + tol:
        active_dims = max(sum(1 for s in spans if s > 0.0), 1)
        beta = float((float(target_volume_ratio) / float(vol)) ** (1.0 / float(active_dims)))
        beta = float(np.clip(beta, 0.0, 1.0))
        out2: list[tuple[float, float]] = []
        for j, ((lo, hi), (gl, gu), span) in enumerate(zip(out, glb, spans)):
            if span <= 0.0:
                out2.append((gl, gu))
                continue
            w = max(hi - lo, 0.0) * beta
            if pin_lo[j] and not pin_hi[j]:
                lo_new = gl
                hi_new = float(np.clip(gl + w, gl, gu))
            elif pin_hi[j] and not pin_lo[j]:
                hi_new = gu
                lo_new = float(np.clip(gu - w, gl, gu))
            elif _pref[j] == "lb":
                lo_new = float(np.clip(hi - w, gl, gu))
                hi_new = float(np.clip(hi, gl, gu))
            elif _pref[j] == "ub":
                lo_new = float(np.clip(lo, gl, gu))
                hi_new = float(np.clip(lo + w, gl, gu))
            else:
                mid = 0.5 * (lo + hi)
                lo_new = float(np.clip(mid - 0.5 * w, gl, gu - w))
                hi_new = float(lo_new + w)
            if hi_new < lo_new:
                lo_new, hi_new = hi_new, lo_new
            out2.append((lo_new, hi_new))
        out = out2

    return out


def _infer_constraint_side_policy(
    *,
    selected_points: np.ndarray,
    selected_features: list[str],
    global_bounds: list[tuple[float, float]],
    pre_constraint_defs: list[dict],
    edge_ratio: float = 0.05,
    probe_anchor_max: int = 16,
    probe_steps: int = 3,
    min_side_samples: int = 24,
    side_rate_min: float = 0.55,
    side_rate_gap: float = 0.12,
    anchor_obj_values: np.ndarray | None = None,
    objective_sense: str = "min",
    anti_anti_threshold: int = 1,
    require_top_obj_support: bool = False,
    top_obj_min_ratio: float = 0.10,
) -> dict | None:
    X = np.asarray(selected_points, dtype=float)
    d = len(global_bounds)
    if (
        X.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] != d
        or d == 0
        or not selected_features
        or len(selected_features) != d
        or not pre_constraint_defs
    ):
        return None

    feas_mask, _, _ = evaluate_constraints_batch(
        X=X,
        var_names=selected_features,
        constraint_defs=pre_constraint_defs,
        scope="pre",
    )
    anchors_all = X[feas_mask]
    feasible_ratio = (
        float(anchors_all.shape[0]) / float(max(X.shape[0], 1))
        if X.shape[0] > 0 else 0.0
    )
    if anchors_all.shape[0] == 0:
        return None

    # obj 값이 주어지면 top-K를 인덱스로 선정(L1).
    top_idx_global: np.ndarray | None = None
    if anchor_obj_values is not None:
        obj_arr = np.asarray(anchor_obj_values, dtype=float).ravel()
        if obj_arr.shape[0] == X.shape[0]:
            obj_feas = obj_arr[feas_mask]
            if obj_feas.size > 0 and np.all(np.isfinite(obj_feas)):
                top_k = int(max(3, int(np.ceil(obj_feas.size * 0.2))))
                top_k = int(min(top_k, obj_feas.size))
                order = np.argsort(obj_feas)
                if str(objective_sense).lower() == "max":
                    order = order[::-1]
                top_idx_global = order[:top_k]

    if anchors_all.shape[0] > int(max(probe_anchor_max, 1)):
        step = max(anchors_all.shape[0] // int(max(probe_anchor_max, 1)), 1)
        anchors = anchors_all[::step][: int(max(probe_anchor_max, 1))]
    else:
        anchors = anchors_all

    steps = int(max(probe_steps, 2))
    t_grid = np.linspace(0.5, 1.0, num=steps, dtype=float)
    min_side_n = int(max(min_side_samples, 1))
    edge_r = float(np.clip(float(edge_ratio), 0.0, 0.5))
    rate_min = float(np.clip(float(side_rate_min), 0.0, 1.0))
    rate_gap = float(np.clip(float(side_rate_gap), 0.0, 1.0))

    probe_points: list[np.ndarray] = []
    probe_dim: list[int] = []
    probe_side: list[str] = []

    near_lb_hits = np.zeros((d,), dtype=int)
    near_ub_hits = np.zeros((d,), dtype=int)
    # L1: top-obj anchor 기반 near-hits (objective 랭킹이 주어진 경우).
    top_near_lb_hits = np.zeros((d,), dtype=int)
    top_near_ub_hits = np.zeros((d,), dtype=int)
    if top_idx_global is not None:
        top_set = set(int(i) for i in top_idx_global.tolist())
        for ia, a in enumerate(anchors_all):
            if ia not in top_set:
                continue
            for j, (g_lb, g_ub) in enumerate(global_bounds):
                gl = float(min(g_lb, g_ub))
                gu = float(max(g_lb, g_ub))
                span = max(gu - gl, 0.0)
                if span <= 0.0:
                    continue
                xj = float(np.clip(a[j], gl, gu))
                if (xj - gl) <= edge_r * span:
                    top_near_lb_hits[j] += 1
                if (gu - xj) <= edge_r * span:
                    top_near_ub_hits[j] += 1
    for a in anchors:
        for j, (g_lb, g_ub) in enumerate(global_bounds):
            gl = float(min(g_lb, g_ub))
            gu = float(max(g_lb, g_ub))
            span = max(gu - gl, 0.0)
            if span <= 0.0:
                continue
            xj = float(np.clip(a[j], gl, gu))
            if (xj - gl) <= edge_r * span:
                near_lb_hits[j] += 1
            if (gu - xj) <= edge_r * span:
                near_ub_hits[j] += 1
            dist_lb = max(xj - gl, 0.0)
            dist_ub = max(gu - xj, 0.0)
            if dist_lb > 0.0:
                for t in t_grid:
                    x = a.copy()
                    x[j] = float(xj - t * dist_lb)
                    probe_points.append(x)
                    probe_dim.append(j)
                    probe_side.append("lb")
            if dist_ub > 0.0:
                for t in t_grid:
                    x = a.copy()
                    x[j] = float(xj + t * dist_ub)
                    probe_points.append(x)
                    probe_dim.append(j)
                    probe_side.append("ub")

    if not probe_points:
        return None

    X_probe = np.asarray(probe_points, dtype=float)
    probe_mask, _, _ = evaluate_constraints_batch(
        X=X_probe,
        var_names=selected_features,
        constraint_defs=pre_constraint_defs,
        scope="pre",
    )

    side_n = {"lb": np.zeros((d,), dtype=int), "ub": np.zeros((d,), dtype=int)}
    side_ok = {"lb": np.zeros((d,), dtype=int), "ub": np.zeros((d,), dtype=int)}
    for idx, ok in enumerate(probe_mask):
        j = int(probe_dim[idx])
        side = probe_side[idx]
        side_n[side][j] += 1
        if bool(ok):
            side_ok[side][j] += 1

    protect_lb = [False] * d
    protect_ub = [False] * d
    prefer_shrink_side: list[str | None] = [None] * d
    score_rows: list[dict] = []
    min_edge_hits = max(2, int(np.ceil(0.2 * anchors.shape[0])))

    for j in range(d):
        n_lb = int(side_n["lb"][j])
        n_ub = int(side_n["ub"][j])
        r_lb = float(side_ok["lb"][j] / max(n_lb, 1))
        r_ub = float(side_ok["ub"][j] / max(n_ub, 1))
        diff = r_ub - r_lb  # +: UB side가 더 feasible

        strong_cmp = (
            n_lb >= min_side_n
            and n_ub >= min_side_n
            and max(r_lb, r_ub) >= rate_min
            and abs(diff) >= rate_gap
        )
        if strong_cmp:
            if diff > 0:
                protect_ub[j] = True
                prefer_shrink_side[j] = "lb"
            else:
                protect_lb[j] = True
                prefer_shrink_side[j] = "ub"

        # 경계 근접 feasible anchor 신호가 충분하면 보조 pin 적용
        if near_lb_hits[j] >= min_edge_hits and not protect_ub[j]:
            protect_lb[j] = True
            if prefer_shrink_side[j] is None:
                prefer_shrink_side[j] = "ub"
        if near_ub_hits[j] >= min_edge_hits and not protect_lb[j]:
            protect_ub[j] = True
            if prefer_shrink_side[j] is None:
                prefer_shrink_side[j] = "lb"

        if protect_lb[j] and protect_ub[j]:
            # 동시 보호 충돌 시 rate가 더 높은 쪽만 유지
            if r_lb > r_ub:
                protect_ub[j] = False
                prefer_shrink_side[j] = "ub"
            elif r_ub > r_lb:
                protect_lb[j] = False
                prefer_shrink_side[j] = "lb"
            else:
                protect_lb[j] = False
                protect_ub[j] = False
                prefer_shrink_side[j] = None

        # L1: top-obj 신호 우선 적용.
        # 같은 dim에서 top-K obj anchors가 한쪽 경계에 몰려있다면 그 side를
        # 보호측으로 강제하고 반대측 shift를 비활성화한다.
        if top_idx_global is not None:
            tlb = int(top_near_lb_hits[j])
            tub = int(top_near_ub_hits[j])
            if tlb >= 1 and tub == 0:
                protect_lb[j] = True
                protect_ub[j] = False
                prefer_shrink_side[j] = None
            elif tub >= 1 and tlb == 0:
                protect_ub[j] = True
                protect_lb[j] = False
                prefer_shrink_side[j] = None

        score_rows.append(
            {
                "dim": int(j),
                "n_lb": n_lb,
                "n_ub": n_ub,
                "rate_lb": r_lb,
                "rate_ub": r_ub,
                "near_lb_hits": int(near_lb_hits[j]),
                "near_ub_hits": int(near_ub_hits[j]),
                "top_near_lb_hits": int(top_near_lb_hits[j]),
                "top_near_ub_hits": int(top_near_ub_hits[j]),
                "protect_lb": bool(protect_lb[j]),
                "protect_ub": bool(protect_ub[j]),
                "prefer_shrink_side": prefer_shrink_side[j],
            }
        )

    # L5: top-obj 지지 게이트 — feasibility-based protect 방향이 top-K objective
    # anchor 신호와 모순하거나 해당 dim에 top 신호가 전혀 없으면 shift 무효화.
    # anchor_obj_values 미제공(top_idx_global is None) 시 기존 동작 유지.
    veto_count = 0
    confirmed_count = 0
    if bool(require_top_obj_support) and top_idx_global is not None:
        top_total = int(top_idx_global.shape[0])
        _ratio = float(np.clip(float(top_obj_min_ratio), 0.0, 1.0))
        min_top_hits = int(max(1, int(np.ceil(top_total * _ratio))))
        for j in range(d):
            if not (protect_lb[j] or protect_ub[j]):
                continue
            tlb = int(top_near_lb_hits[j])
            tub = int(top_near_ub_hits[j])
            veto = False
            confirmed = False
            # (a) top 신호가 양측 모두 threshold 미만 → 근거 부족
            if tlb < min_top_hits and tub < min_top_hits:
                veto = True
            # (b) shift 방향이 top 다수결과 모순
            elif protect_lb[j] and not protect_ub[j] and tub > tlb:
                veto = True
            elif protect_ub[j] and not protect_lb[j] and tlb > tub:
                veto = True
            else:
                # 신호가 protect 방향과 정렬 + threshold 만족
                if protect_lb[j] and not protect_ub[j] and tlb >= min_top_hits and tlb >= tub:
                    confirmed = True
                elif protect_ub[j] and not protect_lb[j] and tub >= min_top_hits and tub >= tlb:
                    confirmed = True
            if veto:
                veto_count += 1
                protect_lb[j] = False
                protect_ub[j] = False
                prefer_shrink_side[j] = None
                if j < len(score_rows):
                    score_rows[j]["protect_lb"] = False
                    score_rows[j]["protect_ub"] = False
                    score_rows[j]["prefer_shrink_side"] = None
                    score_rows[j]["top_obj_veto"] = True
                    score_rows[j]["top_obj_confirmed"] = False
            else:
                if confirmed:
                    confirmed_count += 1
                if j < len(score_rows):
                    score_rows[j]["top_obj_veto"] = False
                    score_rows[j]["top_obj_confirmed"] = bool(confirmed)

    # D-B1 + L3: anti-anti-optimum guard 강화.
    # boundary 근처에 feasible anchor가 anti_thr 이상이면:
    #  - prefer_shrink_side 를 None 으로 비활성화(기존 동작)
    #  - protect_{반대측} 플래그 자체를 해제해 shift 적용도 억제(L3 강화)
    anti_thr = int(max(int(anti_anti_threshold), 1))
    for j in range(d):
        if int(near_lb_hits[j]) >= anti_thr:
            if prefer_shrink_side[j] == "lb":
                prefer_shrink_side[j] = None
            if bool(protect_ub[j]):
                protect_ub[j] = False
        if int(near_ub_hits[j]) >= anti_thr:
            if prefer_shrink_side[j] == "ub":
                prefer_shrink_side[j] = None
            if bool(protect_lb[j]):
                protect_lb[j] = False
        # score_rows 도 동기화
        if j < len(score_rows):
            score_rows[j]["prefer_shrink_side"] = prefer_shrink_side[j]
            score_rows[j]["protect_lb"] = bool(protect_lb[j])
            score_rows[j]["protect_ub"] = bool(protect_ub[j])

    # top-K obj feasible anchor centroid (#A blend 용)
    top_obj_centroid: list | None = None
    if top_idx_global is not None and anchors_all.shape[0] > 0:
        try:
            _top_pts = anchors_all[top_idx_global]
            if _top_pts.ndim == 2 and _top_pts.shape[0] > 0:
                top_obj_centroid = [float(v) for v in np.mean(_top_pts, axis=0).tolist()]
        except Exception:
            top_obj_centroid = None

    base_ret = {
        "protect_lb": protect_lb,
        "protect_ub": protect_ub,
        "prefer_shrink_side": prefer_shrink_side,
        "score_rows": score_rows,
        "anchor_count": int(anchors.shape[0]),
        "probe_count": int(X_probe.shape[0]),
        "feasible_selected_count": int(anchors_all.shape[0]),
        "feasible_ratio": float(feasible_ratio),
        "top_obj_centroid": top_obj_centroid,
        "top_obj_count": (int(top_idx_global.shape[0]) if top_idx_global is not None else 0),
        "top_obj_veto_count": int(veto_count),
        "top_obj_confirmed_count": int(confirmed_count),
    }
    return base_ret


def _apply_constraint_side_shift(
    *,
    selected_bounds: list[tuple[float, float]] | None,
    global_bounds: list[tuple[float, float]],
    protect_lb: list[bool] | None,
    protect_ub: list[bool] | None,
    shift_fraction: float = 1.0,
) -> tuple[list[tuple[float, float]] | None, list[tuple[int, str]]]:
    if (
        selected_bounds is None
        or len(selected_bounds) != len(global_bounds)
        or not isinstance(protect_lb, list)
        or not isinstance(protect_ub, list)
        or len(protect_lb) != len(global_bounds)
        or len(protect_ub) != len(global_bounds)
    ):
        return selected_bounds, []

    frac = float(np.clip(float(shift_fraction), 0.0, 1.0))
    out: list[tuple[float, float]] = []
    shifted: list[tuple[int, str]] = []
    for j, ((s_lb, s_ub), (g_lb, g_ub)) in enumerate(zip(selected_bounds, global_bounds)):
        gl = float(min(g_lb, g_ub))
        gu = float(max(g_lb, g_ub))
        lo = float(np.clip(min(s_lb, s_ub), gl, gu))
        hi = float(np.clip(max(s_lb, s_ub), gl, gu))
        width = max(hi - lo, 0.0)
        if width <= 0.0 or gu <= gl:
            out.append((lo, hi))
            continue

        if frac <= 0.0:
            # L2: shift 전면 비활성 — 원본 bounds 유지
            out.append((lo, hi))
            continue

        if bool(protect_lb[j]) and not bool(protect_ub[j]):
            # target: lo_target=gl, hi_target=gl+width
            lo_target = gl
            hi_target = float(min(gl + width, gu))
            lo_new = float(lo + frac * (lo_target - lo))
            hi_new = float(hi + frac * (hi_target - hi))
            out.append((lo_new, hi_new))
            if abs(lo_new - lo) > 1e-12 or abs(hi_new - hi) > 1e-12:
                shifted.append((j, "lb"))
            continue
        if bool(protect_ub[j]) and not bool(protect_lb[j]):
            hi_target = gu
            lo_target = float(max(gu - width, gl))
            lo_new = float(lo + frac * (lo_target - lo))
            hi_new = float(hi + frac * (hi_target - hi))
            out.append((lo_new, hi_new))
            if abs(lo_new - lo) > 1e-12 or abs(hi_new - hi) > 1e-12:
                shifted.append((j, "ub"))
            continue

        out.append((lo, hi))

    return out, shifted


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))


def _expected_improvement(
    *,
    mu: np.ndarray,
    sigma: np.ndarray,
    best: float,
    objective_sense: str,
) -> np.ndarray:
    eps = 1e-12
    sig = np.maximum(np.asarray(sigma, dtype=float), eps)
    if objective_sense == "max":
        imp = np.asarray(mu, dtype=float) - float(best)
    else:
        imp = float(best) - np.asarray(mu, dtype=float)
    z = imp / sig
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    Phi = _normal_cdf(z)
    ei = imp * Phi + sig * phi
    ei = np.where(sig <= eps, np.maximum(imp, 0.0), ei)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    return ei


def _mask_in_bounds(X: np.ndarray, region_bounds: list[tuple[float, float]] | None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    if region_bounds is None or len(region_bounds) != X.shape[1]:
        return np.zeros((X.shape[0],), dtype=bool)
    mask = np.ones((X.shape[0],), dtype=bool)
    for j, (lb, ub) in enumerate(region_bounds):
        lo = float(min(lb, ub))
        hi = float(max(lb, ub))
        mask &= (X[:, j] >= lo) & (X[:, j] <= hi)
    return mask


def _expand_bounds_around_center(
    *,
    base_bounds: list[tuple[float, float]] | None,
    global_bounds: list[tuple[float, float]],
    scale: float,
    center: np.ndarray,
    min_half_ratio: float = 0.01,
) -> list[tuple[float, float]] | None:
    if base_bounds is None or len(base_bounds) != len(global_bounds):
        return None
    c = np.asarray(center, dtype=float).reshape(-1)
    if c.size != len(global_bounds):
        return None
    out: list[tuple[float, float]] = []
    for j, ((b_lb, b_ub), (g_lb, g_ub)) in enumerate(zip(base_bounds, global_bounds)):
        gl = float(g_lb)
        gu = float(g_ub)
        span_g = max(gu - gl, 1e-12)
        bb_l = float(min(b_lb, b_ub))
        bb_u = float(max(b_lb, b_ub))
        half = 0.5 * (bb_u - bb_l) * float(scale)
        half = max(half, float(min_half_ratio) * span_g)
        cc = float(np.clip(c[j], gl, gu))
        lo = max(cc - half, gl)
        hi = min(cc + half, gu)
        if hi < lo:
            lo, hi = hi, lo
        out.append((lo, hi))
    return out


def _fit_gp_like_additional(
    *,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[GaussianProcessRegressor | None, bool]:
    return fit_gp_with_fallback(X=X, y=y, include_white=False, random_state=int(seed))


def _dedup_rows(X: np.ndarray, decimals: int = 12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.empty((0, X.shape[1] if X.ndim == 2 else 0), dtype=float)
    rounded = np.round(X, decimals=decimals)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    idx = np.sort(idx)
    return X[idx]


def _select_multistarts(
    *,
    points: np.ndarray,
    values: np.ndarray,
    objective_sense: str,
    total: int,
    rng: np.random.Generator,
    deterministic_fraction: float = 0.5,
) -> np.ndarray:
    pts = _dedup_rows(np.asarray(points, dtype=float))
    vals = np.asarray(values, dtype=float).reshape(-1)
    if pts.ndim != 2 or pts.shape[0] == 0 or vals.size == 0:
        return np.empty((0, pts.shape[1] if pts.ndim == 2 else 0), dtype=float)
    n = min(pts.shape[0], vals.size)
    pts = pts[:n]
    vals = vals[:n]
    if str(objective_sense).strip().lower() == "max":
        order = np.argsort(-vals)
    else:
        order = np.argsort(vals)
    n_total = max(int(total), 0)
    if n_total == 0:
        return np.empty((0, pts.shape[1]), dtype=float)
    det_frac = float(np.clip(float(deterministic_fraction), 0.0, 1.0))
    n_det = min(int(round(float(n_total) * det_frac)), n_total, n)
    det_idx = order[:n_det]
    remain = order[n_det:]
    n_rand = min(n_total - n_det, remain.size)
    if n_rand > 0:
        rand_idx = rng.choice(remain, size=n_rand, replace=False)
        pick = np.concatenate([det_idx, rand_idx])
    else:
        pick = det_idx
    if pick.size < min(n_total, n):
        already = set(int(i) for i in pick.tolist())
        fill_pool = [int(i) for i in order.tolist() if int(i) not in already]
        need = min(n_total, n) - pick.size
        if need > 0 and fill_pool:
            pick = np.concatenate([pick, np.asarray(fill_pool[:need], dtype=int)])
    return pts[pick] if pick.size else np.empty((0, pts.shape[1]), dtype=float)


def _resolve_strategy_alias(strategy_id: str, mode: str) -> str:
    sid = str(strategy_id).strip().lower()
    if sid.startswith("s4_obj"):
        return "s4_obj"
    if sid.startswith("s8_obj"):
        return "s8_obj"
    if sid.startswith("s4_pred"):
        return "s4_pred"
    if sid.startswith("s8_pred"):
        return "s8_pred"
    if sid.startswith("s4_"):
        return "s4"
    if sid.startswith("s8_"):
        return "s8"

    m = str(mode).strip().lower()
    mode_map = {
        "dual_refine_ei": "s4",
        "dual_refine_lcb": "s8",
        "dual_gradient_refine": "s8",
        "pred_refine_ei": "s4_pred",
        "pred_refine_lcb": "s8_pred",
        "obj_refine_ei": "s4_obj",
        "obj_refine_lcb": "s8_obj",
    }
    return mode_map.get(m, "s4")


def _resolve_existing_cae_metadata_path(
    *,
    config: ExplorerConfig,
    run_context: RunContext | None,
) -> str:
    if run_context is not None:
        path = get_task_metadata_path(run_context, "CAE")
        if path and os.path.exists(path):
            return path
        raise RuntimeError(
            "Explorer requires existing CAE metadata in run context. "
            "Run CAE task first and then execute Explorer."
        )

    raw = str(config.cae_metadata_path or "").strip()
    if raw:
        candidates = [raw]
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidates.append(os.path.join(project_root, raw))
        for p in candidates:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"CAE metadata not found: {raw}")

    raise RuntimeError(
        "Explorer requires existing CAE metadata. "
        "Provide ExplorerConfig.cae_metadata_path or run via pipeline run_context."
    )


def _load_cae_metadata(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid CAE metadata payload: {path}")
    return payload


def _extract_cae_fields(cae_meta: dict) -> tuple[str, list, list, str]:
    problem_name = str(cae_meta.get("problem", "")).strip()
    inputs = cae_meta.get("inputs", {}) if isinstance(cae_meta.get("inputs", {}), dict) else {}
    variables = inputs.get("variables", [])
    if not isinstance(variables, list):
        variables = []
    constraint_defs = inputs.get("constraint_defs", [])
    if not isinstance(constraint_defs, list):
        constraint_defs = []
    resolved = cae_meta.get("resolved_params", {}) if isinstance(cae_meta.get("resolved_params", {}), dict) else {}
    objective_sense = str(resolved.get("objective_sense", "min")).strip().lower()
    if objective_sense not in {"min", "max"}:
        objective_sense = "min"
    if not problem_name:
        raise RuntimeError("CAE metadata missing required field: problem")
    if len(variables) == 0:
        raise RuntimeError("CAE metadata missing required field: inputs.variables")
    return problem_name, variables, constraint_defs, objective_sense


def _extract_seed_from_cae_metadata(*, cae_meta: dict, cae_meta_path: str) -> int:
    resolved = cae_meta.get("resolved_params", {}) if isinstance(cae_meta.get("resolved_params", {}), dict) else {}
    direct_candidates = [
        resolved.get("seed"),
        cae_meta.get("seed"),
        (cae_meta.get("inputs", {}) if isinstance(cae_meta.get("inputs", {}), dict) else {}).get("seed"),
    ]
    for cand in direct_candidates:
        try:
            if cand is not None:
                return int(cand)
        except Exception:
            pass

    inputs = cae_meta.get("inputs", {}) if isinstance(cae_meta.get("inputs", {}), dict) else {}
    user_ref = str(inputs.get("user_config", "")).strip()
    if user_ref:
        user_cfg_path = user_ref
        if not os.path.isabs(user_cfg_path):
            user_cfg_path = os.path.join(os.path.dirname(cae_meta_path), user_cfg_path)
        if os.path.exists(user_cfg_path):
            with open(user_cfg_path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            if isinstance(user_cfg, dict) and ("seed" in user_cfg):
                return int(user_cfg["seed"])

    raise RuntimeError(
        "CAE metadata missing seed information. "
        "Expected one of: resolved_params.seed, inputs.seed, or inputs.user_config(seed)."
    )


def _resolve_known_optimum(
    known_optimum: object,
    selected_features: list[str],
) -> Optional[list[np.ndarray]]:
    if known_optimum is None:
        return None
    if isinstance(known_optimum, dict):
        values = []
        for feature in selected_features:
            if feature not in known_optimum:
                return None
            values.append(float(known_optimum[feature]))
        return [np.asarray(values, dtype=float)]
    if isinstance(known_optimum, (list, tuple, np.ndarray)):
        # list of dicts or list of arrays
        if isinstance(known_optimum, (list, tuple)) and known_optimum and isinstance(known_optimum[0], dict):
            out = []
            for item in known_optimum:
                values = []
                for feature in selected_features:
                    if feature not in item:
                        return None
                    values.append(float(item[feature]))
                out.append(np.asarray(values, dtype=float))
            return out
        arr = np.asarray(known_optimum, dtype=float)
        if arr.ndim == 1:
            if arr.shape[0] != len(selected_features):
                return None
            return [arr.reshape(-1)]
        if arr.ndim == 2:
            if arr.shape[1] != len(selected_features):
                return None
            return [row.astype(float) for row in arr]
    return None




class ExplorerOrchestrator:
    def __init__(self, config: ExplorerConfig, run_context: RunContext | None = None):
        self.config = config
        self.run_context = run_context

    def run(self) -> dict:
        if self.config.cae is None:
            raise RuntimeError("ExplorerConfig.cae is required.")

        loader = ResultLoader()
        cae_meta_path = _resolve_existing_cae_metadata_path(
            config=self.config,
            run_context=self.run_context,
        )
        cae_meta = _load_cae_metadata(cae_meta_path)
        (
            cae_problem_name,
            cae_variables,
            cae_constraint_defs,
            cae_objective_sense,
        ) = _extract_cae_fields(cae_meta)
        cae_seed = _extract_seed_from_cae_metadata(cae_meta=cae_meta, cae_meta_path=cae_meta_path)
        configured_problem = str(self.config.cae.user.problem_name).strip()
        if configured_problem and configured_problem != cae_problem_name:
            raise RuntimeError(
                "Problem mismatch between Explorer config and CAE metadata: "
                f"config={configured_problem}, cae_metadata={cae_problem_name}"
            )

        doe_meta = {}
        modeler_meta = {}
        doe_df = None
        modeler_df = None
        doe_problem_name = None
        modeler_problem_name = None
        doe_csv_path = None
        modeler_pkl_path = None
        modeler_feas_pkl_path = None
        modeler_task_dir = None

        if self.config.doe_metadata_path:
            print(f"[Explorer] DOE metadata path provided: {self.config.doe_metadata_path}")
            doe_result = loader.load_task(
                task="DOE",
                metadata_path=self.config.doe_metadata_path,
                csv_path=self.config.doe_csv_path,
                allow_latest_fallback=False,
            )
            doe_df = doe_result.df
            doe_meta = doe_result.metadata or {}
            doe_problem_name = doe_result.problem_name
            doe_csv_path = doe_result.csv_path
            if doe_problem_name and str(doe_problem_name).strip() != str(cae_problem_name).strip():
                raise RuntimeError(
                    "Problem mismatch between DOE metadata and CAE metadata: "
                    f"doe={doe_problem_name}, cae={cae_problem_name}"
                )
        elif self.config.doe_csv_path:
            if not os.path.exists(self.config.doe_csv_path):
                raise FileNotFoundError(f"DOE CSV not found: {self.config.doe_csv_path}")
            doe_csv_path = self.config.doe_csv_path
            doe_df = pd.read_csv(doe_csv_path)
            doe_problem_name = (
                self.config.cae.user.problem_name if self.config.cae else None
            )
        elif self.run_context:
            doe_meta_path = get_task_metadata_path(self.run_context, "DOE")
            if not doe_meta_path:
                raise RuntimeError("DOE metadata not found in run context.")
            doe_result = loader.load_task(
                task="DOE",
                metadata_path=doe_meta_path,
                allow_latest_fallback=False,
            )
            doe_df = doe_result.df
            doe_meta = doe_result.metadata or {}
            doe_problem_name = doe_result.problem_name
            doe_csv_path = doe_result.csv_path
            if doe_problem_name and str(doe_problem_name).strip() != str(cae_problem_name).strip():
                raise RuntimeError(
                    "Problem mismatch between DOE metadata and CAE metadata: "
                    f"doe={doe_problem_name}, cae={cae_problem_name}"
                )
        else:
            raise RuntimeError(
                "Explorer requires explicit DOE/Modeler inputs. "
                "Provide paths or run via pipeline run_context."
            )

        if self.config.modeler_metadata_path:
            print(f"[Explorer] Modeler metadata path provided: {self.config.modeler_metadata_path}")
            modeler_task_dir = os.path.dirname(self.config.modeler_metadata_path)
            modeler_result = loader.load_task(
                task="Modeler",
                metadata_path=self.config.modeler_metadata_path,
                allow_latest_fallback=False,
            )
            modeler_df = modeler_result.df
            modeler_meta = modeler_result.metadata or {}
            modeler_problem_name = modeler_result.problem_name
            modeler_pkl_path = self.config.model_pkl_path or modeler_result.pkl_path
            if not modeler_pkl_path:
                model_ref = _artifact_ref(modeler_meta, "model_path")
                if model_ref:
                    modeler_pkl_path = (
                        model_ref if os.path.isabs(model_ref) else os.path.join(modeler_task_dir, model_ref)
                    )
            feas_model_ref = _artifact_ref(modeler_meta, "feas_model_path")
            if feas_model_ref:
                modeler_feas_pkl_path = (
                    feas_model_ref
                    if os.path.isabs(feas_model_ref)
                    else os.path.join(modeler_task_dir, feas_model_ref)
                )
        elif self.config.model_pkl_path:
            if not os.path.exists(self.config.model_pkl_path):
                raise FileNotFoundError(f"Model PKL not found: {self.config.model_pkl_path}")
            modeler_pkl_path = self.config.model_pkl_path
            modeler_meta = {}
        elif self.run_context:
            modeler_meta_path = get_task_metadata_path(self.run_context, "Modeler")
            if not modeler_meta_path:
                raise RuntimeError("Modeler metadata not found in run context.")
            modeler_task_dir = os.path.dirname(modeler_meta_path)
            modeler_result = loader.load_task(
                task="Modeler",
                metadata_path=modeler_meta_path,
                allow_latest_fallback=False,
            )
            modeler_df = modeler_result.df
            modeler_meta = modeler_result.metadata or {}
            modeler_problem_name = modeler_result.problem_name
            modeler_pkl_path = modeler_result.pkl_path
            feas_model_ref = _artifact_ref(modeler_meta, "feas_model_path")
            if feas_model_ref:
                modeler_feas_pkl_path = (
                    feas_model_ref
                    if os.path.isabs(feas_model_ref)
                    else os.path.join(modeler_task_dir, feas_model_ref)
                )
        else:
            raise RuntimeError(
                "Explorer requires explicit DOE/Modeler inputs. "
                "Provide paths or run via pipeline run_context."
            )

        if modeler_feas_pkl_path and not os.path.isabs(modeler_feas_pkl_path):
            base_dir = modeler_task_dir or (
                os.path.dirname(self.config.modeler_metadata_path)
                if self.config.modeler_metadata_path
                else None
            )
            if base_dir:
                modeler_feas_pkl_path = os.path.join(base_dir, modeler_feas_pkl_path)

        if not doe_problem_name:
            doe_problem_name = cae_problem_name
        if not doe_problem_name:
            raise RuntimeError("Problem name not found from DOE/CAE metadata.")
        if str(doe_problem_name).strip() != str(cae_problem_name).strip():
            raise RuntimeError(
                "Problem mismatch between DOE source and CAE metadata: "
                f"doe={doe_problem_name}, cae={cae_problem_name}"
            )
        if modeler_problem_name and str(modeler_problem_name).strip() != str(cae_problem_name).strip():
            raise RuntimeError(
                "Problem mismatch between Modeler metadata and CAE metadata: "
                f"modeler={modeler_problem_name}, cae={cae_problem_name}"
            )

        if not modeler_pkl_path:
            raise RuntimeError("Modeler PKL not found. Check modeler metadata.")

        models, feature_cols = _load_models(modeler_pkl_path)

        selected_features = resolve_selected_features(
            feature_cols=feature_cols,
            doe_df=doe_df,
        )

        # FI scores path resolve (fi_aware 모드용)
        fi_scores_path = self.config.fi_scores_path
        if not fi_scores_path and modeler_task_dir:
            _candidate = os.path.join(modeler_task_dir, "artifacts", "public", "selected_features.csv")
            if os.path.exists(_candidate):
                fi_scores_path = _candidate

        raw_constraint_defs = (
            (doe_meta or {}).get("constraint_defs")
            or (doe_meta or {}).get("inputs", {}).get("constraint_defs")
            or cae_constraint_defs
            or []
        )
        try:
            constraint_defs = validate_constraint_defs(raw_constraint_defs)
        except Exception as exc:
            print(f"[Explorer] constraint_defs validation failed -> skip constraint policy: {exc}")
            constraint_defs = []

        pre_constraint_defs = [
            c for c in constraint_defs
            if str(c.get("scope", "pre")).strip().lower() == "pre"
        ]
        post_constraint_defs = [
            c for c in constraint_defs
            if str(c.get("scope", "pre")).strip().lower() == "post"
        ]
        has_pre_constraints = len(pre_constraint_defs) > 0
        has_post_constraints = len(post_constraint_defs) > 0
        pre_filter_disabled_reason = None

        # pre 제약식을 selected_features 축에서 평가할 수 없는 경우(식 변수 누락)는 pre 필터를 끈다.
        if has_pre_constraints:
            allowed_tokens = {
                "abs", "min", "max", "pow", "sqrt", "sin", "cos", "tan",
                "exp", "log", "pi", "e",
            }
            missing_vars: set[str] = set()
            for c in pre_constraint_defs:
                expr = str(c.get("expr", ""))
                try:
                    names = _extract_expr_vars(expr)
                    req = {n for n in names if n not in allowed_tokens}
                    req_missing = {n for n in req if n not in set(selected_features)}
                    missing_vars.update(req_missing)
                except Exception:
                    # 문법 문제는 DOE에서 이미 fail-fast 됐어야 하므로 여기서는 보수적으로 skip 처리
                    missing_vars.add("__expr_parse_error__")
            if missing_vars:
                print(
                    "[Explorer] pre-constraint filter disabled: "
                    f"missing vars in selected_features -> {sorted(missing_vars)}"
                )
                pre_filter_disabled_reason = (
                    "missing_vars_in_selected_features:"
                    + ",".join(sorted(missing_vars))
                )
                has_pre_constraints = False
                pre_constraint_defs = []

        feasibility_payload = None
        feasibility_model_kind_used = "none"
        feasibility_model_path_used = None
        if has_post_constraints:
            if modeler_feas_pkl_path and os.path.exists(modeler_feas_pkl_path):
                feasibility_payload = _load_feasibility_model(modeler_feas_pkl_path)
                feasibility_model_kind_used = str(feasibility_payload.get("kind", "unknown"))
                feasibility_model_path_used = modeler_feas_pkl_path
                print(f"[Explorer] feasibility model loaded: {modeler_feas_pkl_path}")
            else:
                print("[Explorer] post constraints exist but feasibility model not found; post penalty disabled.")

        variables = None
        if doe_meta:
            variables = doe_meta.get("variables") or doe_meta.get("inputs", {}).get("variables")
        if not variables and cae_variables:
            variables = cae_variables
        if not variables and self.config.cae and self.config.cae.user.variables:
            variables = self.config.cae.user.variables
        if not variables and self.run_context:
            with open(self.run_context.user_config_snapshot_path, "r") as f:
                user_snapshot = json.load(f)
            design_bounds = user_snapshot.get("design_bounds")
            if design_bounds:
                variables = [
                    {"name": name, "lb": bounds[0], "ub": bounds[1]}
                    for name, bounds in design_bounds.items()
                ]

        bounds = resolve_bounds(
            selected_features=selected_features,
            variables=variables,
            df=doe_df,
        )

        rng_seed = int(cae_seed)

        rng = np.random.default_rng(rng_seed)
        has_post_penalty = bool(has_post_constraints and feasibility_payload is not None)

        def _meta_get(key: str, default=None):
            if key in (doe_meta or {}):
                return (doe_meta or {}).get(key, default)
            resolved = (doe_meta or {}).get("resolved_params", {})
            if isinstance(resolved, dict) and key in resolved:
                return resolved.get(key, default)
            return default

        success_mask_base = (
            to_bool_mask(
                doe_df["success"],
                column_name="success",
                warn_prefix="[Explorer][BoolParse]",
            )
            if "success" in doe_df.columns
            else np.ones((len(doe_df),), dtype=bool)
        )
        if "feasible" in doe_df.columns:
            feasible_mask_base = to_bool_mask(
                doe_df["feasible"],
                column_name="feasible",
                warn_prefix="[Explorer][BoolParse]",
            )
            base_mask = success_mask_base & feasible_mask_base
        else:
            feasible_mask_base = success_mask_base.copy()
            base_mask = success_mask_base
        base_n = int(np.sum(base_mask))
        sample_multiplier = self.config.system.sample_multiplier

        nominal_n = int(base_n * float(sample_multiplier)) if sample_multiplier is not None else int(self.config.system.n_samples)
        r_used = 1.0
        r_used_source = "none"

        if sample_multiplier is not None and has_post_constraints:
            r_post = float(np.mean(feasible_mask_base)) if feasible_mask_base.size > 0 else 1.0
            if not np.isfinite(r_post) or r_post <= 0.0:
                r_post = 1.0
            r_used = float(r_post)
            r_used_source = "post_recomputed"
            n_samples = int(np.ceil(max(nominal_n, 1) / r_used))
        elif sample_multiplier is not None and has_pre_constraints:
            r_meta = _meta_get("constraint_rate_hat", _meta_get("constraint_r_hat", 1.0))
            try:
                r_used = float(r_meta)
            except Exception:
                r_used = 1.0
            if not np.isfinite(r_used) or r_used <= 0.0:
                r_used = 1.0
            r_used_source = "doe_metadata"
            n_samples = int(np.ceil(max(nominal_n, 1) / r_used))
        elif sample_multiplier is not None:
            n_samples = int(nominal_n)
        else:
            n_samples = int(self.config.system.n_samples)

        if sample_multiplier is not None:
            n_samples = max(self.config.system.n_samples_min, n_samples)
            n_samples = min(self.config.system.n_samples_max, n_samples)

        boundary_ratio = max(0.0, min(1.0, float(self.config.system.boundary_ratio)))
        n_boundary = int(round(n_samples * boundary_ratio))
        n_boundary = min(max(n_boundary, 0), n_samples)
        n_lhc = n_samples - n_boundary

        spans, _ = compute_spans_lbs(bounds)
        offset = spans * 0.0

        X_boundary_raw = np.empty((0, len(bounds)), dtype=float)
        if n_boundary > 0:
            corner_ratio = max(0.0, min(1.0, float(self.config.system.boundary_corner_ratio)))
            n_corner = int(round(n_boundary * corner_ratio))
            n_partial = n_boundary - n_corner

            boundary_candidates = sample_boundary_corners(bounds, offset=offset)
            if n_corner > 0:
                if boundary_candidates.shape[0] <= n_corner:
                    X_corner = boundary_candidates
                else:
                    idx = rng.choice(boundary_candidates.shape[0], size=n_corner, replace=False)
                    X_corner = boundary_candidates[idx]
            else:
                X_corner = np.empty((0, len(bounds)), dtype=float)

            if n_partial > 0:
                X_base = latin_hypercube_sampling(
                    n_samples=max(n_partial, 1),
                    bounds=bounds,
                    rng=rng,
                    n_divisions=max(n_partial, 1),
                )
                X_partial = sample_boundary_partial(
                    bounds,
                    offset=offset,
                    base_points=X_base,
                    n_samples=n_partial,
                    n_boundary_dims=min(2, len(bounds)),
                    rng=rng,
                )
            else:
                X_partial = np.empty((0, len(bounds)), dtype=float)

            X_boundary_raw = (
                np.vstack([X_corner, X_partial])
                if X_corner.size or X_partial.size
                else np.empty((0, len(bounds)), dtype=float)
            )

        X_lhc_raw = latin_hypercube_sampling(
            n_samples=n_lhc,
            bounds=bounds,
            rng=rng,
            n_divisions=max(n_lhc, 1),
        ) if n_lhc > 0 else np.empty((0, len(bounds)), dtype=float)

        pre_generated = int(X_boundary_raw.shape[0] + X_lhc_raw.shape[0])
        pre_kept = pre_generated
        if has_pre_constraints and pre_generated > 0:
            mask_b, _, _ = evaluate_constraints_batch(
                X=X_boundary_raw,
                var_names=selected_features,
                constraint_defs=pre_constraint_defs,
                scope="pre",
            )
            mask_l, _, _ = evaluate_constraints_batch(
                X=X_lhc_raw,
                var_names=selected_features,
                constraint_defs=pre_constraint_defs,
                scope="pre",
            )
            X_boundary = X_boundary_raw[mask_b]
            X_lhc = X_lhc_raw[mask_l]
            pre_kept = int(X_boundary.shape[0] + X_lhc.shape[0])
        else:
            X_boundary = X_boundary_raw
            X_lhc = X_lhc_raw

        X = np.vstack([X_boundary, X_lhc]) if X_boundary.size or X_lhc.size else X_lhc
        if X.shape[0] == 0:
            raise RuntimeError("Explorer generated zero candidates after pre-constraint filtering.")

        y_mean, y_std = _predict_ensemble(models, X)
        score = y_mean.copy()
        p_feasible_pred = np.ones((X.shape[0],), dtype=float)
        post_lambda_raw = _meta_get("post_lambda", None)
        if post_lambda_raw is None:
            post_lambda = float(self.config.system.post_lambda_default)
            post_lambda_source = "default"
        else:
            try:
                post_lambda = float(post_lambda_raw)
                post_lambda_source = "metadata"
            except Exception:
                post_lambda = float(self.config.system.post_lambda_default)
                post_lambda_source = "default_invalid_metadata"
        if has_post_penalty:
            kind = str(feasibility_payload.get("kind", "none")).strip().lower()
            try:
                if kind == "constant":
                    p0 = float(feasibility_payload.get("constant_prob", 0.5))
                    p_feasible_pred = np.full((X.shape[0],), np.clip(p0, 0.0, 1.0), dtype=float)
                else:
                    clf = feasibility_payload.get("model")
                    if clf is None:
                        raise RuntimeError("feasibility model payload has no model object")
                    p_feasible_pred = np.asarray(clf.predict_proba(X)[:, 1], dtype=float)
                    p_feasible_pred = np.clip(p_feasible_pred, 0.0, 1.0)
                penalty = post_lambda * (1.0 - p_feasible_pred)
                if str(cae_objective_sense).strip().lower() == "min":
                    score = y_mean + penalty
                else:
                    score = y_mean - penalty
            except Exception as exc:
                print(f"[Explorer] post penalty prediction skipped: {exc}")
                has_post_penalty = False
                score = y_mean.copy()
                p_feasible_pred = np.ones((X.shape[0],), dtype=float)

        df = pd.DataFrame(X, columns=selected_features)
        df["pred_mean"] = y_mean
        df["pred_std"] = y_std
        df["pred_score"] = score
        df["p_feasible_pred"] = p_feasible_pred

        objective_sense = (
            str(cae_objective_sense)
        )
        quantile_threshold = float(self.config.system.quantile_threshold)
        if objective_sense == "min":
            q_mask = score <= float(np.quantile(score, 1.0 - quantile_threshold))
        else:
            q_mask = score >= float(np.quantile(score, quantile_threshold))
        n_q_samples = int(np.sum(q_mask))

        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if self.run_context is None:
            design_bounds = None
            if variables:
                design_bounds = {
                    v["name"]: [v["lb"], v["ub"]]
                    for v in variables
                    if isinstance(v, dict) and {"name", "lb", "ub"}.issubset(v.keys())
                }
            user_snapshot = {
                "problem": doe_problem_name,
                "seed": int(rng_seed),
                "objective_sense": objective_sense,
                "task": "Explorer",
            }
            if design_bounds:
                user_snapshot["design_bounds"] = design_bounds
            self.run_context = create_run_context(
                project_root=project_root,
                user_config_snapshot=user_snapshot,
            )
        if get_task_metadata_path(self.run_context, "CAE") is None:
            update_run_index(self.run_context, "CAE", os.path.abspath(cae_meta_path))

        strategy_id = _normalize_strategy_id(self.config.system.strategy_id)
        task_dir = os.path.join(self.run_context.run_root, "Explorer")
        artifacts_root = os.path.join(task_dir, "artifacts")
        public_dir = os.path.join(artifacts_root, "public", strategy_id)
        meta_dir = os.path.join(artifacts_root, "meta", strategy_id)
        debug_dir = os.path.join(artifacts_root, "debug", strategy_id)
        os.makedirs(public_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)
        os.makedirs(debug_dir, exist_ok=True)

        use_timestamp = (
            self.config.cae.system.use_timestamp
            if self.config.cae is not None
            else False
        )
        debug_level = _normalize_debug_level(self.config.system.debug_level)
        save_plot = bool(self.config.system.save_plot)
        dbscan_min_samples = self.config.system.dbscan_min_samples

        x_opt = _resolve_known_optimum(
            self.config.user.known_optimum,
            selected_features,
        )


        # -------------------------------------------------
        # Dual-cluster comparison: model vs objective
        # -------------------------------------------------
        pair_overlay_paths = []
        doe_vs_optimum_plot_paths = []
        pred_stats = {}
        obj_stats = {}
        strategy_params = dict(self.config.system.strategy_params or {})
        strategy_mode_hint = str(strategy_params.get("mode", "")).strip().lower()
        strategy_alias_hint = _resolve_strategy_alias(strategy_id=strategy_id, mode=strategy_mode_hint)
        pred_uq_target_enabled = strategy_alias_hint in {"s4", "s8", "s4_pred", "s8_pred"}
        pred_cluster_beta = float(np.clip(float(strategy_params.get("pred_cluster_beta", 0.2)), 0.0, 2.0))
        pred_cluster_beta_used = float(pred_cluster_beta) if pred_uq_target_enabled else 0.0
        pred_cluster_signal_mode = "score"
        pred_cluster_confidence = None
        pred_cluster_selected_count = 0
        pred_cluster_sigma_mean_selected = None
        pred_refine_shift_norm = None
        pred_obj_miss_case = None
        pred_obj_iou = None
        pred_obj_center_l1_norm = None
        pred_multistart_det_fraction_used = None
        pred_refine_bounds_scale_used = None
        pred_n_starts_used = None
        pred_overconf_guard_triggered = False
        pred_overconf_guard_reason = None
        obj_div_extra = int(max(0, int(strategy_params.get("obj_diversity_extra_clusters", 0))))
        obj_div_weight = float(strategy_params.get("obj_diversity_weight", 0.35))
        obj_div_min_dist = float(strategy_params.get("obj_diversity_min_distance", 0.22))
        obj_div_close_penalty = float(strategy_params.get("obj_diversity_close_penalty", 0.8))
        obj_div_min_dim = int(max(1, int(strategy_params.get("obj_diversity_min_dim", 4))))
        if len(selected_features) < obj_div_min_dim:
            obj_div_extra = 0
        min_topk_count = int(max(1, int(self.config.system.min_topk_count)))
        max_topk_count_used = None
        max_topk_count_cfg = getattr(self.config.system, "max_topk_count", None)
        if max_topk_count_cfg is not None:
            try:
                max_topk_count_used = int(max(1, int(max_topk_count_cfg)))
            except Exception:
                max_topk_count_used = None
        elif bool(getattr(self.config.system, "max_topk_count_dynamic_enabled", True)):
            try:
                dyn_scale = float(getattr(self.config.system, "max_topk_count_dynamic_scale", 40.0))
                dyn_bias = float(getattr(self.config.system, "max_topk_count_dynamic_bias", 40.0))
                dyn_min_raw = int(getattr(self.config.system, "max_topk_count_dynamic_min", 120))
                dyn_max_raw = int(getattr(self.config.system, "max_topk_count_dynamic_max", 320))
                dyn_min = min(dyn_min_raw, dyn_max_raw)
                dyn_max = max(dyn_min_raw, dyn_max_raw)
                dyn_raw = int(round(dyn_scale * float(len(selected_features)) + dyn_bias))
                max_topk_count_used = int(np.clip(dyn_raw, dyn_min, dyn_max))
            except Exception:
                max_topk_count_used = None
        if max_topk_count_used is not None:
            max_topk_count_used = int(max(max_topk_count_used, min_topk_count))
        X_obj = np.empty((0, len(selected_features)))
        y_obj = np.array([], dtype=float)
        try:
            # (1) Model prediction clusters from LHC
            min_samples = dbscan_min_samples or max(5, 2 * len(selected_features))
            pred_cluster_values = np.asarray(score, dtype=float).reshape(-1)
            if pred_uq_target_enabled and pred_cluster_beta_used > 0.0:
                if objective_sense == "max":
                    pred_cluster_values = np.asarray(score, dtype=float).reshape(-1) + pred_cluster_beta_used * np.asarray(y_std, dtype=float).reshape(-1)
                else:
                    pred_cluster_values = np.asarray(score, dtype=float).reshape(-1) - pred_cluster_beta_used * np.asarray(y_std, dtype=float).reshape(-1)
                pred_cluster_signal_mode = "score_beta_sigma"
            X_pred_sel, labels_pred, pred_stats = select_top_clusters(
                X=X,
                y=pred_cluster_values,
                objective_sense=objective_sense,
                quantile_threshold=quantile_threshold,
                min_samples=min_samples,
                bounds=bounds,
                min_topk_count=min_topk_count,
                max_topk_count=max_topk_count_used,
                eps_quantile=float(self.config.system.dbscan_eps_quantile),
            )

            # (2) Objective clusters from executed DOE data
            if doe_df is not None:
                obj_df = doe_df.copy()
                if "feasible" in obj_df.columns:
                    f = obj_df["feasible"]
                    if f.dtype == bool:
                        obj_df = obj_df[f.fillna(False)]
                    else:
                        mask_f = (
                            f.astype(str)
                            .str.strip()
                            .str.lower()
                            .isin({"true", "1", "y", "yes", "t"})
                        )
                        obj_df = obj_df[mask_f]
                if "success" in obj_df.columns:
                    obj_df = obj_df[
                        to_bool_mask(
                            obj_df["success"],
                            column_name="success",
                            warn_prefix="[Explorer][BoolParse]",
                        )
                    ]
                obj_df = obj_df.dropna(subset=["objective"])
                if selected_features:
                    obj_df = obj_df.dropna(subset=selected_features)
                X_obj = obj_df[selected_features].to_numpy(dtype=float) if not obj_df.empty else np.empty((0, len(selected_features)))
                y_obj = obj_df["objective"].to_numpy(dtype=float) if not obj_df.empty else np.array([])
            else:
                X_obj = np.empty((0, len(selected_features)))
                y_obj = np.array([])

            X_obj_sel, labels_obj, obj_stats = select_top_clusters(
                X=X_obj,
                y=y_obj,
                objective_sense=objective_sense,
                quantile_threshold=quantile_threshold,
                min_samples=min_samples,
                bounds=bounds,
                min_topk_count=min_topk_count,
                max_topk_count=max_topk_count_used,
                eps_quantile=float(self.config.system.dbscan_eps_quantile),
                diversity_extra_clusters=obj_div_extra,
                diversity_weight=obj_div_weight,
                diversity_min_distance=obj_div_min_dist,
                diversity_close_penalty=obj_div_close_penalty,
                diversity_min_dim=obj_div_min_dim,
            )

            try:
                sel_idx = np.asarray(pred_stats.get("selected_indices", []), dtype=int).reshape(-1)
                sel_idx = sel_idx[(sel_idx >= 0) & (sel_idx < y_std.shape[0])]
                pred_cluster_selected_count = int(sel_idx.size)
                if sel_idx.size > 0:
                    sigma_all = np.asarray(y_std, dtype=float).reshape(-1)
                    sigma_sel = sigma_all[sel_idx]
                    pred_cluster_sigma_mean_selected = float(np.mean(np.maximum(sigma_sel, 0.0)))
                    sigma_ref = float(np.nanmedian(np.abs(sigma_all[np.isfinite(sigma_all)]))) if np.isfinite(sigma_all).any() else 0.0
                    sigma_ref = max(sigma_ref, 1e-12)
                    conf = 1.0 / (1.0 + float(np.mean(np.maximum(sigma_sel, 0.0) / sigma_ref)))
                    pred_cluster_confidence = float(np.clip(conf, 0.0, 1.0))
            except Exception:
                pred_cluster_confidence = None

            # 대안C: pred cluster 선택 포인트 수 상한 — 과다 선택 시 noise 유입 방지
            _pred_max_sel = int(strategy_params.get("pred_cluster_max_count", 22))
            if X_pred_sel.ndim == 2 and X_pred_sel.shape[0] > _pred_max_sel:
                try:
                    _sel_idx_c = np.asarray(
                        pred_stats.get("selected_indices", []), dtype=int
                    ).reshape(-1)
                    _sel_idx_c = _sel_idx_c[
                        (_sel_idx_c >= 0) & (_sel_idx_c < pred_cluster_values.shape[0])
                    ]
                    if _sel_idx_c.size > _pred_max_sel:
                        _scores_c = pred_cluster_values[_sel_idx_c]
                        if objective_sense == "min":
                            _keep_order = np.argsort(_scores_c)[:_pred_max_sel]
                        else:
                            _keep_order = np.argsort(-_scores_c)[:_pred_max_sel]
                        _kept_idx = _sel_idx_c[_keep_order]
                        X_pred_sel = X[_kept_idx]
                        if labels_pred.size > _pred_max_sel:
                            labels_pred = labels_pred[_keep_order]
                        pred_cluster_selected_count = int(_kept_idx.size)
                        print(
                            f"[Explorer] pred cluster cap applied: "
                            f"{_sel_idx_c.size} → {_kept_idx.size}"
                        )
                except Exception:
                    pass

        except Exception as exc:
            print(f"[Explorer] Dual cluster selection failed: {exc}")
            X_pred_sel = np.empty((0, len(selected_features)))
            labels_pred = np.array([], dtype=int)
            X_obj_sel = np.empty((0, len(selected_features)))
            labels_obj = np.array([], dtype=int)

        format_span_rows(
            kind="Pred",
            spans=pred_stats.get("selected_spans"),
            vols=pred_stats.get("selected_volumes"),
            feature_names=selected_features,
        )
        format_span_rows(
            kind="Obj",
            spans=obj_stats.get("selected_spans"),
            vols=obj_stats.get("selected_volumes"),
            feature_names=selected_features,
        )

        def _mean_vol(vols: list | None) -> float | None:
            if not vols:
                return None
            vals = [v for v in vols if v is not None]
            if not vals:
                return None
            return float(np.mean(vals))

        pred_mean_vol = _mean_vol(pred_stats.get("selected_volumes"))
        obj_mean_vol = _mean_vol(obj_stats.get("selected_volumes"))
        selected_points = np.empty((0, len(selected_features)), dtype=float)

        base_selected_bounds, pred_bounds, obj_bounds = compute_selected_bounds(
            X_pred_sel=X_pred_sel,
            X_obj_sel=X_obj_sel,
        )
        selected_bounds = base_selected_bounds
        strategy_mode = str((self.config.system.strategy_params or {}).get("mode", "")).strip().lower()
        strategy_alias = _resolve_strategy_alias(strategy_id=strategy_id, mode=strategy_mode)
        refine_pre_filter_applied = bool(has_pre_constraints)
        refine_pre_removed_pred = 0
        refine_pre_removed_obj = 0
        refine_pre_removed_selected = 0
        dual_policy_mode = None
        dual_total_starts_target = None
        dual_total_starts_used = None
        dual_np_ratio_used = None
        dual_pred_ratio_used = None
        dual_obj_ratio_used = None
        dual_n_pred_starts = None
        dual_n_obj_starts = None
        dual_disagreement_center_l1_norm = None
        dual_disagreement_iou = None
        dual_disagreement_triggered = False
        dual_center_bias_used = None
        dual_center_tilt_strength_base = None
        dual_center_tilt_strength_used_mean = None
        dual_center_tilt_applied = False
        dual_center_hint = None
        # routed_v2: dual 모드 내부 라우터. 활성화되면 obj_ratio/tilt/cap/expansion/dse2 override
        routed_v2_decision: RouterOutput | None = None

        if strategy_alias in {"s4", "s8", "s4_pred", "s8_pred", "s4_obj", "s8_obj"}:
            source_mode = "dual"
            if strategy_alias in {"s4_pred", "s8_pred"}:
                source_mode = "pred"
            elif strategy_alias in {"s4_obj", "s8_obj"}:
                source_mode = "obj"
            use_pred_model = source_mode in {"dual", "pred"}
            use_obj_model = source_mode in {"dual", "obj"}
            acq_type = "EI" if strategy_alias in {"s4", "s4_pred", "s4_obj"} else "LCB"
            d = max(int(len(selected_features)), 1)
            n_min = max(20, 4 * d + 4)
            min_fit = max(8, d + 2)
            kappa = float((self.config.system.strategy_params or {}).get("lcb_kappa", 0.35))
            ei_xi = float((self.config.system.strategy_params or {}).get("ei_xi", 0.01))
            acq = AcquisitionOptimizer()
            strategy_params_local = dict(self.config.system.strategy_params or {})
            pred_family_alias = strategy_alias in {"s4", "s8", "s4_pred", "s8_pred"}
            pred_multistart_det_fraction = float(
                np.clip(
                    float(strategy_params_local.get("pred_multistart_det_fraction", 0.35 if pred_family_alias else 0.5)),
                    0.0,
                    1.0,
                )
            )
            obj_multistart_det_fraction = float(
                np.clip(
                    float(strategy_params_local.get("obj_multistart_det_fraction", 0.5)),
                    0.0,
                    1.0,
                )
            )
            pred_refine_bounds_scale = float(
                max(
                    float(strategy_params_local.get("pred_refine_bounds_scale", 1.50 if pred_family_alias else 1.15)),
                    1.0,
                )
            )
            obj_refine_bounds_scale_default = float(
                max(float(getattr(self.config.system, "obj_refine_bounds_scale", 1.30)), 1.0)
            )
            obj_refine_bounds_scale = float(
                max(float(strategy_params_local.get("obj_refine_bounds_scale", obj_refine_bounds_scale_default)), 1.0)
            )
            if source_mode == "pred" and strategy_alias in {"s4_pred", "s8_pred"}:
                overconf_guard_enabled = bool(strategy_params_local.get("pred_overconf_guard_enabled", False))
                if overconf_guard_enabled:
                    overconf_conf_high = float(
                        np.clip(float(strategy_params_local.get("pred_overconf_conf_high", 0.58)), 0.0, 1.0)
                    )
                    overconf_iou_high = float(max(float(strategy_params_local.get("pred_overconf_iou_high", 0.24)), 0.0))
                    overconf_scale_boost = float(max(float(strategy_params_local.get("pred_overconf_scale_boost", 0.0)), 0.0))
                    overconf_scale_cap = float(
                        max(
                            float(strategy_params_local.get("pred_overconf_scale_cap", max(pred_refine_bounds_scale, 1.0))),
                            1.0,
                        )
                    )
                    overconf_det_fraction = float(
                        np.clip(
                            float(strategy_params_local.get("pred_overconf_det_fraction", pred_multistart_det_fraction)),
                            0.0,
                            1.0,
                        )
                    )
                    overconf_iou_now = _bounds_iou_ratio(pred_bounds, obj_bounds)
                    if (
                        pred_cluster_confidence is not None
                        and np.isfinite(pred_cluster_confidence)
                        and overconf_iou_now is not None
                        and np.isfinite(overconf_iou_now)
                        and float(pred_cluster_confidence) >= overconf_conf_high
                        and float(overconf_iou_now) >= overconf_iou_high
                    ):
                        pred_refine_bounds_scale = float(
                            np.clip(pred_refine_bounds_scale + overconf_scale_boost, 1.0, overconf_scale_cap)
                        )
                        pred_multistart_det_fraction = float(overconf_det_fraction)
                        pred_overconf_guard_triggered = True
                        pred_overconf_guard_reason = (
                            f"conf={float(pred_cluster_confidence):.3f}, iou={float(overconf_iou_now):.4f}"
                        )
                        print(
                            "[Explorer] pred-overconf guard applied: "
                            f"{pred_overconf_guard_reason} -> "
                            f"scale={pred_refine_bounds_scale:.3f}, det_frac={pred_multistart_det_fraction:.3f}"
                        )
            pred_multistart_det_fraction_used = float(pred_multistart_det_fraction)
            pred_refine_bounds_scale_used = float(pred_refine_bounds_scale)
            if source_mode == "dual":
                dual_policy_mode = str(
                    strategy_params_local.get(
                        "dual_policy_mode",
                        getattr(self.config.system, "dual_policy_mode_default", "routed_v2"),
                    )
                ).strip().lower()
                dual_total_starts_target = int(max(4, int(strategy_params_local.get("dual_total_starts", 40))))
                dual_total_starts_used = int(dual_total_starts_target)
                dual_np_ratio_used = (
                    float(base_n) / float(max(d, 1))
                    if d > 0
                    else None
                )

                obj_ratio_low_np = float(strategy_params_local.get("dual_obj_ratio_low_np", 0.62))
                obj_ratio_mid_np = float(strategy_params_local.get("dual_obj_ratio_mid_np", 0.50))
                obj_ratio_high_np = float(strategy_params_local.get("dual_obj_ratio_high_np", 0.42))
                np_ratio_low = float(strategy_params_local.get("dual_np_ratio_low", 12.0))
                np_ratio_high = float(strategy_params_local.get("dual_np_ratio_high", 24.0))
                high_dim_threshold = int(max(1, int(strategy_params_local.get("dual_high_dim_threshold", 6))))
                high_dim_obj_bonus = float(strategy_params_local.get("dual_high_dim_obj_bonus", 0.08))
                disagree_l1_threshold = float(strategy_params_local.get("dual_disagree_l1_threshold", 0.35))
                disagree_iou_threshold = float(strategy_params_local.get("dual_disagree_iou_threshold", 0.30))
                disagree_obj_bonus = float(strategy_params_local.get("dual_disagree_obj_bonus", 0.08))
                ratio_min = float(strategy_params_local.get("dual_obj_ratio_min", 0.25))
                ratio_max = float(strategy_params_local.get("dual_obj_ratio_max", 0.85))
                # DSE-B: 고제약(crate_hat < high_crate_threshold) 시 tilt_strength 상향
                _dseb_crate_thr = float(np.clip(
                    float(strategy_params_local.get("dsb_high_crate_threshold", 0.85)),
                    0.0, 1.0,
                ))
                _dseb_tilt_base = float(strategy_params_local.get("dual_center_tilt_strength", 0.45))
                _dseb_tilt_high_crate = float(strategy_params_local.get(
                    "dual_center_tilt_strength_high_crate", 0.55
                ))
                try:
                    _dseb_crate_hat = float(_meta_get("constraint_rate_hat", 1.0))
                except Exception:
                    _dseb_crate_hat = 1.0
                _dseb_is_high_crate = bool(
                    has_pre_constraints and np.isfinite(_dseb_crate_hat)
                    and _dseb_crate_hat < _dseb_crate_thr
                )
                _dseb_tilt_eff = _dseb_tilt_high_crate if _dseb_is_high_crate else _dseb_tilt_base
                dual_center_tilt_strength_base = float(
                    np.clip(_dseb_tilt_eff, 0.0, 1.0)
                )

                obj_ratio = float(obj_ratio_mid_np)
                if dual_policy_mode in {"fixed", "fixed50"}:
                    obj_ratio = float(obj_ratio_mid_np)
                elif dual_policy_mode == "routed_v1":
                    # ── II안: 라우팅 피처 기반 obj_ratio 결정 ──
                    _np_ratio = float(dual_np_ratio_used) if dual_np_ratio_used is not None else 0.0
                    if has_pre_constraints:
                        obj_ratio = 0.70
                    elif d <= 3 and _np_ratio > 20.0:
                        obj_ratio = 0.85
                    elif d >= 5 and _np_ratio > 30.0:
                        obj_ratio = 0.45
                    else:
                        obj_ratio = 0.55
                    print(
                        f"[Explorer] routed_v1: p_dim={d}, n/p={_np_ratio:.1f}, "
                        f"constraints={has_pre_constraints} → base obj_ratio={obj_ratio:.2f}"
                    )

                    # 실시간 disagreement 연속 보정
                    pred_center = _bounds_center(pred_bounds)
                    obj_center = _bounds_center(obj_bounds)
                    if pred_center is not None and obj_center is not None:
                        spans = np.array(
                            [max(float(ub) - float(lb), 1e-12) for lb, ub in bounds],
                            dtype=float,
                        )
                        dual_disagreement_center_l1_norm = float(
                            np.mean(np.abs(obj_center - pred_center) / spans)
                        )
                    dual_disagreement_iou = _bounds_iou_ratio(pred_bounds, obj_bounds)
                    # 연속 disagreement 보너스: iou 0.30 이하에서 비례 증가, 최대 +0.25
                    _iou_ref = float(strategy_params_local.get("dual_disagree_iou_ref", 0.30))
                    if dual_disagreement_iou is not None and dual_disagreement_iou < _iou_ref:
                        _cont_bonus = 0.25 * (1.0 - dual_disagreement_iou / _iou_ref)
                        obj_ratio += float(_cont_bonus)
                    dual_disagreement_triggered = bool(
                        dual_disagreement_iou is not None and dual_disagreement_iou < _iou_ref
                    )
                elif dual_policy_mode == "routed_v2":
                    # ── routed_v2: 결정 트리 라우터 (Explorer/executor/routing.py) ──
                    # disagreement 신호를 미리 계산
                    pred_center = _bounds_center(pred_bounds)
                    obj_center = _bounds_center(obj_bounds)
                    if pred_center is not None and obj_center is not None:
                        spans = np.array(
                            [max(float(ub) - float(lb), 1e-12) for lb, ub in bounds],
                            dtype=float,
                        )
                        dual_disagreement_center_l1_norm = float(
                            np.mean(np.abs(obj_center - pred_center) / spans)
                        )
                    dual_disagreement_iou = _bounds_iou_ratio(pred_bounds, obj_bounds)

                    _routed_overrides = strategy_params_local.get("routed_v2_overrides", None)
                    if not isinstance(_routed_overrides, dict):
                        _routed_overrides = None

                    _router_state = RouterInput(
                        p_dim=int(d),
                        has_pre_constraints=bool(has_pre_constraints),
                        has_post_constraints=bool(has_post_constraints),
                        usable_n=int(_meta_get("usable_n", 0) or 0),
                        usable_n_over_p=float(dual_np_ratio_used) if dual_np_ratio_used is not None else 0.0,
                        constraint_rate_hat=float(_meta_get("constraint_rate_hat", 1.0) or 1.0),
                        doe_gate1_pass_rate=_meta_get("doe_gate1_pass_rate", None),
                        doe_gate2_pass_rate=_meta_get("doe_gate2_pass_rate", None),
                        doe_gate1_score_last=_meta_get("doe_gate1_score_last", None),
                        doe_gate2_score_last=_meta_get("doe_gate2_score_last", None),
                        phase2_entered=bool(_meta_get("phase2_entered", False)),
                        phase2_transition_count=int(_meta_get("phase2_transition_count", 0) or 0),
                        doe_collapse_detected_count=int(_meta_get("doe_collapse_detected_count", 0) or 0),
                        doe_diversity_injection_applied_count=int(
                            _meta_get("doe_diversity_injection_applied_count", 0) or 0
                        ),
                        pred_cluster_confidence=pred_cluster_confidence,
                        pred_cluster_sigma_mean_selected=pred_cluster_sigma_mean_selected,
                        pred_refine_shift_norm=None,  # refine 단계에서 assign — routed 시점 미가용
                        pred_obj_iou=dual_disagreement_iou,
                        dual_disagreement_iou=dual_disagreement_iou,
                        dual_disagreement_center_l1_norm=dual_disagreement_center_l1_norm,
                    )
                    routed_v2_decision = route_v2(_router_state, overrides=_routed_overrides)
                    obj_ratio = float(routed_v2_decision.obj_ratio)
                    # tilt override: routed_v2가 high-crate/severe-disag 로직을 포함하므로
                    # 기존 _dseb_tilt_eff 결과를 override
                    dual_center_tilt_strength_base = float(np.clip(
                        routed_v2_decision.tilt_strength, 0.0, 1.0
                    ))
                    print(
                        f"[Explorer] routed_v2: p_dim={d}, n/p={float(dual_np_ratio_used or 0):.1f}, "
                        f"crate={float(_meta_get('constraint_rate_hat', 1.0) or 1.0):.3f}, "
                        f"iou={dual_disagreement_iou}, pred_conf={pred_cluster_confidence}"
                    )
                    print(
                        f"[Explorer] routed_v2 decision: acq={routed_v2_decision.acq_mode}, "
                        f"obj_ratio={routed_v2_decision.obj_ratio:.3f}, "
                        f"tilt={routed_v2_decision.tilt_strength:.3f}, "
                        f"expansion={routed_v2_decision.expansion_mode}, "
                        f"cap={routed_v2_decision.cap_target:.3f}, "
                        f"dse2_iou={routed_v2_decision.dse2_iou_threshold:.3f}"
                    )
                    for _r in routed_v2_decision.reason_trace:
                        print(f"[Explorer]   - {_r}")
                    dual_disagreement_triggered = bool(
                        dual_disagreement_iou is not None
                        and dual_disagreement_iou < float(routed_v2_decision.dse2_iou_threshold)
                    )
                else:
                    np_ratio_now = float(dual_np_ratio_used) if dual_np_ratio_used is not None else 0.0
                    if np_ratio_now <= np_ratio_low:
                        obj_ratio = float(obj_ratio_low_np)
                    elif np_ratio_now >= np_ratio_high:
                        obj_ratio = float(obj_ratio_high_np)
                    else:
                        width = max(np_ratio_high - np_ratio_low, 1e-9)
                        t = float((np_ratio_now - np_ratio_low) / width)
                        obj_ratio = float((1.0 - t) * obj_ratio_low_np + t * obj_ratio_high_np)

                    if d >= high_dim_threshold:
                        obj_ratio += float(high_dim_obj_bonus)

                    pred_center = _bounds_center(pred_bounds)
                    obj_center = _bounds_center(obj_bounds)
                    if pred_center is not None and obj_center is not None:
                        spans = np.array(
                            [max(float(ub) - float(lb), 1e-12) for lb, ub in bounds],
                            dtype=float,
                        )
                        dual_disagreement_center_l1_norm = float(
                            np.mean(np.abs(obj_center - pred_center) / spans)
                        )
                    dual_disagreement_iou = _bounds_iou_ratio(pred_bounds, obj_bounds)
                    dual_disagreement_triggered = bool(
                        (
                            dual_disagreement_center_l1_norm is not None
                            and dual_disagreement_center_l1_norm >= float(disagree_l1_threshold)
                        )
                        or (
                            dual_disagreement_iou is not None
                            and dual_disagreement_iou <= float(disagree_iou_threshold)
                        )
                    )
                    if dual_disagreement_triggered:
                        obj_ratio += float(disagree_obj_bonus)

                    # 대안D: pred/obj 극심 불일치(iou<0.15) 시 obj 편향 강화
                    if (
                        dual_disagreement_iou is not None
                        and dual_disagreement_iou < 0.15
                    ):
                        obj_ratio = max(obj_ratio, 0.70)

                # DSE-B: 고제약(crate_hat < threshold) 시 obj_ratio bonus
                # routed_v2 활성 시 자체 high-crate 로직이 있으므로 중복 적용 방지
                _dseb_obj_bonus = float(strategy_params_local.get(
                    "dual_high_crate_obj_ratio_bonus", 0.10
                ))
                if _dseb_is_high_crate and _dseb_obj_bonus > 0.0 and routed_v2_decision is None:
                    obj_ratio += float(_dseb_obj_bonus)
                    print(
                        f"[Explorer] DSE-B high-crate bonus: crate_hat={_dseb_crate_hat:.3f} "
                        f"→ obj_ratio += {_dseb_obj_bonus:.2f}, tilt={_dseb_tilt_eff:.2f}"
                    )

                obj_ratio = float(np.clip(obj_ratio, ratio_min, ratio_max))

                # #K dim-aware obj floor — multimodal low-dim 비제약 케이스에서
                # dual의 obj 측 비중을 강제로 끌어올림. state-based gate, benchmark
                # 명 분기 아님. has_pre_constraints + p_dim 두 가지만 사용.
                _k_low_dim_max = int(
                    getattr(self.config.system, "dual_obj_floor_low_dim_max", 3)
                )
                _k_low_dim_value = float(
                    getattr(self.config.system, "dual_obj_floor_low_dim_value", 0.90)
                )
                if (
                    not has_pre_constraints
                    and len(selected_features) <= _k_low_dim_max
                    and obj_ratio < _k_low_dim_value
                ):
                    print(
                        f"[Explorer] #K obj-floor: p_dim={len(selected_features)}, "
                        f"obj_ratio {obj_ratio:.2f} → {_k_low_dim_value:.2f}"
                    )
                    obj_ratio = float(np.clip(_k_low_dim_value, 0.0, 1.0))

                # DSE-L6: pred/obj 완전 disjoint 시 dual → obj-only 전환.
                # 상태: dual_disagreement_iou <= disjoint_iou. state-based gate,
                # benchmark 명 분기 아님. disagreement_triggered 조건과 무관하게
                # IoU 0 근방은 blend가 의미 없으므로 obj-only가 안전.
                _l6_disjoint_iou = float(max(
                    float(strategy_params_local.get("dual_obj_only_disjoint_iou", 0.05)),
                    0.0,
                ))
                _l6_applied = False
                if (
                    dual_disagreement_iou is not None
                    and float(dual_disagreement_iou) <= _l6_disjoint_iou
                ):
                    obj_ratio = 1.0
                    _l6_applied = True
                    print(
                        f"[Explorer] DSE-L6 dual→obj-only: iou="
                        f"{float(dual_disagreement_iou):.4f} <= {_l6_disjoint_iou:.2f}"
                    )

                pred_ratio = float(np.clip(1.0 - obj_ratio, 0.0, 1.0))
                dual_obj_ratio_used = float(obj_ratio)
                dual_pred_ratio_used = float(pred_ratio)
                dual_center_bias_used = float(obj_ratio)

                total = int(max(int(dual_total_starts_target), 2))
                if _l6_applied:
                    n_obj_starts = int(total)
                    n_pred_starts = 0
                else:
                    n_obj_starts = int(round(total * obj_ratio))
                    n_obj_starts = int(np.clip(n_obj_starts, 1, total - 1))
                    n_pred_starts = int(total - n_obj_starts)
            elif source_mode == "pred":
                n_pred_starts = 40
                n_obj_starts = 0
            else:
                n_pred_starts = 0
                n_obj_starts = 40

            X_success = np.asarray(X_obj, dtype=float)
            y_success = np.asarray(y_obj, dtype=float).reshape(-1)
            spans_global = np.array([max(ub - lb, 1e-12) for lb, ub in bounds], dtype=float)

            def _post_prob_fn(x_row: np.ndarray) -> float:
                if not has_post_penalty or feasibility_payload is None:
                    return 1.0
                try:
                    kind = str(feasibility_payload.get("kind", "none")).strip().lower()
                    if kind == "constant":
                        return float(np.clip(float(feasibility_payload.get("constant_prob", 0.5)), 0.0, 1.0))
                    clf = feasibility_payload.get("model")
                    if clf is None:
                        return 1.0
                    p = float(np.asarray(clf.predict_proba(np.asarray(x_row, dtype=float).reshape(1, -1))[:, 1], dtype=float)[0])
                    return float(np.clip(p, 0.0, 1.0))
                except Exception:
                    return 1.0

            def _is_pre_feasible_row(x_row: np.ndarray) -> bool:
                if not has_pre_constraints:
                    return True
                try:
                    _payload, ok, _margin = evaluate_constraints_point(
                        x=np.asarray(x_row, dtype=float).reshape(-1),
                        var_names=selected_features,
                        constraint_defs=pre_constraint_defs,
                        scope="pre",
                    )
                    return bool(ok)
                except Exception:
                    return False

            def _filter_points_pre_feasible(points: np.ndarray) -> tuple[np.ndarray, int]:
                pts = np.asarray(points, dtype=float)
                if pts.ndim != 2 or pts.shape[0] == 0:
                    return np.empty((0, d), dtype=float), 0
                if not has_pre_constraints:
                    return pts, 0
                keep = np.array([_is_pre_feasible_row(row) for row in pts], dtype=bool)
                removed = int(pts.shape[0] - int(keep.sum()))
                if int(keep.sum()) <= 0:
                    return np.empty((0, d), dtype=float), removed
                return pts[keep], removed

            def _score_points(points: np.ndarray) -> np.ndarray:
                pts = np.asarray(points, dtype=float)
                if pts.ndim != 2 or pts.shape[0] == 0:
                    return np.array([], dtype=float)
                mu, _ = _predict_ensemble(models, pts)
                out = np.asarray(mu, dtype=float).reshape(-1)
                if has_post_penalty:
                    p = np.asarray([_post_prob_fn(row) for row in pts], dtype=float)
                    pen = float(post_lambda) * (1.0 - np.clip(p, 0.0, 1.0))
                    if objective_sense == "min":
                        out = out + pen
                    else:
                        out = out - pen
                return out

            def _objective_values(points: np.ndarray) -> np.ndarray:
                pts = np.asarray(points, dtype=float)
                if pts.ndim != 2 or pts.shape[0] == 0:
                    return np.array([], dtype=float)
                if X_success.ndim != 2 or X_success.shape[0] == 0 or y_success.size == 0:
                    return _score_points(pts)
                vals = np.empty((pts.shape[0],), dtype=float)
                for i, row in enumerate(pts):
                    dists = np.linalg.norm((X_success - row.reshape(1, -1)) / spans_global.reshape(1, -1), axis=1)
                    vals[i] = float(y_success[int(np.argmin(dists))])
                return vals

            def _build_side_dataset(
                *,
                seed_points: np.ndarray,
                region_bounds: list[tuple[float, float]] | None,
            ) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]] | None]:
                if X_success.ndim != 2 or X_success.shape[0] == 0 or y_success.size == 0:
                    return (
                        np.empty((0, d), dtype=float),
                        np.array([], dtype=float),
                        region_bounds,
                    )
                if seed_points.ndim == 2 and seed_points.shape[0] > 0:
                    center = np.mean(seed_points, axis=0)
                elif region_bounds is not None:
                    center = np.asarray([(lb + ub) * 0.5 for lb, ub in region_bounds], dtype=float)
                else:
                    center = np.mean(X_success, axis=0)

                need = min(int(n_min), int(X_success.shape[0]))
                mask = _mask_in_bounds(X_success, region_bounds)
                X_side = X_success[mask] if mask.size else np.empty((0, d), dtype=float)
                y_side = y_success[mask] if mask.size else np.array([], dtype=float)

                expanded_bounds = region_bounds
                if X_side.shape[0] < need and region_bounds is not None:
                    for scale in (1.25, 1.5, 2.0, 3.0):
                        b = _expand_bounds_around_center(
                            base_bounds=region_bounds,
                            global_bounds=bounds,
                            scale=scale,
                            center=center,
                            min_half_ratio=0.01,
                        )
                        if b is None:
                            continue
                        mask_b = _mask_in_bounds(X_success, b)
                        if mask_b.size == 0:
                            continue
                        X_b = X_success[mask_b]
                        y_b = y_success[mask_b]
                        if X_b.shape[0] > X_side.shape[0]:
                            X_side = X_b
                            y_side = y_b
                            expanded_bounds = b
                        if X_side.shape[0] >= need:
                            break

                if X_side.shape[0] < need:
                    dists = np.linalg.norm((X_success - center.reshape(1, -1)) / spans_global.reshape(1, -1), axis=1)
                    order = np.argsort(dists)
                    keep = order[:need]
                    X_side = X_success[keep]
                    y_side = y_success[keep]
                    expanded_bounds = _bounds_from_points(X_side)

                return X_side, y_side, expanded_bounds

            def _build_union_dataset(
                *,
                region_a: list[tuple[float, float]] | None,
                region_b: list[tuple[float, float]] | None,
                seed_points: np.ndarray,
            ) -> tuple[
                np.ndarray,
                np.ndarray,
                list[tuple[float, float]] | None,
                list[tuple[float, float]] | None,
            ]:
                if X_success.ndim != 2 or X_success.shape[0] == 0 or y_success.size == 0:
                    return (
                        np.empty((0, d), dtype=float),
                        np.array([], dtype=float),
                        region_a,
                        region_b,
                    )

                sp = np.asarray(seed_points, dtype=float)
                if sp.ndim == 2 and sp.shape[0] > 0:
                    center = np.mean(sp, axis=0)
                elif region_a is not None:
                    center = np.asarray([(lb + ub) * 0.5 for lb, ub in region_a], dtype=float)
                elif region_b is not None:
                    center = np.asarray([(lb + ub) * 0.5 for lb, ub in region_b], dtype=float)
                else:
                    center = np.mean(X_success, axis=0)

                need = min(int(n_min), int(X_success.shape[0]))

                def _union_mask(
                    b_a: list[tuple[float, float]] | None,
                    b_b: list[tuple[float, float]] | None,
                ) -> np.ndarray:
                    mask_u = np.zeros((X_success.shape[0],), dtype=bool)
                    if b_a is not None:
                        m_a = _mask_in_bounds(X_success, b_a)
                        if m_a.size:
                            mask_u |= m_a
                    if b_b is not None:
                        m_b = _mask_in_bounds(X_success, b_b)
                        if m_b.size:
                            mask_u |= m_b
                    return mask_u

                expanded_a = region_a
                expanded_b = region_b
                mask_u = _union_mask(expanded_a, expanded_b)
                X_side = X_success[mask_u] if mask_u.size else np.empty((0, d), dtype=float)
                y_side = y_success[mask_u] if mask_u.size else np.array([], dtype=float)

                if X_side.shape[0] < need and (expanded_a is not None or expanded_b is not None):
                    for scale in (1.25, 1.5, 2.0, 3.0):
                        cand_a = (
                            _expand_bounds_around_center(
                                base_bounds=expanded_a,
                                global_bounds=bounds,
                                scale=scale,
                                center=center,
                                min_half_ratio=0.01,
                            )
                            if expanded_a is not None
                            else None
                        )
                        cand_b = (
                            _expand_bounds_around_center(
                                base_bounds=expanded_b,
                                global_bounds=bounds,
                                scale=scale,
                                center=center,
                                min_half_ratio=0.01,
                            )
                            if expanded_b is not None
                            else None
                        )
                        mask_c = _union_mask(cand_a, cand_b)
                        if mask_c.size == 0:
                            continue
                        X_c = X_success[mask_c]
                        y_c = y_success[mask_c]
                        if X_c.shape[0] > X_side.shape[0]:
                            X_side = X_c
                            y_side = y_c
                            expanded_a = cand_a
                            expanded_b = cand_b
                        if X_side.shape[0] >= need:
                            break

                if X_side.shape[0] < need:
                    dists = np.linalg.norm(
                        (X_success - center.reshape(1, -1)) / spans_global.reshape(1, -1),
                        axis=1,
                    )
                    order = np.argsort(dists)
                    keep = order[:need]
                    X_side = X_success[keep]
                    y_side = y_success[keep]
                    derived = _bounds_from_points(X_side)
                    if expanded_a is None:
                        expanded_a = derived
                    if expanded_b is None:
                        expanded_b = derived

                return X_side, y_side, expanded_a, expanded_b

            def _opt_bounds_from_starts(
                *,
                starts: np.ndarray,
                region_bounds: list[tuple[float, float]] | None,
                expand_scale: float = 1.15,
            ) -> tuple[np.ndarray, np.ndarray]:
                base = _bounds_from_points(starts) if starts.ndim == 2 and starts.shape[0] > 0 else None
                if base is None:
                    base = region_bounds
                if base is None:
                    lb = np.asarray([lb_ for lb_, _ in bounds], dtype=float)
                    ub = np.asarray([ub_ for _, ub_ in bounds], dtype=float)
                    return lb, ub
                center = np.mean(starts, axis=0) if starts.ndim == 2 and starts.shape[0] > 0 else np.asarray([(lb + ub) * 0.5 for lb, ub in base], dtype=float)
                b = _expand_bounds_around_center(
                    base_bounds=base,
                    global_bounds=bounds,
                    scale=float(max(expand_scale, 1.0)),
                    center=center,
                    min_half_ratio=0.01,
                )
                use_b = b if b is not None else base
                lb = np.asarray([float(v[0]) for v in use_b], dtype=float)
                ub = np.asarray([float(v[1]) for v in use_b], dtype=float)
                return lb, ub

            def _refine_with_gp(
                *,
                gp_model: GaussianProcessRegressor | None,
                starts: np.ndarray,
                y_train: np.ndarray,
                region_bounds: list[tuple[float, float]] | None,
                expand_scale: float = 1.15,
            ) -> np.ndarray:
                if gp_model is None or starts.ndim != 2 or starts.shape[0] == 0 or y_train.size == 0:
                    return np.empty((0, d), dtype=float)
                if objective_sense == "max":
                    y_best = float(np.max(y_train))
                else:
                    y_best = float(np.min(y_train))
                lb_opt, ub_opt = _opt_bounds_from_starts(
                    starts=starts,
                    region_bounds=region_bounds,
                    expand_scale=float(expand_scale),
                )
                out: list[np.ndarray] = []
                for x0 in starts:
                    x_opt = acq.optimize(
                        model=gp_model,
                        y_best=y_best,
                        lb=lb_opt,
                        ub=ub_opt,
                        starts=np.asarray(x0, dtype=float).reshape(1, -1),
                        objective_sense=objective_sense,
                        acq_type=acq_type,
                        kappa=float(kappa),
                        xi=float(ei_xi),
                        pre_feasible_fn=_is_pre_feasible_row if has_pre_constraints else None,
                        post_feasible_prob_fn=_post_prob_fn if has_post_penalty else None,
                        post_penalty_lambda=float(post_lambda if has_post_penalty else 0.0),
                    )
                    if x_opt is not None:
                        if has_pre_constraints and (not _is_pre_feasible_row(np.asarray(x_opt, dtype=float).reshape(-1))):
                            continue
                        out.append(np.asarray(x_opt, dtype=float).reshape(-1))
                if not out:
                    return np.empty((0, d), dtype=float)
                return _dedup_rows(np.vstack(out))

            pred_seed_pool = np.asarray(X_pred_sel, dtype=float)
            pred_seed_rank_values = np.asarray(score, dtype=float).reshape(-1)
            if strategy_alias in {"s4", "s8", "s4_pred", "s8_pred"} and pred_cluster_signal_mode == "score_beta_sigma":
                if objective_sense == "max":
                    pred_seed_rank_values = np.asarray(score, dtype=float).reshape(-1) + float(pred_cluster_beta_used) * np.asarray(y_std, dtype=float).reshape(-1)
                else:
                    pred_seed_rank_values = np.asarray(score, dtype=float).reshape(-1) - float(pred_cluster_beta_used) * np.asarray(y_std, dtype=float).reshape(-1)
            if not use_pred_model:
                pred_seed_pool = np.empty((0, d), dtype=float)
            elif pred_seed_pool.ndim != 2 or pred_seed_pool.shape[0] == 0:
                order = np.argsort(pred_seed_rank_values)
                if objective_sense == "max":
                    order = order[::-1]
                pred_seed_pool = X[order[: min(80, len(order))]]

            obj_seed_pool = np.asarray(X_obj_sel, dtype=float)
            if use_obj_model and (obj_seed_pool.ndim != 2 or obj_seed_pool.shape[0] == 0):
                if X_success.ndim == 2 and X_success.shape[0] > 0:
                    order_obj = np.argsort(y_success)
                    if objective_sense == "max":
                        order_obj = order_obj[::-1]
                    obj_seed_pool = X_success[order_obj[: min(80, X_success.shape[0])]]
                else:
                    obj_seed_pool = np.empty((0, d), dtype=float)
            if not use_obj_model:
                obj_seed_pool = np.empty((0, d), dtype=float)

            if source_mode == "dual":
                total_target = int(max(int(dual_total_starts_target or 40), 2))
                has_pred_pool = pred_seed_pool.ndim == 2 and pred_seed_pool.shape[0] > 0
                has_obj_pool = obj_seed_pool.ndim == 2 and obj_seed_pool.shape[0] > 0
                if (not has_pred_pool) and has_obj_pool:
                    n_pred_starts = 0
                    n_obj_starts = int(total_target)
                elif has_pred_pool and (not has_obj_pool):
                    n_pred_starts = int(total_target)
                    n_obj_starts = 0
                elif has_pred_pool and has_obj_pool:
                    # keep both sides alive in dual mode
                    n_pred_starts = int(max(int(n_pred_starts), 1))
                    n_obj_starts = int(max(int(n_obj_starts), 1))
                    extra = int(n_pred_starts + n_obj_starts - total_target)
                    if extra > 0:
                        if n_obj_starts >= n_pred_starts:
                            n_obj_starts = int(max(1, n_obj_starts - extra))
                        else:
                            n_pred_starts = int(max(1, n_pred_starts - extra))
                        if (n_pred_starts + n_obj_starts) > total_target:
                            n_pred_starts = int(max(1, total_target - n_obj_starts))
                    elif extra < 0:
                        if (dual_obj_ratio_used or 0.5) >= 0.5:
                            n_obj_starts += int(-extra)
                        else:
                            n_pred_starts += int(-extra)
                else:
                    n_pred_starts = 0
                    n_obj_starts = 0

                used_total = int(max(n_pred_starts + n_obj_starts, 0))
                dual_total_starts_used = int(used_total)
                if used_total > 0:
                    dual_pred_ratio_used = float(n_pred_starts) / float(used_total)
                    dual_obj_ratio_used = float(n_obj_starts) / float(used_total)
                    dual_center_bias_used = float(dual_obj_ratio_used)
                else:
                    dual_pred_ratio_used = None
                    dual_obj_ratio_used = None
                    dual_center_bias_used = None
                dual_n_pred_starts = int(n_pred_starts)
                dual_n_obj_starts = int(n_obj_starts)

            pred_scores = _score_points(pred_seed_pool) if use_pred_model else np.array([], dtype=float)
            obj_scores = _objective_values(obj_seed_pool) if use_obj_model else np.array([], dtype=float)

            pred_starts = np.empty((0, d), dtype=float)
            if n_pred_starts > 0 and pred_seed_pool.shape[0] > 0:
                pred_starts = _select_multistarts(
                    points=pred_seed_pool,
                    values=pred_scores,
                    objective_sense=objective_sense,
                    total=n_pred_starts,
                    rng=rng,
                    deterministic_fraction=pred_multistart_det_fraction,
                )
            pred_n_starts_used = int(pred_starts.shape[0]) if pred_starts.ndim == 2 else 0
            obj_starts = np.empty((0, d), dtype=float)
            if n_obj_starts > 0 and use_obj_model and obj_seed_pool.shape[0] > 0:
                obj_starts = _select_multistarts(
                    points=obj_seed_pool,
                    values=obj_scores,
                    objective_sense=objective_sense,
                    total=n_obj_starts,
                    rng=rng,
                    deterministic_fraction=obj_multistart_det_fraction,
                )

            pred_region = _bounds_from_points(X_pred_sel)
            obj_region = _bounds_from_points(X_obj_sel)
            X_pred_train = np.empty((0, d), dtype=float)
            y_pred_train = np.array([], dtype=float)
            pred_region_used = pred_region
            X_obj_train = np.empty((0, d), dtype=float)
            y_obj_train = np.array([], dtype=float)
            obj_region_used = obj_region
            if source_mode == "dual":
                union_seeds = _safe_stack([pred_starts, obj_starts], n_dim=d)
                (
                    X_union_train,
                    y_union_train,
                    pred_region_used,
                    obj_region_used,
                ) = _build_union_dataset(
                    region_a=pred_region,
                    region_b=obj_region,
                    seed_points=union_seeds,
                )
                X_pred_train = X_union_train
                y_pred_train = y_union_train
                X_obj_train = X_union_train
                y_obj_train = y_union_train
            elif source_mode == "pred":
                X_pred_train, y_pred_train, pred_region_used = _build_side_dataset(
                    seed_points=pred_starts,
                    region_bounds=pred_region,
                )
            elif source_mode == "obj":
                X_obj_train, y_obj_train, obj_region_used = _build_side_dataset(
                    seed_points=obj_starts,
                    region_bounds=obj_region,
                )
            if use_obj_model:
                X_obj_train = np.asarray(X_obj_train, dtype=float)
                y_obj_train = np.asarray(y_obj_train, dtype=float).reshape(-1)

            gp_pred = None
            gp_obj = None
            if use_pred_model and X_pred_train.shape[0] >= min_fit:
                gp_pred, _ = _fit_gp_like_additional(X=X_pred_train, y=y_pred_train, seed=int(rng_seed) + 17)
            if use_obj_model and X_obj_train.shape[0] >= min_fit:
                gp_obj, _ = _fit_gp_like_additional(X=X_obj_train, y=y_obj_train, seed=int(rng_seed) + 29)

            # III-A: dual_acq_split — DUAL 모드에서 pred측 EI, obj측 LCB 분할
            _dual_acq_split = bool(strategy_params_local.get("dual_acq_split", False))
            _acq_type_orig = acq_type

            ref_pred = np.empty((0, d), dtype=float)
            if use_pred_model:
                if _dual_acq_split and source_mode == "dual":
                    acq_type = "EI"  # pred 측: 탐색적
                ref_pred = _refine_with_gp(
                    gp_model=gp_pred,
                    starts=pred_starts,
                    y_train=y_pred_train,
                    region_bounds=pred_region_used,
                    expand_scale=pred_refine_bounds_scale,
                )
                if _dual_acq_split and source_mode == "dual":
                    acq_type = _acq_type_orig  # 복원
                ref_pred, removed_pred = _filter_points_pre_feasible(ref_pred)
                refine_pre_removed_pred += int(removed_pred)
                if ref_pred.shape[0] > 0 and pred_starts.shape[0] > 0:
                    try:
                        spans_eval = spans_global.reshape(1, 1, -1)
                        diff = (ref_pred[:, None, :] - pred_starts[None, :, :]) / spans_eval
                        dmat = np.linalg.norm(diff, axis=2) / np.sqrt(float(max(d, 1)))
                        pred_refine_shift_norm = float(np.mean(np.min(dmat, axis=1)))
                    except Exception:
                        pred_refine_shift_norm = None
            ref_obj = np.empty((0, d), dtype=float)
            if use_obj_model:
                if _dual_acq_split and source_mode == "dual":
                    acq_type = "LCB"  # obj 측: 보수적 수렴
                ref_obj = _refine_with_gp(
                    gp_model=gp_obj,
                    starts=obj_starts,
                    y_train=y_obj_train,
                    region_bounds=obj_region_used,
                    expand_scale=obj_refine_bounds_scale,
                )
                if _dual_acq_split and source_mode == "dual":
                    acq_type = _acq_type_orig  # 복원
                ref_obj, removed_obj = _filter_points_pre_feasible(ref_obj)
                refine_pre_removed_obj += int(removed_obj)

            if source_mode == "dual":
                selected_points = _safe_stack([ref_pred, ref_obj, X_pred_sel, X_obj_sel], n_dim=len(selected_features))
            elif source_mode == "pred":
                selected_points = _safe_stack([ref_pred, X_pred_sel], n_dim=len(selected_features))
            else:
                selected_points = _safe_stack([ref_obj, X_obj_sel], n_dim=len(selected_features))
            if selected_points.shape[0] == 0:
                if source_mode in {"dual", "obj"}:
                    selected_points = _safe_stack([X_obj_sel], n_dim=len(selected_features))
                else:
                    selected_points = _safe_stack([X_pred_sel], n_dim=len(selected_features))
            selected_points, removed_sel = _filter_points_pre_feasible(selected_points)
            refine_pre_removed_selected += int(removed_sel)
            if selected_points.shape[0] == 0:
                if source_mode in {"dual", "obj"}:
                    selected_points = _safe_stack([X_obj_sel], n_dim=len(selected_features))
                else:
                    selected_points = _safe_stack([X_pred_sel], n_dim=len(selected_features))
            selected_bounds = _bounds_from_points(selected_points, q_low=0.01, q_high=0.99)
            if source_mode == "dual" and selected_bounds is not None:
                strategy_params_local = dict(self.config.system.strategy_params or {})
                pred_center = _bounds_center(pred_bounds)
                obj_center = _bounds_center(obj_bounds)
                cur_center = _bounds_center(selected_bounds)
                # EXP-1: 고제약 + low IoU 시 obj_center를 top-K best-objective weighted mean으로 대체
                try:
                    _exp1_k = int(strategy_params_local.get("dual_obj_center_top_k_high_crate", 3))
                    _exp1_iou_thr = float(strategy_params_local.get("dual_obj_center_top_k_iou_thr", 0.20))
                    _exp1_crate_thr = float(strategy_params_local.get("dual_obj_center_top_k_crate_thr", 0.85))
                    _exp1_crate_hat = float(_meta_get("constraint_rate_hat", 1.0))
                except Exception:
                    _exp1_k, _exp1_iou_thr, _exp1_crate_thr, _exp1_crate_hat = 3, 0.20, 0.85, 1.0
                _exp1_iou_cur = _bounds_iou_ratio(pred_bounds, obj_bounds) if pred_bounds is not None and obj_bounds is not None else None
                _exp1_triggered = bool(
                    has_pre_constraints
                    and np.isfinite(_exp1_crate_hat)
                    and _exp1_crate_hat < _exp1_crate_thr
                    and _exp1_iou_cur is not None
                    and _exp1_iou_cur < _exp1_iou_thr
                    and obj_center is not None
                    and isinstance(y_obj, np.ndarray)
                    and y_obj.size >= 1
                    and isinstance(X_obj, np.ndarray)
                    and X_obj.shape[0] >= 1
                    and int(_exp1_k) >= 1
                )
                if _exp1_triggered:
                    try:
                        X_obj_arr = np.asarray(X_obj, dtype=float)
                        y_obj_arr = np.asarray(y_obj, dtype=float).reshape(-1)
                        if objective_sense == "min":
                            order = np.argsort(y_obj_arr)
                            scores = -y_obj_arr
                        else:
                            order = np.argsort(-y_obj_arr)
                            scores = y_obj_arr
                        top_idx = order[: max(1, int(_exp1_k))]
                        if top_idx.size >= 1:
                            top_X = X_obj_arr[top_idx]
                            top_scores = scores[top_idx]
                            _norm = float(np.max(np.abs(top_scores))) + 1e-12
                            _shift = float(np.max(top_scores))
                            w = np.exp((top_scores - _shift) / _norm)
                            w = w / max(float(np.sum(w)), 1e-12)
                            obj_center_topk = np.asarray(
                                np.sum(top_X * w.reshape(-1, 1), axis=0), dtype=float
                            )
                            _delta = float(np.linalg.norm(obj_center_topk - obj_center))
                            print(
                                f"[Explorer] EXP-1 top-K obj center override: k={top_idx.size} "
                                f"iou={_exp1_iou_cur:.3f} crate={_exp1_crate_hat:.3f} "
                                f"center_shift={_delta:.4f}"
                            )
                            obj_center = obj_center_topk
                    except Exception as _e:
                        print(f"[Explorer] EXP-1 top-K fallback failed: {_e}")
                if pred_center is not None and obj_center is not None and cur_center is not None:
                    bias_obj = float(
                        np.clip(
                            float(
                                strategy_params_local.get(
                                    "dual_center_bias_obj_ratio_weight",
                                    dual_obj_ratio_used if dual_obj_ratio_used is not None else 0.5,
                                )
                            ),
                            0.0,
                            1.0,
                        )
                    )
                    dual_center_bias_used = float(bias_obj)
                    target_center = (1.0 - bias_obj) * pred_center + bias_obj * obj_center
                    spans = np.array([max(float(ub) - float(lb), 1e-12) for lb, ub in bounds], dtype=float)
                    disagreement = np.abs(obj_center - pred_center) / spans
                    mean_disagreement = float(max(np.mean(disagreement), 1e-9))
                    aniso_gamma = float(strategy_params_local.get("dual_center_tilt_aniso_gamma", 0.6))
                    base_tilt = float(
                        np.clip(
                            float(
                                dual_center_tilt_strength_base
                                if dual_center_tilt_strength_base is not None
                                else strategy_params_local.get("dual_center_tilt_strength", 0.45)
                            ),
                            0.0,
                            1.0,
                        )
                    )
                    # D6: disagreement 크기에 비례하여 base_tilt 동적 스케일링
                    # mean_disagreement가 클수록 obj 방향으로 더 강하게 tilt
                    _disagree_ref = float(
                        strategy_params_local.get(
                            "dual_tilt_disagree_ref",
                            getattr(self.config.system, "dual_tilt_disagree_ref", 0.10),
                        )
                    )
                    _disagree_scale = float(np.clip(mean_disagreement / max(_disagree_ref, 1e-9), 0.8, 1.5))
                    base_tilt = float(np.clip(base_tilt * _disagree_scale, 0.0, 0.75))
                    tilt_weights = np.clip(
                        base_tilt
                        * (
                            1.0
                            + aniso_gamma * ((disagreement / mean_disagreement) - 1.0)
                        ),
                        0.0,
                        1.0,
                    )
                    dual_center_tilt_strength_used_mean = float(np.mean(tilt_weights))
                    blended_center = (1.0 - tilt_weights) * cur_center + tilt_weights * target_center
                    selected_bounds_tilt = _recenter_bounds_keep_width(
                        base_bounds=selected_bounds,
                        global_bounds=bounds,
                        target_center=blended_center,
                    )
                    if selected_bounds_tilt is not None:
                        selected_bounds = selected_bounds_tilt
                        dual_center_hint = blended_center
                        dual_center_tilt_applied = True

            if strategy_alias in {"s4", "s8", "s4_pred", "s8_pred"}:
                pred_obj_iou = _bounds_iou_ratio(pred_bounds, obj_bounds)
                pred_center = _bounds_center(pred_bounds)
                obj_center = _bounds_center(obj_bounds)
                center_l1 = None
                if pred_center is not None and obj_center is not None:
                    spans = np.array([max(float(ub) - float(lb), 1e-12) for lb, ub in bounds], dtype=float)
                    center_l1 = float(np.mean(np.abs(pred_center - obj_center) / spans))
                pred_obj_center_l1_norm = center_l1
                conf_low = float(np.clip(float(strategy_params_local.get("pred_cluster_confidence_low", 0.35)), 0.0, 1.0))
                shift_hi = float(max(float(strategy_params_local.get("pred_refine_shift_high", 0.35)), 0.0))
                disjoint_iou = float(max(float(strategy_params_local.get("pred_obj_disjoint_iou", 0.05)), 0.0))
                if pred_bounds is None:
                    pred_obj_miss_case = "pred_bounds_empty"
                elif obj_bounds is None:
                    pred_obj_miss_case = "obj_bounds_unavailable"
                elif pred_obj_iou is not None and pred_obj_iou < disjoint_iou:
                    pred_obj_miss_case = "pred_obj_disjoint"
                elif center_l1 is not None and center_l1 >= 0.45:
                    pred_obj_miss_case = "pred_obj_center_shift"
                elif pred_cluster_confidence is not None and pred_cluster_confidence <= conf_low:
                    pred_obj_miss_case = "pred_low_confidence"
                elif pred_refine_shift_norm is not None and pred_refine_shift_norm >= shift_hi:
                    pred_obj_miss_case = "pred_refine_large_shift"
                else:
                    pred_obj_miss_case = "pred_obj_aligned_or_unknown"

                # D-A1: pred-obj disjoint fallback — iou≈0 이면 pred 와 obj 가 완전 다른 영역.
                # obj 는 평가 데이터 기반(ground truth), pred 는 모델 예측. dual 계열도 포함.
                disjoint_fallback_fired = False
                if (
                    strategy_alias in {"s4", "s8", "s4_pred", "s8_pred"}
                    and pred_obj_miss_case == "pred_obj_disjoint"
                    and obj_bounds is not None
                    and selected_bounds is not None
                ):
                    pc_str = (
                        f"{pred_cluster_confidence:.3f}"
                        if pred_cluster_confidence is not None
                        else "None"
                    )
                    print(
                        f"[Explorer] pred-obj disjoint fallback: pred_conf={pc_str}, "
                        f"iou={pred_obj_iou:.4f} → obj_bounds 사용"
                    )
                    selected_bounds = list(obj_bounds)
                    if X_obj_sel is not None and X_obj_sel.shape[0] > 0:
                        selected_points = _safe_stack([X_obj_sel], n_dim=len(selected_features))
                    disjoint_fallback_fired = True

                # D-A2: pred_cluster_confidence danger-zone fallback.
                # 과적합 confidence calibration 실패 구간(pc ∈ [danger_low, danger_high]) 에서
                # iou 도 낮으면 pred 예측을 신뢰할 수 없음. obj_bounds 로 전환.
                if (
                    not disjoint_fallback_fired
                    and strategy_alias in {"s4", "s8", "s4_pred", "s8_pred"}
                    and pred_cluster_confidence is not None
                    and obj_bounds is not None
                    and selected_bounds is not None
                    and pred_obj_iou is not None
                ):
                    _pc_danger_lo = float(np.clip(
                        float(strategy_params_local.get("pred_conf_danger_low", 0.40)),
                        0.0, 1.0,
                    ))
                    _pc_danger_hi = float(np.clip(
                        float(strategy_params_local.get("pred_conf_danger_high", 0.60)),
                        0.0, 1.0,
                    ))
                    _pc_danger_iou_max = float(max(
                        float(strategy_params_local.get("pred_conf_danger_iou_max", 0.15)),
                        0.0,
                    ))
                    if (
                        _pc_danger_lo <= float(pred_cluster_confidence) < _pc_danger_hi
                        and float(pred_obj_iou) < _pc_danger_iou_max
                    ):
                        print(
                            f"[Explorer] pred-conf danger-zone fallback: pred_conf="
                            f"{float(pred_cluster_confidence):.3f} ∈ "
                            f"[{_pc_danger_lo:.2f},{_pc_danger_hi:.2f}), "
                            f"iou={float(pred_obj_iou):.4f} → obj_bounds 사용"
                        )
                        selected_bounds = list(obj_bounds)
                        if X_obj_sel is not None and X_obj_sel.shape[0] > 0:
                            selected_points = _safe_stack([X_obj_sel], n_dim=len(selected_features))

                # DSE-A: high-confidence misleading pred fallback —
                # pred_conf이 높지만(>=0.45) iou가 매우 낮으면(<0.10) EI 표면이
                # 잘못된 클러스터로 과도한 확신을 갖는 패턴. pred 중심을 obj 중심 쪽으로
                # blend하여 over_shrink 완화. (width은 pred 유지)
                _dsea_conf_high = float(np.clip(
                    float(strategy_params_local.get("pred_obj_fallback_conf_high", 0.45)),
                    0.0, 1.0,
                ))
                _dsea_iou_low = float(max(
                    float(strategy_params_local.get("pred_obj_fallback_iou_low", 0.10)),
                    0.0,
                ))
                _dsea_blend_base = float(np.clip(
                    float(strategy_params_local.get("pred_obj_fallback_center_blend", 0.5)),
                    0.0, 1.0,
                ))
                # EXP-2: 고제약 시 blend 상향 (pred center를 obj 쪽으로 더 강하게 끌어당김)
                _exp2_blend_high_crate = float(np.clip(
                    float(strategy_params_local.get("pred_obj_fallback_center_blend_high_crate", 0.75)),
                    0.0, 1.0,
                ))
                _exp2_crate_thr = float(strategy_params_local.get("pred_obj_fallback_crate_thr", 0.85))
                try:
                    _exp2_crate_hat = float(_meta_get("constraint_rate_hat", 1.0))
                except Exception:
                    _exp2_crate_hat = 1.0
                _exp2_is_high_crate = bool(
                    has_pre_constraints
                    and np.isfinite(_exp2_crate_hat)
                    and _exp2_crate_hat < _exp2_crate_thr
                )
                _dsea_blend = _exp2_blend_high_crate if _exp2_is_high_crate else _dsea_blend_base
                if (
                    strategy_alias in {"s4", "s8", "s4_pred", "s8_pred"}
                    and pred_cluster_confidence is not None
                    and pred_cluster_confidence >= _dsea_conf_high
                    and pred_obj_iou is not None
                    and pred_obj_iou < _dsea_iou_low
                    and obj_center is not None
                    and pred_center is not None
                    and selected_bounds is not None
                    and not (
                        pred_obj_miss_case == "pred_obj_disjoint"
                        and pred_cluster_confidence < conf_low
                    )
                ):
                    target_center = (1.0 - _dsea_blend) * pred_center + _dsea_blend * obj_center
                    recentred = _recenter_bounds_keep_width(
                        base_bounds=selected_bounds,
                        global_bounds=bounds,
                        target_center=target_center,
                    )
                    if recentred is not None:
                        print(
                            f"[Explorer] DSE-A pred_obj fallback blend: "
                            f"conf={pred_cluster_confidence:.3f} iou={pred_obj_iou:.4f} "
                            f"blend={_dsea_blend:.2f} "
                            f"({'high-crate' if _exp2_is_high_crate else 'baseline'}) "
                            f"→ center shifted toward obj"
                        )
                        selected_bounds = recentred

        if selected_bounds is None:
            if strategy_alias in {"s4_pred", "s8_pred"} and pred_bounds is not None:
                selected_bounds = pred_bounds
                selected_points = _safe_stack([selected_points, X_pred_sel], n_dim=len(selected_features))
            elif strategy_alias in {"s4_obj", "s8_obj"} and obj_bounds is not None:
                selected_bounds = obj_bounds
                selected_points = _safe_stack([selected_points, X_obj_sel], n_dim=len(selected_features))
            elif strategy_alias in {"s4", "s8"}:
                raise RuntimeError("Dual refinement produced no selected bounds.")
            elif strategy_alias in {"s4_obj", "s8_obj"}:
                raise RuntimeError("Obj refinement produced no selected bounds.")

        bounds_path = None
        vol_ratio = None
        dual_volume_cap_target = None
        dual_volume_ratio_before_cap = None
        # G: 진단 필드 — cap 직전 selected_bounds volume, pred/obj cluster volume,
        # selected centroid의 design 중심 대비 정규화 L1 distance.
        selected_bounds_volume_ratio_before_cap: float | None = None
        pred_bounds_volume_ratio: float | None = None
        obj_bounds_volume_ratio: float | None = None
        selected_bounds_center_l1_to_design_center: float | None = None
        dual_volume_cap_applied = False
        dse_cap_boundary_touch_dims: list[dict[str, object]] = []
        dse_cap_boundary_touch_eps_ratio: float | None = None
        dse_cap_center_realign_applied = False
        dse_cap_center_blend_used: float | None = None
        dse_cap_force_pin_applied = False
        constraint_aware_enabled = bool(getattr(self.config.system, "constraint_aware_enabled", True))
        constraint_aware_policy_applied = False
        constraint_aware_shifted_dims: list[tuple[int, str]] = []
        constraint_aware_probe_count: int | None = None
        constraint_aware_anchor_count: int | None = None
        constraint_aware_side_scores: list[dict] | None = None
        constraint_aware_policy: dict | None = None
        constraint_aware_top_obj_veto_count: int = 0
        constraint_aware_top_obj_confirmed_count: int = 0
        constraint_aware_obj_centroid_blend_applied: bool = False
        constraint_aware_shift_boost_applied: bool = False
        constraint_aware_shift_fraction_used: float | None = None

        if selected_bounds is not None:
            if (
                constraint_aware_enabled
                and has_pre_constraints
                and isinstance(selected_points, np.ndarray)
                and selected_points.ndim == 2
                and selected_points.shape[0] > 0
            ):
                # L1: anchor objective 값(surrogate 예측)을 제공해 top-obj 기반 override 활성화
                _anchor_obj_vals: np.ndarray | None = None
                try:
                    if models is not None and len(models) > 0:
                        _mu_sel, _ = _predict_ensemble(models, selected_points)
                        _anchor_obj_vals = np.asarray(_mu_sel, dtype=float).reshape(-1)
                except Exception:
                    _anchor_obj_vals = None

                constraint_aware_policy = _infer_constraint_side_policy(
                    selected_points=selected_points,
                    selected_features=selected_features,
                    global_bounds=bounds,
                    pre_constraint_defs=pre_constraint_defs,
                    edge_ratio=float(getattr(self.config.system, "constraint_aware_edge_ratio", 0.05)),
                    probe_anchor_max=int(getattr(self.config.system, "constraint_aware_probe_anchor_max", 16)),
                    probe_steps=int(getattr(self.config.system, "constraint_aware_probe_steps", 3)),
                    min_side_samples=int(getattr(self.config.system, "constraint_aware_min_side_samples", 24)),
                    side_rate_min=float(getattr(self.config.system, "constraint_aware_side_rate_min", 0.55)),
                    side_rate_gap=float(getattr(self.config.system, "constraint_aware_side_rate_gap", 0.12)),
                    anchor_obj_values=_anchor_obj_vals,
                    objective_sense=str(objective_sense),
                    anti_anti_threshold=int(
                        getattr(self.config.system, "constraint_aware_anti_anti_threshold", 1)
                    ),
                    require_top_obj_support=bool(
                        getattr(
                            self.config.system,
                            "constraint_aware_require_top_obj_support",
                            False,
                        )
                    ),
                    top_obj_min_ratio=float(
                        getattr(
                            self.config.system,
                            "constraint_aware_top_obj_min_ratio",
                            0.10,
                        )
                    ),
                )
                if isinstance(constraint_aware_policy, dict):
                    constraint_aware_probe_count = int(constraint_aware_policy.get("probe_count", 0))
                    constraint_aware_anchor_count = int(constraint_aware_policy.get("anchor_count", 0))
                    constraint_aware_side_scores = constraint_aware_policy.get("score_rows", [])
                    _pin_lb = [bool(v) for v in constraint_aware_policy.get("protect_lb", [])]
                    _pin_ub = [bool(v) for v in constraint_aware_policy.get("protect_ub", [])]
                    constraint_aware_policy_applied = bool(any(_pin_lb) or any(_pin_ub))
                    if (
                        constraint_aware_policy_applied
                        and bool(getattr(self.config.system, "constraint_aware_pre_shift_enabled", True))
                    ):
                        # L4: 상태 기반 보수 분기 — 제약·저차원·저 feasibility에서 shift 약화
                        _p_dim_now = int(len(selected_features))
                        try:
                            _usable_n_now = float(_meta_get("usable_n", 0) or 0)
                        except Exception:
                            _usable_n_now = 0.0
                        _np_ratio_now = (
                            float(_usable_n_now) / float(max(_p_dim_now, 1))
                            if _usable_n_now and _p_dim_now else 0.0
                        )
                        _feas_ratio = float(constraint_aware_policy.get("feasible_ratio", 1.0))
                        _p_dim_max = int(
                            getattr(self.config.system, "constraint_aware_conservative_p_dim_max", 5)
                        )
                        _np_ratio_max = float(
                            getattr(self.config.system, "constraint_aware_conservative_np_ratio_max", 25.0)
                        )
                        _feas_max = float(
                            getattr(self.config.system, "constraint_aware_conservative_feasible_ratio_max", 0.4)
                        )
                        _is_conservative = bool(
                            has_pre_constraints
                            and _p_dim_now <= _p_dim_max
                            and (_np_ratio_now == 0.0 or _np_ratio_now <= _np_ratio_max)
                            and _feas_ratio <= _feas_max
                        )
                        if _is_conservative:
                            _shift_frac = float(
                                getattr(
                                    self.config.system,
                                    "constraint_aware_conservative_shift_fraction",
                                    0.0,
                                )
                            )
                        else:
                            _shift_frac = float(
                                getattr(self.config.system, "constraint_aware_shift_fraction", 0.5)
                            )
                        # L7-B: top-obj가 모든 protect 방향을 confirm하고 veto가 0개면
                        # 보수 fraction을 boost. policy의 top_obj_confirmed_count/veto_count
                        # 은 protect 신호가 살아있는 dim 기준으로 산정됨.
                        _ca_veto_count = int(constraint_aware_policy.get("top_obj_veto_count", 0) or 0)
                        _ca_confirmed_count = int(
                            constraint_aware_policy.get("top_obj_confirmed_count", 0) or 0
                        )
                        constraint_aware_top_obj_veto_count = _ca_veto_count
                        constraint_aware_top_obj_confirmed_count = _ca_confirmed_count
                        _shift_boost_applied = False
                        if (
                            _ca_veto_count == 0
                            and _ca_confirmed_count >= 1
                        ):
                            # #D 단계화 + #L3 all-confirmed: 3-tier boost.
                            # confirmed == d (전 dim 정합) → 1.0 (#L3, full shift)
                            # confirmed >= ceil(d/2) → 0.85 (강한 다수결, #D)
                            # confirmed >= 1 → 0.70 (약한 신호, 현행)
                            _d_dim = max(int(_p_dim_now), 1)
                            _strong_threshold = int(np.ceil(_d_dim / 2))
                            _strong_threshold = max(1, _strong_threshold)
                            if _ca_confirmed_count >= _d_dim and _d_dim > 0:
                                _boost_frac = float(
                                    getattr(
                                        self.config.system,
                                        "constraint_aware_obj_all_confirmed_shift_fraction",
                                        1.0,
                                    )
                                )
                            elif _ca_confirmed_count >= _strong_threshold:
                                _boost_frac = float(
                                    getattr(
                                        self.config.system,
                                        "constraint_aware_obj_strong_confirmed_shift_fraction",
                                        0.85,
                                    )
                                )
                            else:
                                _boost_frac = float(
                                    getattr(
                                        self.config.system,
                                        "constraint_aware_obj_confirmed_shift_fraction",
                                        0.70,
                                    )
                                )
                            if _boost_frac > _shift_frac:
                                _shift_frac = float(np.clip(_boost_frac, 0.0, 1.0))
                                _shift_boost_applied = True
                        constraint_aware_shift_boost_applied = bool(_shift_boost_applied)
                        constraint_aware_shift_fraction_used = float(_shift_frac)
                        print(
                            f"[Explorer] constraint-aware shift: frac={_shift_frac:.2f} "
                            f"(p_dim={_p_dim_now}, np_ratio={_np_ratio_now:.1f}, "
                            f"feas_ratio={_feas_ratio:.2f}, conservative={_is_conservative}, "
                            f"obj_confirmed={_ca_confirmed_count}, obj_veto={_ca_veto_count}, "
                            f"boost={_shift_boost_applied})"
                        )
                        selected_bounds_shifted, _shifted = _apply_constraint_side_shift(
                            selected_bounds=selected_bounds,
                            global_bounds=bounds,
                            protect_lb=_pin_lb,
                            protect_ub=_pin_ub,
                            shift_fraction=_shift_frac,
                        )
                        if selected_bounds_shifted is not None:
                            selected_bounds = selected_bounds_shifted
                            constraint_aware_shifted_dims = list(_shifted)

                        # L7-A: top-obj 신호가 interior(어떤 edge에도 모이지 않음)이고 veto가
                        # 발동해 protect 신호가 모두 무효화된 경우, selected_bounds 중심을
                        # top-K obj centroid 쪽으로 부분 blend (width 보존). 실패 패턴
                        # 분석에서 (veto>0, shifted=0) 케이스의 over_shrink_fail 회복용.
                        _ca_top_centroid = constraint_aware_policy.get("top_obj_centroid", None)
                        if (
                            _ca_veto_count >= 1
                            and not any(_pin_lb)
                            and not any(_pin_ub)
                            and isinstance(_ca_top_centroid, list)
                            and len(_ca_top_centroid) == len(bounds)
                            and selected_bounds is not None
                        ):
                            _blend = float(
                                getattr(
                                    self.config.system,
                                    "constraint_aware_obj_centroid_blend",
                                    0.30,
                                )
                            )
                            _blend = float(np.clip(_blend, 0.0, 1.0))
                            if _blend > 0.0:
                                # 현재 selected_bounds 중심
                                _cur_center = np.array(
                                    [0.5 * (float(lb) + float(ub)) for lb, ub in selected_bounds],
                                    dtype=float,
                                )
                                _tgt_center = np.array(_ca_top_centroid, dtype=float)
                                _blend_target = (
                                    (1.0 - _blend) * _cur_center + _blend * _tgt_center
                                )
                                _recentred = _recenter_bounds_keep_width(
                                    base_bounds=selected_bounds,
                                    global_bounds=bounds,
                                    target_center=_blend_target,
                                )
                                if _recentred is not None:
                                    selected_bounds = list(_recentred)
                                    constraint_aware_obj_centroid_blend_applied = True
                                    print(
                                        f"[Explorer] L7-A obj-centroid blend: blend={_blend:.2f}, "
                                        f"veto_count={_ca_veto_count}"
                                    )
                    if constraint_aware_policy_applied:
                        _dim_msgs = []
                        for _j, _name in enumerate(selected_features):
                            _lb = bool(_pin_lb[_j]) if _j < len(_pin_lb) else False
                            _ub = bool(_pin_ub[_j]) if _j < len(_pin_ub) else False
                            if _lb or _ub:
                                _dim_msgs.append(f"{_name}(lb={_lb},ub={_ub})")
                        if _dim_msgs:
                            print(
                                "[Explorer] constraint-aware side policy: "
                                + ", ".join(_dim_msgs)
                            )

            expansion_mode = str(getattr(self.config.system, "bounds_expansion_mode", "uniform")).strip().lower()
            # routed_v2 override: 라우터가 expansion_mode를 결정한 경우 우선 적용
            if routed_v2_decision is not None:
                expansion_mode = str(routed_v2_decision.expansion_mode).strip().lower()
                print(
                    f"[Explorer] routed_v2 override expansion_mode={expansion_mode}"
                )
            # EXP-3: 고제약 시 fi_aware → uncertainty_aware 자동 스위치 (state-based)
            try:
                _exp3_crate_hat = float(_meta_get("constraint_rate_hat", 1.0))
            except Exception:
                _exp3_crate_hat = 1.0
            _exp3_crate_thr = float(
                getattr(self.config.system, "bounds_expansion_high_crate_threshold", 0.85)
            )
            _exp3_enabled = bool(
                getattr(self.config.system, "bounds_expansion_high_crate_switch_enabled", True)
            )
            if (
                _exp3_enabled
                and has_pre_constraints
                and np.isfinite(_exp3_crate_hat)
                and _exp3_crate_hat < _exp3_crate_thr
                and expansion_mode == "fi_aware"
            ):
                print(
                    f"[Explorer] EXP-3 high-crate switch: bounds_expansion_mode "
                    f"fi_aware → uncertainty_aware (crate={_exp3_crate_hat:.3f} < {_exp3_crate_thr})"
                )
                expansion_mode = "uncertainty_aware"
            _w_clip_min = float(getattr(self.config.system, "bounds_weight_clip_min", 0.5))
            _w_clip_max = float(getattr(self.config.system, "bounds_weight_clip_max", 2.0))
            dim_weights = None

            if expansion_mode == "fi_aware":
                # FI score 기반 가중치 시도
                fi_scores = None
                if fi_scores_path and os.path.exists(fi_scores_path):
                    try:
                        _fi_df = pd.read_csv(fi_scores_path)
                        if "feature" in _fi_df.columns and "final_score_adj" in _fi_df.columns:
                            fi_scores = dict(zip(
                                _fi_df["feature"].astype(str),
                                _fi_df["final_score_adj"].astype(float),
                            ))
                    except Exception as exc:
                        print(f"[Explorer] FI scores 로드 실패: {exc}")

                if fi_scores:
                    dim_weights = compute_fi_importance_weights(
                        fi_scores=fi_scores,
                        selected_features=selected_features,
                        clip_min=_w_clip_min,
                        clip_max=_w_clip_max,
                    )
                if dim_weights is not None:
                    parts = ", ".join(
                        f"{fn}:{w:.2f}" for fn, w in zip(selected_features, dim_weights)
                    )
                    print(f"[Explorer] bounds_expansion_mode=fi_aware: {parts}")
                else:
                    print("[Explorer] fi_aware 모드에서 유효한 FI score 없음 → uniform fallback")
            elif expansion_mode == "uncertainty_aware":
                dim_weights = compute_gp_boundary_uncertainty(
                    gp_models=[gp_pred, gp_obj],
                    selected_bounds=selected_bounds,
                    clip_min=_w_clip_min,
                    clip_max=_w_clip_max,
                )
                if dim_weights is not None:
                    parts = ", ".join(
                        f"{fn}:{w:.2f}" for fn, w in zip(selected_features, dim_weights)
                    )
                    print(f"[Explorer] bounds_expansion_mode=uncertainty_aware: {parts}")
                else:
                    print("[Explorer] uncertainty_aware 모드에서 유효한 GP uncertainty 없음 → uniform fallback")
            elif expansion_mode == "uniform":
                print("[Explorer] bounds_expansion_mode=uniform")
            else:
                print(
                    f"[Explorer] unknown bounds_expansion_mode='{expansion_mode}' "
                    "→ uniform fallback"
                )

            # EXP-1: selected_points의 중심을 center_hint로 전달하여 비대칭 확장
            _center_hint_arr = None
            if isinstance(selected_points, np.ndarray) and selected_points.ndim == 2 and selected_points.shape[0] > 0:
                _center_hint_arr = np.median(selected_points, axis=0)

            # D4: confidence-adaptive margin — pred confidence가 낮으면 margin 확대
            _margin_ratio = float(self.config.system.bounds_margin_ratio)
            if pred_cluster_confidence is not None and pred_cluster_confidence < 0.4:
                _margin_ratio = max(_margin_ratio, 0.06)

            # DSE-2: dual disagreement이 매우 낮을 때(pred·obj 클러스터 거의 분리)
            # selected_bounds를 pred ∪ obj로 확장. 후속 vol cap이 boundary-pin과 함께
            # 두 region의 corner를 보존할 수 있도록 함. dual 모드(s4/s8)에 한정.
            _dse2_strategy_params = dict(self.config.system.strategy_params or {})
            _dse2_iou_threshold = float(_dse2_strategy_params.get(
                "dse2_low_iou_union_threshold", 0.20
            ))
            # routed_v2 override: 라우터가 더 보수적/공격적 임계값 결정 가능
            if routed_v2_decision is not None:
                _dse2_iou_threshold = float(routed_v2_decision.dse2_iou_threshold)
            if (
                strategy_alias in {"s4", "s8"}
                and pred_bounds is not None
                and obj_bounds is not None
                and dual_disagreement_iou is not None
                and np.isfinite(float(dual_disagreement_iou))
                and float(dual_disagreement_iou) < _dse2_iou_threshold
                and selected_bounds is not None
                and len(selected_bounds) == len(bounds)
                and len(pred_bounds) == len(bounds)
                and len(obj_bounds) == len(bounds)
            ):
                _union_bounds: list[tuple[float, float]] = []
                for (s_lb, s_ub), (p_lb, p_ub), (o_lb, o_ub), (g_lb, g_ub) in zip(
                    selected_bounds, pred_bounds, obj_bounds, bounds
                ):
                    _u_lb = float(max(min(s_lb, p_lb, o_lb), g_lb))
                    _u_ub = float(min(max(s_ub, p_ub, o_ub), g_ub))
                    if _u_ub < _u_lb:
                        _u_lb, _u_ub = _u_ub, _u_lb
                    _union_bounds.append((_u_lb, _u_ub))
                selected_bounds = _union_bounds
                print(
                    f"[Explorer] DSE-2 low-IoU union: iou={float(dual_disagreement_iou):.4f} "
                    f"< {_dse2_iou_threshold:.2f} → selected_bounds = pred ∪ obj "
                    f"(boundary-pin-aware cap 적용 예정)"
                )

            selected_bounds = apply_bounds_margin(
                selected_bounds=selected_bounds,
                bounds=bounds,
                margin_ratio=_margin_ratio,
                min_volume_ratio=float(self.config.system.bounds_min_volume_ratio),
                dim_weights=dim_weights,
                center_hint=_center_hint_arr,
            )

            # G 진단: cap 진입 직전 selected_bounds volume ratio (전 전략 공통)
            try:
                selected_bounds_volume_ratio_before_cap = _volume_ratio_for_bounds(
                    selected_bounds, bounds
                )
            except Exception:
                selected_bounds_volume_ratio_before_cap = None

            # #L2 Pre-cap micro-expansion — has_pre==True AND p_dim>=4 AND
            # pre_cap_ratio < threshold 시, axis-aligned 균일 확장으로 target 부피비
            # 도달. cap이 후속으로 0.25로 줄여주므로 over-wide 위험 0. center 보존.
            # state-based gate, benchmark 명 분기 아님.
            _l2_enabled = bool(
                getattr(self.config.system, "bounds_pre_cap_expand_enabled", True)
            )
            _l2_p_dim_min = int(
                getattr(self.config.system, "bounds_pre_cap_expand_p_dim_min", 4)
            )
            _l2_threshold = float(
                getattr(self.config.system, "bounds_pre_cap_expand_threshold", 0.10)
            )
            _l2_target = float(
                getattr(self.config.system, "bounds_pre_cap_expand_target_ratio", 0.30)
            )
            if (
                _l2_enabled
                and has_pre_constraints
                and len(selected_features) >= _l2_p_dim_min
                and selected_bounds is not None
                and selected_bounds_volume_ratio_before_cap is not None
                and 0.0 < float(selected_bounds_volume_ratio_before_cap) < _l2_threshold
                and 0.0 < _l2_target <= 1.0
            ):
                _cur_ratio = float(selected_bounds_volume_ratio_before_cap)
                _d_dim_l2 = max(int(len(selected_features)), 1)
                # per-dim factor = (target/current)^(1/d)
                _factor = float((_l2_target / max(_cur_ratio, 1e-12)) ** (1.0 / _d_dim_l2))
                _expanded: list[tuple[float, float]] = []
                for (s_lb, s_ub), (g_lb, g_ub) in zip(selected_bounds, bounds):
                    gl = float(min(g_lb, g_ub))
                    gu = float(max(g_lb, g_ub))
                    lo = float(np.clip(min(s_lb, s_ub), gl, gu))
                    hi = float(np.clip(max(s_lb, s_ub), gl, gu))
                    width = max(hi - lo, 0.0)
                    cc = 0.5 * (lo + hi)
                    new_half = 0.5 * width * _factor
                    new_lo = float(np.clip(cc - new_half, gl, gu))
                    new_hi = float(np.clip(cc + new_half, gl, gu))
                    if new_hi < new_lo:
                        new_lo, new_hi = new_hi, new_lo
                    _expanded.append((new_lo, new_hi))
                print(
                    f"[Explorer] #L2 pre-cap expand: pre_cap_ratio "
                    f"{_cur_ratio:.4f} → ~{_l2_target:.2f} (factor {_factor:.3f})"
                )
                selected_bounds = _expanded
                # update before_cap diagnostic to reflect post-expansion value
                try:
                    selected_bounds_volume_ratio_before_cap = _volume_ratio_for_bounds(
                        selected_bounds, bounds
                    )
                except Exception:
                    pass

            # Hard cap policy:
            # - remove all relaxed paths (e.g., 0.30)
            # - ignore strategy/routed overrides
            # - keep fixed target at 24.99%
            dual_cap_raw = 0.2499
            if strategy_alias in {"s4", "s8", "s4_pred", "s8_pred", "s4_obj", "s8_obj"}:  # K: pred/obj 전략에도 volume cap 적용
                try:
                    dual_volume_cap_target = float(dual_cap_raw)
                except Exception:
                    dual_volume_cap_target = None
                if (
                    dual_volume_cap_target is not None
                    and np.isfinite(dual_volume_cap_target)
                    and dual_volume_cap_target > 0.0
                ):
                    dual_volume_ratio_before_cap = _volume_ratio_for_bounds(selected_bounds, bounds)
                    if (
                        dual_volume_ratio_before_cap is not None
                        and np.isfinite(dual_volume_ratio_before_cap)
                        and dual_volume_ratio_before_cap > dual_volume_cap_target
                    ):
                        center_hint = None
                        # EXP-2: pred/obj bounds의 intersection 중심을 우선 사용
                        # intersection 중심이 optimum에 더 가까울 확률이 높음
                        if pred_bounds is not None and obj_bounds is not None:
                            _isect_center = []
                            _isect_valid = True
                            for (p_lb, p_ub), (o_lb, o_ub) in zip(pred_bounds, obj_bounds):
                                i_lb = max(float(p_lb), float(o_lb))
                                i_ub = min(float(p_ub), float(o_ub))
                                if i_lb <= i_ub:
                                    _isect_center.append((i_lb + i_ub) * 0.5)
                                else:
                                    _isect_valid = False
                                    break
                            if _isect_valid and _isect_center:
                                center_hint = np.array(_isect_center, dtype=float)
                        if center_hint is None and isinstance(dual_center_hint, np.ndarray) and dual_center_hint.size == len(bounds):
                            center_hint = np.asarray(dual_center_hint, dtype=float).reshape(-1)
                        elif center_hint is None and isinstance(selected_points, np.ndarray) and selected_points.ndim == 2 and selected_points.shape[0] > 0:
                            center_hint = np.mean(selected_points, axis=0)

                        _force_pin_lo_local: list[bool] | None = None
                        _force_pin_hi_local: list[bool] | None = None
                        _pref_shrink_side_local: list[str | None] | None = None
                        _use_constraint_cap_policy = bool(
                            constraint_aware_enabled
                            and bool(getattr(self.config.system, "constraint_aware_use_for_volume_cap", True))
                            and isinstance(constraint_aware_policy, dict)
                        )
                        if _use_constraint_cap_policy:
                            _raw_lb = list(constraint_aware_policy.get("protect_lb", []))
                            _raw_ub = list(constraint_aware_policy.get("protect_ub", []))
                            _raw_pref = list(constraint_aware_policy.get("prefer_shrink_side", []))
                            _force_pin_lo_local = [
                                bool(_raw_lb[j]) if j < len(_raw_lb) else False
                                for j in range(len(bounds))
                            ]
                            _force_pin_hi_local = [
                                bool(_raw_ub[j]) if j < len(_raw_ub) else False
                                for j in range(len(bounds))
                            ]
                            _pref_shrink_side_local = []
                            for j in range(len(bounds)):
                                _pv = _raw_pref[j] if j < len(_raw_pref) else None
                                _ps = str(_pv).strip().lower() if _pv is not None else ""
                                _pref_shrink_side_local.append(_ps if _ps in {"lb", "ub"} else None)

                        # DSE-A: boundary-touch aware cap center/pin 강화
                        _dse_cap_params = dict(self.config.system.strategy_params or {})
                        _dse_cap_enabled = bool(_dse_cap_params.get("dse_cap_boundary_touch_enabled", True))
                        _dse_cap_eps_ratio = float(np.clip(
                            float(_dse_cap_params.get(
                                "dse_cap_boundary_touch_eps_ratio",
                                _dse_cap_params.get("dse1_boundary_touch_eps_ratio", 0.03),
                            )),
                            0.0,
                            0.2,
                        ))
                        _dse_cap_min_excess = float(max(float(
                            _dse_cap_params.get("dse_cap_boundary_touch_min_excess", 0.008)
                        ), 0.0))
                        _dse_cap_center_blend = float(np.clip(
                            float(_dse_cap_params.get("dse_cap_center_blend", 0.60)),
                            0.0,
                            1.0,
                        ))
                        dse_cap_boundary_touch_eps_ratio = _dse_cap_eps_ratio
                        _cap_excess = float(dual_volume_ratio_before_cap - dual_volume_cap_target)
                        _touch_lb, _touch_ub = _infer_boundary_touch_sides(
                            selected_points=selected_points,
                            global_bounds=bounds,
                            eps_ratio=_dse_cap_eps_ratio,
                        )
                        _touch_dims: list[tuple[int, str]] = []

                        if (
                            _dse_cap_enabled
                            and _cap_excess >= _dse_cap_min_excess
                            and len(selected_bounds) == len(bounds)
                        ):
                            if _force_pin_lo_local is None:
                                _force_pin_lo_local = [False] * len(bounds)
                            if _force_pin_hi_local is None:
                                _force_pin_hi_local = [False] * len(bounds)
                            if _pref_shrink_side_local is None:
                                _pref_shrink_side_local = [None] * len(bounds)

                            _boundary_center: list[float] = []
                            _single_side_touch = False
                            for j, ((s_lb, s_ub), (g_lb, g_ub)) in enumerate(zip(selected_bounds, bounds)):
                                gl = float(min(g_lb, g_ub))
                                gu = float(max(g_lb, g_ub))
                                lo = float(np.clip(min(s_lb, s_ub), gl, gu))
                                hi = float(np.clip(max(s_lb, s_ub), gl, gu))
                                width = min(max(hi - lo, 0.0), max(gu - gl, 0.0))
                                _lb_only = bool(_touch_lb[j] and not _touch_ub[j])
                                _ub_only = bool(_touch_ub[j] and not _touch_lb[j])
                                if _lb_only:
                                    _single_side_touch = True
                                    _touch_dims.append((j, "lb"))
                                    _force_pin_lo_local[j] = True
                                    if _pref_shrink_side_local[j] is None:
                                        _pref_shrink_side_local[j] = "ub"
                                elif _ub_only:
                                    _single_side_touch = True
                                    _touch_dims.append((j, "ub"))
                                    _force_pin_hi_local[j] = True
                                    if _pref_shrink_side_local[j] is None:
                                        _pref_shrink_side_local[j] = "lb"

                                if _lb_only:
                                    _cc = float(np.clip(gl + 0.5 * width, gl, gu))
                                elif _ub_only:
                                    _cc = float(np.clip(gu - 0.5 * width, gl, gu))
                                elif (
                                    isinstance(center_hint, np.ndarray)
                                    and center_hint.size == len(bounds)
                                    and np.isfinite(float(center_hint[j]))
                                ):
                                    _cc = float(np.clip(float(center_hint[j]), gl, gu))
                                else:
                                    _cc = float(np.clip(0.5 * (lo + hi), gl, gu))
                                _boundary_center.append(_cc)

                            if _touch_dims:
                                dse_cap_boundary_touch_dims = [
                                    {
                                        "dim": int(j),
                                        "feature": (
                                            str(selected_features[j])
                                            if j < len(selected_features)
                                            else str(j)
                                        ),
                                        "side": side,
                                    }
                                    for j, side in _touch_dims
                                ]

                            if _single_side_touch and _boundary_center:
                                _boundary_center_arr = np.asarray(_boundary_center, dtype=float).reshape(-1)
                                if (
                                    not isinstance(center_hint, np.ndarray)
                                    or center_hint.size != len(bounds)
                                ):
                                    center_hint = _boundary_center_arr
                                    dse_cap_center_realign_applied = True
                                    dse_cap_center_blend_used = 1.0
                                else:
                                    center_hint = (
                                        (1.0 - _dse_cap_center_blend) * np.asarray(center_hint, dtype=float).reshape(-1)
                                        + _dse_cap_center_blend * _boundary_center_arr
                                    )
                                    dse_cap_center_realign_applied = bool(_dse_cap_center_blend > 0.0)
                                    dse_cap_center_blend_used = _dse_cap_center_blend
                                center_hint = np.asarray(center_hint, dtype=float).reshape(-1)
                                for j, (g_lb, g_ub) in enumerate(bounds):
                                    gl = float(min(g_lb, g_ub))
                                    gu = float(max(g_lb, g_ub))
                                    center_hint[j] = float(np.clip(center_hint[j], gl, gu))
                                print(
                                    "[Explorer] DSE-A cap center realign: "
                                    f"touch_dims={len(_touch_dims)} blend={float(dse_cap_center_blend_used or 0.0):.2f}"
                                )

                        dse_cap_force_pin_applied = bool(
                            isinstance(_force_pin_lo_local, list)
                            and isinstance(_force_pin_hi_local, list)
                            and (any(_force_pin_lo_local) or any(_force_pin_hi_local))
                        )
                        if dse_cap_boundary_touch_dims:
                            print(
                                "[Explorer] DSE-A boundary-touch cap pins: "
                                f"{dse_cap_boundary_touch_dims}"
                            )

                        selected_bounds_cap = _shrink_bounds_to_target_volume(
                            selected_bounds=selected_bounds,
                            global_bounds=bounds,
                            target_volume_ratio=dual_volume_cap_target,
                            center=center_hint,
                            boundary_pin_tol_ratio=float(
                                getattr(self.config.system, "volume_cap_boundary_pin_tol_ratio", 0.03)
                            ),
                            force_pin_lo=_force_pin_lo_local,
                            force_pin_hi=_force_pin_hi_local,
                            preferred_shrink_side=_pref_shrink_side_local,
                        )
                        if selected_bounds_cap is not None:
                            vol_after_cap = _volume_ratio_for_bounds(selected_bounds_cap, bounds)
                            if (
                                vol_after_cap is not None
                                and np.isfinite(vol_after_cap)
                                and vol_after_cap < dual_volume_ratio_before_cap
                            ):
                                selected_bounds = selected_bounds_cap
                                dual_volume_cap_applied = True
                                print(
                                    "[Explorer] dual volume cap applied: "
                                    f"{dual_volume_ratio_before_cap:.4f} -> {vol_after_cap:.4f} "
                                    f"(target={dual_volume_cap_target:.4f})"
                                )

        # DSE-1: boundary-touch shift — selected_points에 design boundary 근접 점이
        # 있는데 post-cap selected_bounds가 해당 boundary에서 떨어져 있으면, width를
        # 유지한 채 그 boundary 쪽으로 shift. volume 변화 없음.
        # 일반화 규칙(state-based): 어떤 차원에서 cluster 점들의 max(또는 min)가
        # design ub(lb)의 ε 이내이지만 현재 bounds가 그 boundary에 닿지 않으면 shift.
        if (
            selected_bounds is not None
            and isinstance(selected_points, np.ndarray)
            and selected_points.ndim == 2
            and selected_points.shape[0] > 0
            and len(selected_bounds) == len(bounds)
            and selected_points.shape[1] == len(bounds)
        ):
            _dse1_strategy_params = dict(self.config.system.strategy_params or {})
            _dse1_eps_ratio = float(_dse1_strategy_params.get(
                "dse1_boundary_touch_eps_ratio", 0.03
            ))
            _dse1_shifted: list[tuple[int, str]] = []
            _dse1_new_bounds: list[tuple[float, float]] = []
            for j, ((s_lb, s_ub), (g_lb, g_ub)) in enumerate(
                zip(selected_bounds, bounds)
            ):
                span_g = max(float(g_ub) - float(g_lb), 1e-12)
                eps_g = float(_dse1_eps_ratio) * span_g
                col = np.asarray(selected_points[:, j], dtype=float)
                col_max = float(np.max(col))
                col_min = float(np.min(col))
                touches_ub = (float(g_ub) - col_max) <= eps_g
                touches_lb = (col_min - float(g_lb)) <= eps_g
                far_from_ub = (float(g_ub) - float(s_ub)) > eps_g
                far_from_lb = (float(s_lb) - float(g_lb)) > eps_g
                width = max(float(s_ub) - float(s_lb), 0.0)
                new_lb, new_ub = float(s_lb), float(s_ub)
                if touches_ub and far_from_ub and not (touches_lb and far_from_lb):
                    new_ub = float(g_ub)
                    new_lb = float(max(float(g_lb), new_ub - width))
                    _dse1_shifted.append((j, "ub"))
                elif touches_lb and far_from_lb and not (touches_ub and far_from_ub):
                    new_lb = float(g_lb)
                    new_ub = float(min(float(g_ub), new_lb + width))
                    _dse1_shifted.append((j, "lb"))
                _dse1_new_bounds.append((new_lb, new_ub))
            if _dse1_shifted:
                selected_bounds = _dse1_new_bounds
                print(
                    f"[Explorer] DSE-1 boundary-touch shift: dims={_dse1_shifted} "
                    f"(width-preserving)"
                )

        if selected_bounds is not None:
            vol_ratio = _volume_ratio_for_bounds(selected_bounds, bounds)
            if selected_features and len(selected_features) == len(selected_bounds):
                formatted = {
                    name: (lb, ub)
                    for name, (lb, ub) in zip(selected_features, selected_bounds)
                }
                print(f"[Explorer] selected_bounds ({strategy_alias}): {formatted}")
            else:
                print(f"[Explorer] selected_bounds ({strategy_alias}): {selected_bounds}")
            if vol_ratio is not None:
                print(f"[Explorer] selected_bounds volume_ratio={vol_ratio:.4f} ({vol_ratio*100:.2f}%)")

            # G 진단: pred/obj cluster volume + selected centroid의 design center 거리
            try:
                pred_bounds_volume_ratio = _volume_ratio_for_bounds(pred_bounds, bounds)
            except Exception:
                pred_bounds_volume_ratio = None
            try:
                obj_bounds_volume_ratio = _volume_ratio_for_bounds(obj_bounds, bounds)
            except Exception:
                obj_bounds_volume_ratio = None
            try:
                _sel_center = _bounds_center(selected_bounds)
                _design_center = _bounds_center(list(bounds))
                if (
                    _sel_center is not None
                    and _design_center is not None
                    and len(bounds) > 0
                ):
                    _spans = np.array(
                        [max(float(ub) - float(lb), 1e-12) for lb, ub in bounds],
                        dtype=float,
                    )
                    selected_bounds_center_l1_to_design_center = float(
                        np.mean(np.abs(_sel_center - _design_center) / _spans)
                    )
            except Exception:
                selected_bounds_center_l1_to_design_center = None
            # persist selected bounds as artifact (json)
            if selected_features and len(selected_features) == len(selected_bounds):
                bounds_payload = {
                    "selected_bounds": {
                        name: {"lb": float(lb), "ub": float(ub)}
                        for name, (lb, ub) in zip(selected_features, selected_bounds)
                    },
                    "volume_ratio": float(vol_ratio) if vol_ratio is not None else None,
                    "bounds_order": list(selected_features),
                    "strategy_alias": strategy_alias,
                }
            else:
                bounds_payload = {
                    "selected_bounds": [
                        {"lb": float(lb), "ub": float(ub)} for lb, ub in selected_bounds
                    ],
                    "volume_ratio": float(vol_ratio) if vol_ratio is not None else None,
                    "bounds_order": list(selected_features),
                    "strategy_alias": strategy_alias,
                }
            bounds_path = os.path.join(public_dir, "selected_bounds.json")
            with open(bounds_path, "w", encoding="utf-8") as f:
                json.dump(bounds_payload, f, indent=2)

        if save_plot:
            try:
                # Pairwise plots for all feature pairs
                if len(selected_features) >= 2 and X_pred_sel.size and X_obj_sel.size:
                    pairs = [(i, j) for i in range(len(selected_features)) for j in range(i + 1, len(selected_features))]
                    for i, j in pairs:
                        out_path = os.path.join(
                            debug_dir,
                            f"pair_dual_{selected_features[i]}_{selected_features[j]}.png",
                        )
                        pair_overlay_paths.append(
                            plot_dual_cluster_pair(
                                X_pred=X_pred_sel,
                                labels_pred=labels_pred,
                                X_obj=X_obj_sel,
                                labels_obj=labels_obj,
                                feature_names=selected_features,
                                pair=(i, j),
                                bounds=bounds,
                                x_opt=x_opt,
                                problem_name=doe_problem_name,
                                project_root=project_root,
                                use_timestamp=bool(use_timestamp),
                                save_path=out_path,
                            )
                        )

                # Selected bounds plot
                if selected_bounds is not None and selected_points is not None and len(selected_features) >= 2:
                    pairs = [(i, j) for i in range(len(selected_features)) for j in range(i + 1, len(selected_features))]
                    for i, j in pairs:
                        out_path = os.path.join(
                            debug_dir,
                            f"pair_bounds_{selected_features[i]}_{selected_features[j]}.png",
                        )
                        pair_overlay_paths.append(
                            plot_bounds_pair(
                                X_points=selected_points,
                                X_pred=X_pred_sel if X_pred_sel.size else None,
                                X_obj=X_obj_sel if X_obj_sel.size else None,
                                feature_names=selected_features,
                                pair=(i, j),
                                bounds=bounds,
                                selected_bounds=selected_bounds,
                                x_opt=x_opt,
                                problem_name=doe_problem_name,
                                project_root=project_root,
                                use_timestamp=bool(use_timestamp),
                                save_path=out_path,
                            )
                        )

                # DOE/Explorer pairwise debug: stage-wise + known optimum + constraint-aware filtering
                try:
                    plot_out = plot_doe_vs_optimum(
                        doe_df=doe_df if doe_df is not None else pd.DataFrame(),
                        explorer_df=df,
                        selected_features=selected_features,
                        known_optimum=self.config.user.known_optimum,
                        objective_sense=objective_sense,
                        out_dir=debug_dir,
                        respect_constraints=bool(has_pre_constraints or has_post_constraints),
                    )
                    doe_vs_optimum_plot_paths = list(plot_out.get("saved", []))
                except Exception as exc:
                    print(f"[Explorer] DOE-vs-optimum plot skipped: {exc}")
            except Exception as exc:
                print(f"[Explorer] Dual cluster plots skipped: {exc}")

        workflow_info = (doe_meta or {}).get("workflow_info", {})
        workflow_info["EXPLORER"] = "LHC"

        saver = ResultSaver(use_timestamp=bool(use_timestamp))
        prev_doe_meta = get_task_metadata_path(self.run_context, "DOE")
        prev_modeler_meta = get_task_metadata_path(self.run_context, "Modeler")
        if not prev_doe_meta:
            prev_doe_meta = self.config.doe_metadata_path
        if not prev_modeler_meta:
            prev_modeler_meta = self.config.modeler_metadata_path

        def _rel_or_abs(path: str | None) -> str | None:
            if not path:
                return None
            try:
                return os.path.relpath(path, task_dir)
            except ValueError:
                return path

        previous = {}
        prev_doe_ref = _rel_or_abs(prev_doe_meta)
        prev_modeler_ref = _rel_or_abs(prev_modeler_meta)
        if prev_doe_ref:
            previous["DOE"] = prev_doe_ref
        if prev_modeler_ref:
            previous["Modeler"] = prev_modeler_ref

        inputs = {
            "user_config": os.path.relpath(
                self.run_context.user_config_snapshot_path,
                task_dir,
            ),
            "system_config_snapshot": {
                "n_samples": n_samples,
                "sample_multiplier": sample_multiplier,
                "quantile_threshold": quantile_threshold,
                "min_topk_count": int(min_topk_count),
                "max_topk_count": (
                    None
                    if getattr(self.config.system, "max_topk_count", None) is None
                    else int(getattr(self.config.system, "max_topk_count"))
                ),
                "max_topk_count_used": (
                    int(max_topk_count_used) if max_topk_count_used is not None else None
                ),
                "dbscan_min_samples": dbscan_min_samples,
                "debug_level": debug_level,
                "strategy_id": strategy_id,
                "probe_multistart": int(self.config.system.probe_multistart),
                "dual_policy_mode_default": str(
                    getattr(self.config.system, "dual_policy_mode_default", "routed_v2")
                ),
                "dual_tilt_disagree_ref": float(
                    getattr(self.config.system, "dual_tilt_disagree_ref", 0.10)
                ),
                "constraint_aware_enabled": bool(getattr(self.config.system, "constraint_aware_enabled", True)),
                "constraint_aware_edge_ratio": float(getattr(self.config.system, "constraint_aware_edge_ratio", 0.05)),
                "constraint_aware_probe_anchor_max": int(getattr(self.config.system, "constraint_aware_probe_anchor_max", 16)),
                "constraint_aware_probe_steps": int(getattr(self.config.system, "constraint_aware_probe_steps", 3)),
                "constraint_aware_min_side_samples": int(getattr(self.config.system, "constraint_aware_min_side_samples", 24)),
                "constraint_aware_side_rate_min": float(getattr(self.config.system, "constraint_aware_side_rate_min", 0.55)),
                "constraint_aware_side_rate_gap": float(getattr(self.config.system, "constraint_aware_side_rate_gap", 0.12)),
                "constraint_aware_pre_shift_enabled": bool(getattr(self.config.system, "constraint_aware_pre_shift_enabled", True)),
                "constraint_aware_use_for_volume_cap": bool(getattr(self.config.system, "constraint_aware_use_for_volume_cap", True)),
                "volume_cap_boundary_pin_tol_ratio": float(
                    getattr(self.config.system, "volume_cap_boundary_pin_tol_ratio", 0.03)
                ),
                "strategy_params": dict(self.config.system.strategy_params or {}),
            },
            "previous": previous,
            "selected_features": selected_features,
            "model_path_input": modeler_pkl_path,
            "doe_csv_input": doe_csv_path,
        }
        # --- Operational metadata ---
        resolved_params = {
            "p_dim": int(len(selected_features)),
            "usable_n": int(base_n),
            "usable_n_over_p": (
                float(base_n) / float(max(len(selected_features), 1))
                if len(selected_features) > 0 else None
            ),
            "has_pre_constraints": bool(has_pre_constraints),
            "has_post_constraints": bool(has_post_constraints),
            "selected_bounds_volume_ratio": vol_ratio,
            "strategy_id": strategy_id,
            "strategy_alias": strategy_alias,
            "strategy_mode": strategy_mode,
            "dual_volume_cap_applied": bool(dual_volume_cap_applied),
            "dse_cap_boundary_touch_dim_count": int(len(dse_cap_boundary_touch_dims)),
            "dse_cap_center_realign_applied": bool(dse_cap_center_realign_applied),
            "dse_cap_force_pin_applied": bool(dse_cap_force_pin_applied),
            "constraint_aware_enabled": bool(constraint_aware_enabled),
            "constraint_aware_policy_applied": bool(constraint_aware_policy_applied),
            "constraint_aware_probe_count": constraint_aware_probe_count,
            "constraint_aware_anchor_count": constraint_aware_anchor_count,
            "constraint_aware_shifted_dims": constraint_aware_shifted_dims,
        }

        # --- Analysis metadata (artifacts/meta/analysis_<strategy>.json) ---
        _explorer_analysis_meta = {
            # Identity/context (analysis-only 파싱 지원)
            "run_id": self.run_context.run_id if self.run_context is not None else None,
            "problem": doe_problem_name,
            "seed": int(rng_seed),
            "objective_sense": objective_sense,
            "strategy_id": strategy_id,
            "strategy_alias": strategy_alias,
            "strategy_mode": strategy_mode,
            # Operational mirror
            "p_dim": int(len(selected_features)),
            "usable_n": int(base_n),
            "usable_n_over_p": (
                float(base_n) / float(max(len(selected_features), 1))
                if len(selected_features) > 0 else None
            ),
            "has_pre_constraints": bool(has_pre_constraints),
            "has_post_constraints": bool(has_post_constraints),
            "selected_bounds_volume_ratio": vol_ratio,
            "selected_bounds_volume_ratio_before_cap": (
                float(selected_bounds_volume_ratio_before_cap)
                if selected_bounds_volume_ratio_before_cap is not None else None
            ),
            "pred_bounds_volume_ratio": (
                float(pred_bounds_volume_ratio)
                if pred_bounds_volume_ratio is not None else None
            ),
            "obj_bounds_volume_ratio": (
                float(obj_bounds_volume_ratio)
                if obj_bounds_volume_ratio is not None else None
            ),
            "selected_bounds_center_l1_to_design_center": (
                float(selected_bounds_center_l1_to_design_center)
                if selected_bounds_center_l1_to_design_center is not None else None
            ),
            "dual_volume_cap_applied": bool(dual_volume_cap_applied),
            "dse_cap_boundary_touch_dims": dse_cap_boundary_touch_dims,
            "dse_cap_boundary_touch_dim_count": int(len(dse_cap_boundary_touch_dims)),
            "dse_cap_boundary_touch_eps_ratio": dse_cap_boundary_touch_eps_ratio,
            "dse_cap_center_realign_applied": bool(dse_cap_center_realign_applied),
            "dse_cap_center_blend_used": dse_cap_center_blend_used,
            "dse_cap_force_pin_applied": bool(dse_cap_force_pin_applied),
            "constraint_aware_enabled": bool(constraint_aware_enabled),
            "constraint_aware_policy_applied": bool(constraint_aware_policy_applied),
            "constraint_aware_probe_count": constraint_aware_probe_count,
            "constraint_aware_anchor_count": constraint_aware_anchor_count,
            "constraint_aware_shifted_dims": constraint_aware_shifted_dims,
            "constraint_aware_top_obj_veto_count": int(constraint_aware_top_obj_veto_count),
            "constraint_aware_top_obj_confirmed_count": int(constraint_aware_top_obj_confirmed_count),
            "constraint_aware_obj_centroid_blend_applied": bool(constraint_aware_obj_centroid_blend_applied),
            "constraint_aware_shift_boost_applied": bool(constraint_aware_shift_boost_applied),
            "constraint_aware_shift_fraction_used": (
                float(constraint_aware_shift_fraction_used)
                if constraint_aware_shift_fraction_used is not None else None
            ),
            "constraint_aware_side_scores": (
                constraint_aware_side_scores if isinstance(constraint_aware_side_scores, list) else []
            ),
            "post_penalty_active": bool(has_post_penalty),
            "post_lambda": float(post_lambda),
            "generated_total_raw": int(pre_generated),
            "generated_total_after_pre": int(X.shape[0]),
            "generated_boundary": int(X_boundary_raw.shape[0]),
            "r_used": float(r_used),
            "min_topk_count": int(min_topk_count),
            "max_topk_count_used": (
                int(max_topk_count_used) if max_topk_count_used is not None else None
            ),
            # Pred
            "pred_cluster_signal_mode": pred_cluster_signal_mode,
            "pred_cluster_beta_used": pred_cluster_beta_used,
            "pred_cluster_confidence": pred_cluster_confidence,
            "pred_cluster_selected_count": pred_cluster_selected_count,
            "pred_cluster_sigma_mean_selected": pred_cluster_sigma_mean_selected,
            "pred_refine_shift_norm": pred_refine_shift_norm,
            "pred_obj_miss_case": pred_obj_miss_case,
            "pred_obj_iou": pred_obj_iou,
            "pred_obj_center_l1_norm": pred_obj_center_l1_norm,
            "pred_multistart_det_fraction_used": pred_multistart_det_fraction_used,
            "pred_refine_bounds_scale_used": pred_refine_bounds_scale_used,
            "pred_n_starts_used": pred_n_starts_used,
            "pred_overconf_guard_triggered": bool(pred_overconf_guard_triggered),
            "pred_overconf_guard_reason": pred_overconf_guard_reason,
            # Dual
            "dual_policy_mode": dual_policy_mode,
            "dual_total_starts_target": dual_total_starts_target,
            "dual_total_starts_used": dual_total_starts_used,
            "dual_np_ratio_used": dual_np_ratio_used,
            "dual_pred_ratio_used": dual_pred_ratio_used,
            "dual_obj_ratio_used": dual_obj_ratio_used,
            "dual_n_pred_starts": dual_n_pred_starts,
            "dual_n_obj_starts": dual_n_obj_starts,
            "dual_disagreement_center_l1_norm": dual_disagreement_center_l1_norm,
            "dual_disagreement_iou": dual_disagreement_iou,
            "dual_disagreement_triggered": bool(dual_disagreement_triggered),
            "dual_center_bias_used": dual_center_bias_used,
            "dual_center_tilt_strength_base": dual_center_tilt_strength_base,
            "dual_center_tilt_strength_used_mean": dual_center_tilt_strength_used_mean,
            "dual_center_tilt_applied": bool(dual_center_tilt_applied),
            "dual_volume_cap_target": dual_volume_cap_target,
            "dual_volume_ratio_before_cap": dual_volume_ratio_before_cap,
            "router_decision": (
                routed_v2_decision.to_dict() if routed_v2_decision is not None else None
            ),
        }
        _safe_strategy_id = _normalize_strategy_id(strategy_id)
        _analysis_filename = f"analysis_{_safe_strategy_id}.json" if _safe_strategy_id else "analysis.json"
        _analysis_path = os.path.join(task_dir, "artifacts", "meta", _analysis_filename)
        os.makedirs(os.path.dirname(_analysis_path), exist_ok=True)
        with open(_analysis_path, "w", encoding="utf-8") as _af:
            json.dump(_explorer_analysis_meta, _af, ensure_ascii=False, indent=2, default=str)
        n_clusters = int(pred_stats.get("n_clusters", 0)) if isinstance(pred_stats, dict) else 0
        cluster_sizes: list[int] = []
        results_summary = {
            "n_candidates_total": int(len(df)),
            "n_q_samples": n_q_samples,
            "n_clusters": n_clusters,
            "cluster_sizes": cluster_sizes,
            "pred_mean_volume": pred_mean_vol,
            "obj_mean_volume": obj_mean_vol,
            "strategy_id": strategy_id,
        }

        public_artifacts = {}
        meta_artifacts = {
            "analysis": os.path.relpath(_analysis_path, task_dir),
        }
        debug_artifacts = {}
        if bounds_path:
            public_artifacts["selected_bounds"] = os.path.relpath(bounds_path, task_dir)
        if pair_overlay_paths:
            debug_artifacts["pair_dual_clusters"] = [os.path.relpath(p, task_dir) for p in pair_overlay_paths]
        if doe_vs_optimum_plot_paths:
            debug_artifacts["doe_vs_optimum_pairwise"] = [
                os.path.relpath(p, task_dir) for p in doe_vs_optimum_plot_paths
            ]

        task_out = saver.save_task_v3(
            run_root=self.run_context.run_root,
            task="Explorer",
            problem_name=doe_problem_name,
            df=df,
            inputs=inputs,
            resolved_params=resolved_params,
            results=results_summary,
            run_tag=strategy_id,
            public_artifacts=public_artifacts,
            meta_artifacts=meta_artifacts,
            debug_artifacts=debug_artifacts,
        )
        update_run_index(self.run_context, "Explorer", task_out["metadata"])

        return {
            "csv": task_out["csv"],
            "metadata": task_out["metadata"],
            "labels": None,
            "strategy_id": strategy_id,
            "selected_bounds_path": bounds_path,
            "selected_bounds_volume_ratio": vol_ratio,
        }
