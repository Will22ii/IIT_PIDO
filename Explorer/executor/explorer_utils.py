from typing import Optional, Tuple

import os
import numpy as np
import pandas as pd


def resolve_selected_features(
    *,
    feature_cols: list[str] | None,
    doe_df: pd.DataFrame | None = None,
) -> list[str]:
    # pkl에서 추출한 feature_cols가 곧 selected_features
    if feature_cols:
        return list(feature_cols)

    # fallback: DOE DataFrame 컬럼에서 추출
    if doe_df is not None:
        ignore_cols = {"id", "objective", "constraints", "feasible", "success", "source", "round", "exec_scope"}
        selected_features = [
            c for c in doe_df.columns
            if c not in ignore_cols and not str(c).startswith("constraint_")
        ]
        if selected_features:
            print("[Explorer] No feature_cols from model; fallback to DOE columns.")
            return selected_features

    raise RuntimeError("Selected features not found. Provide model pkl or DOE data.")


def resolve_bounds(
    *,
    selected_features: list[str],
    variables: Optional[list[dict]],
    df: pd.DataFrame | None,
) -> list[Tuple[float, float]]:
    bounds = []
    if not variables:
        print("[Explorer] Variables not found in DOE metadata.")
        raise RuntimeError("Cannot resolve bounds without DOE variables metadata.")
    for feature in selected_features:
        matched = next(
            (v for v in variables if v.get("name") == feature),
            None,
        )
        if matched:
            bounds.append((matched["lb"], matched["ub"]))
            continue
        print(f"[Explorer] Cannot resolve bounds for feature: {feature}")
        raise RuntimeError(f"Cannot resolve bounds for feature: {feature}")
    return bounds


def format_span_rows(
    *,
    kind: str,
    spans: list | None,
    vols: list | None,
    feature_names: list[str],
) -> None:
    if spans is None:
        return
    for idx, ratios in enumerate(spans):
        if ratios is None:
            continue
        parts = []
        for name, val in zip(feature_names, ratios):
            try:
                parts.append(f"{name}:{float(val):.2f}")
            except Exception:
                parts.append(f"{name}:nan")
        vol = None
        if vols is not None and idx < len(vols):
            vol = vols[idx]
        if vol is None:
            vol_str = "nan"
        else:
            try:
                vol_str = f"{float(vol):.4f}"
            except Exception:
                vol_str = "nan"
        print(f"[Explorer] {kind} cluster{idx+1} span: " + ", ".join(parts) + f" | volume={vol_str}")


def compute_selected_bounds(
    *,
    X_pred_sel: np.ndarray,
    X_obj_sel: np.ndarray,
) -> tuple[list[Tuple[float, float]] | None, list[Tuple[float, float]] | None, list[Tuple[float, float]] | None]:
    pred_bounds = None
    obj_bounds = None
    if X_pred_sel.size:
        pred_bounds = []
        for j in range(X_pred_sel.shape[1]):
            pred_bounds.append(
                (float(X_pred_sel[:, j].min()), float(X_pred_sel[:, j].max()))
            )
    if X_obj_sel.size:
        obj_bounds = []
        for j in range(X_obj_sel.shape[1]):
            obj_bounds.append(
                (float(X_obj_sel[:, j].min()), float(X_obj_sel[:, j].max()))
            )

    selected_bounds = None
    if pred_bounds is not None and obj_bounds is not None:
        selected_bounds = []
        for (p_lb, p_ub), (o_lb, o_ub) in zip(pred_bounds, obj_bounds):
            selected_bounds.append((min(p_lb, o_lb), max(p_ub, o_ub)))
    elif pred_bounds is not None:
        selected_bounds = pred_bounds
    elif obj_bounds is not None:
        selected_bounds = obj_bounds

    return selected_bounds, pred_bounds, obj_bounds


def compute_gp_boundary_uncertainty(
    *,
    gp_models: list,
    selected_bounds: list[Tuple[float, float]],
) -> np.ndarray | None:
    """GP 모델의 경계 불확실성(σ)을 차원별로 측정하여 확장 가중치를 반환한다.

    selected_bounds 각 차원의 lb/ub에서 GP σ를 측정하고,
    σ가 높은 차원에 더 큰 가중치를 부여한다.
    반환값은 합이 d인 가중치 배열 (균등이면 전부 1.0).
    """
    valid_gps = [gp for gp in gp_models if gp is not None]
    if not valid_gps or not selected_bounds:
        return None

    d = len(selected_bounds)
    center = np.array([(lb + ub) * 0.5 for lb, ub in selected_bounds], dtype=float)
    sigmas = np.zeros(d, dtype=float)

    for j in range(d):
        point_lb = center.copy()
        point_lb[j] = selected_bounds[j][0]
        point_ub = center.copy()
        point_ub[j] = selected_bounds[j][1]

        pts = np.vstack([point_lb.reshape(1, -1), point_ub.reshape(1, -1)])
        gp_sigmas = []
        for gp in valid_gps:
            try:
                _, std = gp.predict(pts, return_std=True)
                gp_sigmas.append(float(np.mean(std)))
            except Exception:
                continue
        if gp_sigmas:
            sigmas[j] = float(np.mean(gp_sigmas))

    if sigmas.sum() <= 0.0 or not np.all(np.isfinite(sigmas)):
        return None

    weights = sigmas / sigmas.mean()
    weights = np.clip(weights, 0.3, 3.0)
    weights = weights * (float(d) / weights.sum())
    return weights


def apply_bounds_margin(
    *,
    selected_bounds: list[Tuple[float, float]],
    bounds: list[Tuple[float, float]],
    margin_ratio: float,
    min_volume_ratio: float = 0.20,
    dim_weights: np.ndarray | None = None,
) -> list[Tuple[float, float]]:
    if not selected_bounds or not bounds or len(selected_bounds) != len(bounds):
        return selected_bounds

    min_v = float(np.clip(min_volume_ratio, 0.0, 1.0))

    def _volume_ratio(
        *,
        sel_bounds: list[Tuple[float, float]],
    ) -> float:
        ratios = []
        for (s_lb, s_ub), (g_lb, g_ub) in zip(sel_bounds, bounds):
            g_span = float(g_ub - g_lb)
            if g_span <= 0.0:
                ratios.append(0.0)
                continue
            s_lo = float(min(s_lb, s_ub))
            s_hi = float(max(s_lb, s_ub))
            ratios.append(max(0.0, (s_hi - s_lo) / g_span))
        return float(np.prod(ratios)) if ratios else 0.0

    def _expand_interval_asymmetric(
        *,
        lo: float,
        hi: float,
        gl: float,
        gu: float,
        target_width: float,
    ) -> tuple[float, float]:
        gl = float(gl)
        gu = float(gu)
        lo = float(np.clip(lo, gl, gu))
        hi = float(np.clip(hi, gl, gu))
        if hi < lo:
            lo, hi = hi, lo

        max_w = max(gu - gl, 0.0)
        tgt = float(np.clip(target_width, 0.0, max_w))
        cur_w = max(hi - lo, 0.0)
        if cur_w >= tgt - 1e-12:
            return lo, hi

        need = tgt - cur_w
        left_room = max(lo - gl, 0.0)
        right_room = max(gu - hi, 0.0)

        add_left = min(left_room, 0.5 * need)
        add_right = min(right_room, 0.5 * need)
        lo -= add_left
        hi += add_right
        need -= (add_left + add_right)
        left_room -= add_left
        right_room -= add_right

        if need > 1e-12:
            if right_room >= left_room:
                extra_right = min(right_room, need)
                hi += extra_right
                need -= extra_right
                right_room -= extra_right
                if need > 1e-12:
                    extra_left = min(left_room, need)
                    lo -= extra_left
                    need -= extra_left
            else:
                extra_left = min(left_room, need)
                lo -= extra_left
                need -= extra_left
                left_room -= extra_left
                if need > 1e-12:
                    extra_right = min(right_room, need)
                    hi += extra_right
                    need -= extra_right

        lo = float(np.clip(lo, gl, gu))
        hi = float(np.clip(hi, gl, gu))
        if hi < lo:
            lo, hi = hi, lo
        return lo, hi

    def _current_ratios(sel_bounds: list[Tuple[float, float]]) -> tuple[list[float], list[float]]:
        ratios: list[float] = []
        spans: list[float] = []
        for (s_lb, s_ub), (g_lb, g_ub) in zip(sel_bounds, bounds):
            gl = float(g_lb)
            gu = float(g_ub)
            g_span = max(gu - gl, 0.0)
            spans.append(g_span)
            if g_span <= 0.0:
                ratios.append(0.0)
                continue
            lo = float(np.clip(min(s_lb, s_ub), gl, gu))
            hi = float(np.clip(max(s_lb, s_ub), gl, gu))
            ratios.append(float(np.clip((hi - lo) / g_span, 0.0, 1.0)))
        return ratios, spans

    def _widen_to_ratios(
        *,
        base_bounds: list[Tuple[float, float]],
        target_ratios: list[float],
    ) -> list[Tuple[float, float]]:
        out: list[Tuple[float, float]] = []
        for (s_lb, s_ub), (g_lb, g_ub), t_ratio in zip(base_bounds, bounds, target_ratios):
            gl = float(g_lb)
            gu = float(g_ub)
            g_span = max(gu - gl, 0.0)
            lo = float(np.clip(min(s_lb, s_ub), gl, gu))
            hi = float(np.clip(max(s_lb, s_ub), gl, gu))
            target_w = float(np.clip(t_ratio, 0.0, 1.0)) * g_span
            lo_new, hi_new = _expand_interval_asymmetric(
                lo=lo,
                hi=hi,
                gl=gl,
                gu=gu,
                target_width=target_w,
            )
            out.append((lo_new, hi_new))
        return out

    raw_v = _volume_ratio(sel_bounds=selected_bounds)
    if not np.isfinite(raw_v):
        return selected_bounds
    if raw_v >= min_v:
        return selected_bounds

    expanded = list(selected_bounds)
    base_margin = float(margin_ratio)

    if raw_v < min_v and base_margin > 0.0:
        m = base_margin * max(min_v - raw_v, 0.0) / min_v
        if m > 0.0:
            expanded_try = []
            for (s_lb, s_ub), (g_lb, g_ub) in zip(expanded, bounds):
                gl = float(g_lb)
                gu = float(g_ub)
                g_span = max(gu - gl, 1e-12)
                s_lo = float(min(s_lb, s_ub))
                s_hi = float(max(s_lb, s_ub))
                target_w = max(s_hi - s_lo, 0.0) + 2.0 * m * g_span
                lo, hi = _expand_interval_asymmetric(
                    lo=s_lo,
                    hi=s_hi,
                    gl=gl,
                    gu=gu,
                    target_width=target_w,
                )
                if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
                    return selected_bounds
                expanded_try.append((lo, hi))
            expanded = expanded_try

    cur_v = _volume_ratio(sel_bounds=expanded)
    if not np.isfinite(cur_v):
        return selected_bounds
    if min_v <= 0.0 or cur_v >= min_v:
        return expanded

    d = max(int(len(bounds)), 1)
    alpha = float((max(min_v, 1e-12) / max(cur_v, 1e-12)) ** (1.0 / float(d)))
    if not np.isfinite(alpha):
        return expanded

    ratios_cur, _ = _current_ratios(expanded)
    if dim_weights is not None and len(dim_weights) == d:
        # uncertainty-aware: σ가 높은 차원에 alpha를 더 크게 적용
        w = np.asarray(dim_weights, dtype=float)
        target_ratios = [
            min(1.0, max(0.0, float(r) * max(float(alpha ** w[j]), 1.0)))
            for j, r in enumerate(ratios_cur)
        ]
    else:
        target_ratios = [min(1.0, max(0.0, float(r) * max(alpha, 1.0))) for r in ratios_cur]
    expanded = _widen_to_ratios(base_bounds=expanded, target_ratios=target_ratios)

    cur_v = _volume_ratio(sel_bounds=expanded)
    if not np.isfinite(cur_v):
        return selected_bounds
    if cur_v >= min_v:
        return expanded

    # Floor booster: move each per-dimension ratio toward 1.0 with a shared beta.
    # uncertainty-aware: σ가 높은 차원이 1.0에 더 빨리 접근하도록 가중.
    ratios_cur, _ = _current_ratios(expanded)
    lo_beta = 0.0
    hi_beta = 1.0
    for _ in range(40):
        mid = 0.5 * (lo_beta + hi_beta)
        if dim_weights is not None and len(dim_weights) == d:
            w = np.asarray(dim_weights, dtype=float)
            mid_ratios = [float(r + mid * w[j] / w.mean() * (1.0 - r)) for j, r in enumerate(ratios_cur)]
        else:
            mid_ratios = [float(r + mid * (1.0 - r)) for r in ratios_cur]
        vol_mid = float(np.prod(mid_ratios)) if mid_ratios else 0.0
        if vol_mid >= min_v:
            hi_beta = mid
        else:
            lo_beta = mid
    if dim_weights is not None and len(dim_weights) == d:
        w = np.asarray(dim_weights, dtype=float)
        final_ratios = [float(r + hi_beta * w[j] / w.mean() * (1.0 - r)) for j, r in enumerate(ratios_cur)]
    else:
        final_ratios = [float(r + hi_beta * (1.0 - r)) for r in ratios_cur]
    expanded = _widen_to_ratios(base_bounds=expanded, target_ratios=final_ratios)

    return expanded
