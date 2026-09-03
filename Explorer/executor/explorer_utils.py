from typing import Optional, Tuple

import os
import numpy as np
import pandas as pd

from utils.bool_mask import to_bool_mask


def _unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for name in values:
        if name not in seen:
            unique.append(name)
            seen.add(name)
    return unique


def resolve_selected_features(
    *,
    feature_cols: list[str] | None,
    design_features: list[str] | None = None,
    selected_features_csv_path: str | None = None,
    doe_df: pd.DataFrame | None = None,
) -> list[str]:
    if not design_features:
        raise RuntimeError("CAE design feature list is required.")

    design = [str(f) for f in design_features]
    design_set = set(design)

    selected: list[str]
    source = "cae_context"
    if feature_cols:
        selected = [str(f) for f in feature_cols]
        source = "model_bundle"
    elif selected_features_csv_path:
        if not os.path.exists(selected_features_csv_path):
            raise FileNotFoundError(f"Selected features CSV not found: {selected_features_csv_path}")
        selected_df = pd.read_csv(selected_features_csv_path)
        if "feature" not in selected_df.columns:
            raise RuntimeError(
                "Selected features CSV must include a 'feature' column: "
                f"{selected_features_csv_path}"
            )
        if "selected" in selected_df.columns:
            selected_mask = to_bool_mask(
                selected_df["selected"],
                column_name="selected",
                warn_prefix="[Explorer][SelectedFeatures]",
            )
            selected_df = selected_df.loc[selected_mask]
        selected = selected_df["feature"].dropna().astype(str).tolist()
        source = "selected_features_csv"
    else:
        selected = list(design)

    if not selected:
        raise RuntimeError(f"Selected features are empty (source={source}).")

    selected = _unique_in_order(selected)

    unknown = [name for name in selected if name not in design_set]
    if unknown:
        raise RuntimeError(
            "Selected features must be a subset of CAE design features: "
            f"unknown={unknown}, design_features={design}"
        )

    if doe_df is not None:
        missing = [name for name in selected if name not in doe_df.columns]
        if missing:
            raise RuntimeError(
                "Input CSV missing required feature columns from CAE/selected feature list: "
                f"missing={missing}"
            )
        if "objective" not in doe_df.columns:
            raise RuntimeError("Input CSV must include an 'objective' column.")

    print(f"[Explorer] Active features ({source}, n={len(selected)}): {', '.join(selected)}")
    return selected


def resolve_bounds(
    *,
    selected_features: list[str],
    variables: Optional[list[dict]],
    df: pd.DataFrame | None,
) -> list[Tuple[float, float]]:
    bounds = []
    if not variables:
        print("[Explorer] Variables not found in CAE metadata.")
        raise RuntimeError("Cannot resolve bounds without CAE variable metadata.")
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


def compute_fi_importance_weights(
    *,
    fi_scores: dict[str, float],
    selected_features: list[str],
    clip_min: float = 0.5,
    clip_max: float = 2.0,
) -> np.ndarray | None:
    """FI score 기반 bounds 확장 가중치를 반환한다.

    FI가 높은 feature → 낮은 가중치 (좁힘, 정밀 탐색)
    FI가 낮은 feature → 높은 가중치 (넓힘, 안전 마진)
    반환값은 합이 d인 가중치 배열.
    """
    if not fi_scores or not selected_features:
        return None

    d = len(selected_features)
    scores = np.array(
        [float(fi_scores.get(f, 0.0)) for f in selected_features],
        dtype=float,
    )

    if scores.sum() <= 0.0 or not np.all(np.isfinite(scores)):
        return None

    # 반전: FI 높으면 가중치 낮게, FI 낮으면 가중치 높게
    # 제곱근으로 부드럽게 변환 (선형보다 완만한 차등)
    inv = (1.0 - scores) ** 0.5
    inv = np.clip(inv, 0.05, 0.95)

    weights = inv / inv.mean()
    weights = np.clip(weights, float(clip_min), float(clip_max))
    weights = weights * (float(d) / weights.sum())
    return weights


def compute_gp_boundary_uncertainty(
    *,
    gp_models: list,
    selected_bounds: list[Tuple[float, float]],
    clip_min: float = 0.5,
    clip_max: float = 2.0,
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
    weights = np.clip(weights, float(clip_min), float(clip_max))
    weights = weights * (float(d) / weights.sum())
    return weights


def _expand_bounds_by_side_support(
    *,
    selected_bounds: list[Tuple[float, float]],
    bounds: list[Tuple[float, float]],
    min_volume_ratio: float,
    support_lo: np.ndarray,
    support_hi: np.ndarray,
    step_ratio: float = 0.004,
    max_iter: int = 4000,
) -> list[Tuple[float, float]]:
    """각 변의 지지 가중치에 비례해 최소부피까지 확장한다.

    가중치가 큰 변(그 너머에 GP 상위 후보가 많은 변)이 더 많이 넓혀진다.
    전역 경계는 넘지 않고, 목표 부피를 넘긴 마지막 스텝은 이분법으로 되돌려
    부피가 min_volume_ratio를 초과하지 않게 한다.
    """
    d = len(bounds)
    if d == 0 or len(selected_bounds) != d:
        return selected_bounds
    glb: list[tuple[float, float]] = []
    S0: list[list[float]] = []
    for (s_lb, s_ub), (g_lb, g_ub) in zip(selected_bounds, bounds):
        gl, gu = float(min(g_lb, g_ub)), float(max(g_lb, g_ub))
        lo = float(np.clip(min(s_lb, s_ub), gl, gu))
        hi = float(np.clip(max(s_lb, s_ub), gl, gu))
        glb.append((gl, gu))
        S0.append([lo, hi])
    w_lo = np.maximum(np.asarray(support_lo, dtype=float).reshape(-1), 1e-9)
    w_hi = np.maximum(np.asarray(support_hi, dtype=float).reshape(-1), 1e-9)
    if w_lo.size != d or w_hi.size != d:
        return selected_bounds
    min_v = float(np.clip(min_volume_ratio, 0.0, 1.0))

    def _vol(S: list[list[float]]) -> float:
        r = 1.0
        for (lo, hi), (gl, gu) in zip(S, glb):
            span = max(gu - gl, 1e-12)
            r *= max(hi - lo, 0.0) / span
        return float(r)

    S = [list(x) for x in S0]
    if _vol(S) >= min_v:
        return [tuple(x) for x in S]
    for _ in range(int(max_iter)):
        if _vol(S) >= min_v:
            break
        moves: list[tuple[float, int, int]] = []
        for j in range(d):
            gl, gu = glb[j]
            if S[j][0] > gl + 1e-12:
                moves.append((float(w_lo[j]), j, 0))
            if S[j][1] < gu - 1e-12:
                moves.append((float(w_hi[j]), j, 1))
        if not moves:
            break
        tot = sum(m[0] for m in moves)
        for w, j, side in moves:
            gl, gu = glb[j]
            step = float(step_ratio) * (gu - gl) * (w / tot) * len(moves)
            if side == 0:
                S[j][0] = max(gl, S[j][0] - step)
            else:
                S[j][1] = min(gu, S[j][1] + step)
    if _vol(S) > min_v + 1e-9:
        # 초과분을 시작 상태와의 보간으로 되돌린다. 단, 전역 경계에 도달한 변은
        # 그대로 둔다 — 경계에 붙은 최적점을 0.0001 차이로 잘라내는 사고 방지.
        at_gl = [abs(S[j][0] - glb[j][0]) <= 1e-12 for j in range(d)]
        at_gu = [abs(S[j][1] - glb[j][1]) <= 1e-12 for j in range(d)]

        def _interp(t: float) -> list[list[float]]:
            return [
                [
                    S[j][0] if at_gl[j] else S0[j][0] + t * (S[j][0] - S0[j][0]),
                    S[j][1] if at_gu[j] else S0[j][1] + t * (S[j][1] - S0[j][1]),
                ]
                for j in range(d)
            ]

        if _vol(_interp(0.0)) > min_v:
            # 경계 변만으로 이미 초과 — 되돌릴 수 없으므로 현 상태 유지
            # (호출측 cap 단계가 boundary-pin 인식 축소로 처리한다)
            return [tuple(x) for x in S]
        lo_t, hi_t = 0.0, 1.0
        for _ in range(50):
            mid = 0.5 * (lo_t + hi_t)
            if _vol(_interp(mid)) > min_v:
                hi_t = mid
            else:
                lo_t = mid
        S = _interp(lo_t)
    return [tuple(x) for x in S]


def apply_bounds_margin(
    *,
    selected_bounds: list[Tuple[float, float]],
    bounds: list[Tuple[float, float]],
    margin_ratio: float,
    min_volume_ratio: float = 0.20,
    dim_weights: np.ndarray | None = None,
    center_hint: np.ndarray | None = None,
    side_support_lo: np.ndarray | None = None,
    side_support_hi: np.ndarray | None = None,
) -> list[Tuple[float, float]]:
    if not selected_bounds or not bounds or len(selected_bounds) != len(bounds):
        return selected_bounds

    min_v = float(np.clip(min_volume_ratio, 0.0, 1.0))

    # EXP-1: center_hint 유효성 검증
    _center = None
    if center_hint is not None:
        _c = np.asarray(center_hint, dtype=float).ravel()
        if _c.shape[0] == len(bounds) and np.all(np.isfinite(_c)):
            _center = _c

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
        center_j: float | None = None,
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

        # EXP-1: center hint 기반 비대칭 확장
        # center가 bounds 중심보다 왼쪽이면 left를 더 확장, 오른쪽이면 right를 더 확장
        left_ratio = 0.5
        if center_j is not None and cur_w > 1e-12:
            mid = (lo + hi) * 0.5
            g_span = max(gu - gl, 1e-12)
            # center가 bounds 밖이면 해당 방향으로 더 강하게 확장
            offset = float(np.clip((center_j - mid) / g_span, -0.5, 0.5))
            # offset > 0 → center가 오른쪽 → right 확장 비율 증가
            # 0.5 + offset: [0.0, 1.0] 범위, 기본 0.5에서 center 방향으로 최대 ±0.15 bias
            left_ratio = float(np.clip(0.5 - offset * 0.3, 0.2, 0.8))

        add_left = min(left_room, left_ratio * need)
        add_right = min(right_room, (1.0 - left_ratio) * need)
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
        for j, ((s_lb, s_ub), (g_lb, g_ub), t_ratio) in enumerate(zip(base_bounds, bounds, target_ratios)):
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
                center_j=float(_center[j]) if _center is not None else None,
            )
            out.append((lo_new, hi_new))
        return out

    raw_v = _volume_ratio(sel_bounds=selected_bounds)
    if not np.isfinite(raw_v):
        return selected_bounds
    if raw_v >= min_v:
        return selected_bounds

    # [지지가중 확장] 변별 지지 가중치가 오면 margin/booster 대신 이 경로만 쓴다.
    # margin 사전 확장이 예산을 균등하게 소모해 지지 방향 몫이 줄어드는 것을 막는다.
    if side_support_lo is not None and side_support_hi is not None:
        try:
            return _expand_bounds_by_side_support(
                selected_bounds=selected_bounds,
                bounds=bounds,
                min_volume_ratio=min_v,
                support_lo=side_support_lo,
                support_hi=side_support_hi,
            )
        except Exception:
            pass  # 실패 시 기존 경로로 폴백

    expanded = list(selected_bounds)
    base_margin = float(margin_ratio)

    if raw_v < min_v and base_margin > 0.0:
        m = base_margin * max(min_v - raw_v, 0.0) / min_v
        if m > 0.0:
            expanded_try = []
            for j, ((s_lb, s_ub), (g_lb, g_ub)) in enumerate(zip(expanded, bounds)):
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
                    center_j=float(_center[j]) if _center is not None else None,
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
