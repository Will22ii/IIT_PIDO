from __future__ import annotations

from math import erf, sqrt

import numpy as np


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
