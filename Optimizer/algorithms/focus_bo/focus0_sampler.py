from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable

import numpy as np

from DOE.doe_algorithm.lhs import latin_hypercube_sampling
from DOE.executor.constraint_filter import clamp_ratio
from Optimizer.config import OptimizerSystemConfig
from Optimizer.executor.pre_constraint_policy import PreInequalityPolicy
from utils.boundary_sampling import sample_boundary_corners_random


@dataclass
class Focus0SampleResult:
    X: np.ndarray
    source_labels: list[str]
    constraint_payloads: list[dict]
    margins: np.ndarray
    diagnostics: dict[str, object]


def _resolve_focus0_corner_ratio(
    *,
    system: OptimizerSystemConfig,
    n_samples: int,
    p_dim: int,
) -> tuple[float, str, dict[str, object]]:
    base_ratio = float(getattr(system, "focus0_initial_corner_ratio", 0.10))
    effective_ratio = float(np.clip(base_ratio, 0.0, 1.0))
    policy = "fixed"
    p_eff = max(int(p_dim), 1)
    np_ratio = float(max(int(n_samples), 0)) / float(p_eff)
    context: dict[str, object] = {
        "base_ratio": float(base_ratio),
        "effective_ratio": float(effective_ratio),
        "np_ratio": float(np_ratio),
        "p_dim": int(p_eff),
    }
    if not bool(getattr(system, "focus0_initial_corner_adaptive_enabled", True)):
        return effective_ratio, policy, context

    np_ratio_max = float(getattr(system, "focus0_initial_corner_adaptive_np_ratio_max", 10.0))
    low_dim_max = int(getattr(system, "focus0_initial_corner_adaptive_low_dim_max", 6))
    low_dim_ratio = float(getattr(system, "focus0_initial_corner_adaptive_ratio_low_dim", 0.05))
    if np_ratio <= np_ratio_max:
        if p_eff <= low_dim_max:
            effective_ratio = min(effective_ratio, float(np.clip(low_dim_ratio, 0.0, 1.0)))
            policy = "adaptive_p_le_low_dim_low_np"
        else:
            policy = "adaptive_p_gt_low_dim_keep"
    else:
        policy = "adaptive_np_high_keep"
    context["effective_ratio"] = float(effective_ratio)
    return float(effective_ratio), policy, context


def _filter_pre_constraints(
    *,
    X: np.ndarray,
    var_names: list[str],
    constraint_defs: list[dict],
    has_pre_constraints: bool,
    to_full_x: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, list[dict], np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return (
            np.empty((0, X.shape[1] if X.ndim == 2 else 0), dtype=float),
            [],
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=bool),
        )
    if not has_pre_constraints:
        mask = np.ones((X.shape[0],), dtype=bool)
        return X, [{} for _ in range(X.shape[0])], np.full((X.shape[0],), float("inf")), mask

    policy = PreInequalityPolicy(
        constraint_defs=constraint_defs,
        var_names=var_names,
        to_full_x=to_full_x,
    )
    return policy.filter_matrix(X)


def _extract_pre_constraint_ids(constraints_payloads: list[dict]) -> list[str]:
    ids: set[str] = set()
    for payload in constraints_payloads:
        if not isinstance(payload, dict):
            continue
        for key, cinfo in payload.items():
            if not isinstance(cinfo, dict):
                continue
            if str(cinfo.get("scope", "pre")).strip().lower() != "pre":
                continue
            cid = str(cinfo.get("id", key)).strip()
            if cid:
                ids.add(cid)
    return sorted(ids)


def _resolve_margin_subset_k_values(*, n_constraints: int, random_k_count: int, rng: np.random.Generator) -> list[int]:
    n = int(n_constraints)
    if n <= 0:
        return []
    if n <= 3:
        return list(range(1, n + 1))
    mid_candidates = list(range(2, n))
    n_mid = min(max(int(random_k_count), 0), len(mid_candidates))
    picked_mid: list[int] = []
    if n_mid > 0:
        picked_mid = rng.choice(np.asarray(mid_candidates, dtype=int), size=n_mid, replace=False).astype(int).tolist()
    return [int(k) for k in sorted(set([1, n] + picked_mid))]


def _build_margin_subset_specs(
    *,
    constraint_ids: list[str],
    random_k_count: int,
    rng: np.random.Generator,
) -> list[tuple[tuple[str, ...], float]]:
    if not constraint_ids:
        return []
    k_values = _resolve_margin_subset_k_values(
        n_constraints=len(constraint_ids),
        random_k_count=random_k_count,
        rng=rng,
    )
    subsets: list[tuple[str, ...]] = []
    weights_raw: list[float] = []
    for k in k_values:
        for subset in combinations(constraint_ids, int(k)):
            subsets.append(tuple(str(v) for v in subset))
            weights_raw.append(float(k))
    w_sum = float(sum(weights_raw))
    if not subsets or w_sum <= 0.0:
        return []
    return [(subsets[i], float(weights_raw[i] / w_sum)) for i in range(len(subsets))]


def _allocate_weighted_subset_quotas(
    *,
    n_total: int,
    weights: list[float],
    enforce_min_one: bool,
) -> list[int]:
    if n_total <= 0 or not weights:
        return [0 for _ in weights]
    m = len(weights)
    w = [max(float(x), 0.0) for x in weights]
    w_sum = float(sum(w))
    if w_sum <= 0.0:
        w = [1.0 / float(m) for _ in range(m)]
    else:
        w = [float(x / w_sum) for x in w]

    def _distribute(total: int, base_offset: list[int]) -> list[int]:
        raw = [float(total) * wi for wi in w]
        add = [int(np.floor(v)) for v in raw]
        remain = int(max(total - sum(add), 0))
        frac_order = sorted(
            range(m),
            key=lambda i: (raw[i] - add[i], raw[i]),
            reverse=True,
        )
        for i in frac_order[:remain]:
            add[i] += 1
        return [int(base_offset[i] + add[i]) for i in range(m)]

    if not enforce_min_one:
        return _distribute(int(n_total), [0 for _ in range(m)])
    if n_total >= m:
        return _distribute(int(n_total - m), [1 for _ in range(m)])
    out = [0 for _ in range(m)]
    order = sorted(range(m), key=lambda i: (w[i], -i), reverse=True)
    for i in order[: int(n_total)]:
        out[i] = 1
    return out


def _select_margin_indices(
    *,
    pool_constraints: list[dict],
    pool_margins: list[float],
    n_margin: int,
    system: OptimizerSystemConfig,
    rng: np.random.Generator,
) -> list[int]:
    n_target = int(max(n_margin, 0))
    if n_target <= 0 or not pool_constraints:
        return []

    selected_idx: list[int] = []
    selected_set: set[int] = set()

    def _take(candidates: list[int], k: int) -> list[int]:
        out: list[int] = []
        for cid in candidates:
            if cid in selected_set:
                continue
            out.append(int(cid))
            if len(out) >= int(k):
                break
        return out

    subset_specs: list[tuple[tuple[str, ...], float]] = []
    if bool(getattr(system, "focus0_margin_subset_enabled", True)):
        pre_ids = _extract_pre_constraint_ids(pool_constraints)
        if len(pre_ids) >= 2:
            subset_specs = _build_margin_subset_specs(
                constraint_ids=pre_ids,
                random_k_count=int(getattr(system, "focus0_margin_subset_random_k_count", 2)),
                rng=rng,
            )

    if subset_specs:
        n_constraints = len(_extract_pre_constraint_ids(pool_constraints))
        edge_k = {1, int(n_constraints)}
        subset_priority = sorted(
            range(len(subset_specs)),
            key=lambda gi: (
                0 if len(subset_specs[gi][0]) in edge_k else 1,
                -len(subset_specs[gi][0]),
                subset_specs[gi][0],
            ),
        )
        weights = [float(w) for _subset, w in subset_specs]
        quotas = _allocate_weighted_subset_quotas(
            n_total=n_target,
            weights=weights,
            enforce_min_one=n_target >= len(subset_specs),
        )
        candidate_margin_maps: list[dict[str, float]] = []
        for payload in pool_constraints:
            cur: dict[str, float] = {}
            if isinstance(payload, dict):
                for key, cinfo in payload.items():
                    if not isinstance(cinfo, dict):
                        continue
                    if str(cinfo.get("scope", "pre")).strip().lower() != "pre":
                        continue
                    cid = str(cinfo.get("id", key)).strip()
                    try:
                        mval = float(cinfo.get("margin"))
                    except Exception:
                        continue
                    if cid and np.isfinite(mval):
                        cur[cid] = mval
            candidate_margin_maps.append(cur)

        subset_orders: list[list[int]] = []
        for subset_ids, _w in subset_specs:
            ranked: list[tuple[float, int]] = []
            for idx, m_map in enumerate(candidate_margin_maps):
                vals = [m_map.get(cid) for cid in subset_ids]
                if any(v is None for v in vals):
                    continue
                ranked.append((float(np.min(np.asarray(vals, dtype=float))), int(idx)))
            ranked.sort(key=lambda t: (t[0], t[1]))
            subset_orders.append([idx for _m, idx in ranked])

        allocated = [0 for _ in quotas]
        if n_target >= len(subset_specs):
            for gi in subset_priority:
                picked = _take(subset_orders[gi], 1)
                if picked:
                    selected_idx.extend(picked)
                    selected_set.update(picked)
                    allocated[gi] += len(picked)

        for gi in subset_priority:
            need = max(int(quotas[gi]) - int(allocated[gi]), 0)
            if need <= 0:
                continue
            picked = _take(subset_orders[gi], need)
            if picked:
                selected_idx.extend(picked)
                selected_set.update(picked)
    else:
        order = np.argsort(np.asarray(pool_margins, dtype=float)).astype(int).tolist()
        picked = _take(order, n_target)
        selected_idx.extend(picked)
        selected_set.update(picked)

    return selected_idx[:n_target]


def _sample_initial_corners(
    *,
    bounds: list[tuple[float, float]],
    n_target: int,
    rng: np.random.Generator,
    var_names: list[str],
    constraint_defs: list[dict],
    has_pre_constraints: bool,
    to_full_x: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, list[dict], np.ndarray, dict[str, int]]:
    n_target = int(max(n_target, 0))
    p = len(bounds)
    if n_target <= 0:
        return np.empty((0, p), dtype=float), [], np.empty((0,), dtype=float), {"generated": 0, "feasible": 0}

    n_pool = n_target if not has_pre_constraints else max((2 * n_target), (n_target + 4))
    corner_pool = sample_boundary_corners_random(
        bounds,
        offset=np.zeros((p,), dtype=float),
        n_samples=n_pool,
        rng=rng,
    )
    if corner_pool.size == 0:
        return np.empty((0, p), dtype=float), [], np.empty((0,), dtype=float), {"generated": 0, "feasible": 0}
    corner_pool = np.unique(np.asarray(corner_pool, dtype=float), axis=0)

    if not has_pre_constraints:
        n_take = min(n_target, corner_pool.shape[0])
        if n_take <= 0:
            return np.empty((0, p), dtype=float), [], np.empty((0,), dtype=float), {"generated": int(corner_pool.shape[0]), "feasible": int(corner_pool.shape[0])}
        idx = rng.choice(corner_pool.shape[0], size=n_take, replace=False)
        X = corner_pool[np.asarray(idx, dtype=int)].astype(float)
        return X, [{} for _ in range(X.shape[0])], np.full((X.shape[0],), float("inf")), {
            "generated": int(corner_pool.shape[0]),
            "feasible": int(corner_pool.shape[0]),
        }

    remaining_idx = np.arange(corner_pool.shape[0], dtype=int)
    rows: list[np.ndarray] = []
    payloads: list[dict] = []
    margins: list[float] = []
    generated = 0
    feasible = 0
    for _attempt in range(2):
        need = n_target - len(rows)
        if need <= 0 or remaining_idx.size == 0:
            break
        n_take = min(need, remaining_idx.size)
        pick_pos = rng.choice(remaining_idx.size, size=n_take, replace=False)
        pick_idx = remaining_idx[pick_pos]
        keep_mask = np.ones((remaining_idx.size,), dtype=bool)
        keep_mask[pick_pos] = False
        remaining_idx = remaining_idx[keep_mask]

        X_try = corner_pool[pick_idx]
        X_f, payloads_f, margins_f, mask = _filter_pre_constraints(
            X=X_try,
            var_names=var_names,
            constraint_defs=constraint_defs,
            has_pre_constraints=True,
            to_full_x=to_full_x,
        )
        generated += int(X_try.shape[0])
        feasible += int(np.sum(mask))
        for i in range(X_f.shape[0]):
            if len(rows) >= n_target:
                break
            rows.append(X_f[i].reshape(1, -1))
            payloads.append(payloads_f[i])
            margins.append(float(margins_f[i]))

    if not rows:
        return np.empty((0, p), dtype=float), [], np.empty((0,), dtype=float), {"generated": generated, "feasible": feasible}
    return np.vstack(rows).astype(float), payloads, np.asarray(margins, dtype=float), {
        "generated": int(generated),
        "feasible": int(feasible),
    }


def build_focus0_initial_doe_points(
    *,
    n_samples: int,
    bounds: list[tuple[float, float]],
    var_names: list[str],
    constraint_defs: list[dict],
    rng: np.random.Generator,
    system: OptimizerSystemConfig,
    to_full_x: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Focus0SampleResult:
    n_total = int(max(n_samples, 0))
    p = len(bounds)
    if n_total <= 0:
        return Focus0SampleResult(
            X=np.empty((0, p), dtype=float),
            source_labels=[],
            constraint_payloads=[],
            margins=np.empty((0,), dtype=float),
            diagnostics={"n_samples": 0},
        )

    has_pre_constraints = any(
        str(c.get("scope", "pre")).strip().lower() == "pre"
        for c in (constraint_defs or [])
        if isinstance(c, dict)
    )
    corner_ratio, corner_policy, corner_context = _resolve_focus0_corner_ratio(
        system=system,
        n_samples=n_total,
        p_dim=p,
    )
    n_corner_target = int(round(float(n_total) * float(corner_ratio)))
    n_corner_target = min(max(n_corner_target, 0), n_total)

    X_corner, corner_payloads, corner_margins, corner_diag = _sample_initial_corners(
        bounds=bounds,
        n_target=n_corner_target,
        rng=rng,
        var_names=var_names,
        constraint_defs=list(constraint_defs or []),
        has_pre_constraints=has_pre_constraints,
        to_full_x=to_full_x,
    )
    n_corner_ok = int(X_corner.shape[0])
    n_regular_target = max(n_total - n_corner_ok, 0)

    X_regular = np.empty((0, p), dtype=float)
    regular_payloads: list[dict] = []
    regular_margins = np.empty((0,), dtype=float)
    regular_labels: list[str] = []
    regular_generated = 0
    regular_feasible = 0
    margin_selected = 0

    if n_regular_target > 0:
        if has_pre_constraints:
            probe_base = max(1, int(np.ceil(n_regular_target * float(getattr(system, "focus0_probe_multiplier", 2.0)))))
            probe_size = int(np.ceil(probe_base * float(getattr(system, "focus0_filter_safety", 1.2))))
            max_attempts = min(
                max(2, int(np.ceil(1.0 / max(float(getattr(system, "focus0_filter_r_floor", 0.02)), 1e-6)))),
                50,
            )
            pool_X: list[np.ndarray] = []
            pool_payloads: list[dict] = []
            pool_margins: list[float] = []
            for _attempt in range(max_attempts):
                X_probe = latin_hypercube_sampling(
                    n_samples=probe_size,
                    bounds=bounds,
                    rng=rng,
                    n_divisions=max(probe_size, 1),
                )
                X_f, payloads_f, margins_f, mask = _filter_pre_constraints(
                    X=X_probe,
                    var_names=var_names,
                    constraint_defs=list(constraint_defs or []),
                    has_pre_constraints=True,
                    to_full_x=to_full_x,
                )
                regular_generated += int(X_probe.shape[0])
                regular_feasible += int(np.sum(mask))
                for i in range(X_f.shape[0]):
                    pool_X.append(X_f[i].reshape(-1))
                    pool_payloads.append(payloads_f[i])
                    pool_margins.append(float(margins_f[i]))
                if len(pool_X) >= n_regular_target:
                    break

            if len(pool_X) < n_regular_target:
                raise RuntimeError(
                    "FAILED_FOCUS0_FILTER_MIN: "
                    f"initial feasible points {len(pool_X)} < target {n_regular_target}"
                )

            n_margin = int(round(n_regular_target * float(np.clip(float(getattr(system, "focus0_margin_ratio", 0.30)), 0.0, 1.0))))
            n_margin = min(max(n_margin, 0), n_regular_target)
            selected_idx = _select_margin_indices(
                pool_constraints=pool_payloads,
                pool_margins=pool_margins,
                n_margin=n_margin,
                system=system,
                rng=rng,
            )
            selected_set = set(selected_idx)
            margin_selected = int(len(selected_idx))
            remaining_slots = max(n_regular_target - len(selected_idx), 0)
            remain_candidates = [idx for idx in range(len(pool_X)) if idx not in selected_set]
            if remaining_slots > 0 and remain_candidates:
                fill = rng.choice(
                    np.asarray(remain_candidates, dtype=int),
                    size=min(remaining_slots, len(remain_candidates)),
                    replace=False,
                ).astype(int).tolist()
                selected_idx.extend(fill)
                selected_set.update(fill)
            if len(selected_idx) < n_regular_target:
                fallback = [idx for idx in range(len(pool_X)) if idx not in selected_set]
                selected_idx.extend(fallback[: max(n_regular_target - len(selected_idx), 0)])

            pick_idx = np.asarray(selected_idx[:n_regular_target], dtype=int)
            X_regular = np.vstack([pool_X[i].reshape(1, -1) for i in pick_idx]).astype(float)
            regular_payloads = [pool_payloads[i] for i in pick_idx]
            regular_margins = np.asarray([float(pool_margins[i]) for i in pick_idx], dtype=float)
            margin_idx_set = set(selected_idx[:margin_selected])
            regular_labels = [
                "focus0_margin" if int(i) in margin_idx_set else "focus0_lhs"
                for i in pick_idx.tolist()
            ]
        else:
            X_regular = latin_hypercube_sampling(
                n_samples=n_regular_target,
                bounds=bounds,
                rng=rng,
                n_divisions=max(n_regular_target, 1),
            )
            regular_payloads = [{} for _ in range(X_regular.shape[0])]
            regular_margins = np.full((X_regular.shape[0],), float("inf"), dtype=float)
            regular_labels = ["focus0_lhs"] * int(X_regular.shape[0])
            regular_generated = int(X_regular.shape[0])
            regular_feasible = int(X_regular.shape[0])

    parts = [arr for arr in (X_corner, X_regular) if arr.size > 0]
    X = np.vstack(parts).astype(float) if parts else np.empty((0, p), dtype=float)
    labels = ["focus0_corner"] * int(X_corner.shape[0]) + list(regular_labels)
    payloads = list(corner_payloads) + list(regular_payloads)
    margins = np.concatenate([corner_margins, regular_margins]) if (corner_margins.size or regular_margins.size) else np.empty((0,), dtype=float)

    gen_total = int(corner_diag.get("generated", 0)) + int(regular_generated)
    feas_total = int(corner_diag.get("feasible", 0)) + int(regular_feasible)
    constraint_rate_hat = clamp_ratio(
        feas_total / max(gen_total, 1),
        floor=float(getattr(system, "focus0_filter_r_floor", 0.02)),
    ) if has_pre_constraints else 1.0

    diagnostics = {
        "n_samples": int(n_total),
        "n_generated": int(gen_total),
        "n_feasible": int(feas_total),
        "n_corner_target": int(n_corner_target),
        "n_corner": int(X_corner.shape[0]),
        "n_regular": int(X_regular.shape[0]),
        "n_margin": int(margin_selected),
        "corner_ratio": float(corner_ratio),
        "corner_policy": str(corner_policy),
        "corner_context": dict(corner_context),
        "has_pre_constraints": bool(has_pre_constraints),
        "constraint_rate_hat": float(constraint_rate_hat),
    }
    return Focus0SampleResult(
        X=X,
        source_labels=labels,
        constraint_payloads=payloads,
        margins=np.asarray(margins, dtype=float),
        diagnostics=diagnostics,
    )
