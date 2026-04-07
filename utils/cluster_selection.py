from typing import Tuple

import numpy as np

from utils.dbscan_utils import auto_dbscan_eps_quantile


def select_top_clusters(
    *,
    X: np.ndarray,
    y: np.ndarray,
    objective_sense: str,
    quantile_threshold: float,
    min_samples: int,
    bounds: list[Tuple[float, float]],
    min_topk_count: int,
    eps_quantile: float,
    diversity_extra_clusters: int = 0,
    diversity_weight: float = 0.35,
    diversity_min_distance: float = 0.22,
    diversity_close_penalty: float = 0.8,
    diversity_min_dim: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if X.size == 0 or y.size == 0:
        return np.empty((0, X.shape[1] if X.ndim == 2 else 0)), np.array([], dtype=int), {}
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X/y length mismatch in cluster selection.")

    if objective_sense == "min":
        threshold = float(np.quantile(y, 1.0 - quantile_threshold))
        mask = y <= threshold
        best_idx = int(np.argmin(y))
    else:
        threshold = float(np.quantile(y, quantile_threshold))
        mask = y >= threshold
        best_idx = int(np.argmax(y))

    if mask.sum() < min_topk_count and y.shape[0] >= min_topk_count:
        if objective_sense == "min":
            order = np.argsort(y)
        else:
            order = np.argsort(-y)
        topk_idx = order[:min_topk_count]
        mask = np.zeros_like(mask, dtype=bool)
        mask[topk_idx] = True

    X_q = X[mask]
    y_q = y[mask]
    q_global_idx = np.where(mask)[0]
    if X_q.size == 0:
        return np.empty((0, X.shape[1]), dtype=float), np.array([], dtype=int), {}

    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_q)
    eps_q = float(auto_dbscan_eps_quantile(X_scaled, min_samples, float(eps_quantile)))
    eps = eps_q
    if not np.isfinite(eps) or eps <= 0.0:
        eps = 1e-3
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)

    clusters = []
    unique_labels = [int(v) for v in np.unique(labels) if v != -1]
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        if idx.size == 0:
            continue
        if objective_sense == "min":
            best_local = idx[int(np.argmin(y_q[idx]))]
        else:
            best_local = idx[int(np.argmax(y_q[idx]))]
        clusters.append(
            {
                "label": label,
                "best_val": float(y_q[best_local]),
                "points": X_q[idx],
                "center": np.mean(X_q[idx], axis=0),
                "indices_local": idx.astype(int),
            }
        )

    forced_best = None
    best_cluster_label = None
    if best_idx is not None and mask[best_idx]:
        best_local_idx = int(np.where(mask)[0].tolist().index(best_idx))
        if labels[best_local_idx] == -1:
            forced_best = {
                "label": "noise_best",
                "best_val": float(y_q[best_local_idx]),
                "points": X_q[best_local_idx].reshape(1, -1),
                "indices_local": np.asarray([best_local_idx], dtype=int),
            }
        else:
            best_cluster_label = int(labels[best_local_idx])

    if objective_sense == "min":
        clusters_sorted = sorted(clusters, key=lambda c: c["best_val"])
    else:
        clusters_sorted = sorted(clusters, key=lambda c: c["best_val"], reverse=True)
    rank_map = {c["label"]: i for i, c in enumerate(clusters_sorted)}

    selected = []
    if forced_best is not None:
        forced_best["center"] = np.asarray(forced_best["points"], dtype=float).mean(axis=0)
        selected.append(forced_best)
    elif best_cluster_label is not None:
        best_cluster = next((c for c in clusters_sorted if c["label"] == best_cluster_label), None)
        if best_cluster is not None:
            selected.append(best_cluster)

    if forced_best is not None:
        for c in clusters_sorted:
            if len(selected) >= 2:
                break
            selected.append(c)

    diversity_used = False
    diversity_selected_count = 0
    diversity_selected_min_dist: list[float] = []
    if (
        selected
        and int(diversity_extra_clusters) > 0
        and X.shape[1] >= int(max(diversity_min_dim, 1))
        and len(clusters_sorted) >= 2
    ):
        d_eff = max(int(X.shape[1]), 1)
        spans = np.asarray([max(float(ub - lb), 1e-12) for lb, ub in bounds], dtype=float)
        weight = float(np.clip(diversity_weight, 0.0, 1.0))
        close_pen = float(max(diversity_close_penalty, 0.0))
        dist_floor = float(max(diversity_min_distance, 0.0))

        selected_labels = {str(c.get("label")) for c in selected}
        candidates = [c for c in clusters_sorted if str(c.get("label")) not in selected_labels]
        n_rank = max(len(clusters_sorted) - 1, 1)

        def _norm_min_dist(cand_center: np.ndarray, ref_clusters: list[dict]) -> float:
            dists = []
            for rc in ref_clusters:
                rc_center = np.asarray(rc.get("center"), dtype=float).reshape(-1)
                diff = (cand_center - rc_center) / spans
                d = float(np.linalg.norm(diff) / np.sqrt(float(d_eff)))
                dists.append(max(d, 0.0))
            return float(min(dists)) if dists else 0.0

        for _ in range(int(diversity_extra_clusters)):
            if not candidates:
                break
            best_idx = None
            best_score = -1e18
            best_dist = 0.0
            for i, c in enumerate(candidates):
                rank = int(rank_map.get(c["label"], n_rank))
                quality = 1.0 - (float(rank) / float(n_rank))
                cand_center = np.asarray(c["center"], dtype=float).reshape(-1)
                min_dist = _norm_min_dist(cand_center, selected)
                close_penalty = close_pen * max(dist_floor - min_dist, 0.0)
                score = (1.0 - weight) * quality + weight * min_dist - close_penalty
                if score > best_score:
                    best_score = score
                    best_idx = i
                    best_dist = min_dist
            if best_idx is None:
                break
            chosen = candidates.pop(best_idx)
            selected.append(chosen)
            diversity_used = True
            diversity_selected_count += 1
            diversity_selected_min_dist.append(float(best_dist))

    if not selected:
        return np.empty((0, X.shape[1]), dtype=float), np.array([], dtype=int), {}

    points = []
    labels_out = []
    for idx, c in enumerate(selected):
        pts = c["points"]
        points.append(pts)
        labels_out.append(np.full((pts.shape[0],), idx, dtype=int))
    X_sel = np.vstack(points) if points else np.empty((0, X.shape[1]), dtype=float)
    y_sel = np.concatenate(labels_out) if labels_out else np.array([], dtype=int)

    selected_indices_global: list[int] = []
    for c in selected:
        idx_local = np.asarray(c.get("indices_local", np.array([], dtype=int)), dtype=int).reshape(-1)
        if idx_local.size == 0:
            continue
        idx_local = idx_local[(idx_local >= 0) & (idx_local < q_global_idx.size)]
        if idx_local.size == 0:
            continue
        selected_indices_global.extend(int(v) for v in q_global_idx[idx_local].tolist())
    if selected_indices_global:
        selected_indices_global = sorted(set(selected_indices_global))

    spans = []
    volumes = []
    for c in selected:
        pts = c["points"]
        if pts is None or pts.size == 0:
            spans.append(None)
            volumes.append(None)
            continue
        ratios = []
        for j, (lb, ub) in enumerate(bounds):
            denom = float(ub - lb)
            if denom == 0:
                ratios.append(0.0)
            else:
                ratios.append((pts[:, j].max() - pts[:, j].min()) / denom)
        spans.append(ratios)
        vol = float(np.prod(ratios)) if ratios else 0.0
        volumes.append(vol)

    return X_sel, y_sel, {
        "threshold": threshold,
        "eps": float(eps),
        "n_clusters": len(unique_labels),
        "n_selected_clusters": int(len(selected)),
        "diversity_used": bool(diversity_used),
        "diversity_selected_count": int(diversity_selected_count),
        "diversity_selected_min_dist": diversity_selected_min_dist,
        "selected_indices": selected_indices_global,
        "selected_spans": spans,
        "selected_volumes": volumes,
    }
