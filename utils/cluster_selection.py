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
            }
        else:
            best_cluster_label = int(labels[best_local_idx])

    if objective_sense == "min":
        clusters_sorted = sorted(clusters, key=lambda c: c["best_val"])
    else:
        clusters_sorted = sorted(clusters, key=lambda c: c["best_val"], reverse=True)

    selected = []
    if forced_best is not None:
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
        "selected_spans": spans,
        "selected_volumes": volumes,
    }
