from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from utils.dbscan_utils import auto_dbscan_eps_quantile
from utils.bounds_utils import compute_spans_lbs, clamp_to_bounds


class LocalSampler:
    def __init__(
        self,
        *,
        bounds: list[tuple[float, float]],
        rng: np.random.Generator,
        local_anchor_best_k: int,
        local_anchor_small_k: int,
        local_anchor_best_ratio: float,
        local_anchor_small_ratio: float,
        local_top_p: float,
        local_top_k_min: int,
        local_dbscan_min_samples: int,
        local_dbscan_q_eps: float,
        local_dbscan_eps_max: float,
        local_min_radius_ratio: float,
        local_tol_ratio: float,
    ):
        self.bounds = bounds
        self.rng = rng
        self.local_anchor_best_k = int(local_anchor_best_k)
        self.local_anchor_small_k = int(local_anchor_small_k)
        self.local_anchor_best_ratio = float(local_anchor_best_ratio)
        self.local_anchor_small_ratio = float(local_anchor_small_ratio)
        self.local_top_p = float(local_top_p)
        self.local_top_k_min = int(local_top_k_min)
        self.local_dbscan_min_samples = int(local_dbscan_min_samples)
        self.local_dbscan_q_eps = float(local_dbscan_q_eps)
        self.local_dbscan_eps_max = float(local_dbscan_eps_max)
        self.local_min_radius_ratio = float(local_min_radius_ratio)
        self.local_tol_ratio = float(local_tol_ratio)

    def build_local_plan(
        self,
        *,
        models,
        X_candidate: np.ndarray,
        y_candidate: np.ndarray | None = None,
        n_samples: int,
        objective_sense: str,
        local_radius_ratio: float,
        debug_best_x: np.ndarray | None = None,
        anchor_max: int | None = None,
        best_k: int | None = None,
        small_k: int | None = None,
    ) -> tuple[np.ndarray, Dict[str, float | int | None], Dict[str, object]]:
        if X_candidate is None or X_candidate.size == 0 or n_samples <= 0:
            return (
                np.empty((0, len(self.bounds)), dtype=float),
                {
                    "anchor_count": 0,
                    "local_span_ratio_mean": None,
                    "exec_pairwise_dist_mean": None,
                    "anchor_spread_mean": None,
                    "best_in_topk": None,
                    "best_anchor_dist": None,
                    "best_local_min_dist": None,
                },
                {
                    "anchors": np.empty((0, len(self.bounds)), dtype=float),
                    "anchor_ids": np.array([], dtype=int),
                    "tol_local": 0.0,
                },
            )

        best_in_topk = None
        best_anchor_dist = None
        best_local_min_dist = None

        if y_candidate is not None and y_candidate.size == X_candidate.shape[0]:
            # Force-include a few best points (phase2) even if DBSCAN marks them as noise.
            force_k = min(3, X_candidate.shape[0])
            if objective_sense == "min":
                best_force_idx = np.argsort(y_candidate)[:force_k]
            else:
                best_force_idx = np.argsort(-y_candidate)[:force_k]
            forced_points = X_candidate[best_force_idx]

            topk_indices, topk_pred = self._select_topk_by_true(
                y_candidate=y_candidate,
                objective_sense=objective_sense,
            )
            if debug_best_x is not None:
                if objective_sense == "min":
                    best_idx = int(np.argmin(y_candidate))
                else:
                    best_idx = int(np.argmax(y_candidate))
                best_in_topk = bool(best_idx in set(topk_indices.tolist()))
        else:
            forced_points = None
            topk_indices, topk_pred = self._select_topk_candidates(
                models=models,
                X_candidate=X_candidate,
                objective_sense=objective_sense,
            )
        X_topk = X_candidate[topk_indices]
        anchors, anchor_meta = self._cluster_anchors(
            X_topk=X_topk,
            y_pred=topk_pred,
            objective_sense=objective_sense,
            forced_points=forced_points,
            anchor_max=anchor_max,
            best_k=best_k,
            small_k=small_k,
        )
        if debug_best_x is not None and anchors is not None and anchors.size > 0:
            dists = np.linalg.norm(anchors - debug_best_x.reshape(1, -1), axis=1)
            best_anchor_dist = float(np.min(dists)) if dists.size else None
        X_local, anchor_ids, tol_local = self._sample_local_gaussian(
            anchors=anchors,
            anchor_meta=anchor_meta,
            n_samples=n_samples,
            local_radius_ratio=local_radius_ratio,
        )
        if debug_best_x is not None and X_local is not None and X_local.size > 0:
            dists = np.linalg.norm(X_local - debug_best_x.reshape(1, -1), axis=1)
            best_local_min_dist = float(np.min(dists)) if dists.size else None
        metrics = self._compute_metrics(anchors=anchors, X_local=X_local)
        metrics["best_in_topk"] = best_in_topk
        metrics["best_anchor_dist"] = best_anchor_dist
        metrics["best_local_min_dist"] = best_local_min_dist
        return (
            X_local,
            metrics,
            {
                "anchors": anchors,
                "anchor_ids": anchor_ids,
                "tol_local": tol_local,
            },
        )

    def _select_topk_candidates(
        self,
        *,
        models,
        X_candidate: np.ndarray,
        objective_sense: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_candidate = np.asarray(X_candidate, dtype=float)
        if X_candidate.ndim != 2 or X_candidate.shape[0] == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        n = X_candidate.shape[0]
        k = max(1, int(round(n * self.local_top_p)))
        k = max(k, self.local_top_k_min)

        per_model = []
        for m in models:
            y = np.asarray(m.predict(X_candidate), dtype=float).reshape(-1)
            if objective_sense == "min":
                order = np.argsort(y)
            else:
                order = np.argsort(-y)
            per_model.append(order[:k])

        # Phase1 policy: use UNION of per-model top-k to avoid missing narrow basins.
        union = set()
        for arr in per_model:
            union |= set(arr.tolist())

        if union:
            idx = np.array(sorted(union), dtype=int)
        else:
            idx = np.array([], dtype=int)

        # Return proxy scores for selected candidates (mean prediction across models).
        Yhat = np.vstack([np.asarray(m.predict(X_candidate), dtype=float).reshape(-1) for m in models])
        y_mean = np.mean(Yhat, axis=0)
        return idx, y_mean[idx]

    def _select_topk_by_true(
        self,
        *,
        y_candidate: np.ndarray,
        objective_sense: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_candidate = np.asarray(y_candidate, dtype=float).reshape(-1)
        n = y_candidate.shape[0]
        k = max(1, int(round(n * self.local_top_p)))
        k = max(k, self.local_top_k_min)
        if objective_sense == "min":
            order = np.argsort(y_candidate)
        else:
            order = np.argsort(-y_candidate)
        idx = order[:k].astype(int)
        return idx, y_candidate[idx]

    def _cluster_anchors(
        self,
        *,
        X_topk: np.ndarray,
        y_pred: np.ndarray,
        objective_sense: str,
        forced_points: np.ndarray | None = None,
        anchor_max: int | None = None,
        best_k: int | None = None,
        small_k: int | None = None,
    ) -> tuple[np.ndarray, list[dict]]:
        if X_topk.size == 0:
            return np.empty((0, len(self.bounds)), dtype=float), []

        spans, lbs = compute_spans_lbs(self.bounds)
        X_norm = (X_topk - lbs) / spans

        min_samples = min(self.local_dbscan_min_samples, X_norm.shape[0])
        base_eps = auto_dbscan_eps_quantile(
            X_scaled=X_norm,
            min_samples=min_samples,
            q_eps=self.local_dbscan_q_eps,
        )
        eps = min(float(base_eps), self.local_dbscan_eps_max)

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_norm)
        clusters: list[dict] = []
        for label in np.unique(labels):
            if label == -1:
                continue
            idx = np.where(labels == label)[0]
            if idx.size == 0:
                continue
            # 대표 anchor: cluster 내 y_best
            if y_pred is None or y_pred.size == 0:
                pick = idx[0]
                best_val = float("nan")
            else:
                if objective_sense == "min":
                    pick = idx[int(np.argmin(y_pred[idx]))]
                else:
                    pick = idx[int(np.argmax(y_pred[idx]))]
                best_val = float(y_pred[pick])
            clusters.append(
                {
                    "size": int(idx.size),
                    "best_val": float(best_val),
                    "anchor": X_topk[pick],
                    "idx": idx,
                    "pts": X_topk[idx],
                }
            )

        # Always include forced best points (if provided), even if they are noise.
        forced = []
        forced_set = set()
        if forced_points is not None and forced_points.size > 0:
            for i, p in enumerate(forced_points):
                key = tuple(np.round(p, 12).tolist())
                if key in forced_set:
                    continue
                forced_set.add(key)
                forced.append(p)

        # Optional: force-include best noise point (phase1) if top-k bests are all noise
        forced_noise = []
        if y_pred is not None and y_pred.size == X_topk.shape[0]:
            if objective_sense == "min":
                order = np.argsort(y_pred)
            else:
                order = np.argsort(-y_pred)
            for idx in order[:3]:
                if labels[int(idx)] == -1:
                    p = X_topk[int(idx)]
                    key = tuple(np.round(p, 12).tolist())
                    if key not in forced_set:
                        forced_noise.append(p)
                        forced_set.add(key)
                    break

        if anchor_max is None:
            raise ValueError("anchor_max is required.")
        anchor_max = max(int(anchor_max), 1)

        if best_k is None:
            best_k = int(np.ceil(anchor_max * self.local_anchor_best_ratio))
        best_k = min(max(int(best_k), 0), anchor_max, self.local_anchor_best_k)

        if small_k is None:
            small_k = int(np.ceil(anchor_max * self.local_anchor_small_ratio))
        small_k = min(max(int(small_k), 0), max(anchor_max - best_k, 0), self.local_anchor_small_k)

        if not clusters:
            # fallback: top-k y_best 순으로 anchor_max (with forced points first)
            if y_pred.size:
                if objective_sense == "min":
                    order = np.argsort(y_pred)
                else:
                    order = np.argsort(-y_pred)
            else:
                order = np.arange(X_topk.shape[0])
            anchors = []
            if forced:
                anchors.extend(forced)
            if forced_noise:
                anchors.extend(forced_noise)
            for idx in order:
                p = X_topk[int(idx)]
                key = tuple(np.round(p, 12).tolist())
                if key in forced_set:
                    continue
                anchors.append(p)
                if len(anchors) >= anchor_max + (1 if forced_noise else 0):
                    break
            return (
                np.vstack(anchors) if anchors else np.empty((0, len(self.bounds)), dtype=float),
                [],
            )

        # Select anchors with a hybrid rule:
        # - Keep top-N clusters by y_best (performance)
        # - Protect N smallest clusters (narrow basins)
        # - Fill remaining slots by cluster size (stability/coverage)
        best_k = min(max(int(best_k), 0), anchor_max)
        small_k = min(max(int(small_k), 0), anchor_max - best_k)

        # Sort by y_best (objective-aware). NaN -> worst.
        def _score_key(item: dict) -> float:
            best_val = item.get("best_val", float("nan"))
            if np.isnan(best_val):
                return float("inf") if objective_sense == "min" else float("-inf")
            return best_val

        if objective_sense == "min":
            clusters_by_score = sorted(clusters, key=_score_key)
        else:
            clusters_by_score = sorted(clusters, key=_score_key, reverse=True)

        chosen = []
        chosen_set = set()
        if forced:
            for p in forced:
                key = tuple(np.round(p, 12).tolist())
                if key not in chosen_set:
                    chosen.append({"size": 1, "best_val": float("nan"), "anchor": p, "idx": None, "pts": None})
                    chosen_set.add(key)
        if forced_noise:
            for p in forced_noise:
                key = tuple(np.round(p, 12).tolist())
                if key not in chosen_set:
                    chosen.append({"size": 1, "best_val": float("nan"), "anchor": p, "idx": None, "pts": None})
                    chosen_set.add(key)
        for i, c in enumerate(clusters_by_score):
            if i >= best_k:
                break
            anchor = c["anchor"]
            key = tuple(np.round(anchor, 12).tolist())
            if key not in chosen_set:
                chosen.append(c)
                chosen_set.add(key)

        # Protect small clusters (narrow basins)
        remaining_slots = anchor_max + (1 if forced_noise else 0) - len(chosen)
        if remaining_slots > 0 and small_k > 0:
            clusters_by_size_asc = sorted(clusters, key=lambda x: x["size"])
            picked_small = 0
            for c in clusters_by_size_asc:
                if remaining_slots <= 0 or picked_small >= small_k:
                    break
                anchor = c["anchor"]
                key = tuple(np.round(anchor, 12).tolist())
                if key in chosen_set:
                    continue
                chosen.append(c)
                chosen_set.add(key)
                remaining_slots -= 1
                picked_small += 1

        if remaining_slots > 0:
            clusters_by_size = sorted(clusters, key=lambda x: x["size"], reverse=True)
            for c in clusters_by_size:
                if remaining_slots <= 0:
                    break
                anchor = c["anchor"]
                key = tuple(np.round(anchor, 12).tolist())
                if key in chosen_set:
                    continue
                chosen.append(c)
                chosen_set.add(key)
                remaining_slots -= 1

        anchors = np.vstack([c["anchor"] for c in chosen]) if chosen else np.empty((0, len(self.bounds)), dtype=float)
        return anchors, chosen

    def _sample_local_gaussian(
        self,
        *,
        anchors: np.ndarray,
        n_samples: int,
        local_radius_ratio: float,
        anchor_meta: list[dict],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if anchors.size == 0 or n_samples <= 0:
            return (
                np.empty((0, len(self.bounds)), dtype=float),
                np.array([], dtype=int),
                0.0,
            )

        anchors = np.asarray(anchors, dtype=float)
        dim = anchors.shape[1]
        spans, _ = compute_spans_lbs(self.bounds)
        min_radius = self.local_min_radius_ratio * spans

        n_anchor = anchors.shape[0]
        per_anchor = max(1, int(np.floor(n_samples / n_anchor)))
        remainder = n_samples - per_anchor * n_anchor

        samples = []
        anchor_ids = []
        radius_list = []
        for i, a in enumerate(anchors):
            n_i = per_anchor + (1 if i < remainder else 0)
            if n_i <= 0:
                continue
            meta = anchor_meta[i] if i < len(anchor_meta) else {}
            idx = meta.get("idx")
            if idx is not None:
                pts = meta.get("pts")
                if pts is None:
                    pts = None
                if pts is None:
                    span = spans
                else:
                    span = np.ptp(pts, axis=0) if pts.size else spans
            else:
                span = np.zeros_like(spans)
            radius = float(local_radius_ratio) * span
            radius = np.maximum(radius, min_radius)
            radius_list.append(radius)
            noise = self.rng.normal(loc=0.0, scale=radius, size=(n_i, dim))
            X = a.reshape(1, -1) + noise
            X = clamp_to_bounds(X, self.bounds)
            samples.append(X)
            anchor_ids.append(np.full((X.shape[0],), i, dtype=int))

        if not samples:
            return (
                np.empty((0, dim), dtype=float),
                np.array([], dtype=int),
                0.0,
            )

        X_all = np.vstack(samples)
        anchor_ids_all = np.concatenate(anchor_ids) if anchor_ids else np.array([], dtype=int)

        # local tol (global scalar) based on median radius magnitude
        if radius_list:
            radius_means = [float(np.mean(r)) for r in radius_list]
            tol_local = float(self.local_tol_ratio * np.median(radius_means))
        else:
            tol_local = 0.0

        return X_all, anchor_ids_all, tol_local

    def _compute_metrics(
        self,
        *,
        anchors: np.ndarray,
        X_local: np.ndarray,
    ) -> Dict[str, float | int | None]:
        if X_local.size == 0:
            return {
                "anchor_count": int(anchors.shape[0]) if anchors.size else 0,
                "local_span_ratio_mean": None,
                "exec_pairwise_dist_mean": None,
                "anchor_spread_mean": None,
            }

        spans, _ = compute_spans_lbs(self.bounds)
        local_min = np.min(X_local, axis=0)
        local_max = np.max(X_local, axis=0)
        local_span = local_max - local_min
        span_ratio = np.divide(local_span, spans, out=np.zeros_like(local_span), where=spans > 0)
        span_ratio_mean = float(np.mean(span_ratio))

        if X_local.shape[0] > 1:
            dists = []
            for i in range(X_local.shape[0]):
                for j in range(i + 1, X_local.shape[0]):
                    dists.append(float(np.linalg.norm(X_local[i] - X_local[j])))
            exec_pairwise_mean = float(np.mean(dists)) if dists else None
        else:
            exec_pairwise_mean = None

        if anchors is not None and anchors.size > 0:
            center = np.mean(anchors, axis=0)
            anchor_spread = np.mean(np.linalg.norm(anchors - center, axis=1))
            anchor_spread_mean = float(anchor_spread)
        else:
            anchor_spread_mean = None

        return {
            "anchor_count": int(anchors.shape[0]) if anchors.size else 0,
            "local_span_ratio_mean": span_ratio_mean,
            "exec_pairwise_dist_mean": exec_pairwise_mean,
            "anchor_spread_mean": anchor_spread_mean,
        }
