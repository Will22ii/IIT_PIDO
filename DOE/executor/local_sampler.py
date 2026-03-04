from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from DOE.executor.anchor_refiner import GPAnchorRefiner
from utils.dbscan_utils import auto_dbscan_eps_quantile
from utils.bounds_utils import compute_spans_lbs, clamp_to_bounds


class LocalSampler:
    def __init__(
        self,
        *,
        bounds: list[tuple[float, float]],
        rng: np.random.Generator,
        gp_seed: int,
        local_gp_use_white_kernel: bool = False,
        pre_feasible_fn: Callable[[np.ndarray], bool] | None = None,
        post_feasible_prob_fn: Callable[[np.ndarray], float] | None = None,
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
        local_refine_min_points: int,
        local_cluster_delta_ratio: float,
        local_singleton_box_ratio: float,
        local_phase1_kappa: float,
        local_phase2_kappa: float,
        local_base_perturb_ratio: float,
    ):
        self.bounds = bounds
        self.rng = rng
        self.gp_seed = int(gp_seed)
        self.local_gp_use_white_kernel = bool(local_gp_use_white_kernel)
        self.pre_feasible_fn = pre_feasible_fn
        self.post_feasible_prob_fn = post_feasible_prob_fn
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
        self.local_refine_min_points = int(local_refine_min_points)
        self.local_cluster_delta_ratio = float(local_cluster_delta_ratio)
        self.local_singleton_box_ratio = float(local_singleton_box_ratio)
        self.local_phase1_kappa = float(local_phase1_kappa)
        self.local_phase2_kappa = float(local_phase2_kappa)
        self.local_base_perturb_ratio = float(local_base_perturb_ratio)
        self._refiner = GPAnchorRefiner(
            bounds=self.bounds,
            rng=self.rng,
            random_seed=self.gp_seed,
            refine_min_points=self.local_refine_min_points,
            delta_ratio=self.local_cluster_delta_ratio,
            singleton_box_ratio=self.local_singleton_box_ratio,
            phase1_kappa=self.local_phase1_kappa,
            phase2_kappa=self.local_phase2_kappa,
            q_expand_step=10,
            use_white_kernel=self.local_gp_use_white_kernel,
        )

    def build_local_plan(
        self,
        *,
        models,
        X_candidate: np.ndarray,
        y_candidate: np.ndarray | None = None,
        n_samples: int,
        objective_sense: str,
        phase: int,
        stage_eps: float,
        use_post_penalty: bool = False,
        post_penalty_lambda: float = 0.0,
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

        X_candidate = np.asarray(X_candidate, dtype=float)
        if X_candidate.ndim != 2 or X_candidate.shape[1] != len(self.bounds):
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

        ref_model = models[-1] if models else None
        if ref_model is None:
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
            y_candidate = np.asarray(y_candidate, dtype=float).reshape(-1)
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
            y_candidate = np.asarray(ref_model.predict(X_candidate), dtype=float).reshape(-1)

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
        refined_payload = self._build_refined_payload(
            anchors=anchors,
            anchor_meta=anchor_meta,
            X_source=X_candidate,
            y_source=y_candidate,
            ref_model=ref_model,
            objective_sense=objective_sense,
            phase=int(phase),
            stage_eps=float(stage_eps),
            use_post_penalty=bool(use_post_penalty),
            post_penalty_lambda=float(post_penalty_lambda),
            local_radius_ratio=float(local_radius_ratio),
        )
        anchors_final = refined_payload["anchors"]
        anchor_specs = refined_payload["anchor_specs"]
        if debug_best_x is not None and anchors_final is not None and anchors_final.size > 0:
            dists = np.linalg.norm(anchors_final - debug_best_x.reshape(1, -1), axis=1)
            best_anchor_dist = float(np.min(dists)) if dists.size else None

        X_local, anchor_ids, tol_local = self._sample_local_with_specs(
            anchors=anchors_final,
            anchor_specs=anchor_specs,
            n_samples=n_samples,
            local_radius_ratio=local_radius_ratio,
            phase=int(phase),
        )
        if debug_best_x is not None and X_local is not None and X_local.size > 0:
            dists = np.linalg.norm(X_local - debug_best_x.reshape(1, -1), axis=1)
            best_local_min_dist = float(np.min(dists)) if dists.size else None
        metrics = self._compute_metrics(anchors=anchors_final, X_local=X_local)
        metrics["best_in_topk"] = best_in_topk
        metrics["best_anchor_dist"] = best_anchor_dist
        metrics["best_local_min_dist"] = best_local_min_dist
        return (
            X_local,
            metrics,
            {
                "anchors": anchors_final,
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
        forced_items: list[dict] = []
        forced_set = set()
        if forced_points is not None and forced_points.size > 0:
            for i, p in enumerate(forced_points):
                key = tuple(np.round(p, 12).tolist())
                if key in forced_set:
                    continue
                forced_set.add(key)
                forced_items.append(
                    {
                        "size": 1,
                        "best_val": float("nan"),
                        "anchor": p,
                        "idx": None,
                        "pts": None,
                    }
                )

        # Optional: force-include best noise point (phase1) if top-k bests are all noise
        forced_noise_items: list[dict] = []
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
                        forced_noise_items.append(
                            {
                                "size": 1,
                                "best_val": float("nan"),
                                "anchor": p,
                                "idx": None,
                                "pts": None,
                            }
                        )
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

        # Select anchors with a hybrid rule:
        # - Keep top-N clusters by y_best (performance)
        # - Protect N smallest clusters (narrow basins)
        # - Fill remaining slots by cluster size (stability/coverage)
        best_k = min(max(int(best_k), 0), anchor_max)
        small_k = min(max(int(small_k), 0), anchor_max - best_k)
        target_total = anchor_max + (1 if forced_noise_items else 0)

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
        clusters_by_size_asc = sorted(clusters, key=lambda x: x["size"])
        clusters_by_size_desc = sorted(clusters, key=lambda x: x["size"], reverse=True)

        buckets = {
            "forced": forced_items,
            "forced_noise": forced_noise_items,
            "best": clusters_by_score,
            "small": clusters_by_size_asc,
            "rest": clusters_by_size_desc,
        }
        bucket_limits = {
            "forced": len(forced_items),
            "forced_noise": len(forced_noise_items),
            "best": int(best_k),
            "small": int(small_k),
            "rest": int(target_total),
        }
        bucket_used = {k: 0 for k in buckets.keys()}
        bucket_pos = {k: 0 for k in buckets.keys()}
        bucket_order = ["forced", "forced_noise", "best", "small", "rest"]

        chosen: list[dict] = []
        chosen_set: set[tuple] = set()

        def _next_unique(bucket_name: str) -> dict | None:
            arr = buckets[bucket_name]
            pos = bucket_pos[bucket_name]
            while pos < len(arr):
                cand = arr[pos]
                pos += 1
                key = tuple(np.round(np.asarray(cand["anchor"], dtype=float), 12).tolist())
                if key in chosen_set:
                    continue
                bucket_pos[bucket_name] = pos
                return cand
            bucket_pos[bucket_name] = pos
            return None

        while len(chosen) < target_total:
            progressed = False
            for name in bucket_order:
                if len(chosen) >= target_total:
                    break
                if bucket_used[name] >= bucket_limits[name]:
                    continue
                cand = _next_unique(name)
                if cand is None:
                    continue
                key = tuple(np.round(np.asarray(cand["anchor"], dtype=float), 12).tolist())
                chosen.append(cand)
                chosen_set.add(key)
                bucket_used[name] += 1
                progressed = True
            if not progressed:
                break

        # If cluster buckets are exhausted, fill from raw top-k points.
        if len(chosen) < target_total:
            if y_pred.size:
                if objective_sense == "min":
                    order = np.argsort(y_pred)
                else:
                    order = np.argsort(-y_pred)
            else:
                order = np.arange(X_topk.shape[0])
            for idx in order.tolist():
                if len(chosen) >= target_total:
                    break
                p = np.asarray(X_topk[int(idx)], dtype=float).reshape(-1)
                key = tuple(np.round(p, 12).tolist())
                if key in chosen_set:
                    continue
                chosen.append(
                    {
                        "size": 1,
                        "best_val": float(y_pred[int(idx)]) if y_pred.size else float("nan"),
                        "anchor": p,
                        "idx": None,
                        "pts": None,
                    }
                )
                chosen_set.add(key)

        anchors = np.vstack([c["anchor"] for c in chosen]) if chosen else np.empty((0, len(self.bounds)), dtype=float)
        return anchors, chosen

    def _is_better(self, *, a: float, b: float, objective_sense: str) -> bool:
        if objective_sense == "min":
            return float(a) <= float(b)
        return float(a) >= float(b)

    def _estimate_score(
        self,
        *,
        x: np.ndarray,
        X_source: np.ndarray,
        y_source: np.ndarray,
    ) -> float:
        x = np.asarray(x, dtype=float).reshape(1, -1)
        X_source = np.asarray(X_source, dtype=float)
        y_source = np.asarray(y_source, dtype=float).reshape(-1)
        if X_source.size == 0 or y_source.size == 0:
            return float("nan")
        d = np.linalg.norm(X_source - x, axis=1)
        idx = int(np.argmin(d))
        return float(y_source[idx])

    def _build_refined_payload(
        self,
        *,
        anchors: np.ndarray,
        anchor_meta: list[dict],
        X_source: np.ndarray,
        y_source: np.ndarray,
        ref_model,
        objective_sense: str,
        phase: int,
        stage_eps: float,
        use_post_penalty: bool,
        post_penalty_lambda: float,
        local_radius_ratio: float,
    ) -> dict:
        if anchors is None or anchors.size == 0:
            return {
                "anchors": np.empty((0, len(self.bounds)), dtype=float),
                "anchor_specs": [],
            }

        global_spans, _ = compute_spans_lbs(self.bounds)
        final_anchors: list[np.ndarray] = []
        anchor_specs: list[dict] = []

        for i, base_anchor in enumerate(np.asarray(anchors, dtype=float)):
            meta = anchor_meta[i] if i < len(anchor_meta) else {}
            cluster_pts = np.asarray(meta.get("pts", base_anchor.reshape(1, -1)), dtype=float)

            if phase == 1:
                refine_out = self._refiner.refine_phase1(
                    base_anchor=base_anchor,
                    cluster_points=cluster_pts,
                    X_source=X_source,
                    y_source=y_source,
                    objective_sense=objective_sense,
                    acq_type="LCB",
                    perturb_ratio=self.local_base_perturb_ratio,
                    pre_feasible_fn=self.pre_feasible_fn,
                    post_feasible_prob_fn=self.post_feasible_prob_fn if use_post_penalty else None,
                    post_penalty_lambda=post_penalty_lambda if use_post_penalty else 0.0,
                )
                refined = (
                    np.asarray(refine_out.refined_anchor, dtype=float).reshape(-1)
                    if refine_out.refined_anchor is not None
                    else base_anchor.copy()
                )
                base_score = float(np.asarray(ref_model.predict(base_anchor.reshape(1, -1)), dtype=float).reshape(-1)[0])
                refined_score = float(
                    np.asarray(ref_model.predict(refined.reshape(1, -1)), dtype=float).reshape(-1)[0]
                )
            else:
                refine_out = self._refiner.refine_phase2(
                    base_anchor=base_anchor,
                    cluster_points=cluster_pts,
                    X_source=X_source,
                    y_source=y_source,
                    objective_sense=objective_sense,
                    base_q=float(np.clip(self.local_top_p * 100.0, 0.0, 100.0)),
                    acq_type="LCB",
                    perturb_ratio=self.local_base_perturb_ratio,
                    pre_feasible_fn=self.pre_feasible_fn,
                    post_feasible_prob_fn=self.post_feasible_prob_fn if use_post_penalty else None,
                    post_penalty_lambda=post_penalty_lambda if use_post_penalty else 0.0,
                )
                base_score = self._estimate_score(
                    x=base_anchor,
                    X_source=X_source,
                    y_source=y_source,
                )
                if refine_out.refined_anchor is not None and refine_out.refined_score is not None:
                    refined = np.asarray(refine_out.refined_anchor, dtype=float).reshape(-1)
                    refined_score = float(refine_out.refined_score)
                else:
                    refined = None
                    refined_score = None

            box_lb = np.asarray(refine_out.box_lb, dtype=float).reshape(-1)
            box_ub = np.asarray(refine_out.box_ub, dtype=float).reshape(-1)

            if refined is None:
                anchor_start = len(final_anchors)
                final_anchors.append(base_anchor.copy())
                anchor_specs.append(
                    {
                        "mode": "isotropic",
                        "center": base_anchor.copy(),
                        "box_lb": box_lb,
                        "box_ub": box_ub,
                        "anchor_indices": [anchor_start],
                        "local_radius_ratio": float(local_radius_ratio),
                    }
                )
                continue

            exact_equal = bool(np.array_equal(refined, base_anchor))
            if exact_equal:
                anchor_start = len(final_anchors)
                final_anchors.append(base_anchor.copy())
                anchor_specs.append(
                    {
                        "mode": "isotropic",
                        "center": base_anchor.copy(),
                        "box_lb": box_lb,
                        "box_ub": box_ub,
                        "anchor_indices": [anchor_start],
                        "local_radius_ratio": float(local_radius_ratio),
                    }
                )
                continue

            d_norm = float(np.linalg.norm((refined - base_anchor) / global_spans))
            keep_refined_only = bool(d_norm < float(stage_eps))

            if keep_refined_only:
                ordered = [refined.copy()]
            else:
                if phase == 1:
                    ordered = [base_anchor.copy(), refined.copy()]
                else:
                    ordered = [refined.copy(), base_anchor.copy()]

            anchor_start = len(final_anchors)
            for a in ordered:
                final_anchors.append(np.asarray(a, dtype=float).reshape(-1))
            anchor_indices = list(range(anchor_start, anchor_start + len(ordered)))

            if self._is_better(a=refined_score, b=base_score, objective_sense=objective_sense):
                better = refined.copy()
                worse = base_anchor.copy()
            else:
                better = base_anchor.copy()
                worse = refined.copy()
            vec = better - worse
            norm_vec = float(np.linalg.norm(vec))
            if norm_vec <= 1e-12:
                mode = "isotropic"
                center = better.copy()
                direction = np.zeros_like(center)
            else:
                mode = "anisotropic"
                center = better.copy()
                direction = vec / norm_vec

            anchor_specs.append(
                {
                    "mode": mode,
                    "center": center,
                    "direction": direction,
                    "box_lb": box_lb,
                    "box_ub": box_ub,
                    "anchor_indices": anchor_indices,
                    "local_radius_ratio": float(local_radius_ratio),
                }
            )

        return {
            "anchors": np.vstack(final_anchors) if final_anchors else np.empty((0, len(self.bounds)), dtype=float),
            "anchor_specs": anchor_specs,
        }

    def _sample_local_with_specs(
        self,
        *,
        anchors: np.ndarray,
        anchor_specs: list[dict],
        n_samples: int,
        local_radius_ratio: float,
        phase: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        _ = phase
        if anchors is None or anchors.size == 0 or n_samples <= 0 or not anchor_specs:
            return (
                np.empty((0, len(self.bounds)), dtype=float),
                np.array([], dtype=int),
                0.0,
            )

        anchors = np.asarray(anchors, dtype=float)
        dim = anchors.shape[1]
        spans_global, _ = compute_spans_lbs(self.bounds)
        min_radius = self.local_min_radius_ratio * spans_global

        n_specs = len(anchor_specs)
        per = max(1, int(np.floor(n_samples / max(n_specs, 1))))
        remainder = n_samples - per * n_specs

        all_pts: list[np.ndarray] = []
        all_ids: list[np.ndarray] = []
        scale_refs: list[float] = []

        for i, spec in enumerate(anchor_specs):
            n_i = per + (1 if i < remainder else 0)
            if n_i <= 0:
                continue

            box_lb = np.asarray(spec["box_lb"], dtype=float).reshape(-1)
            box_ub = np.asarray(spec["box_ub"], dtype=float).reshape(-1)
            span_box = np.maximum(box_ub - box_lb, 1e-12)
            mode = str(spec.get("mode", "isotropic")).strip().lower()

            if mode == "anisotropic":
                center = np.asarray(spec.get("center"), dtype=float).reshape(-1)
                direction = np.asarray(spec.get("direction"), dtype=float).reshape(-1)
                n_eff = max(int(np.sum(span_box > 1e-12)), 1)
                rho_perp = float(np.clip(0.03 * (6.0 / float(n_eff)), 0.012, 0.05))
                rho_par = float(np.clip(0.07 * (6.0 / float(n_eff)), 0.028, 0.12))
                sigma_perp = np.maximum(rho_perp * span_box, min_radius)
                l_par = float(np.sqrt(np.sum((direction * span_box) ** 2)))
                sigma_par = max(rho_par * l_par, float(np.mean(min_radius)))
                pts = self._draw_anisotropic_with_reject(
                    center=center,
                    direction=direction,
                    sigma_par=sigma_par,
                    sigma_perp=sigma_perp,
                    box_lb=box_lb,
                    box_ub=box_ub,
                    n_samples=n_i,
                )
                scale_refs.append(float(np.mean(sigma_perp)))
            else:
                center = np.asarray(spec.get("center"), dtype=float).reshape(-1)
                radius = np.maximum(float(local_radius_ratio) * span_box, min_radius)
                pts = self._draw_isotropic_with_reject(
                    center=center,
                    radius=radius,
                    box_lb=box_lb,
                    box_ub=box_ub,
                    n_samples=n_i,
                )
                scale_refs.append(float(np.mean(radius)))

            if pts.size == 0:
                continue

            anchor_indices = list(spec.get("anchor_indices", []))
            if len(anchor_indices) <= 1:
                aid = int(anchor_indices[0]) if anchor_indices else 0
                ids = np.full((pts.shape[0],), aid, dtype=int)
            else:
                anchor_pts = anchors[np.asarray(anchor_indices, dtype=int)]
                d = np.linalg.norm(
                    pts[:, None, :] - anchor_pts[None, :, :],
                    axis=2,
                )
                nearest = np.argmin(d, axis=1)
                ids = np.asarray([anchor_indices[int(k)] for k in nearest.tolist()], dtype=int)

            all_pts.append(pts)
            all_ids.append(ids)

        if not all_pts:
            return (
                np.empty((0, dim), dtype=float),
                np.array([], dtype=int),
                0.0,
            )

        X_all = np.vstack(all_pts).astype(float)
        anchor_ids_all = np.concatenate(all_ids).astype(int)
        if scale_refs:
            tol_local = float(self.local_tol_ratio * np.median(np.asarray(scale_refs, dtype=float)))
        else:
            tol_local = 0.0
        return X_all, anchor_ids_all, tol_local

    def _draw_isotropic_with_reject(
        self,
        *,
        center: np.ndarray,
        radius: np.ndarray,
        box_lb: np.ndarray,
        box_ub: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        center = np.asarray(center, dtype=float).reshape(-1)
        radius0 = np.asarray(radius, dtype=float).reshape(-1)
        lb = np.asarray(box_lb, dtype=float).reshape(-1)
        ub = np.asarray(box_ub, dtype=float).reshape(-1)
        dim = center.shape[0]
        gathered = []
        remain = int(n_samples)
        for attempt in range(2):
            if remain <= 0:
                break
            scale = radius0 if attempt == 0 else (0.8 * radius0)
            n_draw = max(remain * 4, 8)
            raw = center.reshape(1, -1) + self.rng.normal(0.0, scale, size=(n_draw, dim))
            mask = np.all((raw >= lb.reshape(1, -1)) & (raw <= ub.reshape(1, -1)), axis=1)
            accepted = raw[mask]
            accepted = self._filter_pre_feasible_points(accepted)
            if accepted.shape[0] > 0:
                take = min(remain, accepted.shape[0])
                gathered.append(accepted[:take])
                remain -= take
        if remain > 0:
            n_fill = max(remain * 3, remain)
            fill = self.rng.uniform(lb, ub, size=(n_fill, dim))
            fill = self._filter_pre_feasible_points(fill)
            if fill.shape[0] > 0:
                gathered.append(fill[:remain])
                remain -= min(remain, fill.shape[0])
        return np.vstack(gathered) if gathered else np.empty((0, dim), dtype=float)

    def _draw_anisotropic_with_reject(
        self,
        *,
        center: np.ndarray,
        direction: np.ndarray,
        sigma_par: float,
        sigma_perp: np.ndarray,
        box_lb: np.ndarray,
        box_ub: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        center = np.asarray(center, dtype=float).reshape(-1)
        direction = np.asarray(direction, dtype=float).reshape(-1)
        norm_u = float(np.linalg.norm(direction))
        if norm_u <= 1e-12:
            return self._draw_isotropic_with_reject(
                center=center,
                radius=np.asarray(sigma_perp, dtype=float).reshape(-1),
                box_lb=box_lb,
                box_ub=box_ub,
                n_samples=n_samples,
            )
        u = direction / norm_u
        sigma_perp0 = np.asarray(sigma_perp, dtype=float).reshape(-1)
        lb = np.asarray(box_lb, dtype=float).reshape(-1)
        ub = np.asarray(box_ub, dtype=float).reshape(-1)
        dim = center.shape[0]

        gathered = []
        remain = int(n_samples)
        for attempt in range(2):
            if remain <= 0:
                break
            shrink = 1.0 if attempt == 0 else 0.8
            sigma_par_use = max(float(sigma_par) * shrink, 1e-12)
            sigma_perp_use = np.maximum(sigma_perp0 * shrink, 1e-12)
            n_draw = max(remain * 4, 8)
            z = self.rng.normal(0.0, 1.0, size=(n_draw, 1))
            eta = self.rng.normal(0.0, sigma_perp_use.reshape(1, -1), size=(n_draw, dim))
            raw = center.reshape(1, -1) + z * sigma_par_use * u.reshape(1, -1) + eta
            mask = np.all((raw >= lb.reshape(1, -1)) & (raw <= ub.reshape(1, -1)), axis=1)
            accepted = raw[mask]
            accepted = self._filter_pre_feasible_points(accepted)
            if accepted.shape[0] > 0:
                take = min(remain, accepted.shape[0])
                gathered.append(accepted[:take])
                remain -= take
        if remain > 0:
            n_fill = max(remain * 3, remain)
            fill = self.rng.uniform(lb, ub, size=(n_fill, dim))
            fill = self._filter_pre_feasible_points(fill)
            if fill.shape[0] > 0:
                gathered.append(fill[:remain])
                remain -= min(remain, fill.shape[0])
        return np.vstack(gathered) if gathered else np.empty((0, dim), dtype=float)

    def _filter_pre_feasible_points(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.size == 0 or self.pre_feasible_fn is None:
            return X
        keep = np.zeros((X.shape[0],), dtype=bool)
        for i, x in enumerate(X):
            try:
                keep[i] = bool(self.pre_feasible_fn(np.asarray(x, dtype=float).reshape(-1)))
            except Exception:
                keep[i] = False
        return X[keep]

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
