import ast
import json
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd

from utils.result_loader import ResultLoader
from utils.result_saver import ResultSaver
from DOE.doe_algorithm.lhs import latin_hypercube_sampling
from Explorer.config import ExplorerConfig
from utils.boundary_sampling import sample_boundary_corners, sample_boundary_partial
from utils.cluster_selection import select_top_clusters
from utils.bounds_utils import compute_spans_lbs
from utils.dbscan_utils import auto_dbscan_eps_quantile
from DOE.executor.constraint_filter import evaluate_constraints_batch, validate_constraint_defs
from Explorer.executor.explorer_utils import (
    apply_bounds_margin,
    compute_selected_bounds,
    format_span_rows,
    resolve_bounds,
    resolve_selected_features,
)
from Explorer.visualization.explorer_plots import (
    plot_dual_cluster_pair,
    plot_dual_cluster_tsne,
    plot_bounds_pair,
    plot_raw_dbscan,
    plot_raw_known_optimum,
    plot_raw_overlay,
)
from pipeline.run_context import RunContext, get_stage_metadata_path, update_run_index


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


def _to_bool_mask(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.fillna(False).to_numpy(dtype=bool)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "y", "yes", "t"})
        .to_numpy(dtype=bool)
    )


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
        loader = ResultLoader()

        doe_result = None
        modeler_result = None
        doe_meta = {}
        modeler_meta = {}
        doe_df = None
        modeler_df = None
        doe_problem_name = None
        doe_csv_path = None
        modeler_pkl_path = None
        modeler_feas_pkl_path = None
        modeler_stage_dir = None

        if self.config.doe_csv_path:
            if not os.path.exists(self.config.doe_csv_path):
                raise FileNotFoundError(f"DOE CSV not found: {self.config.doe_csv_path}")
            doe_csv_path = self.config.doe_csv_path
            doe_df = pd.read_csv(doe_csv_path)
            if self.config.doe_metadata_path:
                print(f"[Explorer] DOE metadata path provided: {self.config.doe_metadata_path}")
                with open(self.config.doe_metadata_path, "r") as f:
                    doe_meta = json.load(f)
            doe_problem_name = (
                doe_meta.get("problem")
                if doe_meta
                else (self.config.cae.user.problem_name if self.config.cae else None)
            )
        elif self.config.doe_metadata_path:
            print(f"[Explorer] DOE metadata path provided: {self.config.doe_metadata_path}")
            with open(self.config.doe_metadata_path, "r") as f:
                doe_meta = json.load(f)
            if "artifacts" in doe_meta and "results_csv" in doe_meta["artifacts"]:
                stage_dir = os.path.dirname(self.config.doe_metadata_path)
                results_csv = doe_meta["artifacts"]["results_csv"]
                doe_csv_path = os.path.join(stage_dir, results_csv)
                doe_df = pd.read_csv(doe_csv_path)
                doe_problem_name = doe_meta.get("problem")
            else:
                doe_result = loader.load_stage(
                    stage="DOE",
                    metadata_path=self.config.doe_metadata_path,
                    allow_latest_fallback=False,
                )
                doe_df = doe_result.df
                doe_meta = doe_result.metadata or {}
                doe_problem_name = doe_result.problem_name
                doe_csv_path = doe_result.csv_path
        elif self.run_context:
            doe_meta_path = get_stage_metadata_path(self.run_context, "DOE")
            if not doe_meta_path:
                raise RuntimeError("DOE metadata not found in run context.")
            with open(doe_meta_path, "r") as f:
                doe_meta = json.load(f)
            stage_dir = os.path.dirname(doe_meta_path)
            results_csv = doe_meta["artifacts"]["results_csv"]
            doe_csv_path = os.path.join(stage_dir, results_csv)
            doe_df = pd.read_csv(doe_csv_path)
            doe_problem_name = doe_meta.get("problem")
        else:
            if not self.config.cae.system.allow_latest_fallback:
                raise RuntimeError("Latest fallback is disabled. Provide metadata or paths.")
            print("[Explorer] Metadata paths not provided; using latest DOE/Modeler metadata.")
            doe_result = loader.load_stage(stage="DOE", allow_latest_fallback=True)
            doe_df = doe_result.df
            doe_meta = doe_result.metadata or {}
            doe_problem_name = doe_result.problem_name
            doe_csv_path = doe_result.csv_path

        if self.config.model_pkl_path:
            if not os.path.exists(self.config.model_pkl_path):
                raise FileNotFoundError(f"Model PKL not found: {self.config.model_pkl_path}")
            modeler_pkl_path = self.config.model_pkl_path
            if self.config.modeler_metadata_path:
                print(f"[Explorer] Modeler metadata path provided: {self.config.modeler_metadata_path}")
                with open(self.config.modeler_metadata_path, "r") as f:
                    modeler_meta = json.load(f)
                modeler_stage_dir = os.path.dirname(self.config.modeler_metadata_path)
                artifacts_meta = modeler_meta.get("artifacts", {}) if isinstance(modeler_meta, dict) else {}
                results_csv = artifacts_meta.get("results_csv")
                if results_csv:
                    modeler_csv_path = os.path.join(modeler_stage_dir, results_csv)
                    if os.path.exists(modeler_csv_path):
                        modeler_df = pd.read_csv(modeler_csv_path)
                feas_model_path = artifacts_meta.get("feas_model_path") or modeler_meta.get("feas_model_path")
                if feas_model_path:
                    modeler_feas_pkl_path = (
                        feas_model_path
                        if os.path.isabs(feas_model_path)
                        else os.path.join(modeler_stage_dir, feas_model_path)
                    )
            else:
                modeler_meta = {}
        elif self.config.modeler_metadata_path:
            print(f"[Explorer] Modeler metadata path provided: {self.config.modeler_metadata_path}")
            with open(self.config.modeler_metadata_path, "r") as f:
                modeler_meta = json.load(f)
            artifacts_meta = modeler_meta.get("artifacts", {}) if isinstance(modeler_meta, dict) else {}
            modeler_stage_dir = os.path.dirname(self.config.modeler_metadata_path)
            results_csv = artifacts_meta.get("results_csv")
            if results_csv:
                modeler_csv_path = os.path.join(modeler_stage_dir, results_csv)
                if os.path.exists(modeler_csv_path):
                    modeler_df = pd.read_csv(modeler_csv_path)
            model_path = artifacts_meta.get("model_path") or modeler_meta.get("model_path")
            if model_path:
                modeler_pkl_path = (
                    model_path
                    if os.path.isabs(model_path)
                    else os.path.join(modeler_stage_dir, model_path)
                )
            feas_model_path = artifacts_meta.get("feas_model_path") or modeler_meta.get("feas_model_path")
            if feas_model_path:
                modeler_feas_pkl_path = (
                    feas_model_path
                    if os.path.isabs(feas_model_path)
                    else os.path.join(modeler_stage_dir, feas_model_path)
                )
            if modeler_pkl_path is None:
                modeler_result = loader.load_stage(
                    stage="MODELER",
                    metadata_path=self.config.modeler_metadata_path,
                    allow_latest_fallback=False,
                )
                modeler_df = modeler_result.df
                modeler_meta = modeler_result.metadata or {}
                modeler_pkl_path = modeler_result.pkl_path
                if modeler_feas_pkl_path is None:
                    modeler_feas_pkl_path = modeler_meta.get("feas_model_path")
            else:
                modeler_result = None
        elif self.run_context:
            modeler_meta_path = get_stage_metadata_path(self.run_context, "Modeler")
            if not modeler_meta_path:
                raise RuntimeError("Modeler metadata not found in run context.")
            with open(modeler_meta_path, "r") as f:
                modeler_meta = json.load(f)
            modeler_stage_dir = os.path.dirname(modeler_meta_path)
            results_csv = modeler_meta["artifacts"]["results_csv"]
            modeler_csv_path = os.path.join(modeler_stage_dir, results_csv)
            modeler_df = pd.read_csv(modeler_csv_path)
            model_path = modeler_meta.get("artifacts", {}).get("model_path")
            if model_path:
                modeler_pkl_path = os.path.join(modeler_stage_dir, model_path)
            feas_model_path = modeler_meta.get("artifacts", {}).get("feas_model_path")
            if feas_model_path:
                modeler_feas_pkl_path = os.path.join(modeler_stage_dir, feas_model_path)
        else:
            modeler_result = loader.load_stage(stage="MODELER", allow_latest_fallback=True)
            modeler_df = modeler_result.df
            modeler_meta = modeler_result.metadata or {}
            modeler_pkl_path = modeler_result.pkl_path
            modeler_feas_pkl_path = modeler_meta.get("feas_model_path")

        if modeler_feas_pkl_path and not os.path.isabs(modeler_feas_pkl_path):
            base_dir = modeler_stage_dir or (
                os.path.dirname(self.config.modeler_metadata_path)
                if self.config.modeler_metadata_path
                else None
            )
            if base_dir:
                modeler_feas_pkl_path = os.path.join(base_dir, modeler_feas_pkl_path)

        if not doe_problem_name:
            raise RuntimeError("DOE problem name not found.")

        if self.config.cae is None:
            raise RuntimeError("ExplorerConfig.cae is required for seed/objective_sense.")

        if not modeler_pkl_path:
            raise RuntimeError("Modeler PKL not found. Check modeler metadata.")

        models, feature_cols = _load_models(modeler_pkl_path)

        selected_features = resolve_selected_features(
            modeler_meta=modeler_meta,
            modeler_stage_dir=modeler_stage_dir,
            modeler_df=modeler_df,
            feature_cols=feature_cols,
            doe_df=doe_df,
        )

        raw_constraint_defs = (
            (doe_meta or {}).get("constraint_defs")
            or (doe_meta or {}).get("inputs", {}).get("constraint_defs")
            or []
        )
        try:
            constraint_defs = validate_constraint_defs(raw_constraint_defs)
        except Exception as exc:
            print(f"[Explorer] constraint_defs validation failed -> skip constraint policy: {exc}")
            constraint_defs = []

        pre_constraint_defs = [
            c for c in constraint_defs
            if str(c.get("scope", "x_only")).strip().lower() == "x_only"
        ]
        post_constraint_defs = [
            c for c in constraint_defs
            if str(c.get("scope", "x_only")).strip().lower() == "cae_dependent"
        ]
        has_pre_constraints = len(pre_constraint_defs) > 0
        has_post_constraints = len(post_constraint_defs) > 0

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
                has_pre_constraints = False
                pre_constraint_defs = []

        feasibility_payload = None
        if has_post_constraints:
            if modeler_feas_pkl_path and os.path.exists(modeler_feas_pkl_path):
                feasibility_payload = _load_feasibility_model(modeler_feas_pkl_path)
                print(f"[Explorer] feasibility model loaded: {modeler_feas_pkl_path}")
            else:
                print("[Explorer] post constraints exist but feasibility model not found; post penalty disabled.")

        variables = None
        if doe_meta:
            variables = doe_meta.get("variables")
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

        meta_seed = (doe_meta or {}).get("seed")
        seed = self.config.cae.user.seed if self.config.cae else None
        rng_seed = seed if seed is not None else meta_seed
        if rng_seed is None:
            raise RuntimeError("Seed not found in DOE metadata.")

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
            _to_bool_mask(doe_df["success"])
            if "success" in doe_df.columns
            else np.ones((len(doe_df),), dtype=bool)
        )
        if "feasible_final" in doe_df.columns:
            feasible_mask_base = _to_bool_mask(doe_df["feasible_final"])
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
                scope="x_only",
            )
            mask_l, _, _ = evaluate_constraints_batch(
                X=X_lhc_raw,
                var_names=selected_features,
                constraint_defs=pre_constraint_defs,
                scope="x_only",
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
                if str(self.config.cae.user.objective_sense).strip().lower() == "min":
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
            self.config.cae.user.objective_sense
            if self.config.cae
            else (doe_meta or {}).get("objective_sense", "max")
        )
        quantile_threshold = float(self.config.system.quantile_threshold)
        if objective_sense == "min":
            threshold = float(np.quantile(score, 1.0 - quantile_threshold))
            mask = score <= threshold
        else:
            threshold = float(np.quantile(score, quantile_threshold))
            mask = score >= threshold
        df["above_threshold"] = mask

        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        stage_dir = None
        artifacts_dir = None
        if self.run_context:
            stage_dir = os.path.join(self.run_context.run_root, "Explorer")
            artifacts_dir = os.path.join(stage_dir, "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)

        use_timestamp = (
            self.config.cae.system.use_timestamp
            if self.config.cae is not None
            else False
        )
        use_raw = len(selected_features) == 2
        overlay_path = None
        if use_raw:
            overlay_path = plot_raw_overlay(
                X_all=X,
                mask=mask,
                feature_names=selected_features,
                problem_name=doe_problem_name,
                project_root=project_root,
                use_timestamp=bool(use_timestamp),
                save_path=os.path.join(artifacts_dir, "raw_overlay.png") if artifacts_dir else None,
            )

        save_plot = bool(self.config.system.save_plot)
        labels = None
        eps_used = None
        dbscan_plot_path = None
        known_optimum_plot_path = None
        span_heatmap_path = None
        y_gradient_plot_path = None
        dbscan_min_samples = self.config.system.dbscan_min_samples
        if dbscan_min_samples is not None:
            try:
                from sklearn.cluster import DBSCAN
                from sklearn.preprocessing import StandardScaler

                min_samples = dbscan_min_samples or max(5, 2 * len(selected_features))

                mask = df["above_threshold"].to_numpy(dtype=bool)
                X_filtered = X[mask]
                if X_filtered.size:
                    X_scaled = StandardScaler().fit_transform(X_filtered)
                    eps = auto_dbscan_eps_quantile(
                        X_scaled,
                        min_samples,
                        float(self.config.system.dbscan_eps_quantile),
                    )
                    eps_used = eps
                    labels = DBSCAN(
                        eps=eps,
                        min_samples=min_samples,
                    ).fit_predict(X_scaled)
                    cluster_full = np.full(X.shape[0], -1, dtype=int)
                    cluster_full[mask] = labels
                    df["cluster"] = cluster_full

                    if save_plot and use_raw:
                        dbscan_plot_path = plot_raw_dbscan(
                            X_q90=X_filtered,
                            labels=labels,
                            feature_names=selected_features,
                            problem_name=doe_problem_name,
                            project_root=project_root,
                            use_timestamp=bool(use_timestamp),
                            save_path=os.path.join(artifacts_dir, "raw_dbscan.png") if artifacts_dir else None,
                        )
            except Exception as exc:
                print(f"[Explorer] DBSCAN skipped: {exc}")

        x_opt = _resolve_known_optimum(
            self.config.user.known_optimum,
            selected_features,
        )
        if x_opt is not None and save_plot and use_raw:
            try:
                X_q90 = X[mask]
                if X_q90.size:
                    known_optimum_plot_path = plot_raw_known_optimum(
                        X_q90=X_q90,
                        labels=labels if labels is not None else None,
                        x_opt=x_opt,
                        feature_names=selected_features,
                        problem_name=doe_problem_name,
                        project_root=project_root,
                        use_timestamp=bool(use_timestamp),
                        save_path=os.path.join(artifacts_dir, "raw_optimum.png") if artifacts_dir else None,
                    )
            except Exception as exc:
                print(f"[Explorer] Known optimum plot skipped: {exc}")


        # -------------------------------------------------
        # Dual-cluster comparison: model vs objective
        # -------------------------------------------------
        pair_overlay_paths = []
        tsne_overlay_path = None
        pred_stats = {}
        obj_stats = {}
        try:
            # (1) Model prediction clusters from LHC
            min_samples = dbscan_min_samples or max(5, 2 * len(selected_features))
            X_pred_sel, labels_pred, pred_stats = select_top_clusters(
                X=X,
                y=score,
                objective_sense=objective_sense,
                quantile_threshold=quantile_threshold,
                min_samples=min_samples,
                bounds=bounds,
                min_topk_count=int(self.config.system.min_topk_count),
                eps_quantile=float(self.config.system.dbscan_eps_quantile),
            )

            # (2) Objective clusters from executed DOE data
            if doe_df is not None:
                obj_df = doe_df.copy()
                if "feasible_final" in obj_df.columns:
                    f = obj_df["feasible_final"]
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
                    obj_df = obj_df[_to_bool_mask(obj_df["success"])]
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
                min_topk_count=int(self.config.system.min_topk_count),
                eps_quantile=float(self.config.system.dbscan_eps_quantile),
            )
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
        selected_points = None

        selected_bounds, pred_bounds, obj_bounds = compute_selected_bounds(
            X_pred_sel=X_pred_sel,
            X_obj_sel=X_obj_sel,
        )

        if selected_bounds is not None:
            selected_bounds = apply_bounds_margin(
                selected_bounds=selected_bounds,
                bounds=bounds,
                margin_ratio=float(self.config.system.bounds_margin_ratio),
            )

        if selected_bounds is not None:
            # compute bound volume ratio
            ratios = []
            for (lb, ub), (s_lb, s_ub) in zip(bounds, selected_bounds):
                denom = float(ub - lb)
                if denom <= 0:
                    ratios.append(0.0)
                else:
                    ratios.append(max(0.0, float(s_ub - s_lb) / denom))
            vol_ratio = float(np.prod(ratios)) if ratios else 0.0
            if selected_features and len(selected_features) == len(selected_bounds):
                formatted = {
                    name: (lb, ub)
                    for name, (lb, ub) in zip(selected_features, selected_bounds)
                }
                print(f"[Explorer] selected_bounds (union min/max): {formatted}")
            else:
                print(f"[Explorer] selected_bounds (union min/max): {selected_bounds}")
            print(f"[Explorer] selected_bounds volume_ratio={vol_ratio:.4f} ({vol_ratio*100:.2f}%)")
            # persist selected bounds as artifact (json)
            if selected_features and len(selected_features) == len(selected_bounds):
                bounds_payload = {
                    "selected_bounds": {
                        name: {"lb": float(lb), "ub": float(ub)}
                        for name, (lb, ub) in zip(selected_features, selected_bounds)
                    },
                    "volume_ratio": float(vol_ratio),
                    "bounds_order": list(selected_features),
                }
            else:
                bounds_payload = {
                    "selected_bounds": [
                        {"lb": float(lb), "ub": float(ub)} for lb, ub in selected_bounds
                    ],
                    "volume_ratio": float(vol_ratio),
                    "bounds_order": list(selected_features),
                }
            if artifacts_dir:
                bounds_path = os.path.join(artifacts_dir, "selected_bounds.json")
                with open(bounds_path, "w", encoding="utf-8") as f:
                    json.dump(bounds_payload, f, indent=2)
            else:
                project_root = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                out_dir = os.path.join(project_root, "result", "explorer")
                os.makedirs(out_dir, exist_ok=True)
                bounds_path = os.path.join(out_dir, "selected_bounds.json")
                with open(bounds_path, "w", encoding="utf-8") as f:
                    json.dump(bounds_payload, f, indent=2)

        if X_pred_sel.size or X_obj_sel.size:
            if X_pred_sel.size and X_obj_sel.size:
                selected_points = np.vstack([X_pred_sel, X_obj_sel])
            elif X_pred_sel.size:
                selected_points = X_pred_sel
            else:
                selected_points = X_obj_sel

        if save_plot:
            try:
                # Pairwise plots for all feature pairs
                if len(selected_features) >= 2 and X_pred_sel.size and X_obj_sel.size:
                    pairs = [(i, j) for i in range(len(selected_features)) for j in range(i + 1, len(selected_features))]
                    for i, j in pairs:
                        out_path = os.path.join(
                            artifacts_dir,
                            f"pair_dual_{selected_features[i]}_{selected_features[j]}.png",
                        ) if artifacts_dir else None
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
                            artifacts_dir,
                            f"pair_bounds_{selected_features[i]}_{selected_features[j]}.png",
                        ) if artifacts_dir else None
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

                # t-SNE summary (dual)
                tsne_overlay_path = plot_dual_cluster_tsne(
                    X_pred=X_pred_sel,
                    labels_pred=labels_pred,
                    X_obj=X_obj_sel,
                    labels_obj=labels_obj,
                    x_opt=x_opt,
                    problem_name=doe_problem_name,
                    project_root=project_root,
                    use_timestamp=bool(use_timestamp),
                    max_points=int(self.config.system.tsne_max_points),
                    save_path=os.path.join(artifacts_dir, "tsne_dual.png") if artifacts_dir else None,
                )
            except Exception as exc:
                print(f"[Explorer] Dual cluster plots skipped: {exc}")

        workflow_info = (doe_meta or {}).get("workflow_info", {})
        workflow_info["EXPLORER"] = "LHC"

        saver = ResultSaver(use_timestamp=bool(use_timestamp))
        if self.run_context:
            prev_doe_meta = get_stage_metadata_path(self.run_context, "DOE")
            prev_modeler_meta = get_stage_metadata_path(self.run_context, "Modeler")
            prev_doe_ref = (
                os.path.relpath(prev_doe_meta, stage_dir) if prev_doe_meta else None
            )
            prev_modeler_ref = (
                os.path.relpath(prev_modeler_meta, stage_dir) if prev_modeler_meta else None
            )
            inputs = {
                "user_config": os.path.relpath(
                    self.run_context.user_config_snapshot_path,
                    os.path.join(self.run_context.run_root, "Explorer"),
                ),
                "system_config_snapshot": {
                    "n_samples": n_samples,
                    "quantile_threshold": quantile_threshold,
                    "dbscan_min_samples": dbscan_min_samples,
                },
                "previous": {
                    "DOE": prev_doe_ref,
                    "Modeler": prev_modeler_ref,
                },
            }
            resolved_params = {
                "threshold_value": threshold,
                "dbscan_eps_used": eps_used,
                "dbscan_min_samples_used": dbscan_min_samples,
                "base_n_success_feasible": int(base_n),
                "nominal_n": int(nominal_n),
                "target_n_generated": int(n_samples),
                "generated_boundary": int(X_boundary_raw.shape[0]),
                "generated_lhc": int(X_lhc_raw.shape[0]),
                "generated_total_raw": int(pre_generated),
                "generated_total_after_pre": int(X.shape[0]),
                "pre_filter_kept": int(pre_kept),
                "r_used": float(r_used),
                "r_used_source": r_used_source,
                "has_pre_constraints": bool(has_pre_constraints),
                "has_post_constraints": bool(has_post_constraints),
                "post_penalty_active": bool(has_post_penalty),
                "post_lambda": float(post_lambda),
                "post_lambda_source": post_lambda_source,
            }
            n_q_samples = int(mask.sum())
            n_clusters = 0
            cluster_sizes = []
            if labels is not None:
                unique = [int(v) for v in np.unique(labels) if v != -1]
                n_clusters = len(unique)
                cluster_sizes = [int((labels == v).sum()) for v in unique]
            results_summary = {
                "n_candidates_total": int(len(df)),
                "n_q_samples": n_q_samples,
                "n_clusters": n_clusters,
                "cluster_sizes": cluster_sizes,
                "n_generated_raw": int(pre_generated),
                "n_generated_after_pre": int(X.shape[0]),
                "post_penalty_active": bool(has_post_penalty),
            }
            artifacts = {
                "raw_overlay": os.path.relpath(overlay_path, stage_dir) if overlay_path else None,
                "raw_dbscan": os.path.relpath(dbscan_plot_path, stage_dir) if dbscan_plot_path else None,
                "raw_optimum": os.path.relpath(known_optimum_plot_path, stage_dir) if known_optimum_plot_path else None,
                "pair_dual_clusters": [os.path.relpath(p, stage_dir) for p in pair_overlay_paths] if pair_overlay_paths else None,
                "tsne_dual_clusters": os.path.relpath(tsne_overlay_path, stage_dir) if tsne_overlay_path else None,
                "selected_bounds": os.path.relpath(bounds_path, stage_dir) if selected_bounds is not None else None,
            }
            stage_out = saver.save_stage_v2(
                run_root=self.run_context.run_root,
                stage="Explorer",
                problem_name=doe_problem_name,
                df=df,
                inputs=inputs,
                resolved_params=resolved_params,
                results=results_summary,
                artifacts=artifacts,
            )
            update_run_index(self.run_context, "Explorer", stage_out["metadata"])
        else:
            stage_out = saver.save_stage_csv(
                stage="EXPLORER",
                df=df,
                problem_name=doe_problem_name,
                extra_metadata={
                    "seed": rng_seed,
                    "n_samples": n_samples,
                    "workflow_info": workflow_info,
                    "selected_features": selected_features,
                    "model_path": modeler_pkl_path,
                    "training_csv_path": doe_csv_path,
                    "quantile_threshold": quantile_threshold,
                    "objective_sense": objective_sense,
                    "threshold_value": threshold,
                    "dbscan_eps": eps_used,
                    "dbscan_min_samples": dbscan_min_samples,
                    "selected_bounds_path": bounds_path if selected_bounds is not None else None,
                    "selected_bounds_volume_ratio": vol_ratio if selected_bounds is not None else None,
                    "base_n_success_feasible": int(base_n),
                    "nominal_n": int(nominal_n),
                    "target_n_generated": int(n_samples),
                    "generated_boundary": int(X_boundary_raw.shape[0]),
                    "generated_lhc": int(X_lhc_raw.shape[0]),
                    "generated_total_raw": int(pre_generated),
                    "generated_total_after_pre": int(X.shape[0]),
                    "pre_filter_kept": int(pre_kept),
                    "r_used": float(r_used),
                    "r_used_source": r_used_source,
                    "has_pre_constraints": bool(has_pre_constraints),
                    "has_post_constraints": bool(has_post_constraints),
                    "post_penalty_active": bool(has_post_penalty),
                    "post_lambda": float(post_lambda),
                    "post_lambda_source": post_lambda_source,
                },
            )

        return {
            "csv": stage_out["csv"],
            "metadata": stage_out["metadata"],
            "labels": labels,
        }
