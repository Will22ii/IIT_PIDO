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
    kernel_common_best,
    kernel_stable_conservative,
)
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
    plot_bounds_pair,
    plot_raw_dbscan,
    plot_raw_known_optimum,
    plot_raw_overlay,
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
    raw = str(value or "S0_baseline_dual_union").strip()
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw)
    safe = safe.strip("_")
    return safe or "S0_baseline_dual_union"


def _safe_stack(points: list[np.ndarray], n_dim: int) -> np.ndarray:
    valid: list[np.ndarray] = []
    for arr in points:
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float)
        if a.ndim != 2 or a.shape[1] != n_dim or a.shape[0] == 0:
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
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or y.size != X.shape[0] or X.shape[0] < 2:
        return None, False
    dim = int(X.shape[1])
    try:
        gp = GaussianProcessRegressor(
            kernel=kernel_common_best(dim, include_white=False),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=int(seed),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            gp.fit(X, y)
        return gp, False
    except Exception:
        try:
            gp = GaussianProcessRegressor(
                kernel=kernel_stable_conservative(dim, include_white=False),
                alpha=1e-5,
                normalize_y=False,
                n_restarts_optimizer=1,
                random_state=int(seed),
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                gp.fit(X, y)
            return gp, True
        except Exception:
            return None, True


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
    n_det = min(n_total // 2, n)
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
    if sid.startswith("s0_"):
        return "s0"
    if sid.startswith("s2_"):
        return "s2"
    if sid.startswith("s4_"):
        return "s4"
    if sid.startswith("s5_"):
        return "s5"
    if sid.startswith("s8_"):
        return "s8"

    m = str(mode).strip().lower()
    mode_map = {
        "pred_focus": "s2",
        "ei_focus": "s4",
        "dual_refine_ei": "s4",
        "dual_refine_lcb": "s8",
        "wide_quantile": "s5",
        "dual_gradient_refine": "s8",
        "pred_refine_ei": "s4_pred",
        "pred_refine_lcb": "s8_pred",
        "obj_refine_ei": "s4_obj",
        "obj_refine_lcb": "s8_obj",
    }
    return mode_map.get(m, "s0")


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
            threshold = float(np.quantile(score, 1.0 - quantile_threshold))
            mask = score <= threshold
        else:
            threshold = float(np.quantile(score, quantile_threshold))
            mask = score >= threshold
        df["above_threshold"] = mask

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
        keep_debug = debug_level == "full"
        use_raw = len(selected_features) == 2
        overlay_path = None
        if use_raw and keep_debug:
            overlay_path = plot_raw_overlay(
                X_all=X,
                mask=mask,
                feature_names=selected_features,
                problem_name=doe_problem_name,
                project_root=project_root,
                use_timestamp=bool(use_timestamp),
                save_path=os.path.join(debug_dir, "raw_overlay.png"),
            )

        save_plot = bool(self.config.system.save_plot)
        labels = None
        eps_used = None
        dbscan_plot_path = None
        known_optimum_plot_path = None
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
                            save_path=os.path.join(debug_dir, "raw_dbscan.png"),
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
                        save_path=os.path.join(debug_dir, "raw_optimum.png"),
                    )
            except Exception as exc:
                print(f"[Explorer] Known optimum plot skipped: {exc}")


        # -------------------------------------------------
        # Dual-cluster comparison: model vs objective
        # -------------------------------------------------
        pair_overlay_paths = []
        doe_vs_optimum_plot_paths = []
        pred_stats = {}
        obj_stats = {}
        X_obj = np.empty((0, len(selected_features)))
        y_obj = np.array([], dtype=float)
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
        selected_points = np.empty((0, len(selected_features)), dtype=float)

        base_selected_bounds, pred_bounds, obj_bounds = compute_selected_bounds(
            X_pred_sel=X_pred_sel,
            X_obj_sel=X_obj_sel,
        )
        selected_bounds = base_selected_bounds
        strategy_mode = str((self.config.system.strategy_params or {}).get("mode", "")).strip().lower()
        strategy_alias = _resolve_strategy_alias(strategy_id=strategy_id, mode=strategy_mode)

        if strategy_alias == "s0":
            selected_points = _safe_stack([X_pred_sel, X_obj_sel], n_dim=len(selected_features))
            selected_bounds = base_selected_bounds
        elif strategy_alias == "s2":
            selected_points = X_pred_sel
            selected_bounds = _bounds_from_points(X_pred_sel)
        elif strategy_alias == "s5":
            selected_points = _safe_stack([X_pred_sel, X_obj_sel], n_dim=len(selected_features))
            selected_bounds = base_selected_bounds
        elif strategy_alias in {"s4", "s8", "s4_pred", "s8_pred", "s4_obj", "s8_obj"}:
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
            if source_mode == "dual":
                n_pred_starts = 20
                n_obj_starts = 20
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
                    scale=1.15,
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
            ) -> np.ndarray:
                if gp_model is None or starts.ndim != 2 or starts.shape[0] == 0 or y_train.size == 0:
                    return np.empty((0, d), dtype=float)
                if objective_sense == "max":
                    y_best = float(np.max(y_train))
                else:
                    y_best = float(np.min(y_train))
                lb_opt, ub_opt = _opt_bounds_from_starts(starts=starts, region_bounds=region_bounds)
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
                        post_feasible_prob_fn=_post_prob_fn if has_post_penalty else None,
                        post_penalty_lambda=float(post_lambda if has_post_penalty else 0.0),
                    )
                    if x_opt is not None:
                        out.append(np.asarray(x_opt, dtype=float).reshape(-1))
                if not out:
                    return np.empty((0, d), dtype=float)
                return _dedup_rows(np.vstack(out))

            pred_seed_pool = np.asarray(X_pred_sel, dtype=float)
            if not use_pred_model:
                pred_seed_pool = np.empty((0, d), dtype=float)
            elif pred_seed_pool.ndim != 2 or pred_seed_pool.shape[0] == 0:
                order = np.argsort(score)
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
                )
            obj_starts = np.empty((0, d), dtype=float)
            if n_obj_starts > 0 and use_obj_model and obj_seed_pool.shape[0] > 0:
                obj_starts = _select_multistarts(
                    points=obj_seed_pool,
                    values=obj_scores,
                    objective_sense=objective_sense,
                    total=n_obj_starts,
                    rng=rng,
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

            ref_pred = np.empty((0, d), dtype=float)
            if use_pred_model:
                ref_pred = _refine_with_gp(
                    gp_model=gp_pred,
                    starts=pred_starts,
                    y_train=y_pred_train,
                    region_bounds=pred_region_used,
                )
            ref_obj = np.empty((0, d), dtype=float)
            if use_obj_model:
                ref_obj = _refine_with_gp(
                    gp_model=gp_obj,
                    starts=obj_starts,
                    y_train=y_obj_train,
                    region_bounds=obj_region_used,
                )

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
            selected_bounds = _bounds_from_points(selected_points, q_low=0.01, q_high=0.99)
        else:
            selected_points = _safe_stack([X_pred_sel, X_obj_sel], n_dim=len(selected_features))
            selected_bounds = base_selected_bounds

        if selected_bounds is None:
            if strategy_alias in {"s0", "s5"} and base_selected_bounds is not None:
                selected_bounds = base_selected_bounds
                selected_points = _safe_stack([selected_points, X_pred_sel, X_obj_sel], n_dim=len(selected_features))
            elif strategy_alias in {"s2", "s4_pred", "s8_pred"} and pred_bounds is not None:
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

        if selected_bounds is not None:
            selected_bounds = apply_bounds_margin(
                selected_bounds=selected_bounds,
                bounds=bounds,
                margin_ratio=float(self.config.system.bounds_margin_ratio),
                min_volume_ratio=float(self.config.system.bounds_min_volume_ratio),
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
                "dbscan_min_samples": dbscan_min_samples,
                "debug_level": debug_level,
                "strategy_id": strategy_id,
                "probe_multistart": int(self.config.system.probe_multistart),
                "strategy_params": dict(self.config.system.strategy_params or {}),
            },
            "previous": previous,
            "selected_features": selected_features,
            "model_path_input": modeler_pkl_path,
            "doe_csv_input": doe_csv_path,
        }
        resolved_params = {
            "seed": int(rng_seed),
            "objective_sense": objective_sense,
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
            "pre_filter_disabled_reason": pre_filter_disabled_reason,
            "feas_model_kind_used": feasibility_model_kind_used,
            "feas_model_path_used": feasibility_model_path_used,
            "selected_bounds_volume_ratio": vol_ratio,
            "strategy_id": strategy_id,
            "strategy_alias": strategy_alias,
            "strategy_mode": strategy_mode,
            "probe_multistart": int(self.config.system.probe_multistart),
            "workflow_info": workflow_info,
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
            "pred_mean_volume": pred_mean_vol,
            "obj_mean_volume": obj_mean_vol,
            "strategy_id": strategy_id,
            "strategy_alias": strategy_alias,
        }

        public_artifacts = {}
        meta_artifacts = {}
        debug_artifacts = {}
        if bounds_path:
            public_artifacts["selected_bounds"] = os.path.relpath(bounds_path, task_dir)
        if overlay_path:
            debug_artifacts["raw_overlay"] = os.path.relpath(overlay_path, task_dir)
        if dbscan_plot_path:
            debug_artifacts["raw_dbscan"] = os.path.relpath(dbscan_plot_path, task_dir)
        if known_optimum_plot_path:
            debug_artifacts["raw_optimum"] = os.path.relpath(known_optimum_plot_path, task_dir)
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
            "labels": labels,
            "strategy_id": strategy_id,
            "selected_bounds_path": bounds_path,
            "selected_bounds_volume_ratio": vol_ratio,
        }
