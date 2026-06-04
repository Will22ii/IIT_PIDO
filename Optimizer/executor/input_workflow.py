from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from Optimizer.config import OptimizerConfig
from pipeline.run_context import RunContext, get_task_metadata_path
from utils.objective_sense import canonicalize_objective_columns
from utils.result_loader import ResultLoader


@dataclass
class ResolvedOptimizerInputs:
    problem_name: str
    seed: int
    objective_sense: str
    variables: list[dict]
    constraint_defs: list[dict]
    doe_df: pd.DataFrame | None
    doe_metadata_path: str | None
    explorer_metadata_path: str | None
    modeler_metadata_path: str | None
    selected_features: list[str]
    selected_bounds: dict[str, tuple[float, float]]
    bounds_source: str
    bounds_path: str | None
    post_feasibility_payload: dict | None
    post_feasibility_model_path: str | None
    post_feasibility_model_kind: str
    cae_metadata_path: str


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid JSON object payload: {path}")
    return payload


def _resolve_existing_cae_metadata_path(
    *,
    config: OptimizerConfig,
    run_context: RunContext | None,
) -> str:
    if run_context is not None:
        path = get_task_metadata_path(run_context, "CAE")
        if path and os.path.exists(path):
            return path
        raise RuntimeError(
            "Optimizer requires existing CAE metadata in run context. "
            "Run CAE task first and then execute Optimizer."
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
        "Optimizer requires existing CAE metadata. "
        "Provide OptimizerConfig.cae_metadata_path or run via pipeline run_context."
    )


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


def _extract_cae_fields(cae_meta: dict) -> tuple[str, list[dict], list[dict], str]:
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


def _resolve_doe_df_optional(
    *,
    config: OptimizerConfig,
    run_context: RunContext | None,
    doe_dataframe: pd.DataFrame | None,
    cae_problem_name: str,
) -> tuple[pd.DataFrame | None, str | None]:
    if doe_dataframe is not None:
        if not isinstance(doe_dataframe, pd.DataFrame):
            raise TypeError("doe_dataframe must be a pandas DataFrame.")
        return doe_dataframe.copy(), None

    user_csv = str(config.user.doe_csv_path or "").strip()
    cfg_csv = str(config.doe_csv_path or "").strip()
    csv_path = user_csv or cfg_csv
    if csv_path:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"DOE CSV not found: {csv_path}")
        return pd.read_csv(csv_path), None

    meta_path = str(config.doe_metadata_path or "").strip()
    if not meta_path and run_context is not None:
        meta_path = str(get_task_metadata_path(run_context, "DOE") or "").strip()
    if not meta_path:
        return None, None

    loader = ResultLoader()
    doe_result = loader.load_task(
        task="DOE",
        metadata_path=meta_path,
        allow_latest_fallback=False,
    )
    doe_problem_name = str(doe_result.problem_name or "").strip()
    if doe_problem_name and doe_problem_name != str(cae_problem_name).strip():
        raise RuntimeError(
            "Problem mismatch between DOE metadata and CAE metadata: "
            f"doe={doe_problem_name}, cae={cae_problem_name}"
        )
    return doe_result.df.copy(), doe_result.metadata_path


def _parse_selected_bounds_payload(payload: dict) -> tuple[dict[str, tuple[float, float]], list[str]]:
    selected = payload.get("selected_bounds")
    order = payload.get("bounds_order", [])
    if not isinstance(order, list):
        order = []

    out: dict[str, tuple[float, float]] = {}
    out_order: list[str] = []

    if isinstance(selected, dict):
        for name, item in selected.items():
            if not isinstance(item, dict):
                continue
            try:
                lb = float(item["lb"])
                ub = float(item["ub"])
            except Exception:
                continue
            key = str(name)
            out[key] = (float(min(lb, ub)), float(max(lb, ub)))
        if order:
            out_order = [str(name) for name in order if str(name) in out]
        else:
            out_order = list(out.keys())
        return out, out_order

    if isinstance(selected, list):
        if not order:
            raise RuntimeError("selected_bounds list format requires bounds_order.")
        for idx, item in enumerate(selected):
            if idx >= len(order) or not isinstance(item, dict):
                continue
            try:
                lb = float(item["lb"])
                ub = float(item["ub"])
            except Exception:
                continue
            key = str(order[idx])
            out[key] = (float(min(lb, ub)), float(max(lb, ub)))
            out_order.append(key)
        return out, out_order

    raise RuntimeError("Invalid selected_bounds payload format.")


def _resolve_explorer_bounds_optional(
    *,
    config: OptimizerConfig,
    run_context: RunContext | None,
    bounds_override: dict[str, tuple[float, float]] | None,
) -> tuple[dict[str, tuple[float, float]], list[str], str | None, str | None, str]:
    if bounds_override:
        clean: dict[str, tuple[float, float]] = {}
        for key, (lb, ub) in bounds_override.items():
            k = str(key)
            lo = float(min(lb, ub))
            hi = float(max(lb, ub))
            clean[k] = (lo, hi)
        return clean, list(clean.keys()), None, None, "override"

    user_bounds = str(config.user.explorer_bounds_path or "").strip()
    cfg_bounds = str(config.explorer_bounds_path or "").strip()
    bounds_path = user_bounds or cfg_bounds
    if bounds_path:
        payload = _read_json(bounds_path)
        bounds_map, order = _parse_selected_bounds_payload(payload)
        return bounds_map, order, None, bounds_path, "path"

    explorer_meta_path = str(config.explorer_metadata_path or "").strip()
    if not explorer_meta_path and run_context is not None:
        explorer_meta_path = str(get_task_metadata_path(run_context, "Explorer") or "").strip()
    if not explorer_meta_path:
        return {}, [], None, None, "cae"
    if not os.path.exists(explorer_meta_path):
        raise FileNotFoundError(f"Explorer metadata not found: {explorer_meta_path}")

    explorer_meta = _read_json(explorer_meta_path)
    inputs = explorer_meta.get("inputs", {}) if isinstance(explorer_meta.get("inputs", {}), dict) else {}
    selected_features = inputs.get("selected_features", [])
    if not isinstance(selected_features, list):
        selected_features = []
    selected_features = [str(v) for v in selected_features]

    artifacts = explorer_meta.get("artifacts", {}) if isinstance(explorer_meta.get("artifacts", {}), dict) else {}
    selected_bounds_ref = None
    if "selected_bounds" in artifacts:
        selected_bounds_ref = artifacts.get("selected_bounds")
    public_artifacts = artifacts.get("public", {}) if isinstance(artifacts.get("public", {}), dict) else {}
    if selected_bounds_ref is None and "selected_bounds" in public_artifacts:
        selected_bounds_ref = public_artifacts.get("selected_bounds")
    if not isinstance(selected_bounds_ref, str) or not selected_bounds_ref.strip():
        return {}, [], explorer_meta_path, None, "cae"

    bounds_path_abs = selected_bounds_ref
    if not os.path.isabs(bounds_path_abs):
        bounds_path_abs = os.path.normpath(os.path.join(os.path.dirname(explorer_meta_path), bounds_path_abs))
    if not os.path.exists(bounds_path_abs):
        raise FileNotFoundError(f"Explorer selected_bounds not found: {bounds_path_abs}")

    bounds_payload = _read_json(bounds_path_abs)
    bounds_map, order_from_payload = _parse_selected_bounds_payload(bounds_payload)
    if selected_features:
        order = [f for f in selected_features if f in bounds_map]
        if not order:
            order = order_from_payload
    else:
        order = order_from_payload

    return bounds_map, order, explorer_meta_path, bounds_path_abs, "explorer"


def _resolve_modeler_metadata_path_optional(
    *,
    config: OptimizerConfig,
    run_context: RunContext | None,
) -> str | None:
    raw = str(config.modeler_metadata_path or "").strip()
    if raw:
        if not os.path.exists(raw):
            raise FileNotFoundError(f"Modeler metadata not found: {raw}")
        return raw
    if run_context is not None:
        p = get_task_metadata_path(run_context, "Modeler")
        if p and os.path.exists(p):
            return p
    return None


def _resolve_modeler_selected_features_optional(modeler_metadata_path: str | None) -> list[str]:
    if not modeler_metadata_path:
        return []
    payload = _read_json(modeler_metadata_path)
    resolved = payload.get("resolved_params", {}) if isinstance(payload.get("resolved_params", {}), dict) else {}
    selected = resolved.get("selected_features", [])
    if not isinstance(selected, list):
        return []
    return [str(v) for v in selected if str(v).strip()]


def _artifact_ref(metadata: dict | None, key: str) -> str | None:
    if not isinstance(metadata, dict):
        return None
    artifacts = metadata.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return None
    if key in artifacts and isinstance(artifacts.get(key), str):
        return artifacts.get(key)
    for layer in ("public", "meta", "debug"):
        layer_map = artifacts.get(layer, {})
        if isinstance(layer_map, dict) and isinstance(layer_map.get(key), str):
            return layer_map.get(key)
    return None


def _resolve_modeler_feasibility_payload_optional(
    modeler_metadata_path: str | None,
) -> tuple[dict | None, str | None, str]:
    if not modeler_metadata_path:
        return None, None, "none"
    try:
        meta = _read_json(modeler_metadata_path)
    except Exception as exc:
        print(f"[Optimizer] Modeler metadata load failed for feasibility model: {exc}")
        return None, None, "none"

    feas_ref = _artifact_ref(meta, "feas_model_path")
    if not isinstance(feas_ref, str) or not feas_ref.strip():
        return None, None, "none"

    feas_path = feas_ref
    if not os.path.isabs(feas_path):
        feas_path = os.path.normpath(os.path.join(os.path.dirname(modeler_metadata_path), feas_path))
    if not os.path.exists(feas_path):
        print(f"[Optimizer] feasibility model path not found: {feas_path}")
        return None, feas_path, "missing"

    try:
        with open(feas_path, "rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        print(f"[Optimizer] feasibility model load failed: {exc}")
        return None, feas_path, "invalid"

    if not isinstance(payload, dict):
        print("[Optimizer] feasibility model payload is not dict; post penalty disabled.")
        return None, feas_path, "invalid"

    kind = str(payload.get("kind", "unknown")).strip().lower() or "unknown"
    return payload, feas_path, kind


def resolve_optimizer_inputs(
    *,
    config: OptimizerConfig,
    run_context: RunContext | None,
    doe_dataframe: pd.DataFrame | None = None,
    bounds_override: dict[str, tuple[float, float]] | None = None,
) -> ResolvedOptimizerInputs:
    cae_meta_path = _resolve_existing_cae_metadata_path(config=config, run_context=run_context)
    cae_meta = _read_json(cae_meta_path)
    problem_name, variables, constraint_defs, objective_sense = _extract_cae_fields(cae_meta)
    seed = _extract_seed_from_cae_metadata(cae_meta=cae_meta, cae_meta_path=cae_meta_path)

    override_sense = str(config.system.objective_sense_override or "").strip().lower()
    if override_sense in {"min", "max"}:
        objective_sense = override_sense

    doe_df, doe_meta_path = _resolve_doe_df_optional(
        config=config,
        run_context=run_context,
        doe_dataframe=doe_dataframe,
        cae_problem_name=problem_name,
    )

    explorer_bounds, explorer_order, explorer_meta_path, bounds_path, bounds_source = _resolve_explorer_bounds_optional(
        config=config,
        run_context=run_context,
        bounds_override=bounds_override,
    )
    modeler_meta_path = _resolve_modeler_metadata_path_optional(config=config, run_context=run_context)
    modeler_selected_features = _resolve_modeler_selected_features_optional(modeler_meta_path)
    feasibility_payload, feasibility_model_path, feasibility_model_kind = (
        _resolve_modeler_feasibility_payload_optional(modeler_meta_path)
    )

    cae_bounds: dict[str, tuple[float, float]] = {}
    cae_order: list[str] = []
    for v in variables:
        if not isinstance(v, dict):
            continue
        name = str(v.get("name", "")).strip()
        if not name:
            continue
        lb = float(v["lb"])
        ub = float(v["ub"])
        cae_bounds[name] = (float(min(lb, ub)), float(max(lb, ub)))
        cae_order.append(name)

    if explorer_bounds:
        selected_bounds = dict(explorer_bounds)
        selected_features = [f for f in explorer_order if f in selected_bounds]
        if not selected_features:
            selected_features = list(selected_bounds.keys())
    else:
        bounds_source = "cae"
        if modeler_selected_features:
            selected_features = [f for f in modeler_selected_features if f in cae_bounds]
            missing_from_cae = [f for f in modeler_selected_features if f not in cae_bounds]
            if missing_from_cae:
                raise RuntimeError(
                    "Modeler selected_features include variables not in CAE variables: "
                    + ", ".join(missing_from_cae)
                )
        else:
            selected_features = list(cae_order)
        selected_bounds = {f: cae_bounds[f] for f in selected_features}

    if not selected_features:
        raise RuntimeError("Resolved selected_features is empty.")

    for feature in selected_features:
        if feature not in selected_bounds:
            raise RuntimeError(f"Missing bounds for selected feature: {feature}")
        lb, ub = selected_bounds[feature]
        if not np.isfinite(float(lb)) or not np.isfinite(float(ub)):
            raise RuntimeError(f"Invalid bounds for feature '{feature}': ({lb}, {ub})")

    if doe_df is not None:
        objective_col = str(config.system.objective_col).strip()
        doe_df = canonicalize_objective_columns(
            doe_df,
            objective_sense=objective_sense,
            objective_col=objective_col,
        )
        doe_cols = {str(c) for c in doe_df.columns}
        matched_features = [f for f in selected_features if f in doe_cols]
        if len(matched_features) == 0:
            print(
                "[Optimizer] DOE initial guess ignored because no selected feature matched DOE columns."
            )
            doe_df = None
            doe_meta_path = None
        else:
            if len(matched_features) < len(selected_features):
                dropped = [f for f in selected_features if f not in set(matched_features)]
                print(
                    "[Optimizer] DOE partial match detected. "
                    f"Use matched features only: {matched_features} "
                    f"(dropped: {dropped})"
                )
                selected_features = list(matched_features)
                selected_bounds = {f: selected_bounds[f] for f in selected_features}

            keep = list(selected_features)
            if objective_col in doe_df.columns:
                keep.append(objective_col)
            if "objective_raw" in doe_df.columns and "objective_raw" not in keep:
                keep.append("objective_raw")
            doe_df = doe_df[keep].copy()
            doe_df = doe_df.dropna(subset=selected_features).reset_index(drop=True)
            if doe_df.empty:
                print("[Optimizer] DOE initial guess ignored because usable rows are empty after filtering.")
                doe_df = None
                doe_meta_path = None

    return ResolvedOptimizerInputs(
        problem_name=problem_name,
        seed=int(seed),
        objective_sense=objective_sense,
        variables=variables,
        constraint_defs=constraint_defs,
        doe_df=doe_df.copy() if isinstance(doe_df, pd.DataFrame) else None,
        doe_metadata_path=doe_meta_path,
        explorer_metadata_path=explorer_meta_path,
        modeler_metadata_path=modeler_meta_path,
        selected_features=selected_features,
        selected_bounds=selected_bounds,
        bounds_source=str(bounds_source),
        bounds_path=bounds_path,
        post_feasibility_payload=feasibility_payload,
        post_feasibility_model_path=feasibility_model_path,
        post_feasibility_model_kind=str(feasibility_model_kind),
        cae_metadata_path=cae_meta_path,
    )
