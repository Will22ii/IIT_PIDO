from __future__ import annotations

import ast
import json
import os
import pickle
from dataclasses import dataclass

import pandas as pd

from DOE.executor.constraint_filter import validate_constraint_defs
from Explorer.config import ExplorerConfig
from Explorer.executor.explorer_utils import resolve_bounds, resolve_selected_features
from pipeline.run_context import RunContext, get_task_metadata_path
from utils.objective_sense import canonicalize_objective_columns


@dataclass
class ExplorerInputBundle:
    cae_metadata_path: str
    problem_name: str
    cae_variables: list[dict]
    constraint_defs: list[dict]
    pre_constraint_defs: list[dict]
    post_constraint_defs: list[dict]
    objective_sense: str
    raw_objective_sense: str
    seed: int
    input_csv_path: str
    input_df: pd.DataFrame
    model_path: str | None
    models: list
    model_feature_cols: list[str]
    selected_features_csv_path: str | None
    modeler_feasibility_model_path: str | None
    selected_features: list[str]
    fi_scores_path: str | None
    variables: list[dict]
    bounds: list[tuple[float, float]]
    has_pre_constraints: bool
    has_post_constraints: bool
    pre_filter_disabled_reason: str | None


def _load_models(pkl_path: str) -> tuple[list, list[str]]:
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    models = payload.get("models", [])
    feature_cols = payload.get("feature_cols", [])
    return models, feature_cols


def load_feasibility_model(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid feasibility model payload format.")
    return payload


def _extract_expr_vars(expr: str) -> set[str]:
    tree = ast.parse(expr, mode="eval")
    return {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name)
    }


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


def _resolve_input_csv_path(*, config: ExplorerConfig, run_context: RunContext | None) -> str:
    if config.doe_csv_path:
        return str(config.doe_csv_path)
    if run_context:
        candidate = os.path.join(
            run_context.run_root,
            "DOE",
            "artifacts",
            "public",
            "doe_results.csv",
        )
        if os.path.exists(candidate):
            return candidate
    raise RuntimeError(
        "Explorer requires an input CSV. Provide ExplorerConfig.doe_csv_path "
        "or run DOE earlier in the same run_context."
    )


def _resolve_modeler_paths(
    *,
    config: ExplorerConfig,
    run_context: RunContext | None,
) -> tuple[str | None, str | None, str | None]:
    model_path = str(config.model_pkl_path) if config.model_pkl_path else None
    selected_features_csv_path = config.selected_features_csv_path
    feasibility_model_path = None
    modeler_task_dir = None

    if run_context:
        modeler_task_dir = os.path.join(run_context.run_root, "Modeler")
        if not model_path:
            candidate = os.path.join(
                modeler_task_dir,
                "artifacts",
                "public",
                "modeler_selected_models.pkl",
            )
            if os.path.exists(candidate):
                model_path = candidate

        if not selected_features_csv_path:
            candidate = os.path.join(
                modeler_task_dir,
                "artifacts",
                "public",
                "selected_features.csv",
            )
            if os.path.exists(candidate):
                selected_features_csv_path = candidate

        candidate = os.path.join(
            modeler_task_dir,
            "artifacts",
            "public",
            "modeler_feas_models.pkl",
        )
        if os.path.exists(candidate):
            feasibility_model_path = candidate

    if model_path and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model bundle not found: {model_path}")

    return model_path, selected_features_csv_path, feasibility_model_path


def _resolve_constraint_policy(
    *,
    raw_constraint_defs: list,
    selected_features: list[str],
) -> tuple[list[dict], list[dict], list[dict], bool, bool, str | None]:
    try:
        constraint_defs = validate_constraint_defs(raw_constraint_defs or [])
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

    if has_pre_constraints:
        allowed_tokens = {
            "abs", "min", "max", "pow", "sqrt", "sin", "cos", "tan",
            "exp", "log", "pi", "e",
        }
        missing_vars: set[str] = set()
        selected_set = set(selected_features)
        for c in pre_constraint_defs:
            expr = str(c.get("expr", ""))
            try:
                names = _extract_expr_vars(expr)
                req = {n for n in names if n not in allowed_tokens}
                missing_vars.update({n for n in req if n not in selected_set})
            except Exception:
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

    return (
        constraint_defs,
        pre_constraint_defs,
        post_constraint_defs,
        has_pre_constraints,
        has_post_constraints,
        pre_filter_disabled_reason,
    )


def resolve_explorer_inputs(
    *,
    config: ExplorerConfig,
    run_context: RunContext | None,
) -> ExplorerInputBundle:
    if config.cae is None:
        raise RuntimeError("ExplorerConfig.cae is required.")

    cae_meta_path = _resolve_existing_cae_metadata_path(
        config=config,
        run_context=run_context,
    )
    cae_meta = _load_cae_metadata(cae_meta_path)
    (
        cae_problem_name,
        cae_variables,
        cae_constraint_defs,
        cae_objective_sense,
    ) = _extract_cae_fields(cae_meta)
    cae_seed = _extract_seed_from_cae_metadata(cae_meta=cae_meta, cae_meta_path=cae_meta_path)
    configured_problem = str(config.cae.user.problem_name).strip()
    if configured_problem and configured_problem != cae_problem_name:
        raise RuntimeError(
            "Problem mismatch between Explorer config and CAE metadata: "
            f"config={configured_problem}, cae_metadata={cae_problem_name}"
        )

    input_csv_path = _resolve_input_csv_path(config=config, run_context=run_context)
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Explorer input CSV not found: {input_csv_path}")
    input_df = pd.read_csv(input_csv_path)
    input_df = canonicalize_objective_columns(
        input_df,
        objective_sense=cae_objective_sense,
        objective_col="objective",
    )
    print(f"[Explorer] Input CSV: {input_csv_path}")
    if "objective" not in input_df.columns:
        raise RuntimeError("Explorer input CSV must include an 'objective' column.")

    model_path, selected_features_csv_path, feasibility_model_path = _resolve_modeler_paths(
        config=config,
        run_context=run_context,
    )

    models: list = []
    feature_cols: list[str] = []
    if model_path:
        models, feature_cols = _load_models(model_path)
        print(
            f"[Explorer] Model bundle loaded: {model_path} "
            f"(models={len(models)}, features={len(feature_cols)})"
        )
    else:
        print("[Explorer] Model bundle not provided; using objective-data layer only.")

    design_features = [
        str(v.get("name"))
        for v in cae_variables
        if isinstance(v, dict) and str(v.get("name", "")).strip()
    ]
    selected_features = resolve_selected_features(
        feature_cols=feature_cols if feature_cols else None,
        design_features=design_features,
        selected_features_csv_path=selected_features_csv_path,
        doe_df=input_df,
    )
    if feature_cols and [str(f) for f in feature_cols] != [str(f) for f in selected_features]:
        raise RuntimeError(
            "Model bundle feature_cols must match active selected features exactly. "
            f"model_bundle={feature_cols}, active_features={selected_features}"
        )

    (
        constraint_defs,
        pre_constraint_defs,
        post_constraint_defs,
        has_pre_constraints,
        has_post_constraints,
        pre_filter_disabled_reason,
    ) = _resolve_constraint_policy(
        raw_constraint_defs=cae_constraint_defs or [],
        selected_features=selected_features,
    )

    variables = list(cae_variables or [])
    if not variables and config.cae and config.cae.user.variables:
        variables = list(config.cae.user.variables)
    if not variables and run_context:
        with open(run_context.user_config_snapshot_path, "r", encoding="utf-8") as f:
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
        df=input_df,
    )

    return ExplorerInputBundle(
        cae_metadata_path=cae_meta_path,
        problem_name=cae_problem_name,
        cae_variables=cae_variables,
        constraint_defs=constraint_defs,
        pre_constraint_defs=pre_constraint_defs,
        post_constraint_defs=post_constraint_defs,
        objective_sense="min",
        raw_objective_sense=str(cae_objective_sense),
        seed=int(cae_seed),
        input_csv_path=input_csv_path,
        input_df=input_df,
        model_path=model_path,
        models=models,
        model_feature_cols=[str(f) for f in feature_cols],
        selected_features_csv_path=selected_features_csv_path,
        modeler_feasibility_model_path=feasibility_model_path,
        selected_features=selected_features,
        fi_scores_path=config.fi_scores_path or selected_features_csv_path,
        variables=variables,
        bounds=bounds,
        has_pre_constraints=bool(has_pre_constraints),
        has_post_constraints=bool(has_post_constraints),
        pre_filter_disabled_reason=pre_filter_disabled_reason,
    )
