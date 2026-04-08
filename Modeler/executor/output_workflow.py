import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from pipeline.run_context import RunContext, get_task_metadata_path, update_run_index
from utils.result_saver import ResultSaver


@dataclass
class ModelerSaveResult:
    metadata_path: str
    hpo_params_csv_path: str | None


def save_modeler_outputs(
    *,
    saver: ResultSaver,
    run_context: RunContext,
    task_dir: str,
    project_root: str,
    config_system: Any,
    config_user: Any,
    problem_name: str,
    model_name: str,
    doe_meta: dict,
    selected_df: pd.DataFrame,
    selected_features: list[str],
    models_count: int,
    kfold_splits: int,
    kfold_repeats: int,
    cv_policy: dict,
    elite_ratio_eff: float,
    n_elite: int,
    has_post_constraints: bool,
    feas_model_kind: str,
    feas_model_stats: dict,
    hpo_params_used: bool,
    hpo_mode: str,
    hpo_n_trials_effective: int | None,
    hpo_lambda_std_effective: float | None,
    best_params: dict | None,
    model_path: str,
    selected_path: str,
    feas_model_path: str | None,
    processed_path: str | None,
    processed_drop_path: str | None,
    processed_drop_df: pd.DataFrame | None,
    keep_debug: bool,
    perm_path: str | None,
    perm_global_path: str | None,
    perm_elite_path: str | None,
    drop_path: str | None,
    drop_global_path: str | None,
    drop_elite_path: str | None,
    plot_path: str | None,
    drop_plot_path: str | None,
    compare_plot_path: str | None,
    secondary_plot_path: str | None,
) -> ModelerSaveResult:
    base_meta = doe_meta or {}
    workflow_info = base_meta.get("workflow_info", {}) if base_meta else {}
    workflow_info["MODELER"] = model_name

    prev_doe_meta = get_task_metadata_path(run_context, "DOE")
    prev_doe_ref = (
        os.path.relpath(prev_doe_meta, task_dir) if prev_doe_meta else None
    )
    inputs = {
        "user_config": os.path.relpath(
            run_context.user_config_snapshot_path,
            task_dir,
        ),
        "system_config_snapshot": asdict(config_system),
        "previous": {"DOE": prev_doe_ref} if prev_doe_ref else {},
    }
    resolved_params = {
        "model_name": model_name,
        "selected_features": selected_features,
        "use_primary_selection": bool(config_system.use_primary_selection),
        "use_secondary_selection": bool(config_user.use_secondary_selection),
        "fi_low_data_mode": bool(cv_policy["low_data"]),
        "fi_np_ratio": float(cv_policy["np_ratio"]),
        "has_post_constraints": bool(has_post_constraints),
        "workflow_info": workflow_info,
    }
    modeler_analysis_meta = {
        "hpo_used": hpo_params_used,
        "hpo_mode": str(hpo_mode),
        "hpo_n_trials_effective": (
            int(hpo_n_trials_effective) if hpo_n_trials_effective is not None else None
        ),
        "hpo_lambda_std_effective": (
            float(hpo_lambda_std_effective) if hpo_lambda_std_effective is not None else None
        ),
        "kfold_splits": int(kfold_splits),
        "kfold_repeats": int(kfold_repeats),
        "cv_valid_min_est": int(cv_policy["valid_min_est"]),
        "fi_elite_ratio_effective": float(elite_ratio_eff),
        "fi_n_elite": int(n_elite),
        "fi_elite_mode": str(config_system.fi_elite_mode),
        "fi_elite_bonus_beta": float(config_system.fi_elite_bonus_beta),
        "fi_final_score_threshold": float(config_system.fi_final_score_threshold),
        "fi_global_score_floor": float(config_system.fi_global_score_floor),
        "fi_use_score_drop": bool(config_system.fi_use_score_drop),
        "fi_channel_weight_perm": float(config_system.fi_weight_perm),
        "fi_channel_weight_drop": float(config_system.fi_weight_drop),
        "fi_bootstrap_min_freq": float(config_system.fi_bootstrap_min_freq),
        "feas_model_kind": feas_model_kind,
    }
    if "forced_by_min_guard" in selected_df.columns:
        guard_features = selected_df.loc[
            selected_df["forced_by_min_guard"].astype(bool),
            "feature",
        ].astype(str).tolist()
        modeler_analysis_meta["anti_collapse_guard_activated"] = len(guard_features) > 0
        if guard_features:
            modeler_analysis_meta["anti_collapse_guard_features"] = guard_features
    if "bootstrap_enabled_effective" in selected_df.columns:
        modeler_analysis_meta["fi_bootstrap_enabled_effective"] = bool(
            selected_df["bootstrap_enabled_effective"].astype(bool).any()
        )
    if "bootstrap_min_freq_effective" in selected_df.columns:
        vals = pd.to_numeric(selected_df["bootstrap_min_freq_effective"], errors="coerce").dropna()
        if not vals.empty:
            modeler_analysis_meta["fi_bootstrap_min_freq_effective"] = float(vals.iloc[0])
    results_summary = {
        "n_models": int(models_count),
        "n_features_selected": int(len(selected_features)),
        "feas_model_stats": feas_model_stats if has_post_constraints else None,
    }

    public_artifacts = {
        "model_path": os.path.relpath(model_path, task_dir),
        "selected_features": os.path.relpath(selected_path, task_dir),
    }
    if feas_model_path:
        public_artifacts["feas_model_path"] = os.path.relpath(feas_model_path, task_dir)

    meta_artifacts = {}
    analysis_path = os.path.join(task_dir, "artifacts", "meta", "analysis.json")
    os.makedirs(os.path.dirname(analysis_path), exist_ok=True)
    with open(analysis_path, "w", encoding="utf-8") as af:
        json.dump(modeler_analysis_meta, af, ensure_ascii=False, indent=2, default=str)
    meta_artifacts["analysis"] = os.path.relpath(analysis_path, task_dir)
    if processed_path:
        meta_artifacts["importance_processed"] = os.path.relpath(processed_path, task_dir)
    if processed_drop_df is not None and not processed_drop_df.empty and processed_drop_path:
        meta_artifacts["importance_processed_drop"] = os.path.relpath(
            processed_drop_path,
            task_dir,
        )

    hpo_params_csv_path = None
    if hpo_params_used and best_params:
        df_params = pd.DataFrame(
            [{"param": k, "value": v} for k, v in best_params.items()]
        )
        _hpo_dir = os.path.dirname(processed_path) if processed_path else os.path.join(task_dir, "artifacts", "meta")
        hpo_params_csv_path = os.path.join(_hpo_dir, "hpo_best_params.csv")
        df_params.to_csv(hpo_params_csv_path, index=False)
        meta_artifacts["hpo_best_params"] = os.path.relpath(hpo_params_csv_path, task_dir)

    debug_artifacts = {}
    if keep_debug:
        if perm_path and os.path.exists(perm_path):
            debug_artifacts["perm_effect_raw"] = os.path.relpath(perm_path, task_dir)
        if perm_global_path and os.path.exists(perm_global_path):
            debug_artifacts["perm_effect_raw_global"] = os.path.relpath(perm_global_path, task_dir)
        if perm_elite_path and os.path.exists(perm_elite_path):
            debug_artifacts["perm_effect_raw_elite"] = os.path.relpath(perm_elite_path, task_dir)
        if drop_path and os.path.exists(drop_path):
            debug_artifacts["score_drop_raw"] = os.path.relpath(drop_path, task_dir)
        if drop_global_path and os.path.exists(drop_global_path):
            debug_artifacts["score_drop_raw_global"] = os.path.relpath(drop_global_path, task_dir)
        if drop_elite_path and os.path.exists(drop_elite_path):
            debug_artifacts["score_drop_raw_elite"] = os.path.relpath(drop_elite_path, task_dir)
    if plot_path and os.path.exists(plot_path):
        debug_artifacts["feature_selection_plot"] = os.path.relpath(plot_path, task_dir)
    if drop_plot_path and os.path.exists(drop_plot_path):
        debug_artifacts["feature_selection_drop_plot"] = os.path.relpath(drop_plot_path, task_dir)
    if compare_plot_path and os.path.exists(compare_plot_path):
        debug_artifacts["feature_selection_compare_plot"] = os.path.relpath(compare_plot_path, task_dir)
    if secondary_plot_path and os.path.exists(secondary_plot_path):
        debug_artifacts["feature_selection_secondary_plot"] = os.path.relpath(secondary_plot_path, task_dir)

    out = saver.save_task_v3(
        run_root=run_context.run_root,
        task="Modeler",
        problem_name=problem_name,
        df=selected_df,
        inputs=inputs,
        resolved_params=resolved_params,
        results=results_summary,
        public_artifacts=public_artifacts,
        meta_artifacts=meta_artifacts,
        debug_artifacts=debug_artifacts,
    )
    update_run_index(run_context, "Modeler", out["metadata"])

    return ModelerSaveResult(
        metadata_path=str(out["metadata"]),
        hpo_params_csv_path=hpo_params_csv_path,
    )
