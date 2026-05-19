from __future__ import annotations

import json
import os

import pandas as pd

from Optimizer.config import OptimizerConfig
from Optimizer.executor.bo_engine import BOEngineResult
from Optimizer.executor.input_workflow import ResolvedOptimizerInputs
from pipeline.run_context import RunContext, update_run_index
from utils.result_saver import ResultSaver


def save_optimizer_outputs(
    *,
    config: OptimizerConfig,
    run_context: RunContext,
    resolved: ResolvedOptimizerInputs,
    bo_result: BOEngineResult,
) -> dict:
    task_name = "OPT"
    task_dir = os.path.join(run_context.run_root, "OPT")
    artifacts_dir = os.path.join(task_dir, "artifacts")
    public_dir = os.path.join(artifacts_dir, "public")
    meta_dir = os.path.join(artifacts_dir, "meta")
    debug_dir = os.path.join(artifacts_dir, "debug")
    os.makedirs(public_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    best_point_path = os.path.join(public_dir, "best_point.json")
    with open(best_point_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_point": bo_result.best_point,
                "best_objective": float(bo_result.best_objective),
                "best_point_raw": bo_result.best_point_raw,
                "best_objective_raw": float(bo_result.best_objective_raw),
                "objective_sense": str(resolved.objective_sense),
                "post_penalty_active": bool(bo_result.post_penalty_active),
                "post_score_mode": str(bo_result.post_score_mode),
                "post_penalty_lambda": float(bo_result.post_penalty_lambda),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Main CSV는 "OPT 추가 평가점"만 저장한다 (DOE seed 제외).
    opt_point_cols = list(resolved.selected_features) + [
        "objective",
        "objective_raw",
        "objective_effective",
        "p_feasible",
        "pre_feasible",
        "pre_margin",
        "pre_violation",
        "pre_retry_used",
        "pre_generated_count",
        "pre_fallback_used",
        "iter",
        "opt_focus_level",
        "opt_focus_name",
        "focus3_budget_class",
        "focus3_budget_ratio",
        "source_prob_topk",
        "source_prob_boundary",
        "source_prob_random",
        "phase",
        "acq_type",
        "source_mode",
        "segment",
    ]
    history_df = bo_result.history_df.copy()
    for col in opt_point_cols:
        if col not in history_df.columns:
            history_df[col] = pd.NA
    opt_points_df = history_df[opt_point_cols].copy()

    debug_enabled = str(config.system.debug_level).strip().lower() == "on"
    history_full_path = None
    if debug_enabled:
        history_full_path = os.path.join(debug_dir, "optimizer_history_full.csv")
        history_df.to_csv(history_full_path, index=False)

    previous = {}
    if resolved.doe_metadata_path:
        try:
            previous["DOE"] = os.path.relpath(resolved.doe_metadata_path, task_dir)
        except ValueError:
            previous["DOE"] = resolved.doe_metadata_path
    if resolved.explorer_metadata_path:
        try:
            previous["Explorer"] = os.path.relpath(resolved.explorer_metadata_path, task_dir)
        except ValueError:
            previous["Explorer"] = resolved.explorer_metadata_path
    if resolved.modeler_metadata_path:
        try:
            previous["Modeler"] = os.path.relpath(resolved.modeler_metadata_path, task_dir)
        except ValueError:
            previous["Modeler"] = resolved.modeler_metadata_path

    inputs = {
        "user_config": os.path.relpath(run_context.user_config_snapshot_path, task_dir),
        "system_config_snapshot": {
            "acq_type": str(config.system.acq_type),
            "kappa_start": float(config.system.kappa_start),
            "kappa_end": float(config.system.kappa_end),
            "ei_xi": float(config.system.ei_xi),
            "n_restarts": int(config.system.n_restarts),
            "starts_per_iter": int(config.system.starts_per_iter),
            "random_starts_ratio": float(config.system.random_starts_ratio),
            "focus3_budget_policy_enabled": bool(config.system.focus3_budget_policy_enabled),
            "focus3_budget_ultra_low_np": float(config.system.focus3_budget_ultra_low_np),
            "focus3_budget_low_np": float(config.system.focus3_budget_low_np),
            "focus3_budget_normal_np": float(config.system.focus3_budget_normal_np),
            "focus3_ultra_low_source_probs": [
                float(config.system.focus3_ultra_low_source_topk_prob),
                float(config.system.focus3_ultra_low_source_boundary_prob),
                float(config.system.focus3_ultra_low_source_random_prob),
            ],
            "focus3_low_source_probs": [
                float(config.system.focus3_low_source_topk_prob),
                float(config.system.focus3_low_source_boundary_prob),
                float(config.system.focus3_low_source_random_prob),
            ],
            "focus3_normal_source_probs": [
                float(config.system.focus3_normal_source_topk_prob),
                float(config.system.focus3_normal_source_boundary_prob),
                float(config.system.focus3_normal_source_random_prob),
            ],
            "focus3_rich_source_probs": [
                float(config.system.focus3_rich_source_topk_prob),
                float(config.system.focus3_rich_source_boundary_prob),
                float(config.system.focus3_rich_source_random_prob),
            ],
            "init_from_doe_topk": int(config.system.init_from_doe_topk),
            "doe_seed_scope": str(config.system.doe_seed_scope),
            "gp_refit_every": int(config.system.gp_refit_every),
            "objective_col": str(config.system.objective_col),
            "objective_sense_override": config.system.objective_sense_override,
            "enforce_pre_constraints": bool(config.system.enforce_pre_constraints),
            "post_constraint_enabled": bool(config.system.post_constraint_enabled),
            "post_penalty_lambda": float(config.system.post_penalty_lambda),
            "post_p_feasible_min": float(config.system.post_p_feasible_min),
            "post_p_feasible_hard_penalty": float(config.system.post_p_feasible_hard_penalty),
            "post_score_mode": str(config.system.post_score_mode),
            "source_feasible_multiplier": int(config.system.source_feasible_multiplier),
            "source_feasible_retry": int(config.system.source_feasible_retry),
            "source_feasible_min_starts": int(config.system.source_feasible_min_starts),
            "surrogate_only_mode": bool(config.system.surrogate_only_mode),
            "debug_level": str(config.system.debug_level),
        },
        "previous": previous,
        "selected_features": list(resolved.selected_features),
        "selected_bounds_path": resolved.bounds_path,
        "post_feasibility_model_path": resolved.post_feasibility_model_path,
        "post_feasibility_model_kind": str(resolved.post_feasibility_model_kind),
    }

    resolved_params = {
        "seed": int(resolved.seed),
        "objective_sense": str(resolved.objective_sense),
        "n_features": int(len(resolved.selected_features)),
        "n_constraints": int(len(resolved.constraint_defs)),
        "n_samples_requested": int(config.user.n_samples),
        "focus_naming_scheme": "focus1_space_scan__focus2_region_focus__focus3_point_converge__focus4_reserved_final_verify",
        "phase_naming_scheme": "legacy_phase1_phase2_phase3",
    }

    focus_labels: list[str] = []
    if "opt_focus_name" in history_df.columns:
        seen = set()
        for v in history_df["opt_focus_name"].astype(str).tolist():
            if v not in seen:
                focus_labels.append(v)
                seen.add(v)

    phase_labels: list[str] = []
    if "phase" in history_df.columns:
        seen = set()
        for v in history_df["phase"].astype(str).tolist():
            if v not in seen:
                phase_labels.append(v)
                seen.add(v)

    focus3_budget_classes: list[str] = []
    if "focus3_budget_class" in history_df.columns:
        seen = set()
        for v in history_df["focus3_budget_class"].astype(str).tolist():
            if v not in seen:
                focus3_budget_classes.append(v)
                seen.add(v)

    results = {
        "n_iterations": int(bo_result.n_iterations),
        "best_objective": float(bo_result.best_objective),
        "best_objective_raw": float(bo_result.best_objective_raw),
        "best_point": dict(bo_result.best_point),
        "best_point_raw": dict(bo_result.best_point_raw),
        "post_penalty_active": bool(bo_result.post_penalty_active),
        "post_penalty_lambda": float(bo_result.post_penalty_lambda),
        "post_score_mode": str(bo_result.post_score_mode),
        "feasibility_model_kind": str(bo_result.feasibility_model_kind),
        "feasibility_status": str(bo_result.feasibility_status),
        "n_history_rows": int(len(bo_result.history_df)),
        "n_archive_rows": int(len(bo_result.archive_df)),
        "focus_labels": focus_labels,
        "focus3_budget_classes": focus3_budget_classes,
        "phase_labels": phase_labels,
    }

    saver = ResultSaver(use_timestamp=bool(config.cae.system.use_timestamp))
    task_out = saver.save_task_v3(
        run_root=run_context.run_root,
        task=task_name,
        problem_name=resolved.problem_name,
        df=opt_points_df,
        inputs=inputs,
        resolved_params=resolved_params,
        results=results,
        public_artifacts={
            "best_point": best_point_path,
            "optimizer_points": os.path.join(public_dir, "opt_results.csv"),
        },
        debug_artifacts=(
            {"history_full": history_full_path}
            if history_full_path
            else {}
        ),
    )
    update_run_index(run_context, task_name, task_out["metadata"])
    return task_out
