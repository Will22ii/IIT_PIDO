from __future__ import annotations

import json
import os

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
                "objective_sense": str(resolved.objective_sense),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    archive_path = os.path.join(public_dir, "archive_points.csv")
    bo_result.archive_df.to_csv(archive_path, index=False)

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
            "init_from_doe_topk": int(config.system.init_from_doe_topk),
            "doe_seed_scope": str(config.system.doe_seed_scope),
            "gp_refit_every": int(config.system.gp_refit_every),
            "objective_col": str(config.system.objective_col),
            "objective_sense_override": config.system.objective_sense_override,
            "enforce_pre_constraints": bool(config.system.enforce_pre_constraints),
            "surrogate_only_mode": bool(config.system.surrogate_only_mode),
            "debug_level": str(config.system.debug_level),
        },
        "previous": previous,
        "selected_features": list(resolved.selected_features),
        "selected_bounds_path": resolved.bounds_path,
    }

    resolved_params = {
        "seed": int(resolved.seed),
        "objective_sense": str(resolved.objective_sense),
        "n_features": int(len(resolved.selected_features)),
        "n_constraints": int(len(resolved.constraint_defs)),
        "n_samples_requested": int(config.user.n_samples),
    }

    results = {
        "n_iterations": int(bo_result.n_iterations),
        "best_objective": float(bo_result.best_objective),
        "best_point": dict(bo_result.best_point),
        "n_history_rows": int(len(bo_result.history_df)),
        "n_archive_rows": int(len(bo_result.archive_df)),
    }

    saver = ResultSaver(use_timestamp=bool(config.cae.system.use_timestamp))
    task_out = saver.save_task_v3(
        run_root=run_context.run_root,
        task=task_name,
        problem_name=resolved.problem_name,
        df=bo_result.history_df,
        inputs=inputs,
        resolved_params=resolved_params,
        results=results,
        public_artifacts={
            "best_point": best_point_path,
            "archive_points": archive_path,
        },
    )
    update_run_index(run_context, task_name, task_out["metadata"])
    return task_out
