from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

import pandas as pd

from Optimizer.config import OptimizerConfig, split_optimizer_system_config
from Optimizer.algorithms.result import OptimizerAlgorithmResult
from Optimizer.executor.focus_output_workflow import (
    write_focus_bo_debug_artifacts,
    write_focus_bo_meta_artifacts,
)
from Optimizer.executor.input_workflow import ResolvedOptimizerInputs
from pipeline.run_context import RunContext, update_run_index
from utils.result_saver import ResultSaver


def _algorithm_summary_value(bo_result: OptimizerAlgorithmResult, key: str, default: object = "") -> object:
    summary = bo_result.algorithm_summary if isinstance(bo_result.algorithm_summary, dict) else {}
    return summary.get(key, default)


def _optimizer_algorithm_kind(bo_result: OptimizerAlgorithmResult) -> str:
    engine = str(_algorithm_summary_value(bo_result, "engine", "") or "").strip().lower()
    algorithm_id = str(_algorithm_summary_value(bo_result, "algorithm_id", "") or "").strip().lower()
    pipeline_mode = ""
    if isinstance(bo_result.focus_pipeline_summary, dict):
        pipeline_mode = str(bo_result.focus_pipeline_summary.get("mode", "") or "").strip().lower()
    if engine == "optimizer_runtime" or pipeline_mode == "custom_algorithm":
        return "runtime"
    if algorithm_id in {"focus_bo", "default"} or engine == "focus_bo":
        return "focus"
    return "algorithm"


def _optimizer_schema_info(bo_result: OptimizerAlgorithmResult) -> dict[str, str]:
    algorithm_kind = _optimizer_algorithm_kind(bo_result)
    if algorithm_kind == "focus":
        return {
            "optimizer_progress_scheme": "focus",
            "optimizer_output_schema": "optimizer_points_v2_focus",
            "optimizer_debug_schema": "optimizer_history_v2_focus",
        }
    progress_scheme = "custom_algorithm" if algorithm_kind == "runtime" else algorithm_kind
    return {
        "optimizer_progress_scheme": progress_scheme,
        "optimizer_output_schema": "optimizer_points_v2",
        "optimizer_debug_schema": "optimizer_history_v2",
    }


def _optimizer_result_status(
    *,
    history_df: pd.DataFrame,
    bounds_source: str,
    bo_result: OptimizerAlgorithmResult,
    generated_bounds_available: bool = False,
) -> dict[str, object]:
    algorithm_kind = _optimizer_algorithm_kind(bo_result)
    algorithm_id = str(_algorithm_summary_value(bo_result, "algorithm_id", "") or "")
    algorithm_engine = str(_algorithm_summary_value(bo_result, "engine", "") or "")
    selected_bounds_available = str(bounds_source or "").strip().lower() in {"explorer", "path", "override"}
    algorithm_uses_focus_pipeline = algorithm_kind == "focus"
    optimization_bounds_available = bool(
        selected_bounds_available
        or generated_bounds_available
        or not algorithm_uses_focus_pipeline
    )
    final_focus = ""
    if "segment" in history_df.columns and not history_df.empty:
        segments = [
            str(v).strip()
            for v in history_df["segment"].dropna().astype(str).tolist()
            if str(v).strip()
        ]
        if segments:
            final_focus = segments[-1]
    focus3_executed = bool("segment" in history_df.columns and (history_df["segment"].astype(str) == "focus3").any())
    evaluated = bool(len(history_df) > 0)
    if algorithm_uses_focus_pipeline:
        optimized = bool(focus3_executed and optimization_bounds_available)
        status_basis = "focus3_and_bounds" if optimized else "focus_pipeline_incomplete"
    else:
        optimized = bool(evaluated)
        status_basis = "runtime_evaluated" if optimized else "runtime_no_evaluations"

    if optimized:
        result_status = "optimized"
        converged = True
    else:
        result_status = "exploratory_best"
        converged = False
    return {
        "result_status": result_status,
        "converged": bool(converged),
        "final_focus": str(final_focus),
        "selected_bounds_available": bool(selected_bounds_available),
        "generated_bounds_available": bool(generated_bounds_available),
        "optimization_bounds_available": bool(optimization_bounds_available),
        "focus3_executed": bool(focus3_executed),
        "optimizer_algorithm_kind": str(algorithm_kind),
        "optimizer_algorithm_id": str(algorithm_id),
        "optimizer_algorithm_engine": str(algorithm_engine),
        "optimizer_status_basis": str(status_basis),
    }


def _write_optimizer_debug_tables(
    *,
    debug_dir: str,
    history_df: pd.DataFrame,
    bo_result: OptimizerAlgorithmResult,
    final_bounds: dict,
    resolved: ResolvedOptimizerInputs,
) -> dict[str, str]:
    out: dict[str, str] = {}
    if history_df.empty:
        return out

    summary_rows: list[dict[str, object]] = []
    if "segment" in history_df.columns:
        for segment, g in history_df.groupby(history_df["segment"].astype(str), dropna=False):
            y = pd.to_numeric(g.get("objective_raw", pd.Series(dtype=float)), errors="coerce")
            improved = g.get("raw_improved", pd.Series(False, index=g.index))
            summary_rows.append(
                {
                    "segment": str(segment),
                    "count": int(len(g)),
                    "objective_raw_min": float(y.min()) if y.notna().any() else float("nan"),
                    "objective_raw_median": float(y.median()) if y.notna().any() else float("nan"),
                    "objective_raw_max": float(y.max()) if y.notna().any() else float("nan"),
                    "improved_count": int(pd.Series(improved).fillna(False).astype(bool).sum()),
                    "first_iter": int(pd.to_numeric(g.get("iter", pd.Series(dtype=float)), errors="coerce").min()) if "iter" in g.columns else 0,
                    "last_iter": int(pd.to_numeric(g.get("iter", pd.Series(dtype=float)), errors="coerce").max()) if "iter" in g.columns else 0,
                }
            )
    focus_summary_path = os.path.join(debug_dir, "optimizer_focus_summary.csv")
    pd.DataFrame(summary_rows).to_csv(focus_summary_path, index=False)
    out["focus_summary_csv"] = focus_summary_path

    source_rows: list[dict[str, object]] = []
    if {"segment", "source_mode"}.issubset(history_df.columns):
        for (segment, source), g in history_df.groupby(
            [history_df["segment"].astype(str), history_df["source_mode"].astype(str)],
            dropna=False,
        ):
            y = pd.to_numeric(g.get("objective_raw", pd.Series(dtype=float)), errors="coerce")
            improved = g.get("raw_improved", pd.Series(False, index=g.index))
            source_rows.append(
                {
                    "segment": str(segment),
                    "source_mode": str(source),
                    "count": int(len(g)),
                    "objective_raw_min": float(y.min()) if y.notna().any() else float("nan"),
                    "objective_raw_median": float(y.median()) if y.notna().any() else float("nan"),
                    "improved_count": int(pd.Series(improved).fillna(False).astype(bool).sum()),
                }
            )
    source_summary_path = os.path.join(debug_dir, "optimizer_source_summary.csv")
    pd.DataFrame(source_rows).to_csv(source_summary_path, index=False)
    out["source_summary_csv"] = source_summary_path

    debug_plan_path = os.path.join(debug_dir, "optimizer_debug_plan.json")
    with open(debug_plan_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "problem_name": str(resolved.problem_name),
                "objective_sense": str(resolved.objective_sense),
                "selected_features": list(resolved.selected_features),
                "omitted_feature_policy": dict(resolved.omitted_feature_policy),
                "omitted_feature_values": dict(resolved.omitted_feature_values),
                "bounds_source": str(resolved.bounds_source),
                "final_bounds": final_bounds,
                "focus_pipeline_summary": bo_result.focus_pipeline_summary,
                "algorithm_summary": bo_result.algorithm_summary,
                "focus2_summary_keys": sorted(list(bo_result.focus2_summary.keys())) if isinstance(bo_result.focus2_summary, dict) else [],
                "history_columns": list(history_df.columns),
                "n_history_rows": int(len(history_df)),
                "n_archive_rows": int(len(bo_result.archive_df)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    out["debug_plan_json"] = debug_plan_path
    return out


def save_optimizer_outputs(
    *,
    config: OptimizerConfig,
    run_context: RunContext,
    resolved: ResolvedOptimizerInputs,
    bo_result: OptimizerAlgorithmResult,
) -> dict:
    output_t0 = time.perf_counter()
    task_name = "OPT"
    task_dir = os.path.join(run_context.run_root, "OPT")
    artifacts_dir = os.path.join(task_dir, "artifacts")
    public_dir = os.path.join(artifacts_dir, "public")
    meta_dir = os.path.join(artifacts_dir, "meta")
    debug_dir = os.path.join(artifacts_dir, "debug")
    os.makedirs(public_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    history_df = bo_result.history_df.copy()
    algorithm_kind = _optimizer_algorithm_kind(bo_result)
    uses_focus_pipeline = algorithm_kind == "focus"
    schema_info = _optimizer_schema_info(bo_result)
    generated_bounds_available = bool(
        isinstance(bo_result.focus2_summary, dict)
        and bool(bo_result.focus2_summary.get("final_bounds", {}) or {})
    )
    result_status_info = _optimizer_result_status(
        history_df=history_df,
        bounds_source=str(resolved.bounds_source),
        bo_result=bo_result,
        generated_bounds_available=generated_bounds_available,
    )

    best_point_path = os.path.join(public_dir, "best_point.json")
    with open(best_point_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **result_status_info,
                "best_point": bo_result.best_point,
                "best_objective": float(bo_result.best_objective),
                "best_point_raw": bo_result.best_point_raw,
                "best_objective_raw": float(bo_result.best_objective_raw),
                "objective_sense": str(resolved.objective_sense),
                "algorithm_id": result_status_info.get("optimizer_algorithm_id", ""),
                "algorithm_engine": result_status_info.get("optimizer_algorithm_engine", ""),
                "algorithm_kind": result_status_info.get("optimizer_algorithm_kind", ""),
                "goal": bo_result.algorithm_summary.get("goal") if isinstance(bo_result.algorithm_summary, dict) else None,
                "goal_objective": bo_result.algorithm_summary.get("goal_objective") if isinstance(bo_result.algorithm_summary, dict) else None,
                "goal_hit": bool(bo_result.algorithm_summary.get("goal_hit", False)) if isinstance(bo_result.algorithm_summary, dict) else False,
                "goal_first_hit_iter": int(bo_result.algorithm_summary.get("goal_first_hit_iter", 0)) if isinstance(bo_result.algorithm_summary, dict) else 0,
                "goal_early_stopped": bool(bo_result.algorithm_summary.get("goal_early_stopped", False)) if isinstance(bo_result.algorithm_summary, dict) else False,
                "goal_early_stop_reason": str(bo_result.algorithm_summary.get("goal_early_stop_reason", "")) if isinstance(bo_result.algorithm_summary, dict) else "",
                "post_penalty_active": bool(bo_result.post_penalty_active),
                "post_score_mode": str(bo_result.post_score_mode),
                "post_penalty_lambda": float(bo_result.post_penalty_lambda),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    algorithm_result_path = os.path.join(meta_dir, "optimizer_algorithm.json")
    with open(algorithm_result_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "algorithm_id": result_status_info.get("optimizer_algorithm_id", ""),
                "algorithm_engine": result_status_info.get("optimizer_algorithm_engine", ""),
                "algorithm_kind": result_status_info.get("optimizer_algorithm_kind", ""),
                "algorithm_summary": dict(bo_result.algorithm_summary),
                "result_status": result_status_info,
                "schema": dict(schema_info),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    optimizer_system_config_path = os.path.join(meta_dir, "optimizer_system_config.json")
    with open(optimizer_system_config_path, "w", encoding="utf-8") as f:
        json.dump(
            split_optimizer_system_config(
                config.system,
                algorithm_id=str(result_status_info.get("optimizer_algorithm_id", "") or config.system.algorithm_id),
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 기본 CSV는 "OPT 추가 평가점"만 저장한다(DOE seed 제외).
    opt_point_cols = [
        "iter",
        "segment",
        "opt_focus_level",
        "source_mode",
        "success",
        "feasible",
        "objective",
        "objective_raw",
        "p_feasible",
        "pre_feasible",
        "pre_margin",
        "pre_violation",
        "pre_constraint_worst_id",
        "pre_constraint_worst_margin",
    ] + list(resolved.selected_features)
    if "success" not in history_df.columns:
        history_df["success"] = True
    if "feasible" not in history_df.columns:
        if "pre_feasible" in history_df.columns:
            history_df["feasible"] = history_df["pre_feasible"]
        else:
            history_df["feasible"] = True
    for col in opt_point_cols:
        if col not in history_df.columns:
            history_df[col] = pd.NA
    opt_points_df = history_df[opt_point_cols].copy()

    selected_features_path = os.path.join(meta_dir, "selected_features.csv")
    selected_features_df = pd.DataFrame(
        [
            {
                "order": int(idx),
                "feature": str(feature),
                "lb": float(resolved.selected_bounds[feature][0]),
                "ub": float(resolved.selected_bounds[feature][1]),
                "bounds_source": str(resolved.bounds_source),
            }
            for idx, feature in enumerate(resolved.selected_features)
        ]
    )
    selected_features_df.to_csv(selected_features_path, index=False)

    optimizer_inputs_path = os.path.join(meta_dir, "optimizer_inputs.json")
    with open(optimizer_inputs_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "problem_name": str(resolved.problem_name),
                "objective_col": str(config.system.objective_col),
                "objective_sense": str(resolved.objective_sense),
                "selected_features": list(resolved.selected_features),
                "omitted_feature_policy": dict(resolved.omitted_feature_policy),
                "omitted_feature_values": dict(resolved.omitted_feature_values),
                "evaluation_base_values": dict(resolved.evaluation_base_values),
                "selected_bounds": {
                    str(feature): {
                        "lb": float(resolved.selected_bounds[feature][0]),
                        "ub": float(resolved.selected_bounds[feature][1]),
                    }
                    for feature in resolved.selected_features
                },
                "bounds_source": str(resolved.bounds_source),
                "bounds_path": resolved.bounds_path,
                "cae_metadata_path": resolved.cae_metadata_path,
                "doe_metadata_path": resolved.doe_metadata_path,
                "explorer_metadata_path": resolved.explorer_metadata_path,
                "modeler_metadata_path": resolved.modeler_metadata_path,
                "post_feasibility_model_path": resolved.post_feasibility_model_path,
                "post_feasibility_model_kind": str(resolved.post_feasibility_model_kind),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    focus_regions_path = None
    focus_region_output = {}
    final_bounds = {}
    if uses_focus_pipeline:
        focus_meta = write_focus_bo_meta_artifacts(
            meta_dir=meta_dir,
            bo_result=bo_result,
            resolved=resolved,
        )
        focus_regions_path = str(focus_meta["focus_regions_path"])
        focus_bounds_path = str(focus_meta["focus_bounds_path"])
        focus_region_output = dict(focus_meta["focus_region_output"])
        final_bounds = dict(focus_meta["final_bounds"])
    else:
        focus_bounds_path = None

    debug_enabled = str(config.system.debug_level).strip().lower() == "on"
    history_full_path = None
    focus2_debug_artifacts: dict[str, str | list[str]] = {}
    focus3_debug_artifacts: dict[str, str | list[str]] = {}
    optimizer_debug_tables: dict[str, str] = {}
    debug_write_sec = 0.0
    focus2_plot_sec = 0.0
    focus3_plot_sec = 0.0
    if debug_enabled:
        t0 = time.perf_counter()
        history_full_path = os.path.join(debug_dir, "optimizer_history_full.csv")
        history_df.to_csv(history_full_path, index=False)
        debug_write_sec += float(time.perf_counter() - t0)
        t0 = time.perf_counter()
        optimizer_debug_tables = _write_optimizer_debug_tables(
            debug_dir=debug_dir,
            history_df=history_df,
            bo_result=bo_result,
            final_bounds=final_bounds,
            resolved=resolved,
        )
        debug_write_sec += float(time.perf_counter() - t0)
        if uses_focus_pipeline:
            focus_debug = write_focus_bo_debug_artifacts(
                debug_dir=debug_dir,
                history_df=history_df,
                archive_df=bo_result.archive_df.copy(),
                resolved=resolved,
                focus_region_output=focus_region_output,
                final_bounds=final_bounds,
                known_optimum=config.user.known_optimum,
            )
            focus2_debug_artifacts = dict(focus_debug["focus2_debug_artifacts"])
            focus3_debug_artifacts = dict(focus_debug["focus3_debug_artifacts"])
            focus2_plot_sec += float(focus_debug["focus2_plot_sec"])
            focus3_plot_sec += float(focus_debug["focus3_plot_sec"])
        print(
            "[Optimizer][OutputTiming] "
            f"debug_csv={debug_write_sec:.3f}s "
            f"focus2_plots={focus2_plot_sec:.3f}s "
            f"focus3_plots={focus3_plot_sec:.3f}s",
            flush=True,
        )

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
        "system_config_snapshot": asdict(config.system),
        "previous": previous,
        "selected_features": list(resolved.selected_features),
        "selected_bounds_path": resolved.bounds_path,
        "post_feasibility_model_path": resolved.post_feasibility_model_path,
        "post_feasibility_model_kind": str(resolved.post_feasibility_model_kind),
        "bounds_source": str(resolved.bounds_source),
    }

    resolved_params = {
        "seed": int(resolved.seed),
        "objective_sense": str(resolved.objective_sense),
        "n_features": int(len(resolved.selected_features)),
        "n_constraints": int(len(resolved.constraint_defs)),
        "n_samples_requested": int(config.user.n_samples),
        "optimizer_algorithm_id": str(config.system.algorithm_id),
        **schema_info,
    }

    focus3_budget_classes: list[str] = []
    if "focus3_budget_class" in history_df.columns:
        seen = set()
        for v in history_df["focus3_budget_class"].astype(str).tolist():
            if v not in seen:
                focus3_budget_classes.append(v)
                seen.add(v)

    results = {
        "n_iterations": int(bo_result.n_iterations),
        **result_status_info,
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
        "initial_archive_count": int(bo_result.initial_archive_count),
        "final_train_count": int(bo_result.final_train_count),
        "gp_train_cap": int(bo_result.gp_train_cap),
        "algorithm_summary": dict(bo_result.algorithm_summary),
        "focus2_summary": bo_result.focus2_summary,
        "focus_pipeline_summary": bo_result.focus_pipeline_summary,
        "focus3_budget_classes": focus3_budget_classes,
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
        meta_artifacts={
            "selected_features": selected_features_path,
            "optimizer_inputs": optimizer_inputs_path,
            "optimizer_algorithm": algorithm_result_path,
            "optimizer_system_config": optimizer_system_config_path,
            **({"focus_regions": focus_regions_path} if focus_regions_path else {}),
            **({"focus_bounds": focus_bounds_path} if focus_bounds_path else {}),
        },
        debug_artifacts={
            **({"history_full": history_full_path} if history_full_path else {}),
            **optimizer_debug_tables,
            **(
                {
                    "focus2_bounds_evolution_csv": focus2_debug_artifacts.get("bounds_evolution_csv"),
                    "focus2_region_events_csv": focus2_debug_artifacts.get("region_events_csv"),
                    "focus2_region_timeline_plot": focus2_debug_artifacts.get("region_timeline_plot"),
                    "focus2_bounds_pair_plots": focus2_debug_artifacts.get("bounds_pair_plots", []),
                }
                if focus2_debug_artifacts
                else {}
            ),
            **(
                {
                    "focus3_trajectory_csv": focus3_debug_artifacts.get("trajectory_csv"),
                    "focus3_trajectory_pair_plots": focus3_debug_artifacts.get("trajectory_pair_plots", []),
                }
                if focus3_debug_artifacts
                else {}
            ),
        },
    )
    update_run_index(run_context, task_name, task_out["metadata"])
    print(
        "[Optimizer][OutputTiming] "
        f"total={float(time.perf_counter() - output_t0):.3f}s",
        flush=True,
    )
    return task_out
