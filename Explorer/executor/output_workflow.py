from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from pipeline.run_context import RunContext, update_run_index
from utils.result_saver import ResultSaver


@dataclass
class ExplorerOutputResult:
    csv: str
    metadata: str
    selected_bounds_path: str | None


def save_selected_bounds_artifact(
    *,
    public_dir: str,
    selected_features: list[str],
    selected_bounds: list[tuple[float, float]] | None,
    volume_ratio: float | None,
    strategy_alias: str,
) -> str | None:
    if selected_bounds is None:
        return None

    if selected_features and len(selected_features) == len(selected_bounds):
        bounds_payload = {
            "selected_bounds": {
                name: {"lb": float(lb), "ub": float(ub)}
                for name, (lb, ub) in zip(selected_features, selected_bounds)
            },
            "volume_ratio": float(volume_ratio) if volume_ratio is not None else None,
            "bounds_order": list(selected_features),
            "strategy_alias": strategy_alias,
        }
    else:
        bounds_payload = {
            "selected_bounds": [
                {"lb": float(lb), "ub": float(ub)} for lb, ub in selected_bounds
            ],
            "volume_ratio": float(volume_ratio) if volume_ratio is not None else None,
            "bounds_order": list(selected_features),
            "strategy_alias": strategy_alias,
        }

    os.makedirs(public_dir, exist_ok=True)
    bounds_path = os.path.join(public_dir, "selected_bounds.json")
    with open(bounds_path, "w", encoding="utf-8") as f:
        json.dump(bounds_payload, f, indent=2)
    return bounds_path


def save_explorer_outputs(
    *,
    run_context: RunContext,
    task_dir: str,
    problem_name: str,
    use_timestamp: bool,
    strategy_id: str,
    safe_strategy_id: str,
    df: pd.DataFrame,
    inputs: dict[str, Any],
    resolved_params: dict[str, Any],
    analysis_metadata: dict[str, Any],
    results_summary: dict[str, Any],
    selected_bounds_path: str | None,
    pair_overlay_paths: list[str],
    doe_vs_optimum_plot_paths: list[str],
) -> ExplorerOutputResult:
    analysis_filename = (
        f"analysis_{safe_strategy_id}.json"
        if safe_strategy_id
        else "analysis.json"
    )
    analysis_path = os.path.join(task_dir, "artifacts", "meta", analysis_filename)
    os.makedirs(os.path.dirname(analysis_path), exist_ok=True)
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis_metadata, f, ensure_ascii=False, indent=2, default=str)

    public_artifacts: dict[str, Any] = {}
    meta_artifacts = {
        "analysis": os.path.relpath(analysis_path, task_dir),
    }
    debug_artifacts: dict[str, Any] = {}
    if selected_bounds_path:
        public_artifacts["selected_bounds"] = os.path.relpath(selected_bounds_path, task_dir)
    if pair_overlay_paths:
        debug_artifacts["pair_dual_clusters"] = [
            os.path.relpath(p, task_dir) for p in pair_overlay_paths
        ]
    if doe_vs_optimum_plot_paths:
        debug_artifacts["doe_vs_optimum_pairwise"] = [
            os.path.relpath(p, task_dir) for p in doe_vs_optimum_plot_paths
        ]

    saver = ResultSaver(use_timestamp=bool(use_timestamp))
    task_out = saver.save_task_v3(
        run_root=run_context.run_root,
        task="Explorer",
        problem_name=problem_name,
        df=df,
        inputs=inputs,
        resolved_params=resolved_params,
        results=results_summary,
        run_tag=strategy_id,
        public_artifacts=public_artifacts,
        meta_artifacts=meta_artifacts,
        debug_artifacts=debug_artifacts,
    )
    update_run_index(run_context, "Explorer", task_out["metadata"])

    return ExplorerOutputResult(
        csv=task_out["csv"],
        metadata=task_out["metadata"],
        selected_bounds_path=selected_bounds_path,
    )
