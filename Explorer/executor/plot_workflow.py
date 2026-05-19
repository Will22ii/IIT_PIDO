from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from Explorer.visualization.explorer_plots import (
    plot_bounds_pair,
    plot_dual_cluster_pair,
)
from Explorer.visualization.plot_doe_vs_optimum import plot_doe_vs_optimum
from pipeline.run_context import RunContext


@dataclass
class ExplorerPlotResult:
    pair_overlay_paths: list[str]
    doe_vs_optimum_plot_paths: list[str]


def resolve_doe_plot_dataframe(
    *,
    run_context: RunContext | None,
    fallback_df: pd.DataFrame,
) -> pd.DataFrame:
    if run_context is None:
        return fallback_df

    debug_csv_path = os.path.join(
        run_context.run_root,
        "DOE",
        "artifacts",
        "debug",
        "doe_results_internal.csv",
    )
    if not os.path.exists(debug_csv_path):
        return fallback_df

    try:
        debug_df = pd.read_csv(debug_csv_path)
        if "exec_scope" in debug_df.columns:
            print(f"[Explorer] DOE debug CSV for plots: {debug_csv_path}")
            return debug_df
    except Exception as exc:
        print(f"[Explorer] DOE debug CSV for plots skipped: {exc}")
    return fallback_df


def render_explorer_debug_plots(
    *,
    enabled: bool,
    debug_dir: str,
    selected_features: list[str],
    X_pred_sel: np.ndarray,
    labels_pred: np.ndarray,
    X_obj_sel: np.ndarray,
    labels_obj: np.ndarray,
    selected_points: np.ndarray,
    selected_bounds: list[tuple[float, float]] | None,
    bounds: list[tuple[float, float]],
    x_opt: object,
    problem_name: str,
    project_root: str,
    use_timestamp: bool,
    doe_plot_df: pd.DataFrame,
    explorer_df: pd.DataFrame,
    known_optimum: object,
    objective_sense: str,
    respect_constraints: bool,
) -> ExplorerPlotResult:
    pair_overlay_paths: list[str] = []
    doe_vs_optimum_plot_paths: list[str] = []
    if not enabled:
        return ExplorerPlotResult(
            pair_overlay_paths=pair_overlay_paths,
            doe_vs_optimum_plot_paths=doe_vs_optimum_plot_paths,
        )

    try:
        if len(selected_features) >= 2 and X_pred_sel.size and X_obj_sel.size:
            pairs = [
                (i, j)
                for i in range(len(selected_features))
                for j in range(i + 1, len(selected_features))
            ]
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
                        problem_name=problem_name,
                        project_root=project_root,
                        use_timestamp=bool(use_timestamp),
                        save_path=out_path,
                    )
                )

        if selected_bounds is not None and selected_points is not None and len(selected_features) >= 2:
            pairs = [
                (i, j)
                for i in range(len(selected_features))
                for j in range(i + 1, len(selected_features))
            ]
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
                        problem_name=problem_name,
                        project_root=project_root,
                        use_timestamp=bool(use_timestamp),
                        save_path=out_path,
                    )
                )

        try:
            plot_out = plot_doe_vs_optimum(
                doe_df=doe_plot_df if doe_plot_df is not None else pd.DataFrame(),
                explorer_df=explorer_df,
                selected_features=selected_features,
                known_optimum=known_optimum,
                objective_sense=objective_sense,
                out_dir=debug_dir,
                respect_constraints=bool(respect_constraints),
            )
            doe_vs_optimum_plot_paths = list(plot_out.get("saved", []))
        except Exception as exc:
            print(f"[Explorer] DOE-vs-optimum plot skipped: {exc}")
    except Exception as exc:
        print(f"[Explorer] Dual cluster plots skipped: {exc}")

    return ExplorerPlotResult(
        pair_overlay_paths=pair_overlay_paths,
        doe_vs_optimum_plot_paths=doe_vs_optimum_plot_paths,
    )
