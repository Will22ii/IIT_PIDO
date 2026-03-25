import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from Modeler.feature_selection.primary_selection import (
    FeatureSelectionConfig,
    FeatureSelector,
)
from Modeler.executor.importance_analyzer import ImportanceAnalyzer
from Modeler.visualization.feature_selection_plots import (
    plot_drop_effect,
    plot_perm_drop_compare,
    plot_perm_effect,
    plot_secondary_selection,
)


def _normalize_elite_mode(mode: str) -> str:
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"blend", "bonus", "off"}:
        return "bonus"
    return mode_norm


@dataclass
class FIWorkflowResult:
    selected_df: pd.DataFrame
    processed_df: pd.DataFrame
    processed_drop_df: pd.DataFrame
    processed_path: str
    processed_drop_path: str
    perm_path: str
    perm_global_path: str
    perm_elite_path: str
    drop_path: str | None
    drop_global_path: str | None
    drop_elite_path: str | None


def run_fi_selection_workflow(
    *,
    models: list,
    fold_predictions: list[dict[str, Any]],
    y_true: np.ndarray,
    X_ref: pd.DataFrame,
    elite_mask: np.ndarray,
    problem_name: str,
    base_seed: int,
    perm_sample_size: int,
    perm_repeats: int = 1,
    feature_selection_cfg: dict[str, Any],
    use_score_drop: bool,
    low_data: bool,
    n_features: int,
    n_elite: int,
    n_samples: int,
    keep_debug: bool,
    debug_dir: str,
    meta_dir: str,
) -> FIWorkflowResult:
    analyzer = ImportanceAnalyzer(
        perm_sample_size=perm_sample_size,
        perm_repeats=int(perm_repeats),
    )
    elite_mode = _normalize_elite_mode(feature_selection_cfg.get("elite_mode", "bonus"))
    use_elite_scale = elite_mode != "off"

    importance_global = analyzer.run_perm_effect(
        models=models,
        fold_predictions=fold_predictions,
        X_ref=X_ref,
        problem_name=problem_name,
        random_seed=base_seed,
        subset_mask=None,
        scale_label="global",
    )
    if use_elite_scale:
        importance_elite = analyzer.run_perm_effect(
            models=models,
            fold_predictions=fold_predictions,
            X_ref=X_ref,
            problem_name=problem_name,
            random_seed=base_seed,
            subset_mask=elite_mask,
            scale_label="elite",
        )
        perm_imp_df = pd.concat(
            [
                importance_global.get("perm_effect_raw", pd.DataFrame()),
                importance_elite.get("perm_effect_raw", pd.DataFrame()),
            ],
            ignore_index=True,
        )
    else:
        importance_elite = {"perm_effect_raw": pd.DataFrame()}
        perm_imp_df = importance_global.get("perm_effect_raw", pd.DataFrame()).copy()
        print("[Modeler][FI-ELITE] elite_mode=off -> skip elite permutation importance")

    drop_imp_df = pd.DataFrame()
    if bool(use_score_drop):
        drop_global = analyzer.run_score_drop(
            models=models,
            fold_predictions=fold_predictions,
            X_ref=X_ref,
            y_true=y_true,
            problem_name=problem_name,
            random_seed=base_seed,
            subset_mask=None,
            scale_label="global",
        )
        if use_elite_scale:
            drop_elite = analyzer.run_score_drop(
                models=models,
                fold_predictions=fold_predictions,
                X_ref=X_ref,
                y_true=y_true,
                problem_name=problem_name,
                random_seed=base_seed,
                subset_mask=elite_mask,
                scale_label="elite",
            )
            drop_imp_df = pd.concat(
                [
                    drop_global.get("score_drop_raw", pd.DataFrame()),
                    drop_elite.get("score_drop_raw", pd.DataFrame()),
                ],
                ignore_index=True,
            )
        else:
            drop_imp_df = drop_global.get("score_drop_raw", pd.DataFrame()).copy()
            print("[Modeler][FI-ELITE] elite_mode=off -> skip elite score-drop importance")

    selector = FeatureSelector(
        FeatureSelectionConfig(**feature_selection_cfg)
    )

    perm_global_df = perm_imp_df.loc[
        perm_imp_df["scale"].astype(str) == "global"
    ].reset_index(drop=True)
    perm_elite_df = perm_imp_df.loc[
        perm_imp_df["scale"].astype(str) == "elite"
    ].reset_index(drop=True)

    drop_global_df = pd.DataFrame()
    drop_elite_df = pd.DataFrame()
    if bool(use_score_drop) and "scale" in drop_imp_df.columns:
        drop_global_df = drop_imp_df.loc[
            drop_imp_df["scale"].astype(str) == "global"
        ].reset_index(drop=True)
        drop_elite_df = drop_imp_df.loc[
            drop_imp_df["scale"].astype(str) == "elite"
        ].reset_index(drop=True)

    if keep_debug:
        perm_path = os.path.join(debug_dir, "perm_effect_raw.csv")
        perm_global_path = os.path.join(debug_dir, "perm_effect_raw_global.csv")
        perm_elite_path = os.path.join(debug_dir, "perm_effect_raw_elite.csv")
        perm_imp_df.to_csv(perm_path, index=False)
        perm_global_df.to_csv(perm_global_path, index=False)
        perm_elite_df.to_csv(perm_elite_path, index=False)
    else:
        perm_path = os.path.join(meta_dir, "_perm_effect_tmp.csv")
        perm_global_path = os.path.join(meta_dir, "_perm_effect_global_tmp.csv")
        perm_elite_path = os.path.join(meta_dir, "_perm_effect_elite_tmp.csv")

    if keep_debug:
        drop_path = os.path.join(debug_dir, "score_drop_raw.csv")
        drop_global_path = os.path.join(debug_dir, "score_drop_raw_global.csv")
        drop_elite_path = os.path.join(debug_dir, "score_drop_raw_elite.csv")
    else:
        drop_path = os.path.join(meta_dir, "_score_drop_tmp.csv")
        drop_global_path = os.path.join(meta_dir, "_score_drop_global_tmp.csv")
        drop_elite_path = os.path.join(meta_dir, "_score_drop_elite_tmp.csv")
    if bool(use_score_drop):
        if keep_debug:
            drop_imp_df.to_csv(drop_path, index=False)
            drop_global_df.to_csv(drop_global_path, index=False)
            drop_elite_df.to_csv(drop_elite_path, index=False)
    else:
        drop_path = None
        drop_global_path = None
        drop_elite_path = None

    selection_result = selector.run(
        perm_effect_df=perm_global_df,
        perm_effect_elite_df=perm_elite_df,
        score_drop_df=drop_global_df if bool(use_score_drop) else None,
        score_drop_elite_df=drop_elite_df if bool(use_score_drop) else None,
        perm_effect_path=perm_global_path if keep_debug else None,
        score_drop_path=drop_global_path if (keep_debug and bool(use_score_drop)) else None,
        perm_effect_elite_path=perm_elite_path if keep_debug else None,
        score_drop_elite_path=drop_elite_path if (keep_debug and bool(use_score_drop)) else None,
        problem_name=problem_name,
        low_data=bool(low_data),
        n_features=int(n_features),
        n_elite=int(n_elite),
        n_samples=int(n_samples),
    )
    processed_df = selection_result["importance_processed_pred"]
    processed_drop_df = selection_result.get("importance_processed_drop", pd.DataFrame())
    selected_df = selection_result["selected_features"]

    processed_path = os.path.join(meta_dir, "importance_processed.csv")
    processed_df.to_csv(processed_path, index=False)
    processed_drop_path = os.path.join(meta_dir, "importance_processed_drop.csv")
    if processed_drop_df is not None and not processed_drop_df.empty:
        processed_drop_df.to_csv(processed_drop_path, index=False)
    elif os.path.exists(processed_drop_path):
        os.remove(processed_drop_path)

    return FIWorkflowResult(
        selected_df=selected_df,
        processed_df=processed_df,
        processed_drop_df=processed_drop_df,
        processed_path=processed_path,
        processed_drop_path=processed_drop_path,
        perm_path=perm_path,
        perm_global_path=perm_global_path,
        perm_elite_path=perm_elite_path,
        drop_path=drop_path,
        drop_global_path=drop_global_path,
        drop_elite_path=drop_elite_path,
    )


def cleanup_fi_temp_artifacts(
    *,
    keep_debug: bool,
    perm_path: str,
    perm_global_path: str,
    perm_elite_path: str,
    drop_path: str | None,
    drop_global_path: str | None,
    drop_elite_path: str | None,
) -> None:
    if keep_debug:
        return
    if os.path.exists(perm_path):
        os.remove(perm_path)
    if os.path.exists(perm_global_path):
        os.remove(perm_global_path)
    if os.path.exists(perm_elite_path):
        os.remove(perm_elite_path)
    if drop_path and os.path.exists(drop_path):
        os.remove(drop_path)
    if drop_global_path and os.path.exists(drop_global_path):
        os.remove(drop_global_path)
    if drop_elite_path and os.path.exists(drop_elite_path):
        os.remove(drop_elite_path)


def render_fi_debug_plots(
    *,
    keep_debug: bool,
    debug_dir: str,
    primary_selected_df: pd.DataFrame,
    perm_epsilon: float,
    drop_epsilon: float,
    use_score_drop: bool,
    use_secondary_selection: bool = False,
    secondary_diagnostics: list[dict] | None = None,
) -> tuple[str | None, str | None, str | None, str | None]:
    plot_path = None
    drop_plot_path = None
    compare_plot_path = None
    secondary_plot_path = None
    if not keep_debug:
        return plot_path, drop_plot_path, compare_plot_path, secondary_plot_path

    plot_path = os.path.join(debug_dir, "feature_selection_plot.png")
    plot_perm_effect(
        selected_df=primary_selected_df,
        perm_epsilon=perm_epsilon,
        output_path=plot_path,
    )
    if bool(use_score_drop) and ("drop_metric_mean" in primary_selected_df.columns):
        drop_plot_path = os.path.join(debug_dir, "feature_selection_drop_plot.png")
        plot_drop_effect(
            selected_df=primary_selected_df,
            drop_epsilon=drop_epsilon,
            metric_col="drop_metric_mean",
            output_path=drop_plot_path,
        )
        compare_plot_path = os.path.join(debug_dir, "feature_selection_compare_plot.png")
        plot_perm_drop_compare(
            selected_df=primary_selected_df,
            output_path=compare_plot_path,
        )
    if bool(use_secondary_selection):
        sec_df = pd.DataFrame(secondary_diagnostics or [])
        if not sec_df.empty:
            secondary_plot_path = os.path.join(debug_dir, "feature_selection_secondary_plot.png")
            plot_secondary_selection(
                diagnostics_df=sec_df,
                output_path=secondary_plot_path,
            )
    return plot_path, drop_plot_path, compare_plot_path, secondary_plot_path
