from __future__ import annotations

import numpy as np
import pandas as pd

from Modeler.executor.data_workflow import _build_elite_mask
from Modeler.executor.importance_analyzer import ImportanceAnalyzer
from Modeler.executor.trainer import ModelTrainer
from Modeler.feature_selection.primary_selection import (
    FeatureSelectionConfig,
    FeatureSelector,
)


def run_bootstrap_stability(
    *,
    selected_df: pd.DataFrame,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_rounds: int,
    sample_ratio: float,
    min_freq: float,
    base_seed: int,
    model_name: str,
    model_params: dict,
    kfold_splits: int,
    kfold_repeats: int,
    perm_sample_size: int,
    perm_repeats: int,
    fs_config: FeatureSelectionConfig,
    use_score_drop: bool,
    low_data: bool,
    problem_name: str,
    objective_sense: str,
    elite_ratio_base: float,
    elite_min_samples: int,
    n_samples: int = 0,
    rescue_global_floor: float = 0.0,
    rescue_very_low_data_only: bool = True,
    very_low_data_n_threshold: int = 55,
) -> pd.DataFrame:
    """Bootstrap Stability Selection: 서브샘플 K회 반복 → 선택 빈도 < min_freq인 feature 제거."""
    rng = np.random.default_rng(base_seed + 9999)
    n_total = len(df)
    n_sub = max(int(n_total * sample_ratio), kfold_splits + 1)
    all_features = list(selected_df["feature"].astype(str))
    freq = {f: 0 for f in all_features}

    use_elite_scale = str(fs_config.elite_mode).strip().lower() != "off"
    print(f"[Bootstrap] rounds={n_rounds}, sample={n_sub}/{n_total}, min_freq={min_freq}")

    success_count = 0
    for r_idx in range(n_rounds):
        seed_r = base_seed + 7000 + r_idx
        indices = rng.choice(n_total, size=n_sub, replace=False)
        sub_df = df.iloc[indices].reset_index(drop=True)

        y_sub = sub_df[target_col].values
        elite_mask_sub, n_elite_sub, _ = _build_elite_mask(
            y=y_sub,
            objective_sense=objective_sense,
            ratio_base=elite_ratio_base,
            min_samples=elite_min_samples,
        )

        trainer = ModelTrainer(
            base_random_seed=seed_r,
            target_col=target_col,
            feature_cols=feature_cols,
            model_params=model_params,
            model_name=model_name,
            kfold_splits=kfold_splits,
            kfold_repeats=kfold_repeats,
        )
        try:
            train_result = trainer.run(sub_df)
        except Exception:
            continue

        sub_models = train_result["models"]
        analyzer = ImportanceAnalyzer(
            perm_sample_size=perm_sample_size,
            perm_repeats=perm_repeats,
        )

        perm_global = analyzer.run_perm_effect(
            models=sub_models,
            fold_predictions=train_result["fold_predictions"],
            X_ref=sub_df[feature_cols].astype(float),
            problem_name=problem_name,
            random_seed=seed_r,
            subset_mask=None,
            scale_label="global",
        )
        if use_elite_scale:
            perm_elite = analyzer.run_perm_effect(
                models=sub_models,
                fold_predictions=train_result["fold_predictions"],
                X_ref=sub_df[feature_cols].astype(float),
                problem_name=problem_name,
                random_seed=seed_r,
                subset_mask=elite_mask_sub,
                scale_label="elite",
            )
        else:
            perm_elite = {"perm_effect_raw": pd.DataFrame()}

        drop_global_df_sub = None
        drop_elite_df_sub = None
        if use_score_drop:
            drop_g = analyzer.run_score_drop(
                models=sub_models,
                fold_predictions=train_result["fold_predictions"],
                X_ref=sub_df[feature_cols].astype(float),
                y_true=train_result["y_true"],
                problem_name=problem_name,
                random_seed=seed_r,
                subset_mask=None,
                scale_label="global",
            )
            drop_global_df_sub = drop_g.get("score_drop_raw", pd.DataFrame())
            if use_elite_scale:
                drop_e = analyzer.run_score_drop(
                    models=sub_models,
                    fold_predictions=train_result["fold_predictions"],
                    X_ref=sub_df[feature_cols].astype(float),
                    y_true=train_result["y_true"],
                    problem_name=problem_name,
                    random_seed=seed_r,
                    subset_mask=elite_mask_sub,
                    scale_label="elite",
                )
                drop_elite_df_sub = drop_e.get("score_drop_raw", pd.DataFrame())

        perm_g = perm_global.get("perm_effect_raw", pd.DataFrame())
        perm_e = perm_elite.get("perm_effect_raw", pd.DataFrame())

        selector = FeatureSelector(fs_config)
        try:
            sub_result = selector.run(
                perm_effect_df=perm_g,
                perm_effect_elite_df=perm_e,
                score_drop_df=drop_global_df_sub,
                score_drop_elite_df=drop_elite_df_sub,
                problem_name=problem_name,
                low_data=bool(low_data),
                n_features=len(feature_cols),
                n_elite=int(n_elite_sub),
                n_samples=int(n_sub),
                keep_debug=False,
            )
        except Exception:
            continue

        success_count += 1
        sub_sel = sub_result["selected_features"]
        for feat in sub_sel.loc[sub_sel["selected"] == True, "feature"].astype(str):
            if feat in freq:
                freq[feat] += 1

    # 빈도 계산 및 필터
    valid_rounds = max(1, success_count)
    out = selected_df.copy()
    freq_series = out["feature"].astype(str).map(lambda f: freq.get(f, 0) / valid_rounds)
    out["bootstrap_freq"] = freq_series.values

    # very_low_data 구제 적용 여부 판정
    is_very_low = bool(low_data) and int(n_samples) < int(very_low_data_n_threshold)
    rescue_active = float(rescue_global_floor) > 0 and (
        not rescue_very_low_data_only or is_very_low
    )
    # Rescue는 "핵심 상위 feature"로 제한한다.
    # global_score 상위 gap_min_retain개(기본 2개) 안에서만 low-frequency 구제를 허용하여
    # dummy 과구제를 줄이고 core feature 보존을 우선한다.
    rescue_core_k = int(max(int(getattr(fs_config, "gap_min_retain", 2)), 1))
    rank_series = (
        out["global_score"].rank(method="first", ascending=False)
        if "global_score" in out.columns
        else pd.Series(np.arange(1, len(out) + 1), index=out.index, dtype=float)
    )
    out["bootstrap_rescue_core"] = rank_series <= float(rescue_core_k)

    removed = []
    rescued = []
    for idx, row in out.iterrows():
        if row["selected"] and row["bootstrap_freq"] < min_freq:
            gs = float(row.get("global_score", 0.0))
            is_rescue_core = bool(row.get("bootstrap_rescue_core", False))
            if rescue_active and is_rescue_core and gs >= float(rescue_global_floor):
                rescued.append(
                    f"{row['feature']}(freq={row['bootstrap_freq']:.2f},"
                    f"gs={gs:.3f},core={is_rescue_core})"
                )
                continue
            out.at[idx, "selected"] = False
            out.at[idx, "reason"] = "bootstrap_stability_fail"
            removed.append(f"{row['feature']}(freq={row['bootstrap_freq']:.2f})")
    if rescued:
        print(
            "[Bootstrap] rescued "
            f"(global_score >= {rescue_global_floor}, core_rank<={rescue_core_k}): "
            + ", ".join(rescued)
        )
    if removed:
        print(f"[Bootstrap] removed: {', '.join(removed)}")
    elif not rescued:
        print("[Bootstrap] all selected features passed stability check")

    return out
