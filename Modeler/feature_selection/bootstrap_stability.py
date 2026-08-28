from __future__ import annotations

import os

import numpy as np
import pandas as pd

from Modeler.executor.data_workflow import _build_elite_mask
from Modeler.executor.importance_analyzer import ImportanceAnalyzer
from Modeler.executor.trainer import ModelTrainer
from Modeler.feature_selection.primary_selection import (
    FeatureSelectionConfig,
    FeatureSelector,
)


def _resolve_n_jobs(n_jobs: int, n_rounds: int) -> int:
    """실제로 쓸 워커 수. 0/1 이하는 순차 실행, -1은 코어 수의 70%.

    전 코어를 잡으면 워커 내부 모델 학습(BLAS 스레드)과 겹쳐 머신이 포화되므로
    자동 모드는 70%만 쓴다.
    """
    try:
        requested = int(n_jobs)
    except Exception:
        return 1
    if requested == 0 or requested == 1:
        return 1
    if requested < 0:
        cpu = os.cpu_count() or 1
        requested = max(1, int(cpu * 0.7))
    return int(max(1, min(requested, max(1, int(n_rounds)))))


def _model_accepts_n_jobs(model_name: str) -> bool:
    """모델 생성자가 n_jobs 인자를 받는지 시그니처로 판정한다.

    받지 않는 모델(GP 등)에 무조건 주입하면 TypeError로 라운드가 전멸한다
    (20260723 배치 사고). 모델 이름 분기가 아니라 시그니처 검사로 일반화한다.
    판정 불가 시 False(미주입)가 안전한 방향이다.
    """
    import inspect

    try:
        from Modeler.Models.registry import get_model_class

        sig = inspect.signature(get_model_class(model_name).__init__)
        if "n_jobs" in sig.parameters:
            return True
        return any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
    except Exception:
        return False


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
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Bootstrap Stability Selection: 서브샘플 K회 반복 → 선택 빈도 < min_freq인 feature 제거."""
    rng = np.random.default_rng(base_seed + 9999)
    n_total = len(df)
    n_sub = max(int(n_total * sample_ratio), kfold_splits + 1)
    all_features = list(selected_df["feature"].astype(str))
    freq = {f: 0 for f in all_features}

    use_elite_scale = str(fs_config.elite_mode).strip().lower() != "off"

    # 서브샘플 인덱스는 단일 순차 rng에서 나오므로 라운드 실행 순서와 무관하게
    # 먼저 전부 뽑아둔다. 이렇게 해야 병렬 실행에서도 각 라운드가 순차 실행과
    # 동일한 서브샘플을 받는다.
    round_indices = [rng.choice(n_total, size=n_sub, replace=False) for _ in range(n_rounds)]

    n_jobs_eff = _resolve_n_jobs(n_jobs, n_rounds)
    print(
        f"[Bootstrap] rounds={n_rounds}, sample={n_sub}/{n_total}, "
        f"min_freq={min_freq}, n_jobs={n_jobs_eff}"
    )

    def _run_round(r_idx: int, indices: np.ndarray) -> list[str] | None:
        """한 라운드를 실행하고 선택된 feature 목록을 돌려준다. 실패하면 None."""
        seed_r = base_seed + 7000 + r_idx
        sub_df = df.iloc[indices].reset_index(drop=True)

        y_sub = sub_df[target_col].values
        elite_mask_sub, n_elite_sub, _ = _build_elite_mask(
            y=y_sub,
            objective_sense=objective_sense,
            ratio_base=elite_ratio_base,
            min_samples=elite_min_samples,
        )

        round_model_params = dict(model_params or {})
        if n_jobs_eff > 1 and _model_accepts_n_jobs(model_name):
            # 라운드를 병렬로 돌릴 때 모델까지 전 코어를 잡으면 oversubscription이 난다.
            # 단, n_jobs를 받는 모델에만 주입한다. GP처럼 받지 않는 모델에 넣으면
            # 생성자 TypeError로 전 라운드가 조용히 죽는다.
            round_model_params.setdefault("n_jobs", 1)

        trainer = ModelTrainer(
            base_random_seed=seed_r,
            target_col=target_col,
            feature_cols=feature_cols,
            model_params=round_model_params,
            model_name=model_name,
            kfold_splits=kfold_splits,
            kfold_repeats=kfold_repeats,
        )
        try:
            train_result = trainer.run(sub_df)
        except Exception:
            return None

        sub_models = train_result["models"]
        analyzer = ImportanceAnalyzer(
            perm_sample_size=perm_sample_size,
            perm_repeats=perm_repeats,
        )
        X_ref_sub = sub_df[feature_cols].astype(float)

        channels_global = analyzer.run_importance_channels(
            models=sub_models,
            fold_predictions=train_result["fold_predictions"],
            X_ref=X_ref_sub,
            problem_name=problem_name,
            random_seed=seed_r,
            subset_mask=None,
            scale_label="global",
            y_true=train_result["y_true"],
            compute_perm=True,
            compute_drop=bool(use_score_drop),
        )
        if use_elite_scale:
            channels_elite = analyzer.run_importance_channels(
                models=sub_models,
                fold_predictions=train_result["fold_predictions"],
                X_ref=X_ref_sub,
                problem_name=problem_name,
                random_seed=seed_r,
                subset_mask=elite_mask_sub,
                scale_label="elite",
                y_true=train_result["y_true"],
                compute_perm=True,
                compute_drop=bool(use_score_drop),
            )
        else:
            channels_elite = {"perm_effect_raw": pd.DataFrame(), "score_drop_raw": pd.DataFrame()}

        drop_global_df_sub = None
        drop_elite_df_sub = None
        if use_score_drop:
            drop_global_df_sub = channels_global.get("score_drop_raw", pd.DataFrame())
            if use_elite_scale:
                drop_elite_df_sub = channels_elite.get("score_drop_raw", pd.DataFrame())

        perm_g = channels_global.get("perm_effect_raw", pd.DataFrame())
        perm_e = channels_elite.get("perm_effect_raw", pd.DataFrame())

        selector = FeatureSelector(fs_config, base_seed=int(base_seed))
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
            return None

        sub_sel = sub_result["selected_features"]
        return [
            str(f) for f in sub_sel.loc[sub_sel["selected"] == True, "feature"].astype(str)
        ]

    if n_jobs_eff > 1:
        from joblib import Parallel, delayed

        round_outcomes = Parallel(n_jobs=n_jobs_eff)(
            delayed(_run_round)(r_idx, indices)
            for r_idx, indices in enumerate(round_indices)
        )
    else:
        round_outcomes = [_run_round(r_idx, idx) for r_idx, idx in enumerate(round_indices)]

    success_count = 0
    for outcome in round_outcomes:
        if outcome is None:
            continue
        success_count += 1
        for feat in outcome:
            if feat in freq:
                freq[feat] += 1

    if success_count < n_rounds:
        print(
            f"[Bootstrap] rounds failed: {int(n_rounds - success_count)}/{n_rounds} "
            f"(freq denominator={max(1, success_count)})"
        )

    # 실패율 임계 무효화: 성공 라운드가 임계 미만이면 freq는 "측정 실패"이지
    # "선택 빈도 0"이 아니다. 이때 freq로 필터하면 라운드 전멸 상황에서
    # real feature까지 stability_fail로 제거되고, 하류의 topk_half 폴백이
    # 엉뚱한 feature를 강제 선택한다(20260723 배치 사고). freq를 NaN으로 남기고
    # bootstrap 필터 없이 primary 판정을 그대로 통과시킨다.
    min_success_ratio = float(getattr(fs_config, "fi_bootstrap_min_success_ratio", 0.5))
    min_success_rounds = max(1, int(np.ceil(min_success_ratio * n_rounds)))
    if success_count < min_success_rounds:
        print(
            f"[Bootstrap] INVALIDATED: success {success_count}/{n_rounds} < "
            f"required {min_success_rounds} (ratio {min_success_ratio}). "
            "freq를 신뢰할 수 없어 bootstrap 필터를 건너뛴다."
        )
        out = selected_df.copy()
        out["bootstrap_freq"] = np.nan
        out["bootstrap_rescue_core"] = False
        out["bootstrap_invalidated"] = True
        return out

    # 빈도 계산 및 필터
    valid_rounds = max(1, success_count)
    out = selected_df.copy()
    out["bootstrap_invalidated"] = False
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

    # E: Bootstrap-null compound gate
    # 부트스트랩 통과(freq >= min_freq) + null importance 양쪽 채널 모두 실패인 feature를 제거.
    # low-data에서 spurious dummy가 일관적으로 선택되는 패턴을 차단.
    # 단, perm_dominant_override=True인 feature는 면제 (real variable 과제거 방지).
    null_compound_removed = []
    if "null_pass" in out.columns:
        for idx, row in out.iterrows():
            if (
                row["selected"]
                and row["bootstrap_freq"] >= min_freq
                and not bool(row.get("null_pass", True))
                and not bool(row.get("bootstrap_rescue_core", False))
                and not bool(row.get("perm_dominant_override", False))
            ):
                out.at[idx, "selected"] = False
                out.at[idx, "reason"] = "bootstrap_null_compound_fail"
                null_compound_removed.append(
                    f"{row['feature']}(freq={row['bootstrap_freq']:.2f},"
                    f"null_pass=False)"
                )
    if null_compound_removed:
        print(f"[Bootstrap] null compound removed: {', '.join(null_compound_removed)}")

    return out
