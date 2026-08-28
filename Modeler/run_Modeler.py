# Modeler 실행 진입점

import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Modeler.executor.data_workflow import (
    prepare_modeler_data_policy,
)
from Modeler.executor.hpo_workflow import (
    resolve_hpo_params,
)
from Modeler.executor.hpo_runner import supported_hpo_models
from Modeler.executor.input_workflow import (
    ensure_modeler_run_context,
    resolve_modeler_input,
)
from Modeler.executor.model_artifact_workflow import train_and_save_model_artifacts
from Modeler.executor.output_workflow import save_modeler_outputs
from Modeler.feature_selection.finalize_selection import finalize_selected_features
from Modeler.executor.trainer import ModelTrainer
from Modeler.executor.feature_selection_workflow import (
    cleanup_fi_temp_artifacts,
    render_fi_debug_plots,
    run_fi_selection_workflow,
)
from Modeler.feature_selection import (
    SecondarySelectionConfig,
    merge_secondary_features,
    run_secondary_selection,
)
from utils.result_saver import ResultSaver
from Modeler.config import ModelerConfig, build_feature_selection_config
from pipeline.run_context import RunContext


def _normalize_debug_level(value: str | None) -> str:
    level = str(value or "on").strip().lower()
    if level == "full":
        level = "on"
    if level not in {"off", "on"}:
        raise ValueError("Modeler debug_level must be one of: off, on")
    return level


def run_modeler(
    *,
    config: ModelerConfig,
    run_context: RunContext | None = None,
    doe_dataframe: "pd.DataFrame | None" = None,
) -> None:
    print("===================================")
    print(" MODELER 실행 시작")
    print("===================================")

    input_result = resolve_modeler_input(
        config=config,
        run_context=run_context,
        doe_dataframe=doe_dataframe,
    )
    df = input_result.df
    doe_meta = input_result.doe_meta
    csv_path = input_result.csv_path
    problem_name = input_result.problem_name
    base_seed = input_result.base_seed
    cae_variables = input_result.cae_variables
    cae_constraint_defs = input_result.cae_constraint_defs
    cae_objective_sense = input_result.cae_objective_sense
    cae_metadata_path = input_result.cae_metadata_path

    model_name = config.user.model_name
    use_hpo = config.user.use_hpo
    target_col = config.user.target_col
    use_primary_selection = bool(config.system.use_primary_selection)
    use_secondary_selection = bool(config.user.use_secondary_selection)
    if not use_primary_selection and use_secondary_selection:
        print(
            "[Modeler] WARNING: use_secondary_selection=True이지만 "
            "use_primary_selection=False → Secondary Selection도 비활성화됩니다."
        )
        use_secondary_selection = False
    raw_objective_sense = str(cae_objective_sense)
    objective_sense = "min"
    use_timestamp = (
        config.cae.system.use_timestamp if config.cae is not None else False
    )
    hpo_config = config.system.hpo_config
    configured_kfold_splits = config.system.kfold_splits
    configured_kfold_repeats = config.system.kfold_repeats
    fs_config = build_feature_selection_config(config.system)
    perm_sample_size = config.system.perm_sample_size
    perm_repeats = config.system.perm_repeats
    debug_level = _normalize_debug_level(config.system.debug_level)
    keep_debug = debug_level == "on"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if str(model_name).strip().lower() not in supported_hpo_models() and use_hpo:
        use_hpo = False

    data_policy = prepare_modeler_data_policy(
        df=df,
        target_col=target_col,
        doe_meta=doe_meta or {},
        cae_variables=cae_variables,
        cae_constraint_defs=cae_constraint_defs,
        objective_sense=objective_sense,
        system_cfg=config.system,
        keep_debug=keep_debug,
        configured_kfold_splits=int(configured_kfold_splits),
        configured_kfold_repeats=int(configured_kfold_repeats),
    )
    df = data_policy.df
    df_all_for_feas = data_policy.df_all_for_feas
    variables = data_policy.variables
    constraint_defs = data_policy.constraint_defs
    has_post_constraints = data_policy.has_post_constraints
    feature_cols = data_policy.feature_cols
    cv_policy = data_policy.cv_policy
    kfold_splits = data_policy.kfold_splits
    kfold_repeats = data_policy.kfold_repeats
    elite_mask = data_policy.elite_mask
    n_elite = data_policy.n_elite
    elite_ratio_eff = data_policy.elite_ratio_eff

    X = df[feature_cols].values
    y = df[target_col].values

    run_context = ensure_modeler_run_context(
        run_context=run_context,
        project_root=project_root,
        problem_name=problem_name,
        base_seed=int(base_seed),
        objective_sense=raw_objective_sense,
        variables=variables,
        cae_metadata_path=cae_metadata_path,
    )

    hpo_result = resolve_hpo_params(
        use_hpo=bool(use_hpo),
        model_name=model_name,
        hpo_config=hpo_config,
        use_timestamp=bool(use_timestamp),
        project_root=project_root,
        problem_name=problem_name,
        objective_sense=objective_sense,
        target_col=target_col,
        X=X,
        y=y,
        base_seed=int(base_seed),
        kfold_splits=int(kfold_splits),
        low_data=bool(cv_policy["low_data"]),
    )
    best_params = hpo_result.best_params
    hpo_params_used = hpo_result.hpo_params_used
    hpo_mode = hpo_result.hpo_mode
    hpo_n_trials_effective = hpo_result.hpo_n_trials_effective
    hpo_lambda_std_effective = hpo_result.hpo_lambda_std_effective

    saver = ResultSaver(use_timestamp=use_timestamp)
    task_dir = os.path.join(run_context.run_root, "Modeler")
    artifacts_root = os.path.join(task_dir, "artifacts")
    public_dir = os.path.join(artifacts_root, "public")
    meta_dir = os.path.join(artifacts_root, "meta")
    debug_dir = os.path.join(artifacts_root, "debug")
    os.makedirs(public_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # feature selection 분기
    # ------------------------------------------------------------------
    if use_primary_selection:
        # === primary selection ON: FI용 학습 → FI 계산 → primary → (secondary) ===
        trainer = ModelTrainer(
            base_random_seed=base_seed,
            target_col=target_col,
            feature_cols=feature_cols,
            model_params=best_params,
            model_name=model_name,
            kfold_splits=kfold_splits,
            kfold_repeats=kfold_repeats,
        )
        train_result = trainer.run(df)
        models = train_result["models"]
        _np_ratio = float(cv_policy.get("np_ratio", float(len(X)) / max(len(feature_cols), 1)))
        fi_result = run_fi_selection_workflow(
            models=models,
            fold_predictions=train_result["fold_predictions"],
            y_true=train_result["y_true"],
            X_ref=df[feature_cols].astype(float),
            elite_mask=elite_mask,
            problem_name=problem_name,
            base_seed=base_seed,
            perm_sample_size=perm_sample_size,
            perm_repeats=int(perm_repeats),
            fs_config=fs_config,
            use_score_drop=bool(config.system.fi_use_score_drop),
            low_data=bool(cv_policy["low_data"]),
            n_features=len(feature_cols),
            n_elite=int(n_elite),
            n_samples=len(X),
            keep_debug=keep_debug,
            debug_dir=debug_dir,
            meta_dir=meta_dir,
            # bootstrap stability selection
            bootstrap_enabled=bool(config.system.fi_bootstrap_enabled),
            bootstrap_np_ratio=_np_ratio,
            bootstrap_np_threshold=float(config.system.fi_bootstrap_np_threshold),
            bootstrap_rounds=int(config.system.fi_bootstrap_rounds),
            bootstrap_sample_ratio=float(config.system.fi_bootstrap_sample_ratio),
            bootstrap_min_freq=float(config.system.fi_bootstrap_min_freq),
            bootstrap_df=df,
            bootstrap_target_col=target_col,
            bootstrap_model_name=model_name,
            bootstrap_model_params=best_params,
            bootstrap_kfold_splits=int(kfold_splits),
            bootstrap_kfold_repeats=int(kfold_repeats),
            bootstrap_objective_sense=objective_sense,
            bootstrap_elite_ratio_base=float(config.system.fi_elite_ratio_base),
            bootstrap_elite_min_samples=int(config.system.fi_elite_min_samples),
            bootstrap_n_jobs=int(config.system.fi_bootstrap_n_jobs),
        )
        selected_df = fi_result.selected_df
        processed_df = fi_result.processed_df
        processed_drop_df = fi_result.processed_drop_df
        processed_path = fi_result.processed_path
        processed_drop_path = fi_result.processed_drop_path
        perm_path = fi_result.perm_path
        perm_global_path = fi_result.perm_global_path
        perm_elite_path = fi_result.perm_elite_path
        drop_path = fi_result.drop_path
        drop_global_path = fi_result.drop_global_path
        drop_elite_path = fi_result.drop_elite_path

        selection_result = finalize_selected_features(
            selected_df=selected_df,
            feature_cols=feature_cols,
            constraint_defs=constraint_defs,
            public_dir=public_dir,
            keep_debug=keep_debug,
            use_score_drop=bool(config.system.fi_use_score_drop),
        )
        primary_selected_df = selection_result.selected_df.copy()
        selected_df = selection_result.selected_df
        selected_features = selection_result.selected_features
        selected_path = selection_result.selected_path

        secondary_diagnostics: list[dict] = []
        if use_secondary_selection:
            core_features = list(selected_features)
            core_set = set(str(f) for f in core_features)
            secondary_candidates = [
                f for f in feature_cols
                if str(f) not in core_set
            ]
            df_secondary = df.loc[elite_mask].reset_index(drop=True)
            if keep_debug:
                print(
                    "[Modeler][Secondary][DATA] "
                    f"N={len(df_secondary)} core={len(core_features)} "
                    f"candidates={len(secondary_candidates)} "
                    f"elite_ratio_eff={float(elite_ratio_eff):.3f}"
                )
            secondary_cfg = SecondarySelectionConfig(
                target_kr=int(config.system.secondary_target_kr),
                min_repeats=int(config.system.secondary_min_repeats),
                min_delta_r2=float(config.system.secondary_min_delta_r2),
                min_freq=float(config.system.secondary_min_freq),
            )
            secondary_result = run_secondary_selection(
                df=df_secondary,
                target_col=target_col,
                base_seed=int(base_seed),
                model_name=str(model_name),
                model_params=dict(best_params or {}),
                kfold_splits=int(kfold_splits),
                kfold_repeats=int(kfold_repeats),
                core_features=core_features,
                candidate_features=secondary_candidates,
                cfg=secondary_cfg,
                keep_debug=keep_debug,
            )
            selected_df, selected_features = merge_secondary_features(
                selected_df=selected_df,
                selected_features=selected_features,
                secondary_features=secondary_result.selected_features,
            )
            secondary_diagnostics = list(secondary_result.diagnostics or [])
            selected_df.to_csv(selected_path, index=False)

        cleanup_fi_temp_artifacts(
            keep_debug=keep_debug,
            perm_path=perm_path,
            perm_global_path=perm_global_path,
            perm_elite_path=perm_elite_path,
            drop_path=drop_path,
            drop_global_path=drop_global_path,
            drop_elite_path=drop_elite_path,
        )

        plot_path, drop_plot_path, compare_plot_path, secondary_plot_path = render_fi_debug_plots(
            keep_debug=keep_debug,
            debug_dir=debug_dir,
            primary_selected_df=primary_selected_df,
            perm_epsilon=float(config.system.perm_epsilon),
            drop_epsilon=float(config.system.fi_drop_epsilon),
            use_score_drop=bool(config.system.fi_use_score_drop),
            use_secondary_selection=bool(use_secondary_selection),
            secondary_diagnostics=secondary_diagnostics,
        )
        if plot_path:
            print(f"- Saved feature selection plot: {plot_path}")

    else:
        # === primary selection OFF: 전체 feature 사용, FI 스킵 ===
        print(
            "[Modeler] Primary Selection OFF → "
            f"모든 피쳐({len(feature_cols)}개)를 사용하여 모델을 생성합니다."
        )
        selected_features = list(feature_cols)
        selected_df = pd.DataFrame({
            "feature": feature_cols,
            "selected": True,
            "selected_by": "all_features_no_selection",
        })
        selected_path = os.path.join(public_dir, "selected_features.csv")
        selected_df.to_csv(selected_path, index=False)

        # FI 관련 아티팩트 없음
        processed_df = None
        processed_drop_df = None
        processed_path = None
        processed_drop_path = None
        perm_path = None
        perm_global_path = None
        perm_elite_path = None
        drop_path = None
        drop_global_path = None
        drop_elite_path = None
        plot_path = None
        drop_plot_path = None
        compare_plot_path = None
        secondary_plot_path = None

    artifact_result = train_and_save_model_artifacts(
        df=df,
        df_all_for_feas=df_all_for_feas,
        selected_features=selected_features,
        target_col=target_col,
        base_seed=base_seed,
        model_name=model_name,
        best_params=best_params,
        kfold_splits=int(kfold_splits),
        kfold_repeats=int(kfold_repeats),
        public_dir=public_dir,
        problem_name=problem_name,
        has_post_constraints=bool(has_post_constraints),
    )
    model_path = artifact_result.model_path
    feas_model_path = artifact_result.feas_model_path
    feas_model_kind = artifact_result.feas_model_kind
    feas_model_stats = artifact_result.feas_model_stats
    print(f"[Modeler] Selected features ({len(selected_features)}): {', '.join(map(str, selected_features))}")
    print(f"[Modeler] Selected features CSV: {selected_path}")

    save_result = save_modeler_outputs(
        saver=saver,
        run_context=run_context,
        task_dir=task_dir,
        project_root=project_root,
        config_system=config.system,
        config_user=config.user,
        problem_name=problem_name,
        raw_objective_sense=raw_objective_sense,
        internal_objective_sense=objective_sense,
        model_name=model_name,
        doe_meta=doe_meta or {},
        selected_df=selected_df,
        selected_features=selected_features,
        models_count=int(kfold_splits) * int(kfold_repeats),
        kfold_splits=int(kfold_splits),
        kfold_repeats=int(kfold_repeats),
        cv_policy=cv_policy,
        elite_ratio_eff=float(elite_ratio_eff),
        n_elite=int(n_elite),
        has_post_constraints=bool(has_post_constraints),
        feas_model_kind=feas_model_kind,
        feas_model_stats=feas_model_stats,
        hpo_params_used=bool(hpo_params_used),
        hpo_mode=hpo_mode,
        hpo_n_trials_effective=hpo_n_trials_effective,
        hpo_lambda_std_effective=hpo_lambda_std_effective,
        best_params=best_params,
        model_path=model_path,
        selected_path=selected_path,
        feas_model_path=feas_model_path,
        processed_path=processed_path,
        processed_drop_path=processed_drop_path,
        processed_drop_df=processed_drop_df,
        keep_debug=keep_debug,
        perm_path=perm_path,
        perm_global_path=perm_global_path,
        perm_elite_path=perm_elite_path,
        drop_path=drop_path,
        drop_global_path=drop_global_path,
        drop_elite_path=drop_elite_path,
        plot_path=plot_path,
        drop_plot_path=drop_plot_path,
        compare_plot_path=compare_plot_path,
        secondary_plot_path=secondary_plot_path,
    )

    print("===================================")
    print(" MODELER 실행 완료")
    print("===================================")
    return {
        "metadata_path": save_result.metadata_path,
        "model_path": model_path,
        "selected_features_csv_path": selected_path,
        "feas_model_path": feas_model_path,
    }


if __name__ == "__main__":
    raise SystemExit(
        "Modeler/run_Modeler.py is an internal task runner. "
        "Use pipeline/run_pipeline.py as the execution entrypoint."
    )
