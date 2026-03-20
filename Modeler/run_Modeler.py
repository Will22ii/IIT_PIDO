# run_Modeler.py

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Modeler.executor.data_workflow import (
    prepare_modeler_data_policy,
)
from Modeler.executor.hpo_workflow import (
    resolve_hpo_params,
    update_hpo_cache,
)
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
from Modeler.config import ModelerConfig, ModelerSystemConfig, ModelerUserConfig
from CAE_tool_interface.config import CAEConfig, CAEUserConfig, CAESystemConfig
from pipeline.run_context import RunContext


def _normalize_debug_level(value: str | None) -> str:
    level = str(value or "off").strip().lower()
    if level not in {"off", "full"}:
        raise ValueError("Modeler debug_level must be one of: off, full")
    return level


def run_modeler(*, config: ModelerConfig, run_context: RunContext | None = None) -> None:
    print("===================================")
    print(" MODELER 실행 시작")
    print("===================================")

    input_result = resolve_modeler_input(
        config=config,
        run_context=run_context,
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
    use_secondary_selection = bool(config.user.use_secondary_selection)
    objective_sense = str(cae_objective_sense)
    use_timestamp = (
        config.cae.system.use_timestamp if config.cae is not None else False
    )
    hpo_config = config.system.hpo_config
    configured_kfold_splits = config.system.kfold_splits
    configured_kfold_repeats = config.system.kfold_repeats
    feature_selection_cfg = {
        "perm_min_pass_rate": config.system.perm_min_pass_rate,
        "perm_epsilon": config.system.perm_epsilon,
        "use_score_drop": config.system.fi_use_score_drop,
        "drop_metric": config.system.fi_drop_metric,
        "drop_min_pass_rate": config.system.fi_drop_min_pass_rate,
        "drop_epsilon": config.system.fi_drop_epsilon,
        "weight_abs": config.system.fi_weight_abs,
        "weight_quantile": config.system.fi_weight_quantile,
        "weight_rank": config.system.fi_weight_rank,
        "weight_perm": config.system.fi_weight_perm,
        "weight_drop": config.system.fi_weight_drop,
        "weight_global_default": config.system.fi_weight_global_default,
        "weight_global_low": config.system.fi_weight_global_low,
        "weight_global_rich": config.system.fi_weight_global_rich,
        "elite_small_threshold": config.system.fi_elite_small_threshold,
        "elite_rich_threshold": config.system.fi_elite_rich_threshold,
        "elite_mode": config.system.fi_elite_mode,
        "elite_bonus_beta": config.system.fi_elite_bonus_beta,
        "final_score_threshold": config.system.fi_final_score_threshold,
        "global_score_floor": config.system.fi_global_score_floor,
        "quantile_top_ratio_default": config.system.fi_quantile_top_ratio_default,
        "quantile_top_ratio_p_le_6": config.system.fi_quantile_top_ratio_p_le_6,
        "quantile_top_ratio_p_le_12": config.system.fi_quantile_top_ratio_p_le_12,
        "quantile_top_ratio_p_gt_12": config.system.fi_quantile_top_ratio_p_gt_12,
    }
    perm_sample_size = config.system.perm_sample_size
    debug_level = _normalize_debug_level(config.system.debug_level)
    keep_debug = debug_level == "full"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if model_name != "xgb" and use_hpo:
        print("- HPO is XGB-only; disabling HPO for non-XGB model")
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
        objective_sense=objective_sense,
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
        csv_path=csv_path,
        X=X,
        y=y,
        base_seed=int(base_seed),
        kfold_splits=int(kfold_splits),
    )
    best_params = hpo_result.best_params
    hpo_params_used = hpo_result.hpo_params_used
    hpo_signature = hpo_result.hpo_signature
    data_hash = hpo_result.data_hash

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

    print(f"- Trained models: {len(models)}")

    saver = ResultSaver(use_timestamp=use_timestamp)
    task_dir = os.path.join(run_context.run_root, "Modeler")
    artifacts_root = os.path.join(task_dir, "artifacts")
    public_dir = os.path.join(artifacts_root, "public")
    meta_dir = os.path.join(artifacts_root, "meta")
    debug_dir = os.path.join(artifacts_root, "debug")
    os.makedirs(public_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    fi_result = run_fi_selection_workflow(
        models=models,
        fold_predictions=train_result["fold_predictions"],
        y_true=train_result["y_true"],
        X_ref=df[feature_cols].astype(float),
        elite_mask=elite_mask,
        problem_name=problem_name,
        base_seed=base_seed,
        perm_sample_size=perm_sample_size,
        feature_selection_cfg=feature_selection_cfg,
        use_score_drop=bool(config.system.fi_use_score_drop),
        low_data=bool(cv_policy["low_data"]),
        n_features=len(feature_cols),
        n_elite=int(n_elite),
        keep_debug=keep_debug,
        debug_dir=debug_dir,
        meta_dir=meta_dir,
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

    save_result = save_modeler_outputs(
        saver=saver,
        run_context=run_context,
        task_dir=task_dir,
        project_root=project_root,
        config_system=config.system,
        config_user=config.user,
        problem_name=problem_name,
        model_name=model_name,
        doe_meta=doe_meta or {},
        selected_df=selected_df,
        selected_features=selected_features,
        models_count=len(models),
        kfold_splits=int(kfold_splits),
        kfold_repeats=int(kfold_repeats),
        cv_policy=cv_policy,
        elite_ratio_eff=float(elite_ratio_eff),
        n_elite=int(n_elite),
        has_post_constraints=bool(has_post_constraints),
        feas_model_kind=feas_model_kind,
        feas_model_stats=feas_model_stats,
        hpo_params_used=bool(hpo_params_used),
        hpo_signature=hpo_signature,
        data_hash=data_hash,
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

    if hpo_signature and best_params:
        update_hpo_cache(
            project_root=project_root,
            signature=hpo_signature,
            params=best_params,
            metadata_ref=os.path.relpath(save_result.metadata_path, project_root),
        )

    print("===================================")
    print(" MODELER 실행 완료")
    print("===================================")


if __name__ == "__main__":
    from utils.result_loader import ResultLoader

    # Standalone example (custom data path)
    # cfg = ModelerConfig(
    #     user=ModelerUserConfig(model_name="xgb", use_hpo=False, target_col="objective"),
    #     system=ModelerSystemConfig(),
    #     cae=CAEConfig(
    #         user=CAEUserConfig(problem_name="goldstein_price", seed=42, objective_sense="min"),
    #         system=CAESystemConfig(use_timestamp=True, allow_latest_fallback=True),
    #     ),
    #     doe_csv_path="result/run_<id>/DOE/artifacts/public/doe_results.csv",
    #     doe_metadata_path="result/run_<id>/DOE/metadata.json",
    # )

    problem_name = "goldstein_price"
    loader = ResultLoader()
    try:
        doe_result = loader.load_task(
            task="DOE",
            problem_name=problem_name,
            allow_latest_fallback=True,
        )
        doe_meta_path = doe_result.metadata_path
        doe_csv_path = doe_result.csv_path
        run_root = os.path.dirname(os.path.dirname(doe_meta_path))
        cae_meta_path = os.path.join(run_root, "CAE", "metadata.json")
        if not os.path.exists(cae_meta_path):
            raise FileNotFoundError(f"CAE metadata not found: {cae_meta_path}")
        print(f"- Standalone DOE metadata auto-detected: {doe_meta_path}")
    except Exception as exc:
        raise RuntimeError(
            "Standalone Modeler 실행에는 DOE 입력이 필요합니다. "
            "먼저 pipeline/run_pipeline.py 또는 DOE/run_DOE.py를 실행해 "
            "DOE 결과를 생성하거나, cfg의 doe_metadata_path/doe_csv_path를 명시하세요."
        ) from exc

    cfg = ModelerConfig(
        user=ModelerUserConfig(model_name="xgb", use_hpo=False, target_col="objective"),
        system=ModelerSystemConfig(),
        cae=CAEConfig(
            user=CAEUserConfig(problem_name=problem_name, seed=42, objective_sense="min"),
            system=CAESystemConfig(use_timestamp=True, allow_latest_fallback=False),
        ),
        cae_user=None,
        doe_csv_path=doe_csv_path,
        doe_metadata_path=doe_meta_path,
        cae_metadata_path=cae_meta_path,
    )
    run_modeler(config=cfg)
