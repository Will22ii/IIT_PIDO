# run_Modeler.py

import hashlib
import numpy as np
import pandas as pd
import os
import pickle
import json
import re
from dataclasses import asdict
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier

from utils.result_loader import ResultLoader
from Modeler.executor.trainer import ModelTrainer
from Modeler.executor.importance_analyzer import ImportanceAnalyzer
from Modeler.executor.hpo_runner import HPORunner
from utils.result_saver import ResultSaver
from Modeler.executor.feature_selector import (
    FeatureSelector,
    FeatureSelectionConfig,
)
from Modeler.visualization.feature_selection_plots import plot_perm_effect
from Modeler.config import ModelerConfig, ModelerSystemConfig, ModelerUserConfig
from CAE_tool_interface.config import CAEConfig, CAEUserConfig, CAESystemConfig
from pipeline.run_context import RunContext, get_stage_metadata_path, update_run_index

# -------------------------------------------------
# XGB HPO param type spec
# -------------------------------------------------
XGB_PARAM_TYPES = {
    "n_estimators": int,
    "max_depth": int,
    "min_child_weight": int,
    "subsample": float,
    "colsample_bytree": float,
    "learning_rate": float,
    "gamma": float,
}


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_hpo_signature(
    *,
    problem_name: str,
    objective_sense: str,
    model_name: str,
    target_col: str,
    data_hash: str,
) -> str:
    payload = {
        "problem_name": problem_name,
        "objective_sense": objective_sense,
        "model_name": model_name,
        "target_col": target_col,
        "data_hash": data_hash,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_hpo_params_by_signature(
    *,
    signature: str,
    project_root: str,
) -> dict | None:
    result_root = os.path.join(project_root, "result")
    index_path = os.path.join(result_root, "hpo_index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path, "r") as f:
                index_payload = json.load(f)
            entry = index_payload.get(signature)
            if entry:
                params_path = os.path.join(project_root, entry)
                if os.path.exists(params_path):
                    df_params = pd.read_csv(params_path)
                    raw_params = dict(zip(df_params["param"], df_params["value"]))
                    return {
                        k: XGB_PARAM_TYPES[k](v) if k in XGB_PARAM_TYPES else v
                        for k, v in raw_params.items()
                    }
        except (json.JSONDecodeError, OSError):
            pass
    if not os.path.exists(result_root):
        return None
    run_dirs = [
        d for d in os.listdir(result_root)
        if d.startswith("run_") and os.path.isdir(os.path.join(result_root, d))
    ]
    run_dirs.sort(reverse=True)
    for run_dir in run_dirs:
        meta_path = os.path.join(result_root, run_dir, "Modeler", "metadata.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if meta.get("resolved_params", {}).get("hpo_signature") != signature:
            continue
        params_rel = meta.get("artifacts", {}).get("hpo_best_params")
        if not params_rel:
            continue
        params_path = os.path.join(os.path.dirname(meta_path), params_rel)
        if not os.path.exists(params_path):
            continue
        df_params = pd.read_csv(params_path)
        raw_params = dict(zip(df_params["param"], df_params["value"]))
        return {
            k: XGB_PARAM_TYPES[k](v) if k in XGB_PARAM_TYPES else v
            for k, v in raw_params.items()
        }
    return None


def _update_hpo_index(
    *,
    project_root: str,
    signature: str,
    params_path: str,
) -> None:
    result_root = os.path.join(project_root, "result")
    os.makedirs(result_root, exist_ok=True)
    index_path = os.path.join(result_root, "hpo_index.json")
    payload = {}
    if os.path.exists(index_path):
        try:
            with open(index_path, "r") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            payload = {}
    rel_path = os.path.relpath(params_path, project_root)
    payload[signature] = rel_path
    with open(index_path, "w") as f:
        json.dump(payload, f, indent=2)


def _align_df_schema(
    df: pd.DataFrame,
    *,
    variables: list | None,
    target_col: str,
) -> pd.DataFrame:
    if not variables:
        return df

    expected = [v["name"] for v in variables]

    if all(col in df.columns for col in expected):
        ordered = expected + [c for c in df.columns if c not in expected]
        return df[ordered]

    # fallback: rename x_0/x1 style columns by position
    x0_cols = [f"x_{i}" for i in range(len(expected))]
    x1_cols = [f"x{i + 1}" for i in range(len(expected))]
    if all(c in df.columns for c in x0_cols):
        rename_map = {old: new for old, new in zip(x0_cols, expected)}
        print("- Renaming x_0..x_n columns to variable names from metadata")
        df = df.rename(columns=rename_map)
    elif all(c in df.columns for c in x1_cols):
        rename_map = {old: new for old, new in zip(x1_cols, expected)}
        print("- Renaming x1..xn columns to variable names from metadata")
        df = df.rename(columns=rename_map)

    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise RuntimeError(
            "CSV schema mismatch. Missing feature columns: "
            + ", ".join(missing)
            + f" (target_col={target_col})"
        )

    ordered = expected + [c for c in df.columns if c not in expected]
    return df[ordered]


def _to_bool_mask(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.fillna(False).to_numpy(dtype=bool)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "y", "yes", "t"})
        .to_numpy(dtype=bool)
    )


def run_modeler(*, config: ModelerConfig, run_context: RunContext | None = None) -> None:
    print("===================================")
    print(" MODELER 실행 시작")
    print("===================================")

    # -------------------------------------------------
    # 1. DOE 결과 로드
    # -------------------------------------------------
    loader = ResultLoader()
    doe_meta = {}
    doe_result = None
    csv_path = None
    used_input_paths = bool(config.doe_metadata_path or config.doe_csv_path)
    if used_input_paths:
        if config.doe_metadata_path:
            print(f"- DOE metadata path provided: {config.doe_metadata_path}")
            with open(config.doe_metadata_path, "r") as f:
                doe_meta = json.load(f)
            if "artifacts" in doe_meta and "results_csv" in doe_meta["artifacts"]:
                stage_dir = os.path.dirname(config.doe_metadata_path)
                results_csv = doe_meta["artifacts"]["results_csv"]
                csv_path = os.path.join(stage_dir, results_csv)
            else:
                doe_result = loader.load_stage(
                    stage="DOE",
                    metadata_path=config.doe_metadata_path,
                    allow_latest_fallback=False,
                )
                doe_meta = doe_result.metadata or {}
                csv_path = doe_result.csv_path
        if config.doe_csv_path:
            print(f"- DOE CSV path provided: {config.doe_csv_path}")
            csv_path = config.doe_csv_path
        if not csv_path:
            raise RuntimeError("Modeler input CSV path not resolved.")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
    elif run_context:
        doe_meta_path = get_stage_metadata_path(run_context, "DOE")
        if not doe_meta_path:
            raise RuntimeError("DOE metadata not found in run context.")
        with open(doe_meta_path, "r") as f:
            doe_meta = json.load(f)
        stage_dir = os.path.dirname(doe_meta_path)
        results_csv = doe_meta["artifacts"]["results_csv"]
        csv_path = os.path.join(stage_dir, results_csv)
        df = pd.read_csv(csv_path)
    else:
        if not config.cae.system.allow_latest_fallback:
            raise RuntimeError("Latest fallback is disabled. Provide DOE metadata or CSV path.")
        print("- DOE metadata path not provided; using latest DOE metadata")
        doe_result = loader.load_stage(stage="DOE", allow_latest_fallback=True)
        df = doe_result.df
        doe_meta = doe_result.metadata or {}
        csv_path = doe_result.csv_path

    problem_name = doe_meta.get("problem") or config.cae.user.problem_name
    if not problem_name:
        raise RuntimeError("DOE metadata에 problem 정보가 없습니다.")

    if run_context and not used_input_paths:
        with open(run_context.user_config_snapshot_path, "r") as f:
            user_snapshot = json.load(f)
        base_seed = user_snapshot.get("seed", 42)
    else:
        base_seed = config.cae.user.seed

    if csv_path:
        print(f"- DOE CSV: {csv_path}")
    print(f"- Base seed: {base_seed}")
    print(f"- Problem name: {problem_name}")

    model_name = config.user.model_name
    use_hpo = config.user.use_hpo
    target_col = config.user.target_col
    objective_sense = config.cae.user.objective_sense
    use_timestamp = (
        config.cae.system.use_timestamp if config.cae is not None else False
    )
    hpo_config = config.system.hpo_config
    kfold_splits = config.system.kfold_splits
    feature_selection_cfg = {
        "perm_min_pass_rate": config.system.perm_min_pass_rate,
        "perm_epsilon": config.system.perm_epsilon,
    }
    perm_sample_size = config.system.perm_sample_size

    if model_name != "xgb" and use_hpo:
        print("- HPO is XGB-only; disabling HPO for non-XGB model")
        use_hpo = False

    # -------------------------------------------------
    # 2. Feature 정의
    # -------------------------------------------------
    variables = None
    constraint_defs = []
    has_post_constraints = False
    if doe_meta:
        variables = doe_meta.get("variables")
        if not variables:
            variables = doe_meta.get("inputs", {}).get("variables")
        constraint_defs = (
            doe_meta.get("constraint_defs")
            or doe_meta.get("inputs", {}).get("constraint_defs")
            or []
        )
        if isinstance(constraint_defs, list):
            has_post_constraints = any(
                str(c.get("scope", "x_only")).strip().lower() == "cae_dependent"
                for c in constraint_defs
                if isinstance(c, dict)
            )
    df = _align_df_schema(df, variables=variables, target_col=target_col)
    df_all_for_feas = df.copy()

    if target_col not in df.columns:
        raise RuntimeError(f"Target column not found: {target_col}")

    if "success" in df.columns:
        success_mask = _to_bool_mask(df["success"])
        n_before = int(len(df))
        df = df.loc[success_mask].reset_index(drop=True)
        n_after = int(len(df))
        print(f"- Success 필터 적용: {n_before} -> {n_after}")
        if n_after == 0:
            raise RuntimeError("No successful DOE rows available for Modeler training.")

    # post/pre 제약이 도입된 경우 최종 feasible만 학습에 사용
    if "feasible_final" in df.columns:
        feas_mask = _to_bool_mask(df["feasible_final"])
        n_before = int(len(df))
        df = df.loc[feas_mask].reset_index(drop=True)
        n_after = int(len(df))
        print(f"- Feasible_final 필터 적용: {n_before} -> {n_after}")
        if n_after == 0:
            raise RuntimeError("No feasible_final rows available for Modeler training.")

    if len(df) < int(kfold_splits):
        raise RuntimeError(
            f"학습 데이터가 부족합니다: success rows={len(df)} < kfold_splits={kfold_splits}. "
            "DOE n_samples를 늘려주세요."
        )

    if variables:
        feature_cols = [v["name"] for v in variables]
    else:
        ignore_cols = {
            "id",
            target_col,
            "feasible",
            "feasible_pre",
            "feasible_post",
            "feasible_final",
            "success",
            "source",
            "round",
            "exec_scope",
            "margin_pre",
            "margin_post",
            "constraint_margin",
        }
        feature_cols = [c for c in df.columns if c not in ignore_cols]

    if not feature_cols:
        raise RuntimeError("No feature columns resolved for modeling.")

    X = df[feature_cols].values
    y = df[target_col].values

    # -------------------------------------------------
    # 3. HPO 결정 (ON / OFF)
    # -------------------------------------------------
    best_params = None
    hpo_params_used = False
    hpo_signature = None

    if use_hpo:
        data_hash = _file_sha256(csv_path)
        hpo_signature = _build_hpo_signature(
            problem_name=problem_name,
            objective_sense=objective_sense,
            model_name=model_name,
            target_col=target_col,
            data_hash=data_hash,
        )
        reuse_hpo = True
        if hpo_config is not None:
            reuse_hpo = bool(hpo_config.get("reuse_if_same_config", True))
        if reuse_hpo:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            reused = _load_hpo_params_by_signature(
                signature=hpo_signature,
                project_root=project_root,
            )
            if reused:
                best_params = reused
                hpo_params_used = True
                print("- Reusing HPO params (signature match)")

        if best_params is None:
            cfg = hpo_config or {}
            hpo_runner = HPORunner(
                n_trials=cfg.get("n_trials", 20),
                lambda_std=cfg.get("lambda_std", 0.5),
                use_timestamp=use_timestamp,
            )

            hpo_result = hpo_runner.run_xgb(
                X=X,
                y=y,
                base_random_seed=base_seed,
                problem_name=problem_name,
                kfold_splits=kfold_splits,
            )

            best_params = hpo_result["best_params"]
            hpo_params_used = True
            df_params = pd.DataFrame(
                [
                    {"param": k, "value": v}
                    for k, v in hpo_result["best_params"].items()
                ]
            )
            if run_context:
                pass
            else:
                saver = ResultSaver(use_timestamp=use_timestamp)
                hpo_result["hpo_signature"] = hpo_signature
                hpo_result["data_hash"] = data_hash
                saver.save_stage_csv(
                    stage="MODELER",
                    df=df_params,
                    problem_name=f"{problem_name}_hpo_best_xgb",
                    extra_metadata=hpo_result,
                )
            print("- HPO executed")

    elif model_name == "xgb" and not run_context:
        try:
            hpo_loaded = loader.load_stage(
                stage="MODELER",
                problem_name=f"{problem_name}_hpo_best_xgb",
                allow_latest_fallback=bool(config.cae.system.allow_latest_fallback),
            )

            raw_params = dict(
                zip(hpo_loaded.df["param"], hpo_loaded.df["value"])
            )

            best_params = {
                k: XGB_PARAM_TYPES[k](v) if k in XGB_PARAM_TYPES else v
                for k, v in raw_params.items()
            }

            hpo_params_used = True
            print("- Using existing HPO params")

        except FileNotFoundError:
            print("- No HPO params found, using default params")
            best_params = None
            hpo_params_used = False
    else:
        best_params = None
        hpo_params_used = False

    # -------------------------------------------------
    # 4. 모델 학습 
    # -------------------------------------------------
    trainer = ModelTrainer(
        base_random_seed=base_seed,
        target_col=target_col,
        feature_cols=feature_cols,
        model_params=best_params,
        model_name=model_name,
        kfold_splits=kfold_splits,
    )

    train_result = trainer.run(df)
    models = train_result["models"]

    print(f"- Trained models: {len(models)}")

    # -------------------------------------------------
    # 5. Importance 분석
    # -------------------------------------------------
    saver = ResultSaver(use_timestamp=use_timestamp)
    processed_df = None
    selected_df = None
    artifacts = {}

    analyzer = ImportanceAnalyzer(
        perm_sample_size=perm_sample_size,
    )

    X_ref = df[feature_cols].astype(float)
    importance_result = analyzer.run_perm_effect(
        models=models,
        X_ref=X_ref,
        problem_name=problem_name,
        random_seed=base_seed,
    )
    perm_imp_df = importance_result["perm_effect_raw"]

    if run_context:
        pass
    else:
        perm_paths = saver.save_stage_csv(
            stage="MODELER",
            df=perm_imp_df,
            problem_name=f"{problem_name}_perm_effect_raw",
        )

    selector = FeatureSelector(
        FeatureSelectionConfig(**feature_selection_cfg)
    )

    if run_context:
        pass
    else:
        selection_result = selector.run(
            perm_effect_path=perm_paths["csv"],
            problem_name=problem_name,
        )

    if run_context:
        pass
    else:
        processed_df = selection_result["importance_processed"]
        selected_df = selection_result["selected_features"]

        saver.save_stage_csv(
            stage="MODELER",
            df=processed_df,
            problem_name=f"{problem_name}_importance_processed",
        )

    # -------------------------------------------------
    # 8. Selected-feature training + save
    # -------------------------------------------------
    if run_context:
        stage_dir = os.path.join(run_context.run_root, "Modeler")
        artifacts_dir = os.path.join(stage_dir, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        perm_path = os.path.join(artifacts_dir, "perm_effect_raw.csv")
        perm_imp_df.to_csv(perm_path, index=False)
        artifacts["perm_effect_raw"] = os.path.relpath(
            perm_path, stage_dir
        )

        selection_result = selector.run(
            perm_effect_path=perm_path,
            problem_name=problem_name,
        )
        processed_df = selection_result["importance_processed"]
        selected_df = selection_result["selected_features"]

        processed_path = os.path.join(artifacts_dir, "importance_processed.csv")
        processed_df.to_csv(processed_path, index=False)

        artifacts["importance_processed"] = os.path.relpath(
            processed_path, stage_dir
        )

        selected_path = os.path.join(artifacts_dir, "selected_features.csv")
        selected_df.to_csv(selected_path, index=False)
        artifacts["selected_features"] = os.path.relpath(selected_path, stage_dir)

        if hpo_params_used and best_params:
            df_params = pd.DataFrame(
                [{"param": k, "value": v} for k, v in best_params.items()]
            )
            params_path = os.path.join(artifacts_dir, "hpo_best_params.csv")
            df_params.to_csv(params_path, index=False)
            artifacts["hpo_best_params"] = os.path.relpath(params_path, stage_dir)
            if hpo_signature:
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                _update_hpo_index(
                    project_root=project_root,
                    signature=hpo_signature,
                    params_path=params_path,
                )

    # -------------------------------------------------
    # 8.1 Feature selection visualization
    # -------------------------------------------------
    plot_path = None
    if run_context:
        plot_dir = artifacts_dir
        plot_path = os.path.join(plot_dir, "feature_selection_plot.png")
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        plot_dir = os.path.join(project_root, "result", "modeler")
        os.makedirs(plot_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S") if use_timestamp else ""
        suffix = f"_{ts}" if ts else ""
        plot_path = os.path.join(
            plot_dir, f"modeler_feature_selection_{problem_name}{suffix}.png"
        )

    plot_perm_effect(
        selected_df=selected_df,
        perm_epsilon=config.system.perm_epsilon,
        output_path=plot_path,
    )

    if run_context and plot_path:
        artifacts["feature_selection_plot"] = os.path.relpath(
            plot_path, stage_dir
        )
        if plot_path:
            print(f"- Saved feature selection plot: {plot_path}")

    selected_features = selected_df[selected_df["selected"]]["feature"].tolist()
    if not selected_features:
        selected_features = feature_cols
        print("- No selected features found; fallback to all features")
    else:
        print(
            f"- Selected features ({len(selected_features)}/{len(feature_cols)}): "
            f"{', '.join(selected_features)}"
        )

    selected_trainer = ModelTrainer(
        base_random_seed=base_seed,
        target_col=target_col,
        feature_cols=selected_features,
        model_params=best_params,
        model_name=model_name,
        kfold_splits=kfold_splits,
    )
    selected_train_result = selected_trainer.run(df)

    if run_context:
        model_dir = artifacts_dir
        ts = ""
        suffix = ""
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(project_root, "result", "modeler")
        os.makedirs(model_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S") if use_timestamp else ""
        suffix = f"_{ts}" if ts else ""
    model_path = os.path.join(
        model_dir,
        f"modeler_selected_models_{problem_name}{suffix}.pkl",
    )
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "models": selected_train_result["models"],
                "feature_cols": selected_features,
                "model_params": best_params,
                "problem_name": problem_name,
                "model_name": model_name,
            },
            f,
        )
    print(f"- Saved selected-feature models: {model_path}")

    # -------------------------------------------------
    # 8.2 Post feasibility classifier (auto when post constraints exist)
    # -------------------------------------------------
    feas_model_path = None
    feas_model_kind = "none"
    feas_model_stats = {}
    if has_post_constraints:
        df_cls = df_all_for_feas.copy()
        needed_cols = list(selected_features)
        target_feas_col = "feasible_final" if "feasible_final" in df_cls.columns else "feasible"
        if target_feas_col in df_cls.columns:
            needed_cols.append(target_feas_col)
        else:
            print("[Modeler] feasible_final/feasible column not found; fallback constant feasibility model.")

        if all(col in df_cls.columns for col in needed_cols):
            df_cls = df_cls.dropna(subset=needed_cols).reset_index(drop=True)
        else:
            df_cls = pd.DataFrame(columns=needed_cols)

        if target_feas_col in df_cls.columns and len(df_cls) > 0:
            y_cls = _to_bool_mask(df_cls[target_feas_col]).astype(int)
            X_cls = df_cls[selected_features].to_numpy(dtype=float)
        else:
            y_cls = np.asarray([], dtype=int)
            X_cls = np.empty((0, len(selected_features)), dtype=float)

        n_total_cls = int(y_cls.shape[0])
        n_pos_cls = int(np.sum(y_cls == 1))
        n_neg_cls = int(np.sum(y_cls == 0))
        feas_model_stats = {
            "n_total": n_total_cls,
            "n_pos": n_pos_cls,
            "n_neg": n_neg_cls,
        }
        if n_pos_cls < 5 or n_neg_cls < 5:
            print(
                "[Modeler] feasibility class count warning: "
                f"pos={n_pos_cls}, neg={n_neg_cls} (still proceed)"
            )

        if n_total_cls <= 0:
            # 데이터가 전혀 없으면 중립 확률(0.5) 상수 모델로 강행
            const_prob = 0.5
            feas_payload = {
                "kind": "constant",
                "constant_prob": const_prob,
                "feature_cols": selected_features,
                "problem_name": problem_name,
            }
            feas_model_kind = "constant"
        elif n_pos_cls == 0 or n_neg_cls == 0:
            const_prob = 1.0 if n_pos_cls > 0 else 0.0
            dummy = DummyClassifier(strategy="constant", constant=int(const_prob >= 0.5))
            dummy.fit(X_cls, y_cls if y_cls.size > 0 else np.zeros((X_cls.shape[0],), dtype=int))
            feas_payload = {
                "kind": "constant",
                "constant_prob": const_prob,
                "model": dummy,
                "feature_cols": selected_features,
                "problem_name": problem_name,
            }
            feas_model_kind = "constant"
        else:
            clf = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                min_child_weight=1,
                subsample=0.9,
                colsample_bytree=0.9,
                gamma=0.0,
                reg_alpha=0.0,
                reg_lambda=1.0,
                random_state=base_seed,
                n_jobs=-1,
            )
            clf.fit(X_cls, y_cls)
            feas_payload = {
                "kind": "xgb_classifier",
                "model": clf,
                "feature_cols": selected_features,
                "problem_name": problem_name,
            }
            feas_model_kind = "xgb_classifier"

        feas_model_path = os.path.join(
            model_dir,
            f"modeler_feas_models_{problem_name}{suffix}.pkl",
        )
        with open(feas_model_path, "wb") as f:
            pickle.dump(feas_payload, f)
        print(f"- Saved feasibility model: {feas_model_path} (kind={feas_model_kind})")

    if not run_context:
        saver.save_stage_csv(
            stage="MODELER",
            df=selected_df,
            problem_name=f"{problem_name}_selected_features",
            model_path=model_path,
        )

    base_meta = doe_meta or {}
    workflow_info = base_meta.get("workflow_info", {}) if base_meta else {}
    workflow_info["MODELER"] = model_name

    if run_context:
        prev_doe_meta = get_stage_metadata_path(run_context, "DOE")
        prev_doe_ref = (
            os.path.relpath(prev_doe_meta, stage_dir) if prev_doe_meta else None
        )
        inputs = {
            "user_config": os.path.relpath(
                run_context.user_config_snapshot_path,
                os.path.join(run_context.run_root, "Modeler"),
            ),
            "system_config_snapshot": asdict(config.system),
            "previous": {"DOE": prev_doe_ref} if prev_doe_ref else {},
        }
        resolved_params = {
            "hpo_used": hpo_params_used,
            "hpo_signature": hpo_signature,
            "data_hash": data_hash if use_hpo else None,
            "has_post_constraints": bool(has_post_constraints),
            "feas_model_kind": feas_model_kind,
        }
        results_summary = {
            "n_models": len(models),
            "n_features_selected": len(selected_features),
            "feas_model_stats": feas_model_stats if has_post_constraints else None,
        }
        artifacts["model_path"] = os.path.relpath(model_path, stage_dir)
        if feas_model_path:
            artifacts["feas_model_path"] = os.path.relpath(feas_model_path, stage_dir)
        out = saver.save_stage_v2(
            run_root=run_context.run_root,
            stage="Modeler",
            problem_name=problem_name,
            df=selected_df,
            inputs=inputs,
            resolved_params=resolved_params,
            results=results_summary,
            artifacts=artifacts,
        )
        update_run_index(run_context, "Modeler", out["metadata"])
    else:
        combined_meta = dict(base_meta)
        combined_meta.update(
            {
                "created_at": datetime.now().isoformat(),
                "workflow_info": workflow_info,
                "selected_features": selected_features,
                "training_csv_path": csv_path,
                "model_path": model_path,
                "feas_model_path": feas_model_path,
                "has_post_constraints": bool(has_post_constraints),
                "feas_model_kind": feas_model_kind,
                "feas_model_stats": feas_model_stats if has_post_constraints else None,
                "hpo_used": hpo_params_used,
                "model_name": model_name,
            }
        )

        meta_name = f"modeler_metadata_{problem_name}_summary{suffix}.json"
        meta_path = os.path.join(model_dir, meta_name)
        with open(meta_path, "w") as f:
            json.dump(combined_meta, f, indent=2)
        print(f"- Saved modeler summary metadata: {meta_path}")

    # -------------------------------------------------
    # 9. Final report
    # -------------------------------------------------
    if not run_context:
        saver.save_final_txt(
            content_lines=[
                "[MODELER SUMMARY]",
                f"- Models: {model_name} (5-Fold ensemble)",
                f"- Base seed: {base_seed}",
                f"- Selected features: {len(selected_features)}",
                f"- Selected feature names: {', '.join(selected_features)}",
            ],
            problem_name=problem_name,
            workflow_info={
                "DOE": doe_meta.get("algo_name"),
                "MODELER": model_name,
                "EXPLORER": None,
                "OPT": None,
            },
        )

    print("===================================")
    print(" MODELER 실행 완료")
    print("===================================")


if __name__ == "__main__":
    # Standalone example (custom data path)
    # cfg = ModelerConfig(
    #     user=ModelerUserConfig(model_name="xgb", use_hpo=False, target_col="objective"),
    #     system=ModelerSystemConfig(),
    #     cae=CAEConfig(
    #         user=CAEUserConfig(problem_name="goldstein_price", seed=42, objective_sense="min"),
    #         system=CAESystemConfig(use_timestamp=True, allow_latest_fallback=True),
    #     ),
    #     doe_csv_path="C:\\python\\project\\result\\doe\\doe_result_goldstein_price.csv",
    #     doe_metadata_path="C:\\python\\project\\result\\doe\\doe_metadata_goldstein_price.json",
    # )

    cfg = ModelerConfig(
        user=ModelerUserConfig(model_name="xgb", use_hpo=False, target_col="objective"),
        system=ModelerSystemConfig(),
        cae=CAEConfig(
            user=CAEUserConfig(problem_name="goldstein_price", seed=42, objective_sense="min"),
            system=CAESystemConfig(use_timestamp=True, allow_latest_fallback=True),
        ),
        cae_user=None,
        doe_csv_path=None,
        doe_metadata_path=None,
    )
    run_modeler(config=cfg)
