import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

from Modeler.executor.trainer import ModelTrainer


@dataclass
class ModelArtifactResult:
    model_path: str
    feas_model_path: str | None
    feas_model_kind: str
    feas_model_stats: dict


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


def train_and_save_model_artifacts(
    *,
    df: pd.DataFrame,
    df_all_for_feas: pd.DataFrame,
    selected_features: list[str],
    target_col: str,
    base_seed: int,
    model_name: str,
    best_params: dict | None,
    kfold_splits: int,
    kfold_repeats: int,
    public_dir: str,
    problem_name: str,
    has_post_constraints: bool,
) -> ModelArtifactResult:
    selected_trainer = ModelTrainer(
        base_random_seed=base_seed,
        target_col=target_col,
        feature_cols=selected_features,
        model_params=best_params,
        model_name=model_name,
        kfold_splits=kfold_splits,
        kfold_repeats=kfold_repeats,
    )
    selected_train_result = selected_trainer.run(df)

    model_path = os.path.join(public_dir, "modeler_selected_models.pkl")
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

    feas_model_path = None
    feas_model_kind = "none"
    feas_model_stats: dict = {}
    if has_post_constraints:
        df_cls = df_all_for_feas.copy()
        needed_cols = list(selected_features)
        target_feas_col = "feasible"
        if target_feas_col in df_cls.columns:
            needed_cols.append(target_feas_col)
        else:
            print("[Modeler] feasible column not found; fallback constant feasibility model.")

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
            public_dir,
            "modeler_feas_models.pkl",
        )
        with open(feas_model_path, "wb") as f:
            pickle.dump(feas_payload, f)
        print(f"- Saved feasibility model: {feas_model_path} (kind={feas_model_kind})")

    return ModelArtifactResult(
        model_path=model_path,
        feas_model_path=feas_model_path,
        feas_model_kind=feas_model_kind,
        feas_model_stats=feas_model_stats,
    )
