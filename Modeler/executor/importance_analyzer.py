import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.metrics import r2_score


class ImportanceAnalyzer:
    """
    Generate permutation effect deltas from trained models.
    """

    def __init__(self, *, perm_sample_size: int | None = None, perm_repeats: int = 1):
        self.perm_sample_size = perm_sample_size
        self.perm_repeats = max(int(perm_repeats), 1)

    # =================================================
    # Public API
    # =================================================

    def run_perm_effect(
        self,
        *,
        models: List,
        fold_predictions: List[dict],
        X_ref: pd.DataFrame,
        problem_name: str,
        random_seed: int | None = None,
        subset_mask: np.ndarray | None = None,
        scale_label: str = "global",
    ) -> Dict[str, pd.DataFrame]:
        rows = []
        model_by_run = {i: m for i, m in enumerate(models)}
        mask = None
        if subset_mask is not None:
            mask = np.asarray(subset_mask, dtype=bool).reshape(-1)
            if mask.shape[0] != len(X_ref):
                raise RuntimeError(
                    f"subset_mask length mismatch: {mask.shape[0]} != {len(X_ref)}"
                )
        for fold_item in fold_predictions:
            run_id = int(fold_item["run_id"])
            valid_idx = np.asarray(fold_item["valid_idx"], dtype=int)
            if valid_idx.size == 0:
                continue
            if np.max(valid_idx) >= len(X_ref):
                continue
            model = model_by_run.get(run_id)
            if model is None:
                continue

            rng = np.random.default_rng(
                None if random_seed is None else random_seed + run_id
            )
            X_num = self._prepare_input(model, X_ref)
            valid_use = valid_idx
            if mask is not None:
                valid_use = valid_idx[mask[valid_idx]]
            if valid_use.size < 2:
                continue
            if self.perm_sample_size is not None and valid_use.size > int(self.perm_sample_size):
                valid_use = np.sort(
                    rng.choice(valid_use, size=int(self.perm_sample_size), replace=False)
                )
            X_valid = X_num.iloc[valid_use]
            base = X_valid.to_numpy()
            if base.shape[0] == 0:
                continue
            columns = list(X_valid.columns)
            pred_base = np.asarray(model.predict(base), dtype=float).reshape(-1)

            for idx, col in enumerate(columns):
                if self.perm_repeats <= 1:
                    X_perm = base.copy()
                    X_perm[:, idx] = rng.permutation(X_perm[:, idx])
                    pred_perm = np.asarray(model.predict(X_perm), dtype=float).reshape(-1)
                    delta = float(np.mean((pred_base - pred_perm) ** 2))
                else:
                    deltas_k = []
                    for _ in range(self.perm_repeats):
                        X_perm = base.copy()
                        X_perm[:, idx] = rng.permutation(X_perm[:, idx])
                        pred_perm = np.asarray(model.predict(X_perm), dtype=float).reshape(-1)
                        deltas_k.append(float(np.mean((pred_base - pred_perm) ** 2)))
                    delta = float(np.mean(deltas_k))
                rows.append(
                    {
                        "problem": problem_name,
                        "scale": str(scale_label),
                        "method": "PERM",
                        "fold": run_id,
                        "feature": col,
                        "delta": delta,
                    }
                )

        return {
            "perm_effect_raw": pd.DataFrame(
                rows,
                columns=["problem", "scale", "method", "fold", "feature", "delta"],
            ),
        }

    def run_score_drop(
        self,
        *,
        models: List,
        fold_predictions: List[dict],
        X_ref: pd.DataFrame,
        y_true: np.ndarray,
        problem_name: str,
        random_seed: int | None = None,
        subset_mask: np.ndarray | None = None,
        scale_label: str = "global",
    ) -> Dict[str, pd.DataFrame]:
        rows = []

        model_by_run = {i: m for i, m in enumerate(models)}
        mask = None
        if subset_mask is not None:
            mask = np.asarray(subset_mask, dtype=bool).reshape(-1)
            if mask.shape[0] != len(X_ref):
                raise RuntimeError(
                    f"subset_mask length mismatch: {mask.shape[0]} != {len(X_ref)}"
                )
        for fold_item in fold_predictions:
            run_id = int(fold_item["run_id"])
            valid_idx = np.asarray(fold_item["valid_idx"], dtype=int)
            if valid_idx.size == 0:
                continue
            if np.max(valid_idx) >= len(X_ref):
                # split indices are based on full X_ref; skip if subsampled index mismatch
                continue
            model = model_by_run.get(run_id)
            if model is None:
                continue
            rng = np.random.default_rng(
                None if random_seed is None else random_seed + run_id
            )
            X_num = self._prepare_input(model, X_ref)
            valid_use = valid_idx
            if mask is not None:
                valid_use = valid_idx[mask[valid_idx]]
            if valid_use.size < 2:
                continue
            X_valid = X_num.iloc[valid_use]
            y_valid = np.asarray(y_true, dtype=float)[valid_use]
            if y_valid.size < 2:
                continue

            base_pred = np.asarray(fold_item.get("y_pred", []), dtype=float).reshape(-1)
            if base_pred.size != np.asarray(valid_idx, dtype=int).size:
                base_pred = np.asarray(model.predict(X_num.iloc[valid_idx].to_numpy()), dtype=float).reshape(-1)
            if mask is not None:
                base_pred = base_pred[mask[valid_idx]]
            if base_pred.size != y_valid.size:
                base_pred = np.asarray(model.predict(X_valid.to_numpy()), dtype=float).reshape(-1)
            try:
                r2_base = float(r2_score(y_valid, base_pred))
            except Exception:
                continue

            base_arr = X_valid.to_numpy()
            columns = list(X_valid.columns)
            for idx, col in enumerate(columns):
                if self.perm_repeats <= 1:
                    X_perm = base_arr.copy()
                    X_perm[:, idx] = rng.permutation(X_perm[:, idx])
                    pred_perm = np.asarray(model.predict(X_perm), dtype=float).reshape(-1)
                    try:
                        r2_perm = float(r2_score(y_valid, pred_perm))
                    except Exception:
                        continue
                    drop = float(r2_base - r2_perm)
                else:
                    drops_k = []
                    r2_perms_k = []
                    for _ in range(self.perm_repeats):
                        X_perm = base_arr.copy()
                        X_perm[:, idx] = rng.permutation(X_perm[:, idx])
                        pred_perm = np.asarray(model.predict(X_perm), dtype=float).reshape(-1)
                        try:
                            r2_k = float(r2_score(y_valid, pred_perm))
                        except Exception:
                            continue
                        drops_k.append(float(r2_base - r2_k))
                        r2_perms_k.append(r2_k)
                    if not drops_k:
                        continue
                    drop = float(np.mean(drops_k))
                    r2_perm = float(np.mean(r2_perms_k))
                drop_pos = float(max(drop, 0.0))
                rows.append(
                    {
                        "problem": problem_name,
                        "scale": str(scale_label),
                        "method": "R2_DROP",
                        "fold": run_id,
                        "feature": col,
                        "r2_base": r2_base,
                        "r2_perm": r2_perm,
                        "drop": drop,
                        "drop_pos": drop_pos,
                        "drop_sq": float(drop_pos ** 2),
                    }
                )

        return {
            "score_drop_raw": pd.DataFrame(
                rows,
                columns=[
                    "problem",
                    "scale",
                    "method",
                    "fold",
                    "feature",
                    "r2_base",
                    "r2_perm",
                    "drop",
                    "drop_pos",
                    "drop_sq",
                ],
            ),
        }

    # =================================================
    # Internal helpers
    # =================================================

    def _prepare_input(self, model, X: pd.DataFrame) -> pd.DataFrame:
        # keep only the features used by the model (and in correct order)
        missing = [f for f in model.feature_names if f not in X.columns]
        if missing:
            raise RuntimeError(
                "SHAP input missing model features: "
                + ", ".join(missing)
            )
        X_used = X.loc[:, model.feature_names].copy()

        # force numeric and validate shape
        for col in X_used.columns:
            if not np.issubdtype(X_used[col].dtype, np.number):
                X_used[col] = pd.to_numeric(X_used[col], errors="coerce")
        X_num = X_used.astype(float)

        # guard against NaN/inf after coercion
        if not np.isfinite(X_num.to_numpy()).all():
            bad_cols = [
                col
                for col in X_num.columns
                if not np.isfinite(X_num[col].to_numpy()).all()
            ]
            raise RuntimeError(
                "SHAP input contains NaN/inf after numeric coercion in: "
                + ", ".join(bad_cols)
            )

        return X_num
