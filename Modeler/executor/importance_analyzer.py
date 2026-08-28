import numpy as np
import pandas as pd
from typing import Generator, List, Dict
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

    def run_importance_channels(
        self,
        *,
        models: List,
        fold_predictions: List[dict],
        X_ref: pd.DataFrame,
        problem_name: str,
        random_seed: int | None = None,
        subset_mask: np.ndarray | None = None,
        scale_label: str = "global",
        y_true: np.ndarray | None = None,
        compute_perm: bool = True,
        compute_drop: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """perm effect와 R2 drop을 한 번의 permutation pass로 산출한다.

        두 채널은 같은 seed에서 같은 순서로 rng를 소비하므로 생성되는 permutation과
        그 예측값이 비트 단위로 동일하다. 따라서 예측을 공유해도 결과가 바뀌지 않는다.

        두 채널의 비대칭은 그대로 보존한다.
        - fold 탈락 조건(y_valid < 2개, r2_base 계산 실패)은 drop 채널에만 적용된다.
          perm 행은 해당 fold에서도 그대로 생성되어야 한다.
        - repeat 중 일부의 r2 계산이 실패하면 drop은 성공한 repeat만으로 평균을 내지만
          perm의 delta는 항상 전체 repeat 평균이다.
        - base 예측이 서로 다른 양이다. perm은 서브샘플에 대한 재예측을 쓰고,
          drop은 fold 캐시 예측을 mask로 정렬해 쓰되 크기가 맞지 않을 때만 재예측한다.
        """
        perm_rows = []
        drop_rows = []
        want_drop = bool(compute_drop and y_true is not None)

        for run_id, rng, model, valid_idx, valid_use, X_num, X_valid, base_arr, columns, fold_item, mask in self._iter_fold_setup(
            models=models,
            fold_predictions=fold_predictions,
            X_ref=X_ref,
            random_seed=random_seed,
            subset_mask=subset_mask,
            apply_perm_sample_size=True,
        ):
            pred_base = (
                np.asarray(model.predict(base_arr), dtype=float).reshape(-1)
                if compute_perm
                else None
            )

            # drop 채널 fold 준비. 여기서 탈락해도 perm 행 생성은 계속한다.
            drop_ok = False
            y_valid = None
            r2_base = None
            if want_drop:
                y_valid = np.asarray(y_true, dtype=float)[valid_use]
                if y_valid.size >= 2:
                    base_pred = np.asarray(fold_item.get("y_pred", []), dtype=float).reshape(-1)
                    if base_pred.size != valid_idx.size:
                        base_pred = np.asarray(
                            model.predict(X_num.iloc[valid_idx].to_numpy()), dtype=float
                        ).reshape(-1)
                    if mask is not None:
                        base_pred = base_pred[mask[valid_idx]]
                    if base_pred.size != y_valid.size:
                        base_pred = np.asarray(
                            model.predict(X_valid.to_numpy()), dtype=float
                        ).reshape(-1)
                    try:
                        r2_base = float(r2_score(y_valid, base_pred))
                        drop_ok = True
                    except Exception:
                        drop_ok = False

            for idx, col in enumerate(columns):
                deltas_k = []
                drops_k = []
                r2_perms_k = []
                for _ in range(self.perm_repeats):
                    X_perm = base_arr.copy()
                    X_perm[:, idx] = rng.permutation(X_perm[:, idx])
                    pred_perm = np.asarray(model.predict(X_perm), dtype=float).reshape(-1)
                    if compute_perm:
                        deltas_k.append(float(np.mean((pred_base - pred_perm) ** 2)))
                    if drop_ok:
                        try:
                            r2_k = float(r2_score(y_valid, pred_perm))
                        except Exception:
                            continue
                        drops_k.append(float(r2_base - r2_k))
                        r2_perms_k.append(r2_k)

                if compute_perm:
                    perm_rows.append(
                        {
                            "problem": problem_name,
                            "scale": str(scale_label),
                            "method": "PERM",
                            "fold": run_id,
                            "feature": col,
                            "delta": float(np.mean(deltas_k)),
                        }
                    )

                if drop_ok and drops_k:
                    drop = float(np.mean(drops_k))
                    r2_perm = float(np.mean(r2_perms_k))
                    drop_pos = float(max(drop, 0.0))
                    drop_rows.append(
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
            "perm_effect_raw": pd.DataFrame(
                perm_rows,
                columns=["problem", "scale", "method", "fold", "feature", "delta"],
            ),
            "score_drop_raw": pd.DataFrame(
                drop_rows,
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
        out = self.run_importance_channels(
            models=models,
            fold_predictions=fold_predictions,
            X_ref=X_ref,
            problem_name=problem_name,
            random_seed=random_seed,
            subset_mask=subset_mask,
            scale_label=scale_label,
            compute_perm=True,
            compute_drop=False,
        )
        return {"perm_effect_raw": out["perm_effect_raw"]}

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
        out = self.run_importance_channels(
            models=models,
            fold_predictions=fold_predictions,
            X_ref=X_ref,
            problem_name=problem_name,
            random_seed=random_seed,
            subset_mask=subset_mask,
            scale_label=scale_label,
            y_true=y_true,
            compute_perm=False,
            compute_drop=True,
        )
        return {"score_drop_raw": out["score_drop_raw"]}

    # =================================================
    # Internal helpers
    # =================================================

    def _iter_fold_setup(
        self,
        *,
        models: List,
        fold_predictions: List[dict],
        X_ref: pd.DataFrame,
        random_seed: int | None,
        subset_mask: np.ndarray | None,
        apply_perm_sample_size: bool,
    ) -> Generator:
        """Yield per-fold setup data shared by both importance methods.

        Yields
        ------
        run_id, rng, model, valid_idx, valid_use, X_num, X_valid, base_arr, columns, fold_item, mask
        """
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
            if (
                apply_perm_sample_size
                and self.perm_sample_size is not None
                and valid_use.size > int(self.perm_sample_size)
            ):
                valid_use = np.sort(
                    rng.choice(valid_use, size=int(self.perm_sample_size), replace=False)
                )
            X_valid = X_num.iloc[valid_use]
            base_arr = X_valid.to_numpy()
            if base_arr.shape[0] == 0:
                continue
            columns = list(X_valid.columns)
            yield run_id, rng, model, valid_idx, valid_use, X_num, X_valid, base_arr, columns, fold_item, mask

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
