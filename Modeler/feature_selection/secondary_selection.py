from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

from Modeler.executor.trainer import ModelTrainer


@dataclass
class SecondarySelectionConfig:
    n_knots: int = 4
    ridge_alpha: float = 1.0
    null_repeats: int = 30
    min_mean_r2: float = 0.02
    min_delta_vs_null: float = 0.005
    min_freq: float = 0.7
    fold_pass_r2: float = 0.005


@dataclass
class SecondarySelectionResult:
    selected_features: list[str]
    diagnostics: list[dict]


def _stable_token(value: str) -> int:
    token = 0
    for idx, ch in enumerate(str(value), start=1):
        token += idx * ord(ch)
    return int(token % 100_000)


def _make_model(*, cfg: SecondarySelectionConfig) -> Pipeline:
    return Pipeline(
        [
            (
                "spline",
                SplineTransformer(
                    n_knots=int(cfg.n_knots),
                    knots="quantile",
                    degree=3,
                    include_bias=False,
                    extrapolation="linear",
                ),
            ),
            ("ridge", Ridge(alpha=float(cfg.ridge_alpha))),
        ]
    )


def _fold_scores(
    *,
    x: np.ndarray,
    residual: np.ndarray,
    fold_predictions: list[dict],
    cfg: SecondarySelectionConfig,
) -> np.ndarray:
    n = int(len(x))
    idx_all = np.arange(n, dtype=int)
    scores = np.full((len(fold_predictions),), np.nan, dtype=float)

    for i, fold_item in enumerate(fold_predictions):
        valid_idx = np.asarray(fold_item["valid_idx"], dtype=int)
        if valid_idx.size < 2:
            continue
        train_mask = np.ones((n,), dtype=bool)
        train_mask[valid_idx] = False
        train_idx = idx_all[train_mask]
        if train_idx.size < max(12, int(cfg.n_knots) + 2):
            continue
        x_train = x[train_idx]
        if np.unique(x_train).size < 3:
            continue
        x_valid = x[valid_idx]
        y_train = residual[train_idx]
        y_valid = residual[valid_idx]

        try:
            model = _make_model(cfg=cfg)
            model.fit(x_train.reshape(-1, 1), y_train)
            pred = model.predict(x_valid.reshape(-1, 1))
            score = float(r2_score(y_valid, pred))
            if np.isfinite(score):
                scores[i] = score
        except Exception:
            continue
    return scores


def run_secondary_selection(
    *,
    df: pd.DataFrame,
    target_col: str,
    base_seed: int,
    model_name: str,
    model_params: dict,
    kfold_splits: int,
    kfold_repeats: int,
    core_features: list[str],
    candidate_features: list[str],
    cfg: SecondarySelectionConfig,
) -> SecondarySelectionResult:
    diagnostics: list[dict] = []
    selected: list[str] = []
    if len(core_features) == 0 or len(candidate_features) == 0:
        return SecondarySelectionResult(selected_features=selected, diagnostics=diagnostics)

    trainer = ModelTrainer(
        base_random_seed=int(base_seed),
        target_col=target_col,
        feature_cols=core_features,
        model_params=model_params,
        model_name=model_name,
        kfold_splits=int(kfold_splits),
        kfold_repeats=int(kfold_repeats),
    )
    train_out = trainer.run(df)
    residual = np.asarray(train_out["y_true"], dtype=float) - np.asarray(train_out["oof_prediction"], dtype=float)
    fold_predictions = train_out["fold_predictions"]
    n_folds = int(len(fold_predictions))

    print(
        "[Modeler][Secondary] "
        f"enabled=True core={len(core_features)} candidates={len(candidate_features)} "
        f"null_repeats={cfg.null_repeats} folds={n_folds}"
    )

    for feature in candidate_features:
        x = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
        obs = _fold_scores(
            x=x,
            residual=residual,
            fold_predictions=fold_predictions,
            cfg=cfg,
        )
        obs_valid = obs[np.isfinite(obs)]
        if obs_valid.size == 0:
            diag = {
                "feature": feature,
                "mean_r2": float("-inf"),
                "std_r2": float("nan"),
                "ci_low": float("-inf"),
                "null_q95": float("inf"),
                "freq": 0.0,
                "passed": False,
                "fail_reason": "no_valid_fold_score",
            }
            diagnostics.append(diag)
            print(
                "[Modeler][Secondary] "
                f"feature={feature} mean_r2=-inf null_q95=inf freq=0.000 ci_low=-inf "
                "pass=False reason=no_valid_fold_score"
            )
            continue

        mean_r2 = float(np.mean(obs_valid))
        std_r2 = float(np.std(obs_valid, ddof=0))
        freq = float(np.mean(np.where(np.isfinite(obs), obs >= float(cfg.fold_pass_r2), False)))
        ci_low = float(
            mean_r2
            - 1.96 * std_r2 / np.sqrt(max(int(obs_valid.size), 1))
        )

        null_scores: list[float] = []
        token = _stable_token(feature)
        n = int(len(x))
        for j in range(int(cfg.null_repeats)):
            rng = np.random.default_rng(int(base_seed) + 10_007 + token + 97 * (j + 1))
            x_perm = x[rng.permutation(n)]
            null_fold = _fold_scores(
                x=x_perm,
                residual=residual,
                fold_predictions=fold_predictions,
                cfg=cfg,
            )
            valid_null = null_fold[np.isfinite(null_fold)]
            if valid_null.size > 0:
                null_scores.append(float(np.mean(valid_null)))

        null_q95 = float(np.quantile(null_scores, 0.95)) if len(null_scores) > 0 else float("inf")

        cond_mean = bool(mean_r2 >= float(cfg.min_mean_r2))
        cond_null = bool(mean_r2 >= (null_q95 + float(cfg.min_delta_vs_null)))
        cond_freq = bool(freq >= float(cfg.min_freq))
        cond_ci = bool(ci_low > 0.0)
        passed = bool(cond_mean and cond_null and cond_freq and cond_ci)

        fail_reasons = []
        if not cond_mean:
            fail_reasons.append("mean_r2")
        if not cond_null:
            fail_reasons.append("null_margin")
        if not cond_freq:
            fail_reasons.append("freq")
        if not cond_ci:
            fail_reasons.append("ci_low")
        fail_reason = "passed" if passed else ",".join(fail_reasons)

        diag = {
            "feature": feature,
            "mean_r2": mean_r2,
            "std_r2": std_r2,
            "ci_low": ci_low,
            "null_q95": null_q95,
            "freq": freq,
            "passed": passed,
            "fail_reason": fail_reason,
        }
        diagnostics.append(diag)
        if passed:
            selected.append(feature)

        print(
            "[Modeler][Secondary] "
            f"feature={feature} mean_r2={mean_r2:.4f} null_q95={null_q95:.4f} "
            f"freq={freq:.3f} ci_low={ci_low:.4f} pass={passed} reason={fail_reason}"
        )

    print(
        "[Modeler][Secondary] "
        f"selected={len(selected)}/{len(candidate_features)} "
        + (", ".join(selected) if len(selected) > 0 else "none")
    )
    return SecondarySelectionResult(selected_features=selected, diagnostics=diagnostics)


def merge_secondary_features(
    *,
    selected_df: pd.DataFrame,
    selected_features: list[str],
    secondary_features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    if len(secondary_features) == 0:
        out_df = selected_df.copy()
        if "selected_by_secondary" not in out_df.columns:
            out_df["selected_by_secondary"] = False
        return out_df, list(selected_features)

    out_df = selected_df.copy()
    if "selected_by_secondary" not in out_df.columns:
        out_df["selected_by_secondary"] = False
    if "forced_by_constraint" not in out_df.columns:
        out_df["forced_by_constraint"] = False

    merged = list(selected_features)
    existing = set(str(f) for f in merged)

    for feat in secondary_features:
        feat_s = str(feat)
        if feat_s not in existing:
            merged.append(feat_s)
            existing.add(feat_s)

        mask = out_df.get("feature", pd.Series([], dtype=str)).astype(str) == feat_s
        if bool(mask.any()):
            out_df.loc[mask, "selected"] = True
            out_df.loc[mask, "selected_by_secondary"] = True
            if "reason" in out_df.columns:
                out_df.loc[mask, "reason"] = "secondary_pass"
        else:
            out_df = pd.concat(
                [
                    out_df,
                    pd.DataFrame(
                        [
                            {
                                "feature": feat_s,
                                "selected": True,
                                "selected_by_secondary": True,
                                "forced_by_constraint": False,
                                "reason": "secondary_pass",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    return out_df, merged
