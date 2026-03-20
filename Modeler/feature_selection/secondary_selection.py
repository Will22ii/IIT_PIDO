from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from Modeler.executor.trainer import ModelTrainer


@dataclass
class SecondarySelectionConfig:
    # secondary CV 총 fold 수 목표(k*r)
    target_kr: int = 50
    # 최소 repeats
    min_repeats: int = 5
    # delta_r2 평균 최소 기준
    min_delta_r2: float = 0.0
    # delta_r2 > 0 비율 최소 기준
    min_freq: float = 0.7
    eps: float = 1e-12


@dataclass
class SecondarySelectionResult:
    selected_features: list[str]
    diagnostics: list[dict]


def _is_same_fold_layout(
    *,
    fold_predictions_a: list[dict],
    fold_predictions_b: list[dict],
) -> bool:
    if len(fold_predictions_a) != len(fold_predictions_b):
        return False
    for fa, fb in zip(fold_predictions_a, fold_predictions_b):
        va = np.asarray(fa.get("valid_idx", []), dtype=int).reshape(-1)
        vb = np.asarray(fb.get("valid_idx", []), dtype=int).reshape(-1)
        if va.shape != vb.shape:
            return False
        if not np.array_equal(va, vb):
            return False
    return True


def _fold_r2_from_predictions(
    *,
    y_true: np.ndarray,
    fold_predictions: list[dict],
) -> np.ndarray:
    y_arr = np.asarray(y_true, dtype=float).reshape(-1)
    out = np.full((len(fold_predictions),), np.nan, dtype=float)
    for i, fold_item in enumerate(fold_predictions):
        valid_idx = np.asarray(fold_item.get("valid_idx", []), dtype=int).reshape(-1)
        y_pred = np.asarray(fold_item.get("y_pred", []), dtype=float).reshape(-1)
        if valid_idx.size == 0 or y_pred.size == 0:
            continue
        if valid_idx.size != y_pred.size:
            continue
        if np.max(valid_idx, initial=-1) >= int(y_arr.size):
            continue
        y_valid = y_arr[valid_idx]
        if y_valid.size != y_pred.size:
            continue
        if not np.all(np.isfinite(y_valid)) or not np.all(np.isfinite(y_pred)):
            continue
        try:
            s = float(r2_score(y_valid, y_pred))
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(s):
            out[i] = s
    return out


def _rank_selected_features(diagnostics: list[dict]) -> list[str]:
    if len(diagnostics) == 0:
        return []
    df = pd.DataFrame(diagnostics)
    if df.empty or "passed" not in df.columns or "feature" not in df.columns:
        return []
    passed_df = df[df["passed"].astype(bool)].copy()
    if passed_df.empty:
        return []
    passed_df["mean_delta_r2"] = pd.to_numeric(
        passed_df.get("mean_delta_r2", np.nan),
        errors="coerce",
    )
    passed_df["var_delta_r2"] = pd.to_numeric(
        passed_df.get("var_delta_r2", np.nan),
        errors="coerce",
    )
    passed_df = passed_df.replace([np.inf, -np.inf], np.nan)
    passed_df = passed_df.dropna(subset=["feature", "mean_delta_r2", "var_delta_r2"])
    if passed_df.empty:
        return []
    passed_df = passed_df.sort_values(
        by=["mean_delta_r2", "var_delta_r2", "feature"],
        ascending=[False, True, True],
    )
    return passed_df["feature"].astype(str).tolist()


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
    if len(core_features) == 0 or len(candidate_features) == 0:
        return SecondarySelectionResult(selected_features=[], diagnostics=diagnostics)
    if target_col not in df.columns:
        return SecondarySelectionResult(selected_features=[], diagnostics=diagnostics)

    n_samples = int(len(df))
    if n_samples < 2:
        return SecondarySelectionResult(selected_features=[], diagnostics=diagnostics)

    target_kr = max(int(cfg.target_kr), 1)
    min_repeats = max(int(cfg.min_repeats), 1)
    min_delta_r2 = float(cfg.min_delta_r2)
    min_freq = float(np.clip(float(cfg.min_freq), 0.0, 1.0))

    k_eff = max(2, min(int(kfold_splits), n_samples))
    r_from_target = int(math.ceil(float(target_kr) / float(k_eff)))
    r_eff = max(int(kfold_repeats), min_repeats, r_from_target)
    kr_eff = int(k_eff * r_eff)

    print(
        "[Modeler][Secondary] "
        f"enabled=True core={len(core_features)} candidates={len(candidate_features)} "
        f"metric=delta_r2(model2-model1) gate(mean>={min_delta_r2:.6f}, freq>={min_freq:.3f})"
    )
    print(
        "[Modeler][Secondary][CV] "
        f"k={k_eff} r_base={int(kfold_repeats)} r_eff={r_eff} target_kr={target_kr} kr_eff={kr_eff}"
    )

    try:
        base_trainer = ModelTrainer(
            base_random_seed=int(base_seed),
            target_col=target_col,
            feature_cols=list(core_features),
            model_params=dict(model_params or {}),
            model_name=str(model_name),
            kfold_splits=int(k_eff),
            kfold_repeats=int(r_eff),
        )
        base_out = base_trainer.run(df)
    except Exception as exc:  # noqa: BLE001
        for feature in candidate_features:
            diagnostics.append(
                {
                    "feature": str(feature),
                    "mean_delta_r2": float("-inf"),
                    "var_delta_r2": float("inf"),
                    "std_delta_r2": float("nan"),
                    "freq": 0.0,
                    "mean_r2_model1": float("nan"),
                    "mean_r2_model2": float("nan"),
                    "n_pairs": 0,
                    "gate_min_delta_r2": min_delta_r2,
                    "gate_min_freq": min_freq,
                    "gate_target_kr": int(target_kr),
                    "gate_min_repeats": int(min_repeats),
                    "oof_k_used": int(k_eff),
                    "oof_r_used": int(r_eff),
                    "passed": False,
                    "fail_reason": f"model1_train_error:{type(exc).__name__}",
                    "test_available": False,
                }
            )
        return SecondarySelectionResult(selected_features=[], diagnostics=diagnostics)

    y_true = np.asarray(base_out.get("y_true", []), dtype=float).reshape(-1)
    base_fold_predictions = list(base_out.get("fold_predictions", []))
    base_fold_r2 = _fold_r2_from_predictions(y_true=y_true, fold_predictions=base_fold_predictions)

    for feature in candidate_features:
        feature_s = str(feature)
        if feature_s not in df.columns:
            diagnostics.append(
                {
                    "feature": feature_s,
                    "mean_delta_r2": float("-inf"),
                    "var_delta_r2": float("inf"),
                    "std_delta_r2": float("nan"),
                    "freq": 0.0,
                    "mean_r2_model1": float("nan"),
                    "mean_r2_model2": float("nan"),
                    "n_pairs": 0,
                    "gate_min_delta_r2": min_delta_r2,
                    "gate_min_freq": min_freq,
                    "gate_target_kr": int(target_kr),
                    "gate_min_repeats": int(min_repeats),
                    "oof_k_used": int(k_eff),
                    "oof_r_used": int(r_eff),
                    "passed": False,
                    "fail_reason": "missing_feature_column",
                    "test_available": False,
                }
            )
            continue

        try:
            cand_trainer = ModelTrainer(
                base_random_seed=int(base_seed),
                target_col=target_col,
                feature_cols=list(core_features) + [feature_s],
                model_params=dict(model_params or {}),
                model_name=str(model_name),
                kfold_splits=int(k_eff),
                kfold_repeats=int(r_eff),
            )
            cand_out = cand_trainer.run(df)
        except Exception as exc:  # noqa: BLE001
            diagnostics.append(
                {
                    "feature": feature_s,
                    "mean_delta_r2": float("-inf"),
                    "var_delta_r2": float("inf"),
                    "std_delta_r2": float("nan"),
                    "freq": 0.0,
                    "mean_r2_model1": float("nan"),
                    "mean_r2_model2": float("nan"),
                    "n_pairs": 0,
                    "gate_min_delta_r2": min_delta_r2,
                    "gate_min_freq": min_freq,
                    "gate_target_kr": int(target_kr),
                    "gate_min_repeats": int(min_repeats),
                    "oof_k_used": int(k_eff),
                    "oof_r_used": int(r_eff),
                    "passed": False,
                    "fail_reason": f"candidate_train_error:{type(exc).__name__}",
                    "test_available": False,
                }
            )
            continue

        cand_fold_predictions = list(cand_out.get("fold_predictions", []))
        cand_y_true = np.asarray(cand_out.get("y_true", []), dtype=float).reshape(-1)
        if cand_y_true.shape != y_true.shape or not np.allclose(cand_y_true, y_true, equal_nan=True):
            diagnostics.append(
                {
                    "feature": feature_s,
                    "mean_delta_r2": float("-inf"),
                    "var_delta_r2": float("inf"),
                    "std_delta_r2": float("nan"),
                    "freq": 0.0,
                    "mean_r2_model1": float("nan"),
                    "mean_r2_model2": float("nan"),
                    "n_pairs": 0,
                    "gate_min_delta_r2": min_delta_r2,
                    "gate_min_freq": min_freq,
                    "gate_target_kr": int(target_kr),
                    "gate_min_repeats": int(min_repeats),
                    "oof_k_used": int(k_eff),
                    "oof_r_used": int(r_eff),
                    "passed": False,
                    "fail_reason": "y_alignment_mismatch",
                    "test_available": False,
                }
            )
            continue

        if not _is_same_fold_layout(
            fold_predictions_a=base_fold_predictions,
            fold_predictions_b=cand_fold_predictions,
        ):
            diagnostics.append(
                {
                    "feature": feature_s,
                    "mean_delta_r2": float("-inf"),
                    "var_delta_r2": float("inf"),
                    "std_delta_r2": float("nan"),
                    "freq": 0.0,
                    "mean_r2_model1": float("nan"),
                    "mean_r2_model2": float("nan"),
                    "n_pairs": 0,
                    "gate_min_delta_r2": min_delta_r2,
                    "gate_min_freq": min_freq,
                    "gate_target_kr": int(target_kr),
                    "gate_min_repeats": int(min_repeats),
                    "oof_k_used": int(k_eff),
                    "oof_r_used": int(r_eff),
                    "passed": False,
                    "fail_reason": "fold_layout_mismatch",
                    "test_available": False,
                }
            )
            continue

        cand_fold_r2 = _fold_r2_from_predictions(y_true=y_true, fold_predictions=cand_fold_predictions)
        pair_ok = np.isfinite(base_fold_r2) & np.isfinite(cand_fold_r2)
        n_pairs = int(np.sum(pair_ok))
        if n_pairs <= 0:
            diagnostics.append(
                {
                    "feature": feature_s,
                    "mean_delta_r2": float("-inf"),
                    "var_delta_r2": float("inf"),
                    "std_delta_r2": float("nan"),
                    "freq": 0.0,
                    "mean_r2_model1": float("nan"),
                    "mean_r2_model2": float("nan"),
                    "n_pairs": 0,
                    "gate_min_delta_r2": min_delta_r2,
                    "gate_min_freq": min_freq,
                    "gate_target_kr": int(target_kr),
                    "gate_min_repeats": int(min_repeats),
                    "oof_k_used": int(k_eff),
                    "oof_r_used": int(r_eff),
                    "passed": False,
                    "fail_reason": "no_valid_fold_pair",
                    "test_available": False,
                }
            )
            continue

        r2_1 = base_fold_r2[pair_ok]
        r2_2 = cand_fold_r2[pair_ok]
        delta = r2_2 - r2_1

        mean_delta = float(np.mean(delta))
        var_delta = float(np.var(delta, ddof=0))
        std_delta = float(np.std(delta, ddof=0))
        freq = float(np.mean(delta > 0.0))
        mean_r2_1 = float(np.mean(r2_1))
        mean_r2_2 = float(np.mean(r2_2))

        cond_mean = bool(mean_delta >= min_delta_r2)
        cond_freq = bool(freq >= min_freq)
        passed = bool(cond_mean and cond_freq)
        fail_reasons: list[str] = []
        if not cond_mean:
            fail_reasons.append("mean_delta_r2")
        if not cond_freq:
            fail_reasons.append("freq")
        fail_reason = "passed" if passed else ",".join(fail_reasons)

        diag = {
            "feature": feature_s,
            "mean_delta_r2": mean_delta,
            "var_delta_r2": var_delta,
            "std_delta_r2": std_delta,
            "freq": freq,
            "mean_r2_model1": mean_r2_1,
            "mean_r2_model2": mean_r2_2,
            "n_pairs": n_pairs,
            "gate_min_delta_r2": min_delta_r2,
            "gate_min_freq": min_freq,
            "gate_target_kr": int(target_kr),
            "gate_min_repeats": int(min_repeats),
            "oof_k_used": int(k_eff),
            "oof_r_used": int(r_eff),
            "passed": passed,
            "fail_reason": fail_reason,
            "test_available": True,
        }
        diagnostics.append(diag)

        print(
            "[Modeler][Secondary] "
            f"feature={feature_s} mean_delta_r2={mean_delta:.6f} freq={freq:.3f} var={var_delta:.6f} "
            f"r2_1={mean_r2_1:.6f} r2_2={mean_r2_2:.6f} "
            f"pairs={n_pairs} pass={passed} reason={fail_reason}"
        )

    for d in diagnostics:
        if d.get("test_available", False):
            continue
        print(
            "[Modeler][Secondary] "
            f"feature={d.get('feature')} pass=False reason={d.get('fail_reason')}"
        )

    selected = _rank_selected_features(diagnostics)
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
