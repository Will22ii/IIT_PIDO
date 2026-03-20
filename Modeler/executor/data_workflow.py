from dataclasses import dataclass

import numpy as np
import pandas as pd

from Modeler.config import ModelerSystemConfig
from utils.bool_mask import to_bool_mask


@dataclass
class ModelerDataPolicyResult:
    df: pd.DataFrame
    df_all_for_feas: pd.DataFrame
    variables: list | None
    constraint_defs: list
    has_post_constraints: bool
    feature_cols: list[str]
    cv_policy: dict
    kfold_splits: int
    kfold_repeats: int
    elite_mask: np.ndarray
    n_elite: int
    elite_ratio_eff: float


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
    else:
        # Fallback for standalone external CSV:
        # if feature count matches, map by column order.
        ignore_cols = {
            "id",
            target_col,
            "feasible",
            "feasible_pre",
            "feasible_post",
            "success",
            "source",
            "round",
            "exec_scope",
            "margin_pre",
            "margin_post",
            "constraint_margin",
        }
        candidate_cols = [
            c for c in df.columns
            if c not in ignore_cols and not str(c).startswith("constraint_")
        ]
        if len(candidate_cols) >= len(expected):
            picked = candidate_cols[: len(expected)]
            rename_map = {old: new for old, new in zip(picked, expected)}
            print(
                "- Renaming external feature columns by position: "
                + ", ".join(f"{o}->{n}" for o, n in rename_map.items())
            )
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


def _resolve_cv_policy(
    *,
    n_samples: int,
    n_features: int,
    system_cfg: ModelerSystemConfig,
) -> dict:
    min_valid = max(int(system_cfg.cv_min_valid_size), 1)
    low_data_ratio = float(system_cfg.cv_low_data_np_ratio)
    p_dim = max(int(n_features), 1)
    np_ratio = float(n_samples) / float(p_dim)
    low_data = bool(np_ratio < low_data_ratio)

    if not bool(system_cfg.cv_dynamic_policy):
        k = max(int(system_cfg.kfold_splits), 2)
        r = max(int(system_cfg.kfold_repeats), 1)
        valid_min = int(n_samples // k) if k > 0 else 0
        warn_valid = bool(valid_min < min_valid)
        return {
            "kfold_splits": int(k),
            "kfold_repeats": int(r),
            "min_valid_target": int(min_valid),
            "valid_min_est": int(valid_min),
            "low_data": bool(low_data),
            "np_ratio": float(np_ratio),
            "warn_valid": bool(warn_valid),
            "dynamic": False,
        }

    # Dynamic split policy driven by per-fold valid size target.
    if n_samples >= 5 * min_valid:
        k = 5
    elif n_samples >= 4 * min_valid:
        k = 4
    else:
        k = 3

    if n_samples < k:
        raise RuntimeError(
            f"학습 데이터가 부족합니다: rows={n_samples} < required_splits={k}"
        )

    repeat_map_low = {3: 10, 4: 8, 5: 6}
    repeat_map_normal = {3: 5, 4: 4, 5: 3}
    repeat_map = repeat_map_low if low_data else repeat_map_normal
    r = int(repeat_map.get(k, 1))

    valid_min = int(n_samples // k)
    warn_valid = bool(valid_min < min_valid)
    return {
        "kfold_splits": int(k),
        "kfold_repeats": int(r),
        "min_valid_target": int(min_valid),
        "valid_min_est": int(valid_min),
        "low_data": bool(low_data),
        "np_ratio": float(np_ratio),
        "warn_valid": bool(warn_valid),
        "dynamic": True,
    }


def resolve_cv_policy_for_subset(
    *,
    n_samples: int,
    n_features: int,
    system_cfg: ModelerSystemConfig,
) -> tuple[dict, int, int]:
    policy = _resolve_cv_policy(
        n_samples=int(n_samples),
        n_features=int(n_features),
        system_cfg=system_cfg,
    )
    k = int(policy["kfold_splits"])
    r = int(policy["kfold_repeats"])
    if int(n_samples) < k:
        raise RuntimeError(
            f"학습 데이터가 부족합니다: rows={int(n_samples)} < kfold_splits={k}. "
            "DOE n_samples를 늘려주세요."
        )
    return policy, k, r


def _build_elite_mask(
    *,
    y: np.ndarray,
    objective_sense: str,
    ratio_base: float,
    min_samples: int,
) -> tuple[np.ndarray, int, float]:
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    n = int(y_arr.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=bool), 0, 0.0

    ratio = float(max(ratio_base, float(min_samples) / float(n)))
    ratio = float(np.clip(ratio, 0.0, 1.0))
    n_elite = int(np.ceil(ratio * n))
    n_elite = min(max(n_elite, 1), n)

    sense = str(objective_sense).strip().lower()
    if sense == "max":
        idx = np.argsort(-y_arr)[:n_elite]
    else:
        idx = np.argsort(y_arr)[:n_elite]
    mask = np.zeros((n,), dtype=bool)
    mask[idx] = True
    return mask, int(n_elite), float(ratio)


def _resolve_fi_quantile_top_ratio(
    *,
    low_data: bool,
    p_dim: int,
    system_cfg: ModelerSystemConfig,
) -> float:
    if not bool(low_data):
        return float(np.clip(system_cfg.fi_quantile_top_ratio_default, 0.05, 1.0))
    if p_dim <= 6:
        return float(np.clip(system_cfg.fi_quantile_top_ratio_p_le_6, 0.05, 1.0))
    if p_dim <= 12:
        return float(np.clip(system_cfg.fi_quantile_top_ratio_p_le_12, 0.05, 1.0))
    return float(np.clip(system_cfg.fi_quantile_top_ratio_p_gt_12, 0.05, 1.0))


def _normalize_pair_weights(a: float, b: float) -> tuple[float, float]:
    denom = float(a) + float(b)
    if denom <= 0.0:
        return 1.0, 0.0
    return float(a) / denom, float(b) / denom


def _normalize_triplet_weights(a: float, b: float, c: float) -> tuple[float, float, float]:
    denom = float(a) + float(b) + float(c)
    if denom <= 0.0:
        return 1.0, 0.0, 0.0
    return float(a) / denom, float(b) / denom, float(c) / denom


def prepare_modeler_data_policy(
    *,
    df: pd.DataFrame,
    target_col: str,
    doe_meta: dict,
    cae_variables: list | None,
    cae_constraint_defs: list | None,
    objective_sense: str,
    system_cfg: ModelerSystemConfig,
    keep_debug: bool,
    configured_kfold_splits: int,
    configured_kfold_repeats: int,
) -> ModelerDataPolicyResult:
    variables = None
    constraint_defs = list(cae_constraint_defs or [])
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
                str(c.get("scope", "pre")).strip().lower() == "post"
                for c in constraint_defs
                if isinstance(c, dict)
            )
    if not variables and isinstance(cae_variables, list):
        variables = cae_variables
    if not has_post_constraints and isinstance(constraint_defs, list):
        has_post_constraints = any(
            str(c.get("scope", "pre")).strip().lower() == "post"
            for c in constraint_defs
            if isinstance(c, dict)
        )

    df = _align_df_schema(df, variables=variables, target_col=target_col)
    df_all_for_feas = df.copy()

    if target_col not in df.columns:
        raise RuntimeError(f"Target column not found: {target_col}")

    if "success" in df.columns:
        success_mask = to_bool_mask(
            df["success"],
            column_name="success",
            warn_prefix="[Modeler][BoolParse]",
        )
        n_before = int(len(df))
        df = df.loc[success_mask].reset_index(drop=True)
        n_after = int(len(df))
        print(f"- Success 필터 적용: {n_before} -> {n_after}")
        if n_after == 0:
            raise RuntimeError("No successful DOE rows available for Modeler training.")

    # post/pre 제약이 도입된 경우 최종 feasible만 학습에 사용
    if "feasible" in df.columns:
        feas_mask = to_bool_mask(
            df["feasible"],
            column_name="feasible",
            warn_prefix="[Modeler][BoolParse]",
        )
        n_before = int(len(df))
        df = df.loc[feas_mask].reset_index(drop=True)
        n_after = int(len(df))
        print(f"- Feasible 필터 적용: {n_before} -> {n_after}")
        if n_after == 0:
            raise RuntimeError("No feasible rows available for Modeler training.")

    if variables:
        feature_cols = [v["name"] for v in variables]
    else:
        ignore_cols = {
            "id",
            target_col,
            "feasible",
            "feasible_pre",
            "feasible_post",
            "success",
            "source",
            "round",
            "exec_scope",
            "margin_pre",
            "margin_post",
            "constraint_margin",
        }
        feature_cols = [
            c for c in df.columns
            if c not in ignore_cols and not str(c).startswith("constraint_")
        ]

    if not feature_cols:
        raise RuntimeError("No feature columns resolved for modeling.")

    cv_policy = _resolve_cv_policy(
        n_samples=int(len(df)),
        n_features=int(len(feature_cols)),
        system_cfg=system_cfg,
    )
    kfold_splits = int(cv_policy["kfold_splits"])
    kfold_repeats = int(cv_policy["kfold_repeats"])
    if len(df) < int(kfold_splits):
        raise RuntimeError(
            f"학습 데이터가 부족합니다: rows={len(df)} < kfold_splits={kfold_splits}. "
            "DOE n_samples를 늘려주세요."
        )
    if bool(cv_policy["warn_valid"]):
        print(
            "[Modeler][CV-WARN] "
            f"N={len(df)} p={len(feature_cols)} "
            f"k={kfold_splits} valid_min={cv_policy['valid_min_est']} "
            f"< target={cv_policy['min_valid_target']}. Proceed anyway."
        )
    print(
        "[Modeler][CV] "
        f"dynamic={cv_policy['dynamic']} "
        f"N={len(df)} p={len(feature_cols)} "
        f"N_over_p={cv_policy['np_ratio']:.3f} "
        f"k={kfold_splits} r={kfold_repeats} "
        f"valid_min={cv_policy['valid_min_est']} "
        f"low_data={cv_policy['low_data']} "
        f"(configured_k={configured_kfold_splits}, configured_r={configured_kfold_repeats})"
    )
    if bool(cv_policy["low_data"]):
        print(
            "[Modeler][FI] low-data mode enabled: "
            "SHAP/gain hard-gating is disabled (informational ranking only)."
        )

    y = df[target_col].values
    elite_mask, n_elite, elite_ratio_eff = _build_elite_mask(
        y=y,
        objective_sense=objective_sense,
        ratio_base=float(system_cfg.fi_elite_ratio_base),
        min_samples=int(system_cfg.fi_elite_min_samples),
    )
    print(
        "[Modeler][FI] "
        f"elite_ratio_eff={elite_ratio_eff:.3f} "
        f"n_elite={n_elite}/{len(y)} "
        f"(base={system_cfg.fi_elite_ratio_base}, min={system_cfg.fi_elite_min_samples})"
    )
    if keep_debug:
        p_dim = max(int(len(feature_cols)), 1)
        fi_top_ratio = _resolve_fi_quantile_top_ratio(
            low_data=bool(cv_policy["low_data"]),
            p_dim=p_dim,
            system_cfg=system_cfg,
        )
        w_abs_n, w_q_n, w_r_n = _normalize_triplet_weights(
            float(system_cfg.fi_weight_abs),
            float(system_cfg.fi_weight_quantile),
            float(system_cfg.fi_weight_rank),
        )
        w_drop_raw = float(system_cfg.fi_weight_drop) if bool(system_cfg.fi_use_score_drop) else 0.0
        w_perm_n, w_drop_n = _normalize_pair_weights(
            float(system_cfg.fi_weight_perm),
            w_drop_raw,
        )
        elite_mode = str(getattr(system_cfg, "fi_elite_mode", "bonus")).strip().lower()
        if elite_mode not in {"blend", "bonus", "off"}:
            elite_mode = "bonus"
        elite_bonus_beta = float(np.clip(float(getattr(system_cfg, "fi_elite_bonus_beta", 0.3)), 0.0, 1.0))
        if bool(cv_policy["low_data"]) or int(n_elite) < int(system_cfg.fi_elite_small_threshold):
            w_global = float(system_cfg.fi_weight_global_low)
        elif int(n_elite) >= int(system_cfg.fi_elite_rich_threshold):
            w_global = float(system_cfg.fi_weight_global_rich)
        else:
            w_global = float(system_cfg.fi_weight_global_default)
        w_global = float(np.clip(w_global, 0.0, 1.0))
        w_elite = float(np.clip(1.0 - w_global, 0.0, 1.0))
        print(
            "[Modeler][FI-POLICY] "
            f"perm_eps={system_cfg.perm_epsilon:.3f} "
            f"drop_eps={system_cfg.fi_drop_epsilon:.3f} "
            f"tau={system_cfg.fi_final_score_threshold:.3f} "
            f"global_floor={system_cfg.fi_global_score_floor:.3f} "
            f"top_ratio={fi_top_ratio:.2f}"
        )
        print(
            "[Modeler][FI-ELITE] "
            f"mode={elite_mode} "
            f"bonus_beta={elite_bonus_beta:.2f}"
        )
        print(
            "[Modeler][FI-WEIGHT] "
            f"fold(abs/q/r)={w_abs_n:.2f}/{w_q_n:.2f}/{w_r_n:.2f} "
            f"channel(perm/drop)={w_perm_n:.2f}/{w_drop_n:.2f} "
            f"scale(global/elite)={w_global:.2f}/{w_elite:.2f}"
        )

    return ModelerDataPolicyResult(
        df=df,
        df_all_for_feas=df_all_for_feas,
        variables=variables,
        constraint_defs=constraint_defs if isinstance(constraint_defs, list) else [],
        has_post_constraints=bool(has_post_constraints),
        feature_cols=feature_cols,
        cv_policy=cv_policy,
        kfold_splits=int(kfold_splits),
        kfold_repeats=int(kfold_repeats),
        elite_mask=elite_mask,
        n_elite=int(n_elite),
        elite_ratio_eff=float(elite_ratio_eff),
    )
