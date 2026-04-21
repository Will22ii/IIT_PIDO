from dataclasses import dataclass

from CAE_tool_interface.config import CAEConfig, CAEUserConfig


@dataclass
class ModelerUserConfig:
    # 모델 종류 이름 (예: "xgb")
    model_name: str = "xgb"
    # HPO 사용 여부
    use_hpo: bool = False
    # 목표 컬럼명
    target_col: str = "objective"
    # Secondary Selection 사용 여부
    use_secondary_selection: bool = False


@dataclass
class ModelerSystemConfig:
    # 1) HPO / CV
    hpo_config: dict | None = None
    kfold_splits: int = 5
    kfold_repeats: int = 2
    cv_dynamic_policy: bool = True
    cv_min_valid_size: int = 14
    cv_low_data_np_ratio: float = 15.0

    # 2) FI 기본 채널 (perm/drop)
    perm_sample_size: int = 1000
    perm_repeats: int = 5
    perm_min_pass_rate: float = 0.6
    perm_epsilon: float = 0.05
    fi_use_score_drop: bool = True
    fi_drop_metric: str = "drop_sq"
    fi_drop_min_pass_rate: float = 0.6
    fi_drop_epsilon: float = 0.06
    fi_drop_min_pass_rate_very_low_data: float = 0.35
    fi_drop_epsilon_very_low_data: float = 0.02

    # 3) FI 가중치
    fi_weight_abs: float = 0.7
    fi_weight_quantile: float = 0.15
    fi_weight_rank: float = 0.15
    fi_weight_perm: float = 0.80
    fi_weight_drop: float = 0.20
    fi_weight_perm_low_data: float = 0.85
    fi_weight_drop_low_data: float = 0.15

    # 4) global/elite 결합
    fi_weight_global_default: float = 0.6
    fi_weight_global_low: float = 0.6
    fi_weight_global_rich: float = 0.5
    fi_elite_small_threshold: int = 40
    fi_elite_rich_threshold: int = 80
    fi_elite_mode: str = "bonus"
    fi_elite_bonus_beta: float = 0.35
    fi_elite_var_penalty_enabled: bool = True
    fi_elite_var_threshold: float = 0.25
    fi_elite_var_penalty_scale: float = 0.15

    # 5) 최종 선택 컷
    fi_final_score_threshold: float = 0.58
    fi_global_score_floor: float = 0.25

    # 6) Stability gate
    fi_stability_enabled: bool = True
    fi_stability_rule: str = "or"
    fi_stability_very_low_data_n_threshold: int = 55
    fi_stability_rule_very_low_data: str = "and"
    fi_stability_perm_min_rate_very_low_data: float = 0.60
    fi_stability_drop_min_rate_very_low_data: float = 0.35
    fi_stability_rule_low_data: str = "and"
    fi_stability_perm_min_rate_low_data: float = 0.60
    fi_stability_drop_min_rate_low_data: float = 0.44
    fi_stability_rule_normal: str = "or"
    fi_stability_perm_min_rate_normal: float = 0.80
    fi_stability_drop_min_rate_normal: float = 0.60

    # 6-1) 채널 불일치/중복 감쇠
    fi_disagreement_penalty_enabled: bool = True
    fi_disagreement_threshold: float = 0.20
    fi_disagreement_penalty_scale: float = 0.80
    fi_redundancy_dampening_enabled: bool = True
    fi_redundancy_perm_floor: float = 0.85
    fi_redundancy_drop_ceil: float = 0.40
    fi_redundancy_dampening_factor: float = 0.5
    # 6-1a) perm-dominant override gating (L1)
    fi_perm_dominant_perm_threshold: float = 0.93
    fi_perm_dominant_gap_ceil: float = 0.25

    # 6-2) score gap / veto
    fi_gap_filter_enabled: bool = True
    fi_gap_threshold_very_low_data: float = 0.08
    fi_gap_threshold_normal: float = 0.12
    fi_gap_global_floor: float = 0.79
    fi_gap_global_floor_very_low_data: float = 0.60
    fi_gap_global_floor_p_le_4: float = 0.85
    fi_gap_global_floor_p_ge_8: float = 0.75
    fi_gap_min_retain: int = 2
    fi_drop_veto_enabled: bool = True
    fi_drop_veto_threshold: float = 0.03
    fi_perm_var_penalty_very_low_data_enabled: bool = True
    fi_perm_var_penalty_very_low_data_scale: float = 0.35

    # 7) Bootstrap stability
    fi_bootstrap_enabled: bool = True
    fi_bootstrap_np_threshold: float = 50.0
    fi_bootstrap_rounds: int = 20
    fi_bootstrap_sample_ratio: float = 0.8
    fi_bootstrap_min_freq: float = 0.60
    fi_bootstrap_min_freq_low_data: float = 0.78
    fi_bootstrap_min_freq_very_low_data: float = 0.55
    fi_bootstrap_rescue_global_floor: float = 0.78
    fi_bootstrap_rescue_very_low_data_only: bool = False
    fi_bootstrap_rescue_perm_floor: float = 0.85
    fi_bootstrap_rescue_min_freq: float = 0.30

    # 8) Null gate
    fi_null_enabled: bool = True
    fi_null_mode: str = "soft"
    fi_null_quantile: float = 0.90
    fi_null_shuffle_runs_low_data: int = 50
    fi_null_shuffle_runs_normal: int = 30
    fi_null_alpha_low_data: float = 0.55
    fi_null_alpha_normal: float = 0.12
    fi_null_apply_to: str = "both"
    fi_null_pre_elite_ratio: float = 0.5

    # 9) Elite subset
    fi_elite_ratio_base: float = 0.30
    fi_elite_min_samples: int = 30

    # 10) Quantile(top-ratio)
    fi_quantile_top_ratio_default: float = 0.50
    fi_quantile_top_ratio_p_le_6: float = 0.45
    fi_quantile_top_ratio_p_le_12: float = 0.40
    fi_quantile_top_ratio_p_gt_12: float = 0.35

    # 11) 디버그
    debug_level: str = "full"

    # 12) Primary / Secondary selection
    use_primary_selection: bool = True
    secondary_target_kr: int = 50
    secondary_min_repeats: int = 5
    secondary_min_delta_r2: float = 0.0
    secondary_min_freq: float = 0.7

    # 13) Anti-collapse guard behaviour (L2)
    # False: 품질 후보(=bootstrap/null 미실패, null_pass=True)가 부족하면
    # min_features 미달을 허용. True: score 상위 feature로 강제 보충(legacy).
    fi_guard_force_fill: bool = False


def build_feature_selection_config(system: "ModelerSystemConfig") -> "FeatureSelectionConfig":
    """ModelerSystemConfig → FeatureSelectionConfig 변환."""
    from Modeler.feature_selection.primary_selection import FeatureSelectionConfig

    return FeatureSelectionConfig(
        # perm 채널 (접두사 없음)
        perm_min_pass_rate=system.perm_min_pass_rate,
        perm_epsilon=system.perm_epsilon,
        # drop 채널
        use_score_drop=system.fi_use_score_drop,
        drop_metric=system.fi_drop_metric,
        drop_min_pass_rate=system.fi_drop_min_pass_rate,
        drop_epsilon=system.fi_drop_epsilon,
        drop_min_pass_rate_very_low_data=system.fi_drop_min_pass_rate_very_low_data,
        drop_epsilon_very_low_data=system.fi_drop_epsilon_very_low_data,
        # fold vote weights
        weight_abs=system.fi_weight_abs,
        weight_quantile=system.fi_weight_quantile,
        weight_rank=system.fi_weight_rank,
        # channel merge weights
        weight_perm=system.fi_weight_perm,
        weight_drop=system.fi_weight_drop,
        weight_perm_low_data=system.fi_weight_perm_low_data,
        weight_drop_low_data=system.fi_weight_drop_low_data,
        # scale merge weights
        weight_global_default=system.fi_weight_global_default,
        weight_global_low=system.fi_weight_global_low,
        weight_global_rich=system.fi_weight_global_rich,
        # elite
        elite_small_threshold=system.fi_elite_small_threshold,
        elite_rich_threshold=system.fi_elite_rich_threshold,
        elite_mode=system.fi_elite_mode,
        elite_bonus_beta=system.fi_elite_bonus_beta,
        elite_var_penalty_enabled=system.fi_elite_var_penalty_enabled,
        elite_var_threshold=system.fi_elite_var_threshold,
        elite_var_penalty_scale=system.fi_elite_var_penalty_scale,
        # decision guards
        final_score_threshold=system.fi_final_score_threshold,
        global_score_floor=system.fi_global_score_floor,
        # stability gate
        stability_enabled=system.fi_stability_enabled,
        stability_rule=system.fi_stability_rule,
        stability_very_low_data_n_threshold=system.fi_stability_very_low_data_n_threshold,
        stability_rule_very_low_data=system.fi_stability_rule_very_low_data,
        stability_perm_min_rate_very_low_data=system.fi_stability_perm_min_rate_very_low_data,
        stability_drop_min_rate_very_low_data=system.fi_stability_drop_min_rate_very_low_data,
        stability_rule_low_data=system.fi_stability_rule_low_data,
        stability_perm_min_rate_low_data=system.fi_stability_perm_min_rate_low_data,
        stability_drop_min_rate_low_data=system.fi_stability_drop_min_rate_low_data,
        stability_rule_normal=system.fi_stability_rule_normal,
        stability_perm_min_rate_normal=system.fi_stability_perm_min_rate_normal,
        stability_drop_min_rate_normal=system.fi_stability_drop_min_rate_normal,
        # disagreement penalty
        disagreement_penalty_enabled=system.fi_disagreement_penalty_enabled,
        disagreement_threshold=system.fi_disagreement_threshold,
        disagreement_penalty_scale=system.fi_disagreement_penalty_scale,
        # perm-dominant override gating (L1)
        perm_dominant_perm_threshold=system.fi_perm_dominant_perm_threshold,
        perm_dominant_gap_ceil=system.fi_perm_dominant_gap_ceil,
        # drop veto
        drop_veto_enabled=system.fi_drop_veto_enabled,
        drop_veto_threshold=system.fi_drop_veto_threshold,
        # perm var penalty
        perm_var_penalty_very_low_data_enabled=system.fi_perm_var_penalty_very_low_data_enabled,
        perm_var_penalty_very_low_data_scale=system.fi_perm_var_penalty_very_low_data_scale,
        # redundancy dampening
        redundancy_dampening_enabled=system.fi_redundancy_dampening_enabled,
        redundancy_perm_floor=system.fi_redundancy_perm_floor,
        redundancy_drop_ceil=system.fi_redundancy_drop_ceil,
        redundancy_dampening_factor=system.fi_redundancy_dampening_factor,
        # gap filter
        gap_filter_enabled=system.fi_gap_filter_enabled,
        gap_threshold_very_low_data=system.fi_gap_threshold_very_low_data,
        gap_threshold_normal=system.fi_gap_threshold_normal,
        gap_global_floor=system.fi_gap_global_floor,
        gap_global_floor_very_low_data=system.fi_gap_global_floor_very_low_data,
        gap_global_floor_p_le_4=system.fi_gap_global_floor_p_le_4,
        gap_global_floor_p_ge_8=system.fi_gap_global_floor_p_ge_8,
        gap_min_retain=system.fi_gap_min_retain,
        # null importance
        null_enabled=system.fi_null_enabled,
        null_mode=system.fi_null_mode,
        null_quantile=system.fi_null_quantile,
        null_shuffle_runs_low_data=system.fi_null_shuffle_runs_low_data,
        null_shuffle_runs_normal=system.fi_null_shuffle_runs_normal,
        null_alpha_low_data=system.fi_null_alpha_low_data,
        null_alpha_normal=system.fi_null_alpha_normal,
        null_apply_to=system.fi_null_apply_to,
        null_pre_elite_ratio=system.fi_null_pre_elite_ratio,
        # quantile policy
        # bootstrap rescue
        bootstrap_min_freq_low_data=system.fi_bootstrap_min_freq_low_data,
        bootstrap_min_freq_very_low_data=system.fi_bootstrap_min_freq_very_low_data,
        fi_bootstrap_rescue_global_floor=system.fi_bootstrap_rescue_global_floor,
        fi_bootstrap_rescue_very_low_data_only=system.fi_bootstrap_rescue_very_low_data_only,
        fi_bootstrap_rescue_perm_floor=system.fi_bootstrap_rescue_perm_floor,
        fi_bootstrap_rescue_min_freq=system.fi_bootstrap_rescue_min_freq,
        quantile_top_ratio_default=system.fi_quantile_top_ratio_default,
        quantile_top_ratio_p_le_6=system.fi_quantile_top_ratio_p_le_6,
        quantile_top_ratio_p_le_12=system.fi_quantile_top_ratio_p_le_12,
        quantile_top_ratio_p_gt_12=system.fi_quantile_top_ratio_p_gt_12,
    )


@dataclass
class ModelerConfig:
    # 모델러 사용자 설정
    user: ModelerUserConfig
    # 모델러 시스템 설정
    system: ModelerSystemConfig
    # CAE 설정
    cae: CAEConfig
    # CAE 사용자 설정 (선택)
    cae_user: CAEUserConfig | None = None
    # DOE CSV 경로 (선택)
    doe_csv_path: str | None = None
    # DOE 메타데이터 경로 (선택)
    doe_metadata_path: str | None = None
    # CAE 메타데이터 경로 (선택, run_context가 없을 때 필수)
    cae_metadata_path: str | None = None
