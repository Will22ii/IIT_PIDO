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
    # 보조 선택 사용 여부
    use_secondary_selection: bool = False


@dataclass
class ModelerSystemConfig:
    # -----------------------------
    # 1) HPO / 교차검증
    # -----------------------------
    # HPO 상세 설정(옵션, model_name별 registry가 있는 경우 사용)
    # - n_trials: 일반 모드 trial 수(기본 20)
    # - lambda_std: robust objective 가중치(기본 0.5)
    # - reuse_if_same_config: signature 일치 시 캐시 재사용 여부(기본 True)
    # - low_data_constrained_enabled: 저데이터 제약 HPO 활성화(기본 True)
    # - low_data_n_trials: 저데이터 모드 trial 수(기본 10)
    # - low_data_lambda_std: 저데이터 모드 lambda_std(기본 lambda_std와 동일)
    # - sampler: HPO sampler 선택(tpe|cmaes, 기본 tpe)
    # - search_space: 일반 모드 탐색공간 override(dict)
    # - low_data_search_space: 저데이터 모드 탐색공간 override(dict)
    # - models: 모델별 override 예: {"gp": {"search_space": {...}}}
    hpo_config: dict | None = None
    # 기본 CV split/repeat
    kfold_splits: int = 5
    kfold_repeats: int = 2
    # 동적 CV 정책
    cv_dynamic_policy: bool = True
    # fold별 최소 valid 샘플 목표(동적 k 결정 기준)
    cv_min_valid_size: int = 14
    # 저데이터 기준: usable N / p < ratio
    cv_low_data_np_ratio: float = 15.0

    # -----------------------------
    # 2) FI 기본 채널(perm / drop)
    # -----------------------------
    # permutation importance 평가 샘플 수(각 fold valid에서 최대)
    perm_sample_size: int = 1000
    # permutation 반복 횟수(fold당). 높을수록 importance 추정이 안정적. very_low_data에서 효과적.
    perm_repeats: int = 5
    # perm 채널 통과 기준
    perm_min_pass_rate: float = 0.6
    perm_epsilon: float = 0.05
    # drop 채널 설정(OOF/valid R2 drop 기반)
    fi_use_score_drop: bool = True
    fi_drop_metric: str = "drop_sq"
    fi_drop_min_pass_rate: float = 0.6
    fi_drop_epsilon: float = 0.06
    # very_low_data 전용 drop 채널 임계값(n_samples < stability_very_low_data_n_threshold일 때)
    fi_drop_min_pass_rate_very_low_data: float = 0.35
    fi_drop_epsilon_very_low_data: float = 0.02

    # -----------------------------
    # 3) FI 가중 투표(채널 내부)
    # -----------------------------
    # fold vote 가중치 (합 1 권장): abs / quantile / rank
    fi_weight_abs: float = 0.7
    fi_weight_quantile: float = 0.15
    fi_weight_rank: float = 0.15
    # 채널 결합 가중치 (perm / drop, 합 1 권장)
    fi_weight_perm: float = 0.80
    fi_weight_drop: float = 0.20
    # low_data 전용 채널 가중치(low_data=True일 때 fi_weight_perm/drop 대신 사용)
    fi_weight_perm_low_data: float = 0.85
    fi_weight_drop_low_data: float = 0.15

    # -----------------------------
    # 4) FI 스케일 결합(global / elite)
    # -----------------------------
    # 기본 결합 가중치(합 1 권장)
    fi_weight_global_default: float = 0.6
    # 저데이터/작은 elite 시 global 가중치
    fi_weight_global_low: float = 0.6
    # 충분한 elite 시 global 가중치
    fi_weight_global_rich: float = 0.5
    # elite 크기 임계(작음/충분)
    fi_elite_small_threshold: int = 40
    fi_elite_rich_threshold: int = 80
    # elite 최종 점수 결합 모드: blend | bonus | off
    # - blend: 기존 global/elite 가중합
    # - bonus: final = global + beta * max(elite - global, 0)
    # - off: final = global
    fi_elite_mode: str = "bonus"
    # elite bonus 계수(beta), [0, 1]로 내부 clip
    fi_elite_bonus_beta: float = 0.35
    # elite variance penalty: fold별 vote 분산이 높은 feature에 패널티
    fi_elite_var_penalty_enabled: bool = True
    fi_elite_var_threshold: float = 0.25
    fi_elite_var_penalty_scale: float = 0.15

    # -----------------------------
    # 5) FI 최종 선택 가드
    # -----------------------------
    # final/global 컷오프
    fi_final_score_threshold: float = 0.58
    fi_global_score_floor: float = 0.25

    # -----------------------------
    # 6) FI Stability Gate
    # -----------------------------
    # stability gate 사용 여부
    fi_stability_enabled: bool = True
    # stability 결합 규칙(legacy fallback): or | and
    fi_stability_rule: str = "or"
    # -----------------------------
    # 3단계 stability 분기 (n_samples 기반)
    # very_low_data: n_samples < threshold → or 규칙, 느슨한 임계값(six_hump 류)
    # low_data     : low_data=True AND n_samples >= threshold → and 규칙, 중간 임계값(cantilever 류)
    # normal_data  : low_data=False → and 규칙, 엄격한 임계값
    fi_stability_very_low_data_n_threshold: int = 55
    fi_stability_rule_very_low_data: str = "and"
    fi_stability_perm_min_rate_very_low_data: float = 0.60
    fi_stability_drop_min_rate_very_low_data: float = 0.35
    fi_stability_rule_low_data: str = "and"
    fi_stability_perm_min_rate_low_data: float = 0.60
    fi_stability_drop_min_rate_low_data: float = 0.44
    fi_stability_rule_normal: str = "or"
    fi_stability_perm_min_rate_normal: float = 0.78
    fi_stability_drop_min_rate_normal: float = 0.60
    # -----------------------------
    # 채널 불일치 패널티 (disagreement penalty)
    # perm/drop 불일치가 클수록 final_score 감산
    fi_disagreement_penalty_enabled: bool = True
    fi_disagreement_threshold: float = 0.25   # 이 이상 불일치면 패널티 시작
    fi_disagreement_penalty_scale: float = 0.40  # 패널티 강도 (B: 0.55→0.40, 과도한 감점 완화)
    # redundancy-aware disagreement 완화(L1)
    # perm이 높지만 drop이 매우 낮은 경우 (변수 간 정보 중복 의심) disagreement penalty 감면
    fi_redundancy_dampening_enabled: bool = True
    fi_redundancy_perm_floor: float = 0.78   # perm_rate >= 이 값 (rosenbrock x1 같은 borderline multicollinearity 포착)
    fi_redundancy_drop_ceil: float = 0.50    # drop_rate < 이 값 (A: 0.40→0.50, 다중공선성 감지 범위 확대)
    fi_redundancy_dampening_factor: float = 0.5  # penalty에 곱할 감면 계수
    # adaptive score gap filter(L2)
    # 선택된 feature 간 점수 갭이 클 때, 갭 아래의 저점수 feature를 제거
    fi_gap_filter_enabled: bool = True
    fi_gap_threshold_very_low_data: float = 0.08
    fi_gap_threshold_normal: float = 0.12
    fi_gap_global_floor: float = 0.79  # global_score가 이 값 미만인 feature만 제거 대상
    # very_low_data에서는 더 보수적으로 tail을 제거하기 위해 별도 floor 사용
    fi_gap_global_floor_very_low_data: float = 0.60
    # p_dim 기반 adaptive gap floor:
    # - 저차원(<=4): dummy 억제를 위해 floor를 더 엄격하게 사용(max)
    # - 고차원(>=8): real 과삭제 방지를 위해 floor를 완화(min)
    fi_gap_global_floor_p_le_4: float = 0.85
    fi_gap_global_floor_p_ge_8: float = 0.75
    fi_gap_min_retain: int = 2         # 제거 후 최소 유지 feature 수
    # very_low_data drop veto: drop_rate가 극도로 낮은 feature를 perm 점수와 무관하게 기각
    fi_drop_veto_enabled: bool = True
    fi_drop_veto_threshold: float = 0.03  # 이 미만이면 veto (very_low_data에서만 동작)
    # very_low_data perm vote variance penalty: fold간 perm vote 분산이 높은 feature에 패널티
    fi_perm_var_penalty_very_low_data_enabled: bool = True
    fi_perm_var_penalty_very_low_data_scale: float = 0.35  # E: 0.20→0.35, very_low_data dummy 분산 패널티 강화

    # -----------------------------
    # bootstrap stability selection
    # -----------------------------
    # N/p ≤ threshold일 때 bootstrap 서브샘플링으로 feature 선택 안정성 검증
    fi_bootstrap_enabled: bool = True
    fi_bootstrap_np_threshold: float = 10.0   # N/p ≤ 이 값일 때 발동
    fi_bootstrap_rounds: int = 20             # 반복 횟수
    fi_bootstrap_sample_ratio: float = 0.8    # 서브샘플 비율
    fi_bootstrap_min_freq: float = 0.73       # 최소 선택 빈도 (미만이면 제거) (E: 0.80→0.73 원복, real variable 과제거 방지)
    fi_bootstrap_min_freq_very_low_data: float = 0.55  # very_low_data 전용 완화 임계값
    # very_low_data 구제: bootstrap_freq < min_freq여도 global_score가 이 값 이상이면 유지
    fi_bootstrap_rescue_global_floor: float = 0.83
    fi_bootstrap_rescue_very_low_data_only: bool = True  # very_low_data에서만 구제 적용

    # -----------------------------
    # 7) FI null(soft) gate
    # -----------------------------
    # null importance 기반 soft penalty 사용 여부
    fi_null_enabled: bool = True
    # null 모드: soft | hard | off(권장: soft)
    fi_null_mode: str = "soft"
    # null 비교 분위수(예: 0.90 = null90)
    fi_null_quantile: float = 0.90  # C: 0.85→0.90, null 기준 상향으로 dummy 억제 강화
    # null 셔플 반복 수(low-data / normal)
    fi_null_shuffle_runs_low_data: int = 50
    fi_null_shuffle_runs_normal: int = 30
    # soft penalty 강도(low-data / normal)
    fi_null_alpha_low_data: float = 0.55  # C: 0.45→0.55, low-data dummy 감점 강화
    fi_null_alpha_normal: float = 0.12
    # null 기준 점수 축: global_score | final_score | both
    # both: global_score에 pre-elite penalty 적용 + 이후 final_score에도 적용
    fi_null_apply_to: str = "both"
    # both 모드 시 pre-elite penalty 비율 (alpha의 이 비율만큼 pre-elite에 적용)
    fi_null_pre_elite_ratio: float = 0.5

    # -----------------------------
    # 8) FI Elite subset 정책
    # -----------------------------
    # elite 기본 비율(목적값 기준 상위 구간)
    fi_elite_ratio_base: float = 0.30
    # elite 최소 샘플 수
    fi_elite_min_samples: int = 30

    # -----------------------------
    # 9) FI Quantile(top-ratio) 정책
    # -----------------------------
    # normal-data 기본 top 비율
    fi_quantile_top_ratio_default: float = 0.50
    # low-data에서 변수 수(p)에 따른 top 비율
    fi_quantile_top_ratio_p_le_6: float = 0.45
    fi_quantile_top_ratio_p_le_12: float = 0.40
    fi_quantile_top_ratio_p_gt_12: float = 0.35

    # -----------------------------
    # 10) 디버그
    # -----------------------------
    # 디버그 산출물/plot 생성 여부: off | on
    debug_level: str = "on"

    # -----------------------------
    # 11) primary / secondary selection
    # -----------------------------
    # primary selection 사용 여부(False이면 FI 전체 스킵, 모든 feature로 모델 생성)
    use_primary_selection: bool = True
    # secondary selection은 user config의 use_secondary_selection으로 제어
    # model1=f(X_primary), model2=f(X_primary+dj)
    # - secondary 후보 dj를 하나씩 추가한 model2와 model1을 같은 CV split에서 비교
    # - delta_r2 = r2(model2) - r2(model1)
    secondary_target_kr: int = 50
    secondary_min_repeats: int = 5
    secondary_min_delta_r2: float = 0.0
    secondary_min_freq: float = 0.7


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
        # fold vote 가중치
        weight_abs=system.fi_weight_abs,
        weight_quantile=system.fi_weight_quantile,
        weight_rank=system.fi_weight_rank,
        # channel merge 가중치
        weight_perm=system.fi_weight_perm,
        weight_drop=system.fi_weight_drop,
        weight_perm_low_data=system.fi_weight_perm_low_data,
        weight_drop_low_data=system.fi_weight_drop_low_data,
        # scale merge 가중치
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
        # decision guard
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
        # drop veto
        drop_veto_enabled=system.fi_drop_veto_enabled,
        drop_veto_threshold=system.fi_drop_veto_threshold,
        # perm 분산 penalty
        perm_var_penalty_very_low_data_enabled=system.fi_perm_var_penalty_very_low_data_enabled,
        perm_var_penalty_very_low_data_scale=system.fi_perm_var_penalty_very_low_data_scale,
        # redundancy 완화
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
        # quantile 정책
        # bootstrap 구제
        bootstrap_min_freq_very_low_data=system.fi_bootstrap_min_freq_very_low_data,
        fi_bootstrap_rescue_global_floor=system.fi_bootstrap_rescue_global_floor,
        fi_bootstrap_rescue_very_low_data_only=system.fi_bootstrap_rescue_very_low_data_only,
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
