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
    # -----------------------------
    # 1) HPO / CV
    # -----------------------------
    # HPO 상세 설정(옵션, model_name=xgb일 때만 사용)
    # - n_trials: normal 모드 trial 수 (기본 20)
    # - lambda_std: robust objective 가중치 (기본 0.5)
    # - reuse_if_same_config: signature 일치 시 캐시 재사용 여부 (기본 True)
    # - low_data_constrained_enabled: low-data constrained HPO 활성화 (기본 True)
    # - low_data_n_trials: low-data 모드 trial 수 (기본 10)
    # - low_data_lambda_std: low-data 모드 lambda_std (기본 lambda_std와 동일)
    # - search_space: normal 모드 탐색공간 override(dict)
    # - low_data_search_space: low-data 모드 탐색공간 override(dict)
    hpo_config: dict | None = None
    # 기본 CV split/repeat
    kfold_splits: int = 5
    kfold_repeats: int = 2
    # 동적 CV 정책
    cv_dynamic_policy: bool = True
    # fold별 최소 valid 샘플 목표(동적 k 결정 기준)
    cv_min_valid_size: int = 14
    # low-data 기준: usable N / p < ratio
    cv_low_data_np_ratio: float = 15.0

    # -----------------------------
    # 2) FI 기본 채널(perm / drop)
    # -----------------------------
    # permutation importance 평가 샘플 수(각 fold valid에서 최대)
    perm_sample_size: int = 1000
    # permutation 반복 횟수(fold당). 높을수록 importance 추정이 안정적. very_low_data에서 효과적.
    perm_repeats: int = 10
    # perm 채널 통과 기준
    perm_min_pass_rate: float = 0.6
    perm_epsilon: float = 0.05
    # drop 채널 설정(OOF/valid R2 drop 기반)
    fi_use_score_drop: bool = True
    fi_drop_metric: str = "drop_sq"
    fi_drop_min_pass_rate: float = 0.6
    fi_drop_epsilon: float = 0.06
    # very_low_data 전용 drop 채널 임계값 (n_samples < stability_very_low_data_n_threshold일 때)
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
    # low_data 전용 채널 가중치 (low_data=True일 때 fi_weight_perm/drop 대신 사용)
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
    fi_final_score_threshold: float = 0.55
    fi_global_score_floor: float = 0.25

    # -----------------------------
    # 6) FI Stability Gate
    # -----------------------------
    # Stability gate 사용 여부
    fi_stability_enabled: bool = True
    # Stability 결합 규칙 (legacy fallback): or | and
    fi_stability_rule: str = "or"
    # -----------------------------
    # 3단계 stability 분기 (n_samples 기반)
    # very_low_data: n_samples < threshold → or rule, 느슨한 임계값 (six_hump 류)
    # low_data     : low_data=True AND n_samples >= threshold → and rule, 중간 임계값 (cantilever 류)
    # normal_data  : low_data=False → and rule, 엄격한 임계값
    fi_stability_very_low_data_n_threshold: int = 55
    fi_stability_rule_very_low_data: str = "or"
    fi_stability_perm_min_rate_very_low_data: float = 0.65
    fi_stability_drop_min_rate_very_low_data: float = 0.25
    fi_stability_rule_low_data: str = "and"
    fi_stability_perm_min_rate_low_data: float = 0.60
    fi_stability_drop_min_rate_low_data: float = 0.44
    fi_stability_rule_normal: str = "or"
    fi_stability_perm_min_rate_normal: float = 0.80
    fi_stability_drop_min_rate_normal: float = 0.60
    # -----------------------------
    # 채널 불일치 패널티 (disagreement penalty)
    # perm/drop 불일치가 클수록 final_score 감산
    fi_disagreement_penalty_enabled: bool = True
    fi_disagreement_threshold: float = 0.25   # 이 이상 불일치면 패널티 시작
    fi_disagreement_penalty_scale: float = 0.55  # 패널티 강도
    # very_low_data drop veto: drop_rate가 극도로 낮은 feature를 perm 점수와 무관하게 기각
    fi_drop_veto_enabled: bool = True
    fi_drop_veto_threshold: float = 0.03  # 이 미만이면 veto (very_low_data에서만 동작)
    # very_low_data perm vote variance penalty: fold간 perm vote 분산이 높은 feature에 패널티
    fi_perm_var_penalty_very_low_data_enabled: bool = True
    fi_perm_var_penalty_very_low_data_scale: float = 0.20

    # -----------------------------
    # 7) FI Null(soft) Gate
    # -----------------------------
    # null importance 기반 soft penalty 사용 여부
    fi_null_enabled: bool = True
    # null 모드: soft | hard | off (권장: soft)
    fi_null_mode: str = "soft"
    # null 비교 분위수(예: 0.90 = null90)
    fi_null_quantile: float = 0.90
    # null 셔플 반복 수(low-data / normal)
    fi_null_shuffle_runs_low_data: int = 50
    fi_null_shuffle_runs_normal: int = 30
    # soft penalty 강도(low-data / normal)
    fi_null_alpha_low_data: float = 0.40
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
    # 디버그 출력 수준: off | full
    debug_level: str = "full"

    # -----------------------------
    # 11) Secondary Selection
    # -----------------------------
    # model1=f(X_primary), model2=f(X_primary+dj)
    # - secondary 후보 dj를 하나씩 추가한 model2와 model1을 같은 CV split에서 비교
    # - delta_r2 = r2(model2) - r2(model1)
    secondary_target_kr: int = 50
    secondary_min_repeats: int = 5
    secondary_min_delta_r2: float = 0.0
    secondary_min_freq: float = 0.7


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
