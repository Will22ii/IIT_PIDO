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
    # HPO 상세 설정(옵션)
    hpo_config: dict | None = None
    # 기본 K-fold 수
    kfold_splits: int = 5
    # 기본 반복 수(동적 CV 사용 시 내부 재계산 가능)
    kfold_repeats: int = 2
    # 동적 CV 정책 사용 여부
    cv_dynamic_policy: bool = True
    # fold별 최소 valid 샘플 목표
    cv_min_valid_size: int = 14
    # low-data 기준: usable N / p < ratio
    cv_low_data_np_ratio: float = 15.0

    # -----------------------------
    # 2) 기본 FI 채널(perm/drop)
    # -----------------------------
    # permutation 중요도 평가 샘플 수(각 fold valid에서 최대 샘플 수)
    perm_sample_size: int = 1000
    # perm 채널 최소 통과율
    perm_min_pass_rate: float = 0.6
    # perm abs 임계값(정규화 점수 기준)
    perm_epsilon: float = 0.05
    # drop 채널 사용 여부 (OOF/valid R2 drop 기반)
    fi_use_score_drop: bool = True
    # drop 채널 metric: drop | drop_sq
    fi_drop_metric: str = "drop_sq"
    # drop 채널 최소 통과율
    fi_drop_min_pass_rate: float = 0.6
    # drop abs 임계값
    fi_drop_epsilon: float = 0.06

    # -----------------------------
    # 3) 가중 투표(채널 내부)
    # -----------------------------
    # fold vote 가중치 (합 1 권장)
    fi_weight_abs: float = 0.75
    fi_weight_quantile: float = 0.15
    fi_weight_rank: float = 0.10
    # 채널 결합 가중치 (perm/drop, 합 1 권장)
    fi_weight_perm: float = 0.7
    fi_weight_drop: float = 0.3

    # -----------------------------
    # 4) 스케일 결합(global/elite)
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
    # elite bonus 모드 계수(beta), [0, 1]로 내부 clip
    fi_elite_bonus_beta: float = 0.30

    # -----------------------------
    # 5) 최종 선택 가드
    # -----------------------------
    # 최종 점수 컷
    fi_final_score_threshold: float = 0.61
    # 전역 점수 하한
    fi_global_score_floor: float = 0.25

    # -----------------------------
    # 6) Elite subset 정책
    # -----------------------------
    # elite 기본 비율(목적값 기준 상위 구간)
    fi_elite_ratio_base: float = 0.30
    # elite 최소 샘플 수
    fi_elite_min_samples: int = 30

    # -----------------------------
    # 7) Low-data quantile 완화
    # -----------------------------
    # low-data가 아닐 때 기본 top 비율
    fi_quantile_top_ratio_default: float = 0.50
    # low-data에서 변수 수(p)에 따른 top 비율
    fi_quantile_top_ratio_p_le_6: float = 0.60
    fi_quantile_top_ratio_p_le_12: float = 0.50
    fi_quantile_top_ratio_p_gt_12: float = 0.40

    # -----------------------------
    # 8) 디버그
    # -----------------------------
    # 디버그 출력 수준: off | full
    debug_level: str = "full"

    # -----------------------------
    # 9) Secondary Selection
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
