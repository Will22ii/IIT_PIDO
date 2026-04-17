from dataclasses import dataclass, field

from CAE_tool_interface.config import CAEConfig


@dataclass
class ExplorerUserConfig:
    # 알려진 최적점 (시각화용, 옵션)
    known_optimum: object | None = None


@dataclass
class ExplorerSystemConfig:
    # 1) 샘플링 크기
    n_samples: int = 10000
    sample_multiplier: float | None = 20.0
    n_samples_min: int = 1000
    n_samples_max: int = 10000

    # 2) 경계 샘플링
    boundary_ratio: float = 0.2
    boundary_corner_ratio: float = 0.5

    # 3) bounds 확장/클리핑
    bounds_margin_ratio: float = 0.03
    bounds_min_volume_ratio: float = 0.249
    bounds_expansion_mode: str = "uniform"
    bounds_weight_clip_min: float = 0.7
    bounds_weight_clip_max: float = 1.5
    # EXP-3: 고제약(crate_hat < threshold) 시 fi_aware → uncertainty_aware 자동 스위치
    bounds_expansion_high_crate_switch_enabled: bool = True
    bounds_expansion_high_crate_threshold: float = 0.85
    # pre-constraint 기반 side-aware 보정 (전략 공통)
    constraint_aware_enabled: bool = True
    # 경계 인접 판정 비율(설계공간 span 대비)
    constraint_aware_edge_ratio: float = 0.05
    # side probe 설정(선택점 기반 제약 샘플링)
    constraint_aware_probe_anchor_max: int = 16
    constraint_aware_probe_steps: int = 3
    constraint_aware_min_side_samples: int = 24
    # side 보호 판정 기준(rate_winner >= min, diff >= gap)
    constraint_aware_side_rate_min: float = 0.55
    constraint_aware_side_rate_gap: float = 0.12
    # 보호 side가 추론되면 cap 이전에 boundary 쪽으로 width-preserving shift
    constraint_aware_pre_shift_enabled: bool = True
    # volume-cap shrink 단계에서 side-aware pin/절단 우선순위 적용
    constraint_aware_use_for_volume_cap: bool = True
    # volume-cap 내부 boundary pin 허용 오차(기존 하드코딩 0.02 대체)
    volume_cap_boundary_pin_tol_ratio: float = 0.02

    # 4) 후보 선택
    quantile_threshold: float = 0.85
    min_topk_count: int = 30
    # DBSCAN 입력 후보 상한 (None이면 동적식 사용)
    max_topk_count: int | None = None
    max_topk_count_dynamic_enabled: bool = True
    max_topk_count_dynamic_scale: float = 60.0
    max_topk_count_dynamic_bias: float = 40.0
    max_topk_count_dynamic_min: int = 120
    max_topk_count_dynamic_max: int = 500

    # 5) 군집화
    dbscan_min_samples: int = 2
    dbscan_eps_quantile: float = 0.9

    # 6) post 제약 fallback
    post_lambda_default: float = 2.0

    # 7) 실행/메타
    save_plot: bool = True
    debug_level: str = "off"
    strategy_id: str = "S4_dual"
    # Dual source_mode 기본 정책 라우터
    dual_policy_mode_default: str = "routed_v2"
    probe_multistart: int = 20
    obj_refine_bounds_scale: float = 1.30
    # Dual tilt disagreement reference (mean disagreement / ref, clipped to [0.8, 1.5])
    # 전략 파라미터에 값이 없으면 이 시스템 기본값을 사용
    dual_tilt_disagree_ref: float = 0.10
    strategy_params: dict[str, object] = field(default_factory=dict)


@dataclass
class ExplorerConfig:
    user: ExplorerUserConfig
    system: ExplorerSystemConfig
    cae: CAEConfig
    cae_metadata_path: str | None = None
    doe_csv_path: str | None = None
    doe_metadata_path: str | None = None
    model_pkl_path: str | None = None
    modeler_metadata_path: str | None = None
    fi_scores_path: str | None = None
