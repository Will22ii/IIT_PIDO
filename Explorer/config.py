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
    bounds_min_volume_ratio: float = 0.2499
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
    # L2: shift 이동 비율(1.0=전체 width 이동, 0.0=shift 비활성). 기본 0.5로
    # 부분 이동하여 비보호측 optimum coverage 리스크를 줄인다.
    constraint_aware_shift_fraction: float = 0.5
    # L3: near_{opposite}_hits 이 anti_threshold 이상이면 protect 플래그 자체를
    # 해제해 shift 전면 무효화(기존 D-B1 prefer-shrink-side 무효화만이었음)
    constraint_aware_anti_anti_threshold: int = 1
    # L5: top-obj anchor 기반 shift veto — feasibility 기반 shift 방향이 top-K
    # objective anchor 신호와 모순하거나 해당 dim에 top 신호가 전혀 없으면
    # protect 플래그 자체를 해제해 shift 무효화. False 로 두면 기존 동작.
    constraint_aware_require_top_obj_support: bool = True
    # top anchor 집합 대비 최소 near-hit 비율 (그 미만이면 신호 부재로 간주)
    constraint_aware_top_obj_min_ratio: float = 0.10
    # L7-A: veto가 발동하고 top-obj anchor가 interior(어떤 edge에도 모이지 않음)인
    # 경우, 기본 selected_bounds 중심을 top-obj centroid 쪽으로 부분 blend.
    # 0.0=비활성, 1.0=완전 recenter. width는 보존된다.
    constraint_aware_obj_centroid_blend: float = 0.40
    # L7-B: 모든 protect dim이 top-obj 다수결로 confirmed이고 veto가 0개일 때
    # 사용할 boost된 shift_fraction. 0.0~1.0. 0.5 미만이면 보수 분기 값 유지.
    constraint_aware_obj_confirmed_shift_fraction: float = 0.70
    # L4: 제약·저차원·저 feasibility 구조에서만 L2/L3 보수 분기를 강제
    constraint_aware_conservative_p_dim_max: int = 5
    constraint_aware_conservative_np_ratio_max: float = 25.0
    constraint_aware_conservative_feasible_ratio_max: float = 0.3
    # 보수 분기 시 shift_fraction (0.0=완전 비활성, 0.3=부분 유지)
    constraint_aware_conservative_shift_fraction: float = 0.3
    # volume-cap shrink 단계에서 side-aware pin/절단 우선순위 적용
    constraint_aware_use_for_volume_cap: bool = True
    # volume-cap 내부 boundary pin 허용 오차(기존 하드코딩 0.02 대체)
    volume_cap_boundary_pin_tol_ratio: float = 0.03

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
