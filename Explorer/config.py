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
    # GP refinement/uncertainty X scaling. 좌표 정규화는 GP fit/predict 내부에서만 적용하고,
    # Explorer bounds와 CAE 좌표는 raw 값을 유지한다.
    gp_x_scaling_enabled: bool = True
    gp_x_scaling_mode: str = "bounds"
    gp_x_scaling_auto_span_ratio_threshold: float = 3.0
    gp_x_scaling_span_floor: float = 1e-12
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
    # L5: top-objective anchor 기반 shift veto. feasibility 기반 shift 방향이 top-K
    # objective anchor 신호와 모순하거나 해당 dim에 top 신호가 전혀 없으면
    # protect 플래그 자체를 해제해 shift를 무효화한다. False로 두면 기존 동작.
    constraint_aware_require_top_obj_support: bool = True
    # top anchor 집합 대비 최소 near-hit 비율(그 미만이면 신호 부재로 간주)
    constraint_aware_top_obj_min_ratio: float = 0.10
    # L7-A: veto가 발동했고 실제 shift가 일어나지 않은 경우, selected_bounds
    # 중심을 top-objective centroid 쪽으로 부분 blend한다. 0.0=비활성, 1.0=완전 recenter.
    # width는 보존된다. confirmed protect dim과의 상충을 줄이기 위해 기본값은 0.30이다.
    constraint_aware_obj_centroid_blend: float = 0.30
    # L7-B: 모든 protect dim이 top-obj 다수결로 confirmed이고 veto가 0개일 때
    # 사용할 boost된 shift_fraction. 0.0~1.0. 0.5 미만이면 보수 분기 값을 유지한다.
    constraint_aware_obj_confirmed_shift_fraction: float = 0.70
    # #D L7-B 단계화: confirmed_count 가 ceil(d/2) 이상이면 강한 boost.
    # 약한 신호(>=1)는 0.70, 강한 신호(>=ceil(d/2))는 0.85.
    constraint_aware_obj_strong_confirmed_shift_fraction: float = 0.85
    # #L3: 전 protect dim이 confirmed인 경우 full shift (all-confirmed → 1.0)
    constraint_aware_obj_all_confirmed_shift_fraction: float = 1.0
    # #L9: DSE-L6 dispersion-aware 확장 — 두 cluster가 모두 dispersed이고
    # 거의 disjoint인 경우 dual blend가 가짜 중심 잡음. iou ≤ 0.05 (L6 strict)
    # 대신 다음 조건을 모두 만족해도 pure objective path를 적용한다:
    #   has_pre_constraints AND p_dim >= 4
    #   AND dual_disagreement_iou < l9_iou_max
    #   AND obj_bounds_widths_min > l9_obj_min_threshold
    #   AND pred_cluster_selected_count >= l9_pred_n_sel_min
    # state-only 정책이며 benchmark 이름/seed 분기가 아니다.
    dual_l9_extended_iou_max: float = 0.10
    dual_l9_extended_obj_min_threshold: float = 0.50
    dual_l9_extended_pred_n_sel_min: int = 20
    dual_l9_extended_p_dim_min: int = 4
    # #K: dim-aware objective floor. has_pre==False AND p_dim <= max_dim 인 경우
    # dual의 obj_ratio 하한값. multimodal low-dim에서 objective 측 우월성을 확보한다.
    dual_obj_floor_low_dim_max: int = 3
    dual_obj_floor_low_dim_value: float = 0.90
    # #L2: pre-cap micro-expansion — has_pre==True AND p_dim>=min_dim 일 때
    # cap 직전 selected_bounds 부피비가 threshold 미만이면 target_pre_cap_ratio 까지
    # axis-aligned 균일 확장. 이후 cap이 0.25로 축소.
    bounds_pre_cap_expand_enabled: bool = True
    bounds_pre_cap_expand_p_dim_min: int = 4
    bounds_pre_cap_expand_threshold: float = 0.10
    bounds_pre_cap_expand_target_ratio: float = 0.30
    # #M: feasibility-aware floor expansion direction — has_pre==True AND p_dim>=min_dim
    # AND side_scores 존재 시, |rate_ub - rate_lb| >= rate_diff_min 인 dim의
    # apply_bounds_margin center_hint를 feasible side로 override한다. floor expansion이
    # feasibility 방향으로 우선 확장되어 optimum corner 누락을 막는다.
    # state-based gate이며 benchmark 이름 분기가 아니다. (모든 dim의 UB가 saturated인
    # CB류 패턴에서 b1_ub 절단 회복 목적)
    feasibility_floor_hint_enabled: bool = True
    feasibility_floor_hint_p_dim_min: int = 4
    feasibility_floor_hint_rate_diff_min: float = 0.5
    feasibility_floor_hint_bias: float = 0.90
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
    # cap 이후 selected/top point가 현재 bounds 밖 한쪽 side에 남아 있으면,
    # 부피를 유지한 채 해당 side로 width-preserving shift한다.
    post_cap_side_mass_shift_enabled: bool = True
    post_cap_side_mass_quantile: float = 0.98
    post_cap_side_mass_trigger_ratio: float = 0.0
    post_cap_side_mass_pad_ratio: float = 0.001
    # 제약 저차원 문제에서 cap/shift 이후 보호해야 할 설계 경계가 다시 잘리면,
    # selected/top anchor의 boundary-touch 신호를 근거로 width를 유지한 채 edge로 복원한다.
    constrained_boundary_preserve_enabled: bool = True
    constrained_boundary_preserve_max_dim: int = 4
    constrained_boundary_preserve_touch_eps_ratio: float = 0.10
    constrained_boundary_preserve_shift_fraction: float = 1.0
    constrained_boundary_preserve_require_policy_or_touch: bool = True
    # 저차원·무제약에서 cap 이후 bounds가 한쪽으로 과하게 치우치면,
    # 부피와 width는 유지하고 설계공간 중심 쪽으로 recenter한다.
    # SH처럼 multimodal 저차원 문제에서 과도한 one-sided shrink를 줄이기 위한
    # state-only guard이며 benchmark 이름/정답 좌표를 쓰지 않는다.
    low_dim_balance_guard_enabled: bool = True
    low_dim_balance_guard_max_dim: int = 3
    low_dim_balance_guard_center_l1_threshold: float = 0.20
    low_dim_balance_guard_min_pre_cap_ratio: float = 0.27
    low_dim_balance_guard_min_improvement: float = 0.03
    low_dim_balance_guard_recenter_blend: float = 1.0
    # cap shrink가 명시 적용되지 않았더라도 최종 bounds가 이미 cap 근처이고
    # 중심 편향이 크면 같은 guard를 적용한다. apply_bounds_margin의 min-volume
    # floor만으로 24.99% 근처가 된 over-shrink 케이스를 회복하기 위한 조건이다.
    low_dim_balance_guard_near_cap_enabled: bool = True
    low_dim_balance_guard_near_cap_min_ratio: float = 0.98
    low_dim_balance_guard_near_cap_max_ratio: float = 1.01
    # recenter 후보가 기존 selected/top anchor를 과하게 버리면 적용하지 않는다.
    # Goldstein류 off-center optimum에서 center recenter가 좋은 anchor를 자르는 것을 막는다.
    low_dim_balance_guard_anchor_veto_enabled: bool = True
    low_dim_balance_guard_anchor_min_retained_ratio: float = 0.80
    low_dim_balance_guard_anchor_min_count: int = 4

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
    debug_level: str = "on"
    # 시스템 기본 strategy. standalone-safe하며 Modeler 미동반 환경에서도 동작한다.
    # run_pipelines 등 AION 모드(Modeler task 동반) entry point에서는 호출자가
    # S4_dual로 명시 override한다. (예: pipeline/run_pipelines.py argparse default)
    strategy_id: str = "S4_obj"
    # dual source_mode 기본 정책 라우터
    dual_policy_mode_default: str = "routed_v2"
    probe_multistart: int = 20
    obj_refine_bounds_scale: float = 1.30
    # dual tilt disagreement 기준값(mean disagreement / ref, [0.8, 1.5]로 clip)
    # 전략 파라미터에 값이 없으면 이 시스템 기본값을 사용
    dual_tilt_disagree_ref: float = 0.10
    # AION mode 선택 router signal. standalone/direct pipeline 실행이 DOE metadata
    # diagnostic에 의존하지 않도록 기본값은 off다.
    enable_doe_router_signals: bool = False
    strategy_params: dict[str, object] = field(default_factory=dict)


@dataclass
class ExplorerConfig:
    user: ExplorerUserConfig
    system: ExplorerSystemConfig
    cae: CAEConfig
    cae_metadata_path: str | None = None
    doe_csv_path: str | None = None
    selected_features_csv_path: str | None = None
    model_pkl_path: str | None = None
    fi_scores_path: str | None = None
