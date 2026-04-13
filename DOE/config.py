from dataclasses import dataclass

from CAE_tool_interface.config import CAEConfig, CAEUserConfig


@dataclass
class DOEUserConfig:
    # DOE 알고리즘 이름 (예: "lhs")
    algo_name: str = "lhs"
    # 추가 DOE 사용 여부
    use_additional: bool = False


@dataclass
class DOESystemConfig:
    # 1) DOE 기본
    n_samples: int = 120
    force_baseline_initial: bool = False
    initial_corner_ratio: float = 0.03
    initial_corner_adaptive_enabled: bool = False
    initial_corner_adaptive_np_ratio_max: float = 10.0
    initial_corner_adaptive_low_dim_max: int = 6
    initial_corner_adaptive_ratio_low_dim: float = 0.05

    # 2) Additional DOE: 단계/배분
    additional_init_ratio: float = 0.4
    additional_exec_ratio: float = 0.1
    dynamic_exec_allocation_enabled: bool = True
    additional_exec_round_shares: tuple[float, ...] = (0.24, 0.19, 0.17, 0.15, 0.13, 0.12)
    exec_eff_weight_gate1: float = 0.45
    exec_eff_weight_gate2: float = 0.55
    exec_eff_gain_k: float = 0.7
    exec_eff_clip_min: float = 0.85
    exec_eff_clip_max: float = 1.20
    exec_stage_min_ratio: float = 0.08
    exec_stage_max_ratio: float = 0.40
    additional_initial_probe_multiplier: float = 2.0
    success_rate_floor: float = 0.02
    global_boundary_ratio: float = 0.08
    global_margin_ratio: float = 0.3
    global_margin_subset_enabled: bool = True
    global_margin_subset_random_k_count: int = 2
    global_top_ratio: float = 0.25
    global_boundary_corner_ratio: float = 0.5
    global_boundary_constraint_boost: float = 1.2
    global_boundary_margin_cross_ratio: float = 0.6
    global_boundary_margin_near_tol: float = 0.03
    global_margin_obj_alpha: float = 0.75

    # 3) Additional DOE: plan 크기
    plan_base_k: float = 200.0
    plan_remaining_cap: float = 4.0
    plan_decay: float = 0.9
    plan_filter_safety: float = 1.2
    plan_filter_r_floor: float = 0.02
    max_additional_stages: int = 12

    # 4) Gate
    gate1_ratio: float = 0.3
    gate1_pass_ratio: float = 0.6
    gate2_k: int = 2
    gate2_cdf_level: float = 0.9
    gate2_ratio_threshold: float = 0.9
    gate2_relax_factor: float = 1.1

    # 5) Local planner: 앵커/클러스터
    local_anchor_max_base: int = 6
    local_anchor_max_decay: float = 0.9
    local_anchor_best_ratio: float = 0.35
    local_anchor_small_ratio: float = 0.2
    local_top_p: float = 0.3
    local_dbscan_min_samples: int = 2
    local_dbscan_q_eps: float = 0.65
    local_dbscan_eps_max: float = 0.25

    # 6) Local planner: 반경/중복 제어
    local_radius_ratio_phase1: float = 0.25
    local_radius_ratio_phase2: float = 0.15
    local_min_radius_ratio: float = 0.08
    local_tol_ratio: float = 0.3

    # 7) Local planner: GP refine
    local_refine_min_points: int = 15
    local_cluster_delta_ratio: float = 0.03
    local_singleton_box_ratio: float = 0.05
    local_phase1_kappa: float = 0.8
    local_phase2_kappa: float = 0.6
    local_base_perturb_ratio: float = 0.05
    local_gp_use_white_kernel: bool = False

    # 8) Local planner: 제약 재시도
    local_constraint_retry_count: int = 1
    local_constraint_shrink_factor: float = 0.75
    local_constraint_min_factor: float = 2.0
    local_exec_pick_mode: str = "random"

    # 9) Post 제약 penalty
    post_use_penalty: bool = True
    post_lambda_init: float = 2.0
    post_lambda_min: float = 0.25
    post_lambda_max: float = 8.0
    post_lambda_power: float = 1.0
    post_feasible_rate_floor: float = 0.05
    post_clf_min_samples: int = 30
    post_clf_min_pos: int = 5
    post_clf_min_neg: int = 5

    # 10) Phase/예산
    budget_policy: str = "consume_all"
    phase1_global_ratio: float = 0.8
    phase2_global_ratio: float = 0.30
    # True면 phase=2에서 boundary 샘플 할당을 0으로 강제
    phase2_disable_boundary_sampling: bool = True
    phase2_gate1_score_min: float = 0.55
    phase2_gate2_score_min: float = 0.75
    phase2_gate2_score_sticky_min: float = 0.65
    phase2_min_usable_np_ratio: float = 10.0
    phase2_np_ratio_cap_scale: float = 0.75
    phase2_min_used_budget_ratio: float = 0.30
    phase2_min_used_budget_ratio_high_crate: float = 0.50
    phase2_high_crate_threshold: float = 0.85
    # DOE-2: gate2 saturation 가드
    # phase2 진입 후 gate2_ema가 포화(0.92+) + 고제약 + 예산 잔여 충분 시
    # 다음 stage에 diversity_boost 강제 (위양성 수렴 보정)
    phase2_g2_saturation_threshold: float = 0.92
    phase2_g2_saturation_min_remaining: float = 0.20
    # ─────────────────────────────────────────────────────
    # Bucket minima 정책 (DOE-3 대체)
    # ─────────────────────────────────────────────────────
    # phase별/제약상태별 각 bucket의 최소 개수를 보장. n_exec_floor 계산 근거.
    # remaining 부족 시 drop priority (top > margin > boundary > random) 순으로 graceful degrade
    global_bucket_minima_enabled: bool = True
    global_bucket_minima_strict: bool = False   # True면 minima 미달 시 FAILED (기본 relax)
    # high_crate 판정 (bucket minima 전용, DSE-C의 phase2 진입 threshold와 별개)
    global_boundary_high_crate_threshold: float = 0.80   # crate_hat < 0.80 → high_crate
    # phase1 minima
    phase1_min_top: int = 1
    phase1_min_margin: int = 1              # has_pre_constraints일 때만 활성
    phase1_min_boundary_any: int = 1        # constraint + normal(crate>=0.80) 또는 no_constraint
    phase1_min_boundary_classic_hc: int = 1 # high_crate 전용
    phase1_min_boundary_cross_hc: int = 1   # high_crate 전용
    # phase2 minima (boundary는 phase2_disable_boundary_sampling=True로 이미 off)
    phase2_min_top: int = 1
    phase2_min_margin: int = 1              # has_pre_constraints일 때만 활성
    # planning pool headroom (filter/dedup 손실 감안)
    global_bucket_plan_multiplier: int = 3
    gate_smoothing_enabled: bool = True
    gate_ema_alpha: float = 0.35
    gate_ema_warmup_stages: int = 2
    gate_smoothing_use_for_phase2: bool = True
    gate_smoothing_use_for_stop: bool = True
    collapse_span_ratio_threshold: float = 0.22
    collapse_anchor_streak_threshold: int = 2
    collapse_min_stage: int = 2
    diversity_injection_ratio: float = 0.20
    diversity_injection_min_points: int = 2
    diversity_injection_max_ratio: float = 0.40
    diversity_boundary_floor_ratio: float = 0.12
    min_additional_rounds: int = 6
    early_stop_min_used_budget_ratio: float = 0.50
    early_stop_min_usable_np_ratio: float = 20.0
    stop_span_ratio_threshold: float = 0.20
    stop_anchor_spread_streak: int = 5
    stop_min_usable_np_ratio: float = 20.0

    # 10-1) Probe stage
    probe_stage_enabled: bool = True
    probe_top_ratio: float = 0.3
    probe_max_points: int = 6
    probe_min_range_ratio: float = 0.3
    probe_std_scale: float = 2.0
    probe_perturb_ratio: float = 0.02

    # 11) 기타
    additional_cfg: dict | None = None
    debug_level: str = "off"


@dataclass
class DOEConfig:
    cae: CAEConfig
    cae_user: CAEUserConfig | None
    user: DOEUserConfig
    system: DOESystemConfig
    cae_output: dict | None = None
    cae_metadata_path: str | None = None
