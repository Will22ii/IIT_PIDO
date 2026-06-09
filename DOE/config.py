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
    initial_corner_ratio: float = 0.1
    initial_corner_adaptive_enabled: bool = True
    initial_corner_adaptive_np_ratio_max: float = 10.0
    initial_corner_adaptive_low_dim_max: int = 6
    initial_corner_adaptive_ratio_low_dim: float = 0.05

    # 2) Additional DOE: 단계/배분
    additional_init_ratio: float = 0.3
    additional_exec_ratio: float = 0.1
    additional_initial_probe_multiplier: float = 2.0
    # n_exec_local_floor: 최소 서로 다른 anchor에서 각 1개 sampling (다양성 보장)
    local_exec_min_anchors: int = 2
    success_rate_floor: float = 0.02
    global_boundary_ratio: float = 0.05
    global_margin_ratio: float = 0.3
    global_margin_subset_enabled: bool = True
    global_margin_subset_random_k_count: int = 2
    global_top_ratio: float = 0.25
    global_boundary_corner_ratio: float = 0.5
    global_boundary_constraint_boost: float = 12.0
    global_boundary_margin_cross_ratio: float = 0.6
    global_boundary_margin_near_tol: float = 0.03
    global_margin_obj_alpha: float = 0.5

    # 3) Additional DOE: plan 크기
    plan_base_k: float = 200.0
    plan_remaining_cap: float = 4.0
    plan_decay: float = 0.9
    plan_filter_safety: float = 1.2
    plan_filter_r_floor: float = 0.02
    pre_equality_boost_base: float = 2.0
    pre_equality_boost_max: float = 8.0
    pre_equality_warning_threshold: int = 3
    max_additional_stages: int = 10

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
    local_constraint_retry_count: int = 2
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
    # gate1이 threshold 바로 아래인 경우, gate2 strong 신호면 phase2 진입 rescue 허용
    phase2_gate1_rescue_enabled: bool = True
    phase2_gate1_rescue_band: float = 0.03
    phase2_gate1_rescue_gate2_min: float = 0.85
    phase2_gate2_score_min: float = 0.75
    phase2_gate2_score_sticky_min: float = 0.65
    phase2_min_usable_np_ratio: float = 10.0
    phase2_np_ratio_cap_scale: float = 0.74
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
    # budget_policy == "adaptive_early_stop"일 때 추가 라운드/예산 소진 기준
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
    debug_level: str = "on"


def build_additional_cfg_from_system(
    system: DOESystemConfig,
    *,
    initial_corner_ratio: float | None = None,
    initial_corner_ratio_base: float | None = None,
    initial_corner_ratio_policy: str | None = None,
    initial_corner_np_ratio: float | None = None,
    initial_corner_p_dim: int | None = None,
) -> dict:
    cfg = {
        "init_ratio": system.additional_init_ratio,
        "exec_ratio": system.additional_exec_ratio,
        "initial_corner_ratio": (
            float(initial_corner_ratio)
            if initial_corner_ratio is not None
            else float(system.initial_corner_ratio)
        ),
        "initial_probe_multiplier": system.additional_initial_probe_multiplier,
        "local_exec_min_anchors": system.local_exec_min_anchors,
        "success_rate_floor": system.success_rate_floor,
        "global_boundary_ratio": system.global_boundary_ratio,
        "global_boundary_constraint_boost": system.global_boundary_constraint_boost,
        "global_boundary_margin_cross_ratio": system.global_boundary_margin_cross_ratio,
        "global_boundary_margin_near_tol": system.global_boundary_margin_near_tol,
        "global_margin_ratio": system.global_margin_ratio,
        "global_margin_obj_alpha": system.global_margin_obj_alpha,
        "global_margin_subset_enabled": system.global_margin_subset_enabled,
        "global_margin_subset_random_k_count": system.global_margin_subset_random_k_count,
        "global_top_ratio": system.global_top_ratio,
        "global_boundary_corner_ratio": system.global_boundary_corner_ratio,
        "plan_base_k": system.plan_base_k,
        "plan_remaining_cap": system.plan_remaining_cap,
        "plan_decay": system.plan_decay,
        "plan_filter_safety": system.plan_filter_safety,
        "plan_filter_r_floor": system.plan_filter_r_floor,
        "pre_equality_boost_base": system.pre_equality_boost_base,
        "pre_equality_boost_max": system.pre_equality_boost_max,
        "pre_equality_warning_threshold": system.pre_equality_warning_threshold,
        "max_additional_stages": system.max_additional_stages,
        "gate1_ratio": system.gate1_ratio,
        "gate1_pass_ratio": system.gate1_pass_ratio,
        "local_anchor_max_base": system.local_anchor_max_base,
        "local_anchor_max_decay": system.local_anchor_max_decay,
        "local_anchor_best_ratio": system.local_anchor_best_ratio,
        "local_anchor_small_ratio": system.local_anchor_small_ratio,
        "local_radius_ratio_phase1": system.local_radius_ratio_phase1,
        "local_radius_ratio_phase2": system.local_radius_ratio_phase2,
        "local_top_p": system.local_top_p,
        "local_dbscan_min_samples": system.local_dbscan_min_samples,
        "local_dbscan_q_eps": system.local_dbscan_q_eps,
        "local_dbscan_eps_max": system.local_dbscan_eps_max,
        "local_min_radius_ratio": system.local_min_radius_ratio,
        "local_tol_ratio": system.local_tol_ratio,
        "local_refine_min_points": system.local_refine_min_points,
        "local_cluster_delta_ratio": system.local_cluster_delta_ratio,
        "local_singleton_box_ratio": system.local_singleton_box_ratio,
        "local_phase1_kappa": system.local_phase1_kappa,
        "local_phase2_kappa": system.local_phase2_kappa,
        "local_base_perturb_ratio": system.local_base_perturb_ratio,
        "local_gp_use_white_kernel": system.local_gp_use_white_kernel,
        "local_constraint_retry_count": system.local_constraint_retry_count,
        "local_constraint_shrink_factor": system.local_constraint_shrink_factor,
        "local_constraint_min_factor": system.local_constraint_min_factor,
        "local_exec_pick_mode": system.local_exec_pick_mode,
        "post_use_penalty": system.post_use_penalty,
        "post_lambda_init": system.post_lambda_init,
        "post_lambda_min": system.post_lambda_min,
        "post_lambda_max": system.post_lambda_max,
        "post_lambda_power": system.post_lambda_power,
        "post_feasible_rate_floor": system.post_feasible_rate_floor,
        "post_clf_min_samples": system.post_clf_min_samples,
        "post_clf_min_pos": system.post_clf_min_pos,
        "post_clf_min_neg": system.post_clf_min_neg,
        "gate2_k": system.gate2_k,
        "gate2_cdf_level": system.gate2_cdf_level,
        "gate2_ratio_threshold": system.gate2_ratio_threshold,
        "gate2_relax_factor": system.gate2_relax_factor,
        "budget_policy": system.budget_policy,
        "phase1_global_ratio": system.phase1_global_ratio,
        "phase2_global_ratio": system.phase2_global_ratio,
        "phase2_disable_boundary_sampling": system.phase2_disable_boundary_sampling,
        "phase2_gate1_score_min": system.phase2_gate1_score_min,
        "phase2_gate1_rescue_enabled": system.phase2_gate1_rescue_enabled,
        "phase2_gate1_rescue_band": system.phase2_gate1_rescue_band,
        "phase2_gate1_rescue_gate2_min": system.phase2_gate1_rescue_gate2_min,
        "phase2_gate2_score_min": system.phase2_gate2_score_min,
        "phase2_gate2_score_sticky_min": system.phase2_gate2_score_sticky_min,
        "gate_smoothing_enabled": system.gate_smoothing_enabled,
        "gate_ema_alpha": system.gate_ema_alpha,
        "gate_ema_warmup_stages": system.gate_ema_warmup_stages,
        "gate_smoothing_use_for_phase2": system.gate_smoothing_use_for_phase2,
        "gate_smoothing_use_for_stop": system.gate_smoothing_use_for_stop,
        "collapse_span_ratio_threshold": system.collapse_span_ratio_threshold,
        "collapse_anchor_streak_threshold": system.collapse_anchor_streak_threshold,
        "collapse_min_stage": system.collapse_min_stage,
        "diversity_injection_ratio": system.diversity_injection_ratio,
        "diversity_injection_min_points": system.diversity_injection_min_points,
        "diversity_injection_max_ratio": system.diversity_injection_max_ratio,
        "diversity_boundary_floor_ratio": system.diversity_boundary_floor_ratio,
        "min_additional_rounds": system.min_additional_rounds,
        "phase2_min_usable_np_ratio": system.phase2_min_usable_np_ratio,
        "phase2_np_ratio_cap_scale": system.phase2_np_ratio_cap_scale,
        "phase2_min_used_budget_ratio": system.phase2_min_used_budget_ratio,
        "phase2_min_used_budget_ratio_high_crate": system.phase2_min_used_budget_ratio_high_crate,
        "phase2_high_crate_threshold": system.phase2_high_crate_threshold,
        "phase2_g2_saturation_threshold": system.phase2_g2_saturation_threshold,
        "phase2_g2_saturation_min_remaining": system.phase2_g2_saturation_min_remaining,
        "global_bucket_minima_enabled": system.global_bucket_minima_enabled,
        "global_bucket_minima_strict": system.global_bucket_minima_strict,
        "global_boundary_high_crate_threshold": system.global_boundary_high_crate_threshold,
        "phase1_min_top": system.phase1_min_top,
        "phase1_min_margin": system.phase1_min_margin,
        "phase1_min_boundary_any": system.phase1_min_boundary_any,
        "phase1_min_boundary_classic_hc": system.phase1_min_boundary_classic_hc,
        "phase1_min_boundary_cross_hc": system.phase1_min_boundary_cross_hc,
        "phase2_min_top": system.phase2_min_top,
        "phase2_min_margin": system.phase2_min_margin,
        "global_bucket_plan_multiplier": system.global_bucket_plan_multiplier,
        "early_stop_min_used_budget_ratio": system.early_stop_min_used_budget_ratio,
        "early_stop_min_usable_np_ratio": system.early_stop_min_usable_np_ratio,
        "stop_span_ratio_threshold": system.stop_span_ratio_threshold,
        "stop_anchor_spread_streak": system.stop_anchor_spread_streak,
        "stop_min_usable_np_ratio": system.stop_min_usable_np_ratio,
        "probe_stage_enabled": system.probe_stage_enabled,
        "probe_top_ratio": system.probe_top_ratio,
        "probe_max_points": system.probe_max_points,
        "probe_min_range_ratio": system.probe_min_range_ratio,
        "probe_std_scale": system.probe_std_scale,
        "probe_perturb_ratio": system.probe_perturb_ratio,
    }
    if initial_corner_ratio_base is not None:
        cfg["initial_corner_ratio_base"] = float(initial_corner_ratio_base)
    if initial_corner_ratio_policy is not None:
        cfg["initial_corner_ratio_policy"] = str(initial_corner_ratio_policy)
    if initial_corner_np_ratio is not None:
        cfg["initial_corner_np_ratio"] = float(initial_corner_np_ratio)
    if initial_corner_p_dim is not None:
        cfg["initial_corner_p_dim"] = int(initial_corner_p_dim)
    return cfg


@dataclass
class DOEConfig:
    cae: CAEConfig
    cae_user: CAEUserConfig | None
    user: DOEUserConfig
    system: DOESystemConfig
    cae_output: dict | None = None
    cae_metadata_path: str | None = None
