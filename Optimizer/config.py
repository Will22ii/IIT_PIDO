from dataclasses import asdict, dataclass, field
from typing import Any

from CAE_tool_interface.config import CAEConfig


@dataclass
class OptimizerUserConfig:
    # Optimizer 반복 횟수(=새 점 탐색 횟수)
    n_samples: int = 30
    # Standalone 입력: DOE 결과 CSV 경로 (옵션)
    doe_csv_path: str | None = None
    # Standalone 입력: Explorer selected bounds JSON 경로 (옵션)
    explorer_bounds_path: str | None = None
    # Debug/benchmark용 known optimum. 있으면 pair plot에 marker로 표시한다.
    known_optimum: Any | None = None
    # Optional target objective. If provided, any Optimizer algorithm can stop
    # early after reaching this quality target and observing no further best
    # improvement. None means "use full budget".
    goal: float | None = None
    # Backward-compatible alias. Prefer `goal`.
    goal_objective: float | None = None


@dataclass(frozen=True)
class OptimizerSystemConfigView:
    """Non-mutating view of Optimizer system config ownership boundaries."""

    schema: str
    algorithm_id: str
    algorithm_contract: str
    common: dict[str, object]
    algorithm: dict[str, object]
    focus_bo: dict[str, object]
    compatibility: dict[str, object]


@dataclass
class OptimizerSystemConfig:
    """Optimizer internal/system tuning.

    Service-facing knobs should stay small: algorithm_id, focus_pipeline,
    debug_level, and user.n_samples/direct input paths. Most fields below are
    algorithm tuning values and should be changed through presets, not ordinary
    UI controls.
    """

    # ------------------------------------------------------------------
    # Compatibility/default BO controls
    # ------------------------------------------------------------------
    # Removal candidates:
    # - acq_type/kappa_start/kappa_end are still used by the generic acquisition
    #   scaffold and Focus3 fallback/recovery. Remove only after Focus3 owns all
    #   acquisition scheduling through focus3_* fields.
    # - starts_per_iter/random_starts_ratio are legacy default-BO start controls.
    #   Focus3 primarily uses plan-pool/refine starts, but fallback paths still
    #   read these values.
    # Default acquisition 정책: auto | LCB | EI. Focus3는 focus3_acq_type을 우선 사용한다.
    acq_type: str = "auto"
    # LCB kappa 스케줄 (legacy/default path 및 Focus3 recovery base에서 사용)
    kappa_start: float = 2.5
    kappa_end: float = 0.6
    # EI 파라미터
    ei_xi: float = 0.01
    # acquisition 다중 시작점 최적화 파라미터
    starts_per_iter: int = 32
    random_starts_ratio: float = 0.7
    # Focus3 source mixture 사용 여부. False면 legacy source_topk/boundary/random 기본값을 사용한다.
    source_mixture_enabled: bool = True

    # ------------------------------------------------------------------
    # Service-facing algorithm selection
    # ------------------------------------------------------------------
    # focus_bo is the built-in default algorithm. Other algorithms can be
    # registered as long as they consume the common Optimizer inputs and return
    # the common Optimizer result contract.
    algorithm_id: str = "focus_bo"
    algorithm_params: dict[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Focus BO orchestration
    # ------------------------------------------------------------------
    # auto | focus0,focus1,focus3 | list/tuple. 어떤 focus를 실행할지는 focus_pipeline이 결정한다.
    focus_pipeline: str | list[str] | tuple[str, ...] = "auto"
    # auto planner profile:
    # - standalone: user-provided/standalone evidence is treated conservatively;
    #   external bounds may still be validated/refined by Focus2.
    # - aion: AION Explorer bounds are trusted; Focus3 receives most budget
    #   directly unless the objective archive is too sparse.
    focus_planner_profile: str = "standalone"
    # Focus budget은 고정 fraction보다 archive/bounds/budget 상태를 보고 동적으로 배분한다.
    focus_budget_adaptive_enabled: bool = True
    focus_budget_selected_bounds_focus3_fraction: float = 0.70
    focus_budget_generated_bounds_focus3_fraction: float = 0.50
    focus_budget_low_np_focus3_fraction: float = 0.35
    # Used only when focus2 is explicitly included with external selected bounds.
    # auto trusts external bounds and goes directly to Focus3 after any needed archive fill.
    focus_budget_focus2_fraction_selected_bounds: float = 0.15
    focus_budget_focus2_fraction_generated_bounds: float = 0.25
    focus_budget_focus1_fraction_low_archive: float = 0.45
    focus_budget_focus1_fraction_normal: float = 0.30

    # ------------------------------------------------------------------
    # Focus0/Focus1: objective archive bootstrap/classification
    # ------------------------------------------------------------------
    # Focus0/1: active bounds 안에서 BO 시작 전 objective archive를 보강한다.
    focus0_enabled: bool = True
    focus0_min_np_ratio: float = 1.0
    focus0_min_points: int = 3
    focus1_enabled: bool = True
    focus1_target_np_ratio: float = 10.0
    # AION trusted-bounds mode only. If Explorer already produced bounds and
    # objective archive density is at least this ratio, auto planner skips
    # Focus1/2 and sends the remaining optimizer budget to Focus3.
    focus_aion_focus1_target_np_ratio: float = 5.0
    focus1_max_budget_fraction: float = 0.50
    focus1_min_gp_points: int = 3
    focus1_batch_size: int = 10
    focus1_candidate_pool_min: int = 2048
    focus1_candidate_pool_per_dim: int = 100
    focus1_candidate_pool_max: int = 20000
    focus1_beta: float = 1.0
    focus1_tau_quantile: float = 0.20
    focus1_good_ratio: float = 0.20
    focus1_uncertain_ratio: float = 0.60
    focus1_bad_ratio: float = 0.20
    focus1_min_rms_distance: float = 0.03
    focus1_early_stop_enabled: bool = True
    focus1_early_stop_uncertain_ratio: float = 0.25
    focus1_early_stop_min_good_candidates: int = 10
    # Focus1 조기 종료는 분류가 안정되어 보여도 최소 archive 밀도를 만족해야 허용한다.
    # p=5이면 기본 50개 archive 전에는 Focus2/3로 넘기지 않는다.
    focus1_early_stop_min_data_ratio: float = 10.0
    focus0_initial_corner_ratio: float = 0.10
    focus0_initial_corner_adaptive_enabled: bool = True
    focus0_initial_corner_adaptive_np_ratio_max: float = 10.0
    focus0_initial_corner_adaptive_low_dim_max: int = 6
    focus0_initial_corner_adaptive_ratio_low_dim: float = 0.05
    focus0_margin_ratio: float = 0.30
    focus0_margin_subset_enabled: bool = True
    focus0_margin_subset_random_k_count: int = 2
    focus0_probe_multiplier: float = 2.0
    focus0_filter_safety: float = 1.2
    focus0_filter_r_floor: float = 0.02

    # Optional objective-goal early stop. Active only when user.goal or
    # user.goal_objective is provided. This is useful for benchmark/production
    # runs with an explicit target quality; normal runs without a target keep
    # using the full budget.
    goal_early_stop_enabled: bool = True
    goal_early_stop_patience: int = 5
    # Built-in focus_bo only: record goal hits in earlier focus stages, but allow
    # early stop only from this focus level onward. User algorithms can use
    # GoalMonitor directly and keep the default "anytime" behavior.
    goal_early_stop_min_focus: int = 2

    # GP X scaling 정책.
    # - auto: active bounds span 차이가 충분히 클 때만 [0, 1] 좌표계에서 학습/예측한다.
    # - bounds: 항상 bounds scaling을 적용한다.
    # - off/raw: raw 설계변수 좌표계 그대로 GP를 fit한다.
    # CAE 평가, pre-constraint, output/debug 값은 계속 raw 설계변수로 유지한다.
    gp_x_scaling_enabled: bool = True
    gp_x_scaling_mode: str = "auto"
    gp_x_scaling_auto_span_ratio_threshold: float = 3.0
    gp_x_scaling_span_floor: float = 1e-12

    # ------------------------------------------------------------------
    # Focus3: final selected-bounds BO
    # ------------------------------------------------------------------
    # Audit label only. Runtime behavior is controlled by the focus3_* knobs.
    focus3_profile_id: str = "default"
    # Legacy source mixture 기본 확률 (topk / boundary / random).
    # Removal candidate: focus3_budget_policy_enabled=True가 고정되면
    # focus3_*_source_* 확률만 남기고 이 fallback trio는 제거한다.
    # focus3_budget_policy_enabled=False일 때 fallback으로 사용한다.
    source_topk_prob: float = 0.60
    source_boundary_prob: float = 0.25
    source_random_prob: float = 0.15
    # focus3 budget class별 source mixture. n_opt / p 기준.
    focus3_budget_policy_enabled: bool = True
    focus3_budget_ultra_low_np: float = 3.0
    focus3_budget_low_np: float = 8.0
    focus3_budget_normal_np: float = 20.0
    focus3_min_budget_np_ratio: float = 2.0
    focus3_min_budget: int = 5
    focus3_ultra_low_source_topk_prob: float = 0.85
    focus3_ultra_low_source_boundary_prob: float = 0.10
    focus3_ultra_low_source_random_prob: float = 0.05
    focus3_low_source_topk_prob: float = 0.75
    focus3_low_source_boundary_prob: float = 0.15
    focus3_low_source_random_prob: float = 0.10
    focus3_normal_source_topk_prob: float = 0.60
    focus3_normal_source_boundary_prob: float = 0.25
    focus3_normal_source_random_prob: float = 0.15
    focus3_rich_source_topk_prob: float = 0.45
    focus3_rich_source_boundary_prob: float = 0.35
    focus3_rich_source_random_prob: float = 0.20
    # Focus3 plan/refine: 큰 source별 X_plan을 GP로 scoring한 뒤 일부만 L-BFGS-B refine.
    focus3_plan_pool_min_per_source: int = 1024
    focus3_plan_pool_per_dim: int = 80
    focus3_plan_pool_max_per_source: int = 4096
    # Legacy hard override. Prefer min/per_dim/max policy above.
    # Removal candidate once old configs no longer set this field.
    focus3_plan_pool_per_source: int | None = None
    focus3_refine_starts: int = 30
    # Focus3는 매번 multi-start L-BFGS-B를 돌리지 않는다. 기본은
    # 2번 discrete GP scoring 후 1번 refine.
    focus3_discrete_enabled: bool = True
    focus3_refine_every: int = 2
    # 최근 discrete/refine 성과를 보고 expensive L-BFGS-B refine 주기를 조절한다.
    focus3_refine_adaptive_enabled: bool = True
    focus3_refine_adaptive_window: int = 80
    focus3_refine_adaptive_min_samples: int = 6
    focus3_refine_adaptive_min_every: int = 1
    focus3_refine_adaptive_max_every: int = 4
    focus3_refine_adaptive_better_every: int = 1
    focus3_refine_adaptive_worse_every: int = 4
    focus3_refine_adaptive_advantage_ratio: float = 1.50
    focus3_refine_adaptive_disadvantage_ratio: float = 0.50
    # Refine 후보가 boundary/random 중심이면 비싼 L-BFGS-B를 생략하고 GP scoring 결과를
    # 그대로 사용한다. RB/GP처럼 budget이 큰 무제약 run에서 runtime을 크게 줄이기 위한 장치다.
    focus3_refine_source_filter_enabled: bool = True
    focus3_refine_allowed_sources: str = "topk,best_local,correlated_local,local_probe"
    # Random start는 local refine에서 개선 기여가 낮게 관측되었다. 무제약 Focus3에서는
    # random source의 best acquisition score가 topk/best_local보다 충분히 좋을 때만
    # L-BFGS-B refine start quota를 유지한다.
    focus3_no_constraint_random_refine_gate_enabled: bool = True
    focus3_no_constraint_random_refine_gate_margin_ratio: float = 0.08
    focus3_no_constraint_random_discrete_gate_enabled: bool = True
    focus3_no_constraint_random_discrete_gate_margin_ratio: float = 0.08
    # 최근 실제 source 성과를 보고 refine start quota를 보정한다. Acquisition score만으로
    # random을 계속 refine하는 현상을 막기 위한 장치이며, 무제약 Focus3에만 적용한다.
    focus3_source_performance_policy_enabled: bool = True
    focus3_source_performance_recover_only: bool = False
    focus3_source_performance_window: int = 120
    focus3_source_performance_min_focus3_evals: int = 60
    focus3_source_performance_min_source_count: int = 12
    focus3_source_performance_poor_improve_rate: float = 0.006
    focus3_source_performance_random_penalty_fraction: float = 0.85
    focus3_source_performance_random_min_quota_fraction: float = 0.03
    focus3_source_performance_local_probe_enabled: bool = True
    focus3_source_performance_local_probe_min_count: int = 12
    focus3_source_performance_local_probe_poor_improve_rate: float = 0.006
    focus3_source_performance_local_probe_penalty_fraction: float = 0.70
    focus3_source_performance_local_probe_min_quota_fraction: float = 0.02
    focus3_source_performance_boundary_enabled: bool = True
    focus3_source_performance_boundary_min_count: int = 12
    focus3_source_performance_boundary_poor_improve_rate: float = 0.003
    focus3_source_performance_boundary_penalty_fraction: float = 0.85
    focus3_source_performance_boundary_min_quota_fraction: float = 0.01
    # Source performance가 random/boundary를 둘 다 penalize하더라도, 현재 best가
    # goal 근처까지 온 경우에는 productive한 source만 제한적으로 보호한다. 최근
    # no_dummy standalone 결과에서 Rosenbrock boundary 보호는 개선 없이 best-plan을
    # 많이 차지했으므로, standalone에서는 boundary 최근 개선률이 최소 기준을 넘을
    # 때만 보호한다. AION trusted-bounds override는 별도 profile에서 유지된다.
    focus3_source_performance_best_plan_filter_enabled: bool = True
    focus3_source_performance_best_plan_filter_local_probe_enabled: bool = True
    focus3_source_performance_best_plan_filter_near_goal_enabled: bool = True
    focus3_source_performance_best_plan_filter_near_goal_margin_ratio: float = 0.15
    focus3_source_performance_best_plan_filter_keep_one_when_both_penalized: bool = True
    focus3_source_performance_best_plan_filter_near_goal_preferred_source: str = "boundary"
    focus3_source_performance_best_plan_filter_near_goal_boundary_require_ok: bool = True
    focus3_source_performance_best_plan_filter_near_goal_boundary_min_improve_rate: float = 0.003
    focus3_source_performance_best_plan_filter_near_goal_random_fallback_enabled: bool = False
    focus3_source_performance_best_plan_filter_near_goal_random_min_advantage: float = 0.01
    focus3_refine_cooldown_enabled: bool = True
    focus3_refine_cooldown_window: int = 50
    focus3_refine_cooldown_worse_count: int = 12
    focus3_refine_cooldown_min_focus3_evals: int = 80
    # L-BFGS-B acquisition refine이 runtime을 지배할 수 있으므로 start 수는 유지하되
    # 각 local refine의 최대 반복/함수평가를 제한한다.
    focus3_refine_maxiter: int = 35
    focus3_refine_maxfun: int = 90
    # Focus3 acquisition: auto는 상태에 따라 LCB/EI/MEAN으로 전환한다.
    focus3_acq_type: str = "auto"
    focus3_auto_ei_min_data_ratio: float = 15.0
    focus3_auto_ei_max_volume_ratio: float = 0.25
    focus3_auto_ei_max_mean_width_ratio: float = 0.75
    focus3_auto_mean_enabled: bool = True
    focus3_auto_mean_min_data_ratio: float = 20.0
    focus3_auto_mean_max_volume_ratio: float = 0.20
    focus3_auto_mean_best_local_required: bool = True
    # Goal 근처에서 recovery가 켜지면 더 넓게 찾는 LCB보다 exploitation을 우선한다.
    # Rosenbrock no_dummy는 bounds 안에 optimum이 있어도 후반 LCB 탐색에 묶여
    # 1.20 근처에서 멈추는 케이스가 많았다.
    focus3_near_goal_exploitation_enabled: bool = True
    focus3_near_goal_exploitation_margin_ratio: float = 0.10
    focus3_near_goal_exploitation_acq: str = "MEAN"
    focus3_near_goal_exploitation_min_focus3_evals: int = 60
    # Focus3 source ratio = budget class + data reliability + GP/recent improvement 보정.
    focus3_source_adaptive_enabled: bool = True
    focus3_data_ratio_low: float = 5.0
    focus3_data_ratio_good: float = 15.0
    focus3_low_reliability_boundary_bonus: float = 0.00
    focus3_low_reliability_random_bonus: float = 0.10
    focus3_mid_reliability_random_bonus: float = 0.03
    focus3_gp_fallback_boundary_bonus: float = 0.00
    focus3_gp_fallback_random_bonus: float = 0.15
    focus3_recent_improvement_topk_bonus: float = 0.10
    # Constraint가 없는 문제에서는 boundary source가 개선 효율이 낮을 수 있으므로
    # 최종 source mixture에 cap/floor를 적용한다. Constraint 문제에서는 적용하지 않는다.
    focus3_no_constraint_source_cap_enabled: bool = True
    focus3_no_constraint_boundary_max: float = 0.06
    focus3_no_constraint_topk_min: float = 0.62
    focus3_no_constraint_random_min: float = 0.06
    # 무제약 문제에서는 boundary source가 acquisition상 약간 좋아 보이는 정도로
    # 최종 후보를 이기지 못하게 한다. topk/random보다 이 margin 이상 좋을 때만 허용.
    focus3_no_constraint_boundary_gate_enabled: bool = True
    focus3_no_constraint_boundary_gate_margin_ratio: float = 0.10
    # 무제약 문제에서는 boundary score에 직접 penalty를 더해 GP uncertainty만으로
    # boundary가 best_plan을 과도하게 이기지 못하게 한다. 낮은 acquisition score가 우수하다.
    focus3_no_constraint_boundary_score_penalty_enabled: bool = True
    focus3_no_constraint_boundary_score_penalty_ratio: float = 0.25
    # Discrete Focus3 compares the best candidate from each source pool. If
    # random gets the same pool size as structured sources, it can win simply
    # through broad coverage. Keep random available, but smaller in unconstrained
    # local search where topk/best_local have shown better improvement rates.
    focus3_no_constraint_random_pool_ratio: float = 0.35
    # RB처럼 narrow valley를 가진 문제에서는 top-k 전체보다 현재 best 주변을 더 촘촘히
    # 파는 source가 필요하다. best_local은 topk quota 일부를 가져와 좁은 sigma로 후보를 만든다.
    focus3_best_local_enabled: bool = True
    focus3_best_local_prob: float = 0.44
    focus3_best_local_max_prob: float = 0.65
    focus3_best_local_min_focus3_evals: int = 3
    focus3_best_local_min_data_ratio: float = 5.0
    focus3_best_local_sigma: float = 0.015
    focus3_best_local_sigma_schedule_enabled: bool = True
    focus3_best_local_sigma_mid_eval: int = 80
    focus3_best_local_sigma_mid: float = 0.010
    focus3_best_local_sigma_late_eval: int = 250
    focus3_best_local_sigma_late: float = 0.004
    focus3_best_local_sigma_recover_strong_multiplier: float = 0.85
    focus3_best_local_top_count: int = 5
    focus3_best_local_pool_ratio: float = 0.90
    focus3_best_local_anisotropic_enabled: bool = True
    focus3_best_local_elite_std_scale: float = 0.75
    focus3_best_local_max_sigma: float = 0.08
    focus3_best_local_anchor_best_prob: float = 0.45
    # best_local quota 일부를 elite covariance/PCA 방향 source로 분리한다.
    # 함수명을 보지 않고 최근 elite archive의 형상만 사용해 curved valley를 따라간다.
    focus3_correlated_local_enabled: bool = True
    focus3_correlated_local_min_dim: int = 5
    focus3_correlated_local_recover_only: bool = True
    focus3_correlated_local_min_no_improve: int = 80
    focus3_correlated_local_min_data_ratio: float = 8.0
    focus3_correlated_local_best_local_fraction: float = 0.35
    focus3_correlated_local_max_prob: float = 0.26
    focus3_correlated_local_near_goal_fraction_multiplier: float = 0.50
    focus3_correlated_local_near_goal_max_prob: float = 0.16
    focus3_correlated_local_min_prob: float = 0.03
    focus3_correlated_local_pool_ratio: float = 0.75
    focus3_correlated_local_elite_count: int = 16
    focus3_correlated_local_sigma_scale: float = 0.85
    focus3_correlated_local_max_step_ratio: float = 0.18
    focus3_correlated_local_jitter_ratio: float = 0.001
    # Focus2 fallback bounds는 evidence가 약하므로 boundary를 과신하지 않는다.
    focus3_fallback_bounds_source_policy_enabled: bool = True
    focus3_fallback_bounds_boundary_max: float = 0.03
    focus3_fallback_bounds_topk_min: float = 0.40
    focus3_fallback_bounds_best_local_min: float = 0.25
    focus3_fallback_bounds_random_min: float = 0.25
    # Random source는 pure random 대신 acquisition filter + archive coverage로 refine start를 고른다.
    focus3_random_cover_enabled: bool = True
    focus3_random_cover_acq_quantile: float = 0.70
    focus3_random_cover_reference_max: int = 1000
    # Focus3 최종 후보가 기존 archive와 너무 가까우면 다음 후보로 교체한다.
    focus3_dedup_enabled: bool = True
    focus3_min_candidate_rms_distance: float = 0.01
    focus3_dedup_random_attempts: int = 64
    # Focus3 정체 시 boundary/random 비율과 kappa를 일시적으로 올려 recovery 모드로 전환한다.
    focus3_recover_enabled: bool = True
    focus3_recover_window: int = 60
    # 너무 이른 recovery 전환을 막는다. 충분한 Focus3 history가 쌓인 뒤에만
    # stagnation 판정을 적용한다.
    focus3_recover_min_history: int = 90
    focus3_recover_tol: float = 1e-8
    focus3_recover_relative_tol: float = 1e-4
    focus3_recover_boundary_bonus: float = 0.15
    focus3_recover_random_bonus: float = 0.06
    # Mild/strong 2단 recovery. 짧은 정체에서는 boundary 과투입을 막고,
    # 긴 정체에서만 기존 수준에 가까운 탐색 강화로 전환한다.
    focus3_recover_strong_no_improve: int = 180
    focus3_recover_mild_boundary_scale: float = 0.15
    focus3_recover_mild_random_scale: float = 0.50
    focus3_recover_mild_kappa_multiplier: float = 1.05
    focus3_recover_kappa_multiplier: float = 1.25
    focus3_recover_max_kappa: float = 2.50
    # 무제약 recovery는 boundary보다 random/topk 쪽으로 기울인다.
    focus3_no_constraint_recover_boundary_scale: float = 0.0
    focus3_no_constraint_recover_random_scale: float = 0.15
    focus3_no_constraint_recover_topk_bonus: float = 0.18
    focus3_no_constraint_recover_strong_no_improve: int = 720
    focus3_no_constraint_recover_best_local_mild_bonus: float = 0.16
    focus3_no_constraint_recover_best_local_strong_bonus: float = 0.28
    focus3_no_constraint_recover_best_local_max: float = 0.65
    focus3_no_constraint_recover_best_local_late_no_improve: int = 450
    focus3_no_constraint_recover_best_local_late_bonus: float = 0.06
    focus3_no_constraint_recover_best_local_late_max: float = 0.75
    focus3_recover_random_discrete_gate_enabled: bool = True
    focus3_recover_random_discrete_gate_margin_ratio: float = 0.06
    focus3_recover_random_discrete_gate_min_no_improve: int = 180
    # Recovery 중에는 best 주변의 축/대각 probe 후보를 별도 source로 넣어
    # 단순 best-local gaussian이 놓치는 좁은 valley 방향을 보완한다.
    focus3_local_probe_enabled: bool = True
    focus3_local_probe_min_no_improve: int = 80
    focus3_local_probe_prob: float = 0.12
    focus3_local_probe_max_prob: float = 0.20
    focus3_local_probe_refine_floor_fraction: float = 0.10
    focus3_local_probe_late_no_improve: int = 320
    focus3_local_probe_late_refine_floor_fraction: float = 0.16
    focus3_local_probe_pool_max: int = 192
    focus3_local_probe_step_ratio: float = 0.030
    focus3_local_probe_min_step_ratio: float = 0.0015
    focus3_local_probe_scales: str = "1.0,0.5,0.25,0.1,0.05,0.025,0.01,1.75,2.5"

    # ------------------------------------------------------------------
    # Focus2: region manager / TuRBO-lite
    # ------------------------------------------------------------------
    # Focus2: Focus1 regions 안에서 region별 local GP 경쟁(TuRBO-lite)을 수행한다.
    focus2_enabled: bool = True
    focus2_min_total_budget: int = 10
    # Focus3가 Focus2 generated bounds에 의존하는 경우, Focus3 reserve가 Focus2
    # 예산을 0으로 밀어내지 못하게 보장할 최소 Focus2 샘플 수.
    focus2_min_budget_with_focus3: int = 1
    # Standalone selected-bounds validation path also keeps at least a tiny
    # Focus2 budget when Focus2 is part of the auto plan.
    focus2_min_budget_selected_bounds_with_focus3: int = 1
    focus2_budget_fraction: float = 0.30
    focus2_kappa_min: float = 1.50
    focus2_kappa_start: float = 2.50
    focus2_kappa_end: float = 1.20
    # Focus2 region output 정책: 실제 archive support가 있는 good regions만 다음 단계로 넘긴다.
    focus2_region_min_archive_points: int = 5
    focus2_region_min_good_candidates: int = 10
    focus2_region_min_volume_ratio: float = 0.05
    focus2_region_max_regions: int = 5
    focus2_region_candidate_pool_min: int = 4096
    focus2_region_candidate_pool_per_dim: int = 200
    focus2_region_candidate_pool_max: int = 30000
    focus2_region_uncertain_seed_fraction: float = 0.30
    focus2_region_cluster_radius: float = 0.20
    focus2_region_retry_enabled: bool = True
    focus2_region_retry_radius_scale: float = 1.50
    focus2_region_retry_uncertain_seed_fraction: float = 0.60
    focus2_region_retry_min_good_scale: float = 0.50
    focus2_region_retry_expand_ratio: float = 1.30
    focus2_region_quantile_low: float = 0.10
    focus2_region_quantile_high: float = 0.90
    focus2_region_expand_ratio: float = 1.15
    focus2_local_pool_size: int = 512
    focus2_local_topk_ratio: float = 0.30
    focus2_local_boundary_ratio: float = 0.35
    focus2_local_random_ratio: float = 0.35
    focus2_min_candidate_rms_distance: float = 0.02
    focus2_region_no_improve_tolerance: int = 8
    focus2_region_min_evals_before_drop: int = 5
    focus2_early_stop_enabled: bool = True
    focus2_early_stop_min_evals: int = 20
    focus2_early_stop_min_evals_per_region: int = 3
    focus2_early_stop_min_selected_region_evals: int = 3
    focus2_early_stop_min_active_bounds_updates: int = 2
    focus2_early_stop_require_all_active_sampled: bool = True
    focus2_early_stop_max_active_regions: int = 1
    focus2_early_stop_max_volume_ratio: float = 0.25
    focus2_early_stop_score_gap: float = 0.35
    focus2_early_stop_min_archive_points: int = 12
    focus2_final_archive_weight: float = 0.70
    focus2_final_support_weight: float = 0.15
    focus2_final_success_weight: float = 0.10
    focus2_final_no_improve_penalty: float = 0.08
    focus2_final_dropped_penalty: float = 0.25
    # Focus2가 generated bounds를 만들 때 단일 region을 과신하지 않도록
    # survivor 상위 region들을 union하고, 너무 좁은 bounds는 보수적으로 확장한다.
    focus2_final_union_enabled: bool = True
    focus2_final_union_top_k: int = 2
    # dropped region도 final_score penalty를 받은 상태로 union 경쟁에는 남긴다.
    # 조기 drop이 과했을 때 좋은 basin을 완전히 잃지 않기 위한 안전장치다.
    focus2_final_include_dropped_competitive: bool = True
    focus2_final_union_use_parent_bounds: bool = False
    # Final union bounds를 그대로 넘기기 전에 union 내부 archive 우수점으로 한 번 더 다듬는다.
    # 이후 expand/min-volume을 다시 적용하므로 너무 좁아지는 위험은 아래 bounds policy가 흡수한다.
    focus2_final_archive_trim_enabled: bool = True
    focus2_final_archive_trim_top_fraction: float = 0.50
    # Archive trim은 적은 점에 끌리면 위험하므로 충분한 support가 있을 때만 쓴다.
    focus2_final_archive_trim_min_points: int = 20
    focus2_final_archive_trim_min_per_dim: float = 4.0
    focus2_final_archive_trim_keep_min_points: int = 10
    focus2_final_archive_trim_keep_min_per_dim: float = 2.0
    focus2_final_archive_trim_quantile_low: float = 0.10
    focus2_final_archive_trim_quantile_high: float = 0.90
    # 0이면 union bounds 유지, 1이면 archive top quantile bounds로 완전 대체.
    # 기본은 center/width를 약하게만 보정한다.
    focus2_final_archive_trim_blend: float = 0.30
    focus2_final_bounds_expand_ratio: float = 1.20
    focus2_final_bounds_min_volume_ratio: float = 0.45
    focus2_final_bounds_max_volume_ratio: float = 1.0
    # Focus2 fallback은 evidence가 약하다는 뜻이므로 final bounds를 더 보수적으로 유지한다.
    focus2_final_fallback_archive_trim_blend: float = 0.0
    focus2_final_fallback_bounds_expand_ratio: float = 1.50
    focus2_final_fallback_min_volume_ratio: float = 0.55
    # Fallback으로 분류되지는 않았지만 최종 선택 region의 archive support가 약하면
    # Focus3 입력 bounds를 덜 공격적으로 좁힌다.
    focus2_final_low_support_policy_enabled: bool = True
    focus2_final_low_support_min_archive_points: int = 8
    focus2_final_low_support_min_per_dim: float = 4.0
    focus2_final_low_support_union_top_k: int = 3
    focus2_final_low_support_archive_trim_blend: float = 0.05
    focus2_final_low_support_bounds_expand_ratio: float = 1.35
    focus2_final_low_support_min_volume_ratio: float = 0.40
    # Volume ratio만 맞추면 특정 축이 좁게 잘릴 수 있다. Evidence가 약한 final bounds는
    # 축별 최소 coverage도 강제해 hard miss를 줄인다.
    focus2_final_axis_coverage_enabled: bool = True
    focus2_final_axis_coverage_low_support_min_width_ratio: float = 0.55
    focus2_final_axis_coverage_fallback_min_width_ratio: float = 0.75
    focus2_merge_enabled: bool = True
    focus2_merge_duplicate_pair_threshold: int = 2
    focus2_merge_shared_pair_threshold: int = 2
    focus2_merge_same_basin_min_score: float = 0.75
    focus2_merge_best_point_rms: float = 0.05
    focus2_merge_objective_gap_ratio: float = 0.10
    focus2_merge_expand_ratio: float = 1.15
    focus2_merge_top_fraction: float = 0.50
    focus2_merge_max_volume_ratio: float = 0.25
    focus2_active_bounds_enabled: bool = True
    focus2_active_bounds_min_archive_points: int = 12
    focus2_active_bounds_min_region_evals: int = 3
    focus2_active_bounds_top_fraction: float = 0.50
    focus2_active_bounds_min_top_points: int = 5
    focus2_active_bounds_quantile_low: float = 0.10
    focus2_active_bounds_quantile_high: float = 0.90
    focus2_active_bounds_expand_ratio: float = 1.20
    focus2_active_bounds_min_volume_ratio: float = 0.10
    focus2_active_bounds_max_volume_ratio: float = 0.25
    focus2_active_bounds_update_rate: float = 0.35
    focus2_scheduler_enabled: bool = True
    focus2_scheduler_initial_full_round: bool = True
    focus2_scheduler_archive_weight: float = 0.50
    focus2_scheduler_support_weight: float = 0.15
    focus2_scheduler_success_weight: float = 0.15
    focus2_scheduler_exploration_weight: float = 0.10
    focus2_scheduler_active_bounds_weight: float = 0.10
    focus2_scheduler_no_improve_penalty: float = 0.10
    focus2_scheduler_duplicate_penalty: float = 0.03

    # ------------------------------------------------------------------
    # Shared source-pool and stagnation tuning
    # ------------------------------------------------------------------
    # Removal candidates:
    # - source_stagnation_* overlaps conceptually with focus3_recover_*.
    #   Merge into focus3_recover_* after benchmark validation.
    # - source_pool_size/source_topk_fraction/source_topk_perturb_sigma/
    #   source_boundary_near_ratio are Focus3-specific in practice. Rename to
    #   focus3_source_* and keep aliases only for one compatibility window.
    # 정체(stagnation) 감지 시 탐색 강화. Focus3 adaptive source policy에서 사용한다.
    source_stagnation_window: int = 8
    source_stagnation_tol: float = 1e-8
    source_stagnation_boundary_bonus: float = 0.00
    source_stagnation_random_bonus: float = 0.02
    # source pool 생성 파라미터
    source_pool_size: int = 24
    source_topk_fraction: float = 0.20
    source_topk_perturb_sigma: float = 0.08
    source_boundary_near_ratio: float = 0.03
    # pre-constraint 후보 생성 정책 (focus3)
    source_feasible_multiplier: int = 3
    source_feasible_retry: int = 3
    source_feasible_min_starts: int = 1
    # pre-constraint가 있을 때 feasible이지만 제약 경계에 가까운 후보를
    # 별도 source로 일부 투입한다. 최종 CAE 전 hard feasible check는 항상 유지한다.
    constraint_boundary_source_enabled: bool = True
    constraint_boundary_acq_quantile: float = 0.75
    focus2_constraint_boundary_select_prob: float = 0.25
    focus3_constraint_boundary_prob: float = 0.20
    pre_final_feasible_attempts: int = 256

    # ------------------------------------------------------------------
    # Archive / GP train-set policy
    # ------------------------------------------------------------------
    # DOE 데이터에서 초기 학습점으로 반드시 보존할 상위 개수
    init_from_doe_topk: int = 20
    # DOE objective warm-start / GP train 상한: min(init_train_np_ratio * p, init_max_points)
    init_train_np_ratio: float = 20.0
    # 절대 상한. p가 커질 때 GP 학습 비용이 폭증하는 것을 막는다.
    init_max_points: int = 500
    # cap 초과 시 objective 우수점 비율. 나머지는 diversity로 채운다.
    init_train_top_fraction: float = 0.70
    # BO가 새로 평가한 최근 점을 GP train subset에 보존할 비율.
    gp_train_recent_fraction: float = 0.25
    # DOE seed 사용 범위: in_bounds | all
    doe_seed_scope: str = "in_bounds"
    # GP 재학습 주기 (iteration 단위)
    gp_refit_every: int = 1

    # ------------------------------------------------------------------
    # Input/objective and constraint policy
    # ------------------------------------------------------------------
    # DOE objective 컬럼명
    objective_col: str = "objective"
    # None이면 CAE metadata objective_sense 사용.
    # Removal candidate: CAE metadata should remain the objective-sense source
    # of truth. Keep this only while benchmarking opposite-sense experiments.
    objective_sense_override: str | None = None
    # 현재는 기본 OFF (추후 pre/post 제약 로직 확장용)
    enforce_pre_constraints: bool = False
    # post feasibility penalty 사용 여부
    post_constraint_enabled: bool = True
    # score/acquisition 패널티 강도
    post_penalty_lambda: float = 2.0
    # p_feasible이 임계치보다 낮으면 추가 강한 패널티 부여
    post_p_feasible_min: float = 0.0
    post_p_feasible_hard_penalty: float = 0.0
    # post score 적용 방식: add_penalty(기본)
    post_score_mode: str = "add_penalty"

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Output/debug
    # ------------------------------------------------------------------
    # 중복 판정 반올림 자릿수
    dedup_decimals: int = 12
    # 디버그 산출물 생성 여부: off | on
    debug_level: str = "on"
    # 긴 Optimizer run 진행 로그 주기. 0이면 progress 로그를 끈다.
    optimizer_progress_log_every: int = 25
    # Optimizer 병목 분석용 timing 로그. Public 결과에는 들어가지 않고
    # debug full history와 console progress에만 기록한다.
    optimizer_timing_log_enabled: bool = True


OPTIMIZER_COMMON_SYSTEM_FIELDS: set[str] = {
    "algorithm_id",
    "algorithm_params",
    "objective_col",
    "objective_sense_override",
    "enforce_pre_constraints",
    "post_constraint_enabled",
    "post_penalty_lambda",
    "post_p_feasible_min",
    "post_p_feasible_hard_penalty",
    "post_score_mode",
    "goal_early_stop_enabled",
    "goal_early_stop_patience",
    "dedup_decimals",
    "debug_level",
    "optimizer_progress_log_every",
    "optimizer_timing_log_enabled",
}

OPTIMIZER_FOCUS_BO_SYSTEM_PREFIXES: tuple[str, ...] = (
    "focus_",
    "focus0_",
    "focus1_",
    "focus2_",
    "focus3_",
)

OPTIMIZER_FOCUS_BO_SYSTEM_FIELDS: set[str] = {
    "acq_type",
    "kappa_start",
    "kappa_end",
    "ei_xi",
    "starts_per_iter",
    "random_starts_ratio",
    "source_mixture_enabled",
    "source_topk_prob",
    "source_boundary_prob",
    "source_random_prob",
    "source_stagnation_window",
    "source_stagnation_tol",
    "source_stagnation_boundary_bonus",
    "source_stagnation_random_bonus",
    "source_pool_size",
    "source_topk_fraction",
    "source_topk_perturb_sigma",
    "source_boundary_near_ratio",
    "source_feasible_multiplier",
    "source_feasible_retry",
    "source_feasible_min_starts",
    "constraint_boundary_source_enabled",
    "constraint_boundary_acq_quantile",
    "pre_final_feasible_attempts",
    "init_from_doe_topk",
    "init_train_np_ratio",
    "init_max_points",
    "init_train_top_fraction",
    "gp_train_recent_fraction",
    "doe_seed_scope",
    "gp_refit_every",
    "gp_x_scaling_enabled",
    "gp_x_scaling_mode",
    "gp_x_scaling_auto_span_ratio_threshold",
    "gp_x_scaling_span_floor",
    "goal_early_stop_min_focus",
}


def _optimizer_system_config_payload(system: OptimizerSystemConfig) -> dict[str, object]:
    if isinstance(system, OptimizerSystemConfig):
        return asdict(system)
    if hasattr(system, "__dict__"):
        return dict(vars(system))
    return {}


def optimizer_common_system_snapshot(system: OptimizerSystemConfig) -> dict[str, object]:
    payload = _optimizer_system_config_payload(system)
    return {
        key: payload[key]
        for key in sorted(OPTIMIZER_COMMON_SYSTEM_FIELDS)
        if key in payload
    }


def optimizer_focus_bo_system_snapshot(system: OptimizerSystemConfig) -> dict[str, object]:
    payload = _optimizer_system_config_payload(system)
    return {
        key: value
        for key, value in sorted(payload.items())
        if key in OPTIMIZER_FOCUS_BO_SYSTEM_FIELDS
        or key.startswith(OPTIMIZER_FOCUS_BO_SYSTEM_PREFIXES)
    }


def split_optimizer_system_config(
    system: OptimizerSystemConfig,
    *,
    algorithm_id: str | None = None,
) -> dict[str, object]:
    """Return an audit snapshot split by the common contract and algorithm knobs."""

    return asdict(optimizer_system_config_view(system, algorithm_id=algorithm_id))


def optimizer_system_config_view(
    system: OptimizerSystemConfig,
    *,
    algorithm_id: str | None = None,
) -> OptimizerSystemConfigView:
    """Build a structured view without changing the flat runtime config shape."""

    payload = _optimizer_system_config_payload(system)
    common = optimizer_common_system_snapshot(system)
    focus_bo = optimizer_focus_bo_system_snapshot(system)
    assigned = set(common) | set(focus_bo)
    compatibility = {
        key: value
        for key, value in sorted(payload.items())
        if key not in assigned
    }
    selected_algorithm_id = str(algorithm_id or payload.get("algorithm_id", "focus_bo") or "focus_bo")
    if selected_algorithm_id in {"focus_bo", "default"}:
        algorithm = dict(focus_bo)
        algorithm_contract = "focus_bo_builtin"
    else:
        algorithm = {
            "algorithm_params": dict(payload.get("algorithm_params", {}) or {}),
        }
        algorithm_contract = "optimizer_runtime"
    return OptimizerSystemConfigView(
        schema="optimizer_system_config_v1",
        algorithm_id=selected_algorithm_id,
        algorithm_contract=algorithm_contract,
        common=common,
        algorithm=algorithm,
        focus_bo=focus_bo,
        compatibility=compatibility,
    )


@dataclass
class OptimizerConfig:
    user: OptimizerUserConfig
    system: OptimizerSystemConfig
    cae: CAEConfig
    # Sequential 연계용 metadata 경로들 (옵션).
    # Removal candidate for service mode: prefer run_context + direct public/meta
    # artifact paths. Keep while standalone/internal task resume still depends on it.
    cae_metadata_path: str | None = None
    doe_metadata_path: str | None = None
    explorer_metadata_path: str | None = None
    modeler_metadata_path: str | None = None
    # 직접 경로 주입 (옵션).
    # Duplicate with OptimizerUserConfig paths. Long-term policy should keep one
    # user/input-layer path source and remove the duplicate config-level aliases.
    doe_csv_path: str | None = None
    explorer_bounds_path: str | None = None
