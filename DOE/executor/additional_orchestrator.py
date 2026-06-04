# DOE/executor/additional_orchestrator.py

from __future__ import annotations

import inspect
from itertools import combinations
import warnings
from typing import Any, Callable, Optional

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor

from utils.boundary_sampling import sample_boundary_corners, sample_boundary_corners_random, sample_boundary_partial
from utils.bounds_utils import compute_spans_lbs, clamp_to_bounds
from utils.objective_sense import canonical_objective_for_result
from DOE.executor.anchor_refiner import (
    AcquisitionOptimizer,
    fit_gp_with_fallback,
    kernel_common_best,
    kernel_stable_conservative,
)
from DOE.executor.constraint_filter import (
    clamp_ratio,
    evaluate_constraints_batch,
    evaluate_constraints_point,
    has_constraint_defs,
)
from DOE.executor.dataset_store import DatasetStore
from DOE.executor.eval_sanitizer import sanitize_evaluate_output
from DOE.executor.plan_builder import PlanBuilder
from DOE.executor.execution_budget import ExecutionBudget
from DOE.executor.local_sampler import LocalSampler



class AdditionalDOEOrchestrator:
    """
    Additional DOE Orchestrator

    핵심 동작 요약:
    - 초기 DOE: 사용자 선택 sampler로 total_budget * init_ratio 만큼 실행
    - 추가 DOE는 stage 단위로 반복
    - X_plan: stage당 1회 생성, 중복 허용 (dedup 없음)
      * 현재 phase 비율로 global/local 분할
    - Gate 평가: X_plan으로 1회 평가
      * gate_stop이면 즉시 종료
      * phase 전환은 다음 stage부터 적용
    - X_exec: X_plan에서 선별 (중복 제거는 X_exec에서만)
      * n_exec = min(exec_ratio * total_budget, remaining)
      * global/local 비율은 현재 phase 기준
      * 부족하면 최대 3회 재시도 후 종료
    - tol: stage 기반으로 계산하며 X_exec dedup에만 적용
    - budget: 시도 횟수 기준으로 소모 (success 여부와 무관)
    - 실패(success=False)일 때 objective는 inf로 저장
    """
    def __init__(
        self,
        *,
        bounds: list[tuple[float, float]],
        sampler: Callable[..., np.ndarray],  # initial DOE sampler (lhs/rs)
        evaluate_func: Callable[[np.ndarray], Dict[str, Any]],
        feasibility_func: Callable[[Optional[Dict]], bool],
        surrogate_factory: Any,
        gate1: Any,
        gate2: Any,
        gate_manager: Any,
        rng: np.random.Generator,
        total_budget: int,
        init_ratio: float = 0.3,
        exec_ratio: float = 0.1,
        initial_corner_ratio: float = 0.05,
        global_random_ratio: float = 0.3,
        global_boundary_ratio: float = 0.1,
        global_boundary_constraint_boost: float = 1.2,
        global_boundary_margin_cross_ratio: float = 0.5,
        global_boundary_margin_near_tol: float = 0.05,
        global_margin_ratio: float = 0.2,
        global_margin_obj_alpha: float = 0.6,
        global_margin_subset_enabled: bool = True,
        global_margin_subset_random_k_count: int = 2,
        global_top_ratio: float = 0.2,
        global_boundary_corner_ratio: float = 0.5,
        plan_base_k: float = 200.0,
        plan_remaining_cap: float = 3.0,
        plan_decay: float = 0.85,
        budget_policy: str = "consume_all",
        phase1_global_ratio: Optional[float] = None,
        phase2_global_ratio: Optional[float] = None,
        phase2_disable_boundary_sampling: bool = True,
        phase2_gate1_score_min: float = 0.55,
        phase2_gate1_rescue_enabled: bool = True,
        phase2_gate1_rescue_band: float = 0.03,
        phase2_gate1_rescue_gate2_min: float = 0.85,
        phase2_gate2_score_min: float = 0.75,
        phase2_gate2_score_sticky_min: float = 0.65,
        gate_smoothing_enabled: bool = True,
        gate_ema_alpha: float = 0.35,
        gate_ema_warmup_stages: int = 2,
        gate_smoothing_use_for_phase2: bool = True,
        gate_smoothing_use_for_stop: bool = True,
        collapse_span_ratio_threshold: float = 0.22,
        collapse_anchor_streak_threshold: int = 2,
        collapse_min_stage: int = 2,
        diversity_injection_ratio: float = 0.20,
        diversity_injection_min_points: int = 2,
        diversity_injection_max_ratio: float = 0.40,
        diversity_boundary_floor_ratio: float = 0.12,
        min_additional_rounds: int = 2,
        phase2_min_usable_np_ratio: float = 15.0,
        phase2_np_ratio_cap_scale: float = 0.74,
        phase2_min_used_budget_ratio: float = 0.3,
        phase2_min_used_budget_ratio_high_crate: float = 0.5,
        phase2_high_crate_threshold: float = 0.85,
        phase2_g2_saturation_threshold: float = 0.92,
        phase2_g2_saturation_min_remaining: float = 0.20,
        global_bucket_minima_enabled: bool = True,
        global_bucket_minima_strict: bool = False,
        global_boundary_high_crate_threshold: float = 0.80,
        phase1_min_top: int = 1,
        phase1_min_margin: int = 1,
        phase1_min_boundary_any: int = 1,
        phase1_min_boundary_classic_hc: int = 1,
        phase1_min_boundary_cross_hc: int = 1,
        phase2_min_top: int = 1,
        phase2_min_margin: int = 1,
        global_bucket_plan_multiplier: int = 3,
        early_stop_min_used_budget_ratio: float = 0.5,
        early_stop_min_usable_np_ratio: float = 20.0,
        stop_span_ratio_threshold: float = 0.3,
        stop_anchor_spread_streak: int = 2,
        stop_min_usable_np_ratio: float = 20.0,
        probe_stage_enabled: bool = True,
        probe_top_ratio: float = 0.3,
        probe_max_points: int = 5,
        probe_min_range_ratio: float = 0.25,
        probe_std_scale: float = 2.0,
        probe_perturb_ratio: float = 0.02,
        initial_probe_multiplier: float = 2.0,
        local_exec_min_anchors: int = 2,
        plan_filter_safety: float = 1.2,
        plan_filter_r_floor: float = 0.02,
        success_rate_floor: float = 0.02,
        max_additional_stages: int = 10,
        var_names: list[str] | None = None,
        constraint_defs: list[dict] | None = None,
        # local building knobs
        local_anchor_max_base: int = 8,
        local_anchor_max_decay: float = 0.9,
        local_anchor_best_k: int = 3,
        local_anchor_small_k: int = 2,
        local_anchor_best_ratio: float = 0.35,
        local_anchor_small_ratio: float = 0.2,
        local_radius_ratio_phase1: float = 0.5,
        local_radius_ratio_phase2: float = 0.3,
        local_top_p: float = 0.1,
        local_top_k_min: int = 10,
        local_dbscan_min_samples: int = 5,
        local_dbscan_q_eps: float = 0.85,
        local_dbscan_eps_max: float = 0.5,
        local_min_radius_ratio: float = 0.01,
        local_tol_ratio: float = 0.2,
        local_refine_min_points: int = 15,
        local_cluster_delta_ratio: float = 0.02,
        local_singleton_box_ratio: float = 0.03,
        local_phase1_kappa: float = 0.75,
        local_phase2_kappa: float = 0.5,
        local_base_perturb_ratio: float = 0.02,
        local_constraint_retry_count: int = 1,
        local_constraint_shrink_factor: float = 0.5,
        local_constraint_min_factor: float = 2.0,
        local_exec_pick_mode: str = "random",
        post_use_penalty: bool = True,
        post_lambda_init: float = 2.0,
        post_lambda_min: float = 0.25,
        post_lambda_max: float = 8.0,
        post_lambda_power: float = 1.0,
        post_feasible_rate_floor: float = 0.05,
        post_clf_min_samples: int = 30,
        post_clf_min_pos: int = 5,
        post_clf_min_neg: int = 5,
        hpo_runner: Optional[Any] = None,
        force_baseline: bool = False,
        local_gp_seed: int = 42,
        local_gp_use_white_kernel: bool = False,
    ):
        self.bounds = bounds
        self.sampler = sampler

        self.evaluate_func = evaluate_func
        self.feasibility_func = feasibility_func

        self.surrogate_factory = surrogate_factory
        self.gate1 = gate1
        self.gate2 = gate2
        self.gate_manager = gate_manager

        self.rng = rng

        self.total_budget = int(total_budget)
        self.init_ratio = float(init_ratio)
        self.exec_ratio = float(exec_ratio)
        self._initial_budget_used = 0
        self._additional_budget_total = 0
        self.initial_corner_ratio = float(initial_corner_ratio)
        if not (0.0 <= self.initial_corner_ratio <= 1.0):
            raise ValueError("initial_corner_ratio must be in [0, 1]")
        self.global_random_ratio = float(global_random_ratio)
        self.global_boundary_ratio = float(global_boundary_ratio)
        self.global_boundary_constraint_boost = float(global_boundary_constraint_boost)
        self.global_boundary_margin_cross_ratio = float(global_boundary_margin_cross_ratio)
        self.global_boundary_margin_near_tol = float(global_boundary_margin_near_tol)
        self.global_margin_ratio = float(global_margin_ratio)
        self.global_margin_obj_alpha = float(global_margin_obj_alpha)
        self.global_margin_subset_enabled = bool(global_margin_subset_enabled)
        self.global_margin_subset_random_k_count = int(global_margin_subset_random_k_count)
        if self.global_margin_subset_random_k_count < 0:
            raise ValueError("global_margin_subset_random_k_count must be >= 0")
        self.global_top_ratio = float(global_top_ratio)
        self.global_boundary_corner_ratio = float(global_boundary_corner_ratio)
        self.plan_base_k = float(plan_base_k)
        self.plan_remaining_cap = float(plan_remaining_cap)
        self.plan_decay = float(plan_decay)
        policy = str(budget_policy).strip().lower()
        if policy not in {"consume_all", "adaptive_early_stop"}:
            raise ValueError("budget_policy must be one of: consume_all, adaptive_early_stop")
        self.budget_policy = policy
        self.exec_min = 4
        if phase1_global_ratio is None or phase2_global_ratio is None:
            raise ValueError("phase1_global_ratio and phase2_global_ratio are required")
        self.phase1_global_ratio = float(phase1_global_ratio)
        self.phase2_global_ratio = float(phase2_global_ratio)
        self.phase2_disable_boundary_sampling = bool(phase2_disable_boundary_sampling)
        self.phase2_gate1_score_min = float(phase2_gate1_score_min)
        self.phase2_gate1_rescue_enabled = bool(phase2_gate1_rescue_enabled)
        self.phase2_gate1_rescue_band = float(phase2_gate1_rescue_band)
        self.phase2_gate1_rescue_gate2_min = float(phase2_gate1_rescue_gate2_min)
        self.phase2_gate2_score_min = float(phase2_gate2_score_min)
        self.phase2_gate2_score_sticky_min = float(phase2_gate2_score_sticky_min)
        if not (0.0 <= self.phase2_gate1_score_min <= 1.0):
            raise ValueError("phase2_gate1_score_min must be in [0, 1]")
        if not (0.0 <= self.phase2_gate1_rescue_band <= 1.0):
            raise ValueError("phase2_gate1_rescue_band must be in [0, 1]")
        if not (0.0 <= self.phase2_gate1_rescue_gate2_min <= 1.0):
            raise ValueError("phase2_gate1_rescue_gate2_min must be in [0, 1]")
        if not (0.0 <= self.phase2_gate2_score_min <= 1.0):
            raise ValueError("phase2_gate2_score_min must be in [0, 1]")
        if not (0.0 <= self.phase2_gate2_score_sticky_min <= 1.0):
            raise ValueError("phase2_gate2_score_sticky_min must be in [0, 1]")
        self.gate_smoothing_enabled = bool(gate_smoothing_enabled)
        self.gate_ema_alpha = float(gate_ema_alpha)
        if not (0.0 < self.gate_ema_alpha <= 1.0):
            raise ValueError("gate_ema_alpha must be in (0, 1]")
        self.gate_ema_warmup_stages = int(gate_ema_warmup_stages)
        if self.gate_ema_warmup_stages < 0:
            raise ValueError("gate_ema_warmup_stages must be >= 0")
        self.gate_smoothing_use_for_phase2 = bool(gate_smoothing_use_for_phase2)
        self.gate_smoothing_use_for_stop = bool(gate_smoothing_use_for_stop)
        self.collapse_span_ratio_threshold = float(collapse_span_ratio_threshold)
        if self.collapse_span_ratio_threshold < 0.0:
            raise ValueError("collapse_span_ratio_threshold must be >= 0")
        self.collapse_anchor_streak_threshold = int(collapse_anchor_streak_threshold)
        if self.collapse_anchor_streak_threshold < 1:
            raise ValueError("collapse_anchor_streak_threshold must be >= 1")
        self.collapse_min_stage = int(collapse_min_stage)
        if self.collapse_min_stage < 1:
            raise ValueError("collapse_min_stage must be >= 1")
        self.diversity_injection_ratio = float(diversity_injection_ratio)
        if self.diversity_injection_ratio < 0.0:
            raise ValueError("diversity_injection_ratio must be >= 0")
        self.diversity_injection_min_points = int(diversity_injection_min_points)
        if self.diversity_injection_min_points < 0:
            raise ValueError("diversity_injection_min_points must be >= 0")
        self.diversity_injection_max_ratio = float(diversity_injection_max_ratio)
        if not (0.0 <= self.diversity_injection_max_ratio <= 1.0):
            raise ValueError("diversity_injection_max_ratio must be in [0, 1]")
        self.diversity_boundary_floor_ratio = float(diversity_boundary_floor_ratio)
        if not (0.0 <= self.diversity_boundary_floor_ratio <= 1.0):
            raise ValueError("diversity_boundary_floor_ratio must be in [0, 1]")
        self.min_additional_rounds = int(min_additional_rounds)
        self.phase2_min_usable_np_ratio = float(phase2_min_usable_np_ratio)
        if self.phase2_min_usable_np_ratio < 0.0:
            raise ValueError("phase2_min_usable_np_ratio must be >= 0")
        self.phase2_np_ratio_cap_scale = float(phase2_np_ratio_cap_scale)
        if self.phase2_np_ratio_cap_scale <= 0.0:
            raise ValueError("phase2_np_ratio_cap_scale must be > 0")
        self.phase2_min_used_budget_ratio = float(phase2_min_used_budget_ratio)
        if not (0.0 <= self.phase2_min_used_budget_ratio <= 1.0):
            raise ValueError("phase2_min_used_budget_ratio must be in [0, 1]")
        self.phase2_min_used_budget_ratio_high_crate = float(phase2_min_used_budget_ratio_high_crate)
        if not (0.0 <= self.phase2_min_used_budget_ratio_high_crate <= 1.0):
            raise ValueError("phase2_min_used_budget_ratio_high_crate must be in [0, 1]")
        self.phase2_high_crate_threshold = float(phase2_high_crate_threshold)
        if not (0.0 <= self.phase2_high_crate_threshold <= 1.0):
            raise ValueError("phase2_high_crate_threshold must be in [0, 1]")
        # DOE-2 gate2 saturation guard params
        self.phase2_g2_saturation_threshold = float(phase2_g2_saturation_threshold)
        if not (0.0 <= self.phase2_g2_saturation_threshold <= 1.0):
            raise ValueError("phase2_g2_saturation_threshold must be in [0, 1]")
        self.phase2_g2_saturation_min_remaining = float(phase2_g2_saturation_min_remaining)
        if not (0.0 <= self.phase2_g2_saturation_min_remaining <= 1.0):
            raise ValueError("phase2_g2_saturation_min_remaining must be in [0, 1]")
        # Bucket minima (DOE-3 대체)
        self.global_bucket_minima_enabled = bool(global_bucket_minima_enabled)
        self.global_bucket_minima_strict = bool(global_bucket_minima_strict)
        self.global_boundary_high_crate_threshold = float(global_boundary_high_crate_threshold)
        if not (0.0 <= self.global_boundary_high_crate_threshold <= 1.0):
            raise ValueError("global_boundary_high_crate_threshold must be in [0, 1]")
        self.phase1_min_top = int(max(0, int(phase1_min_top)))
        self.phase1_min_margin = int(max(0, int(phase1_min_margin)))
        self.phase1_min_boundary_any = int(max(0, int(phase1_min_boundary_any)))
        self.phase1_min_boundary_classic_hc = int(max(0, int(phase1_min_boundary_classic_hc)))
        self.phase1_min_boundary_cross_hc = int(max(0, int(phase1_min_boundary_cross_hc)))
        self.phase2_min_top = int(max(0, int(phase2_min_top)))
        self.phase2_min_margin = int(max(0, int(phase2_min_margin)))
        self.global_bucket_plan_multiplier = int(max(1, int(global_bucket_plan_multiplier)))
        # 상태 추적
        self._g2_saturation_trigger_count = 0
        self._last_phase1_n_exec: int | None = None
        self._bucket_min_relaxed_count = 0
        self.early_stop_min_used_budget_ratio = float(early_stop_min_used_budget_ratio)
        if not (0.0 <= self.early_stop_min_used_budget_ratio <= 1.0):
            raise ValueError("early_stop_min_used_budget_ratio must be in [0, 1]")
        self.early_stop_min_usable_np_ratio = float(early_stop_min_usable_np_ratio)
        if self.early_stop_min_usable_np_ratio < 0.0:
            raise ValueError("early_stop_min_usable_np_ratio must be >= 0")
        self.stop_span_ratio_threshold = float(stop_span_ratio_threshold)
        self.stop_anchor_spread_streak = int(stop_anchor_spread_streak)
        self.stop_min_usable_np_ratio = float(stop_min_usable_np_ratio)
        if self.stop_min_usable_np_ratio < 0.0:
            raise ValueError("stop_min_usable_np_ratio must be >= 0")
        self.probe_stage_enabled = bool(probe_stage_enabled)
        self.probe_top_ratio = float(probe_top_ratio)
        self.probe_max_points = max(int(probe_max_points), 1)
        self.probe_min_range_ratio = float(probe_min_range_ratio)
        self.probe_std_scale = float(probe_std_scale)
        self.probe_perturb_ratio = float(probe_perturb_ratio)
        self._anchor_spread_zero_streak = 0
        self._probe_stage_done = False
        self._ever_phase2 = False
        self._collapse_diversity_pending = False
        self.initial_probe_multiplier = float(initial_probe_multiplier)
        self.local_exec_min_anchors = int(max(0, int(local_exec_min_anchors)))
        self.plan_filter_safety = float(plan_filter_safety)
        self.plan_filter_r_floor = float(plan_filter_r_floor)
        self.success_rate_floor = float(success_rate_floor)
        if not (0.0 <= self.success_rate_floor <= 1.0):
            raise ValueError("success_rate_floor must be in [0, 1]")
        self.max_additional_stages = int(max_additional_stages)
        self.var_names = list(var_names or [f"x{i+1}" for i in range(len(bounds))])
        self.constraint_defs = list(constraint_defs or [])
        self.has_constraints = has_constraint_defs(self.constraint_defs)
        self.has_pre_constraints = any(
            str(c.get("scope", "pre")).strip().lower() == "pre"
            for c in self.constraint_defs
        )
        self.has_post_constraints = any(
            str(c.get("scope", "pre")).strip().lower() == "post"
            for c in self.constraint_defs
        )
        self.constraint_rate_hat = 1.0
        self._constraint_gen_total = 0
        self._constraint_feas_total = 0

        self.local_anchor_max_base = int(local_anchor_max_base)
        self.local_anchor_max_decay = float(local_anchor_max_decay)
        self.local_anchor_best_k = int(local_anchor_best_k)
        self.local_anchor_small_k = int(local_anchor_small_k)
        self.local_anchor_best_ratio = float(local_anchor_best_ratio)
        self.local_anchor_small_ratio = float(local_anchor_small_ratio)
        self.local_radius_ratio_phase1 = float(local_radius_ratio_phase1)
        self.local_radius_ratio_phase2 = float(local_radius_ratio_phase2)
        self.local_top_p = float(local_top_p)
        self.local_top_k_min = int(local_top_k_min)
        self.local_dbscan_min_samples = int(local_dbscan_min_samples)
        self.local_dbscan_q_eps = float(local_dbscan_q_eps)
        self.local_dbscan_eps_max = float(local_dbscan_eps_max)
        self.local_min_radius_ratio = float(local_min_radius_ratio)
        self.local_tol_ratio = float(local_tol_ratio)
        self.local_refine_min_points = int(local_refine_min_points)
        self.local_cluster_delta_ratio = float(local_cluster_delta_ratio)
        self.local_singleton_box_ratio = float(local_singleton_box_ratio)
        self.local_phase1_kappa = float(local_phase1_kappa)
        self.local_phase2_kappa = float(local_phase2_kappa)
        self.local_base_perturb_ratio = float(local_base_perturb_ratio)
        self.local_constraint_retry_count = int(local_constraint_retry_count)
        self.local_constraint_shrink_factor = float(local_constraint_shrink_factor)
        self.local_constraint_min_factor = float(local_constraint_min_factor)
        mode = str(local_exec_pick_mode).strip().lower()
        if mode not in {"random", "distance"}:
            raise ValueError("local_exec_pick_mode must be one of: random, distance")
        self.local_exec_pick_mode = mode
        self.post_use_penalty = bool(post_use_penalty)
        self.post_lambda_init = float(post_lambda_init)
        self.post_lambda_min = float(post_lambda_min)
        self.post_lambda_max = float(post_lambda_max)
        self.post_lambda_power = float(post_lambda_power)
        self.post_feasible_rate_floor = float(post_feasible_rate_floor)
        self.post_clf_min_samples = int(post_clf_min_samples)
        self.post_clf_min_pos = int(post_clf_min_pos)
        self.post_clf_min_neg = int(post_clf_min_neg)
        self.post_lambda_current = float(post_lambda_init)
        self.post_feasible_rate_hat = 1.0
        self._post_feas_total = 0
        self._post_eval_total = 0
        self._post_model = None
        self.post_policy_log: list[dict] = []
        self.force_baseline = bool(force_baseline)
        self.local_gp_seed = int(local_gp_seed)
        self.local_gp_use_white_kernel = bool(local_gp_use_white_kernel)

        self.store = DatasetStore(dim=len(bounds))
        self.budget = ExecutionBudget(total=self.total_budget)

        self.hpo_runner = hpo_runner
        self._hpo_done = False
        self.hpo_min_samples = 10
        self.failure_reason: str | None = None
        self.local_metrics: list[dict] = []
        self.gate_history: list[dict] = []
        self._probe_acq = AcquisitionOptimizer()
        self._anchor_spread_zero_streak_max = 0
        self._gate_eval_count = 0
        self._gate1_ema: float | None = None
        self._gate2_ema: float | None = None

        # Global plan builder is fixed LHS (per your scenario)
        self.plan_builder = PlanBuilder(bounds=bounds, rng=rng)

        self.local_sampler = LocalSampler(
            bounds=self.bounds,
            rng=self.rng,
            gp_seed=self.local_gp_seed,
            local_gp_use_white_kernel=self.local_gp_use_white_kernel,
            pre_feasible_fn=self._is_pre_feasible,
            post_feasible_prob_fn=self._post_feasible_prob,
            local_anchor_best_k=self.local_anchor_best_k,
            local_anchor_small_k=self.local_anchor_small_k,
            local_anchor_best_ratio=self.local_anchor_best_ratio,
            local_anchor_small_ratio=self.local_anchor_small_ratio,
            local_top_p=self.local_top_p,
            local_top_k_min=self.local_top_k_min,
            local_dbscan_min_samples=self.local_dbscan_min_samples,
            local_dbscan_q_eps=self.local_dbscan_q_eps,
            local_dbscan_eps_max=self.local_dbscan_eps_max,
            local_min_radius_ratio=self.local_min_radius_ratio,
            local_tol_ratio=self.local_tol_ratio,
            local_refine_min_points=self.local_refine_min_points,
            local_cluster_delta_ratio=self.local_cluster_delta_ratio,
            local_singleton_box_ratio=self.local_singleton_box_ratio,
            local_phase1_kappa=self.local_phase1_kappa,
            local_phase2_kappa=self.local_phase2_kappa,
            local_base_perturb_ratio=self.local_base_perturb_ratio,
        )

        # Exec selection is handled inline (rank + random mix)


    @staticmethod
    def _extract_pre_constraint_ids(constraints_payloads: list[dict]) -> list[str]:
        ids: set[str] = set()
        for payload in constraints_payloads:
            if not isinstance(payload, dict):
                continue
            for key, cinfo in payload.items():
                if not isinstance(cinfo, dict):
                    continue
                scope = str(cinfo.get("scope", "pre")).strip().lower()
                if scope != "pre":
                    continue
                cid = str(cinfo.get("id", key)).strip()
                if cid:
                    ids.add(cid)
        return sorted(ids)

    def _resolve_margin_subset_k_values(self, n_constraints: int) -> list[int]:
        n = int(n_constraints)
        if n <= 0:
            return []
        if n <= 3:
            return list(range(1, n + 1))
        mid_candidates = list(range(2, n))
        n_mid = min(max(self.global_margin_subset_random_k_count, 0), len(mid_candidates))
        picked_mid: list[int] = []
        if n_mid > 0:
            arr = np.asarray(mid_candidates, dtype=int)
            picked_mid = (
                self.rng.choice(arr, size=n_mid, replace=False).astype(int).tolist()
            )
        k_vals = sorted(set([1, n] + picked_mid))
        return [int(k) for k in k_vals]

    def _build_margin_subset_specs(self, constraint_ids: list[str]) -> list[tuple[tuple[str, ...], float]]:
        if not constraint_ids:
            return []
        k_values = self._resolve_margin_subset_k_values(len(constraint_ids))
        if not k_values:
            return []
        subsets: list[tuple[str, ...]] = []
        weights_raw: list[float] = []
        for k in k_values:
            for subset in combinations(constraint_ids, int(k)):
                subsets.append(tuple(subset))
                weights_raw.append(float(k))
        if not subsets:
            return []
        w_sum = float(sum(weights_raw))
        if w_sum <= 0.0:
            return []
        return [(subsets[i], float(weights_raw[i] / w_sum)) for i in range(len(subsets))]

    @staticmethod
    def _allocate_weighted_subset_quotas(
        *,
        n_total: int,
        weights: list[float],
        enforce_min_one: bool,
    ) -> list[int]:
        if n_total <= 0 or not weights:
            return [0 for _ in weights]
        m = len(weights)
        w = [max(float(x), 0.0) for x in weights]
        w_sum = float(sum(w))
        if w_sum <= 0.0:
            w = [1.0 / float(m) for _ in range(m)]
        else:
            w = [float(x / w_sum) for x in w]

        def _distribute(total: int, base_offset: list[int]) -> list[int]:
            raw = [float(total) * wi for wi in w]
            add = [int(np.floor(v)) for v in raw]
            remain = int(max(total - sum(add), 0))
            frac_order = sorted(
                range(m),
                key=lambda i: (raw[i] - add[i], raw[i]),
                reverse=True,
            )
            for i in frac_order[:remain]:
                add[i] += 1
            return [int(base_offset[i] + add[i]) for i in range(m)]

        if not enforce_min_one:
            return _distribute(int(n_total), [0 for _ in range(m)])

        if n_total >= m:
            # 각 subset 최소 1개를 보장하되, 총량은 n_total로 유지
            return _distribute(int(n_total - m), [1 for _ in range(m)])

        # 총량이 subset 수보다 작으면 가중치가 큰 subset부터 1개 배정
        out = [0 for _ in range(m)]
        order = sorted(range(m), key=lambda i: (w[i], -i), reverse=True)
        for i in order[: int(n_total)]:
            out[i] = 1
        return out

    def _compute_n_exec_for_stage(
        self,
        *,
        round_idx: int,
        remaining: int,
        phase: int,
    ) -> tuple[int, dict[str, int | bool | float]]:
        """Dynamic stage allocation with priority:
        1) min floors, 2) phase ratio(strict if possible), 3) prefer >=3 stages,
        4) prefer around 5 stages and N/p in [5, 7] (soft).
        """

        def _split_exec(
            *,
            n_exec_try: int,
            strict_ratio: bool,
        ) -> tuple[bool, int, int, float]:
            n_i = int(max(n_exec_try, 0))
            if n_i <= 0:
                return False, 0, 0, 0.0
            g_floor = int(global_floor)
            l_floor = int(local_floor)
            g_min = int(max(g_floor, 0))
            g_max = int(n_i - max(l_floor, 0))
            if g_min > g_max:
                return False, 0, 0, 0.0

            # phase1: global ratio lower-bound
            if int(phase) == 1:
                r = float(np.clip(self.phase1_global_ratio, 0.0, 1.0))
                g_ratio_min = int(np.ceil(r * float(n_i) - 1e-12)) if strict_ratio else 0
                g_pick = int(max(g_min, g_ratio_min))
                if g_pick > g_max:
                    return False, 0, 0, 0.0
                l_pick = int(n_i - g_pick)
                return True, g_pick, l_pick, float(g_pick) / float(max(n_i, 1))

            # phase2: global ratio upper-bound (local-priority)
            r = float(np.clip(self.phase2_global_ratio, 0.0, 1.0))
            g_ratio_max = int(np.floor(r * float(n_i) + 1e-12)) if strict_ratio else n_i
            g_cap = int(min(g_max, g_ratio_max))
            if g_min > g_cap:
                return False, 0, 0, 0.0
            g_pick = int(g_min)  # maximize local by minimizing global
            l_pick = int(n_i - g_pick)
            return True, g_pick, l_pick, float(g_pick) / float(max(n_i, 1))

        def _strict_exec_floor() -> int:
            g_floor_i = int(global_floor)
            l_floor_i = int(local_floor)
            base = int(max(g_floor_i + l_floor_i, self.exec_min))
            if int(phase) == 1:
                r = float(np.clip(self.phase1_global_ratio, 0.0, 1.0))
                if g_floor_i > 0 and r <= 1e-12:
                    return int(remaining_i + 1)
                if l_floor_i > 0 and (1.0 - r) <= 1e-12:
                    return int(remaining_i + 1)
                by_g = int(np.ceil(g_floor_i / max(r, 1e-12) - 1e-9)) if g_floor_i > 0 else 0
                by_l = int(np.ceil(l_floor_i / max(1.0 - r, 1e-12) - 1e-9)) if l_floor_i > 0 else 0
                return int(max(base, by_g, by_l))
            r = float(np.clip(self.phase2_global_ratio, 0.0, 1.0))
            if g_floor_i > 0 and r <= 1e-12:
                return int(remaining_i + 1)
            if l_floor_i > 0 and (1.0 - r) <= 1e-12:
                return int(remaining_i + 1)
            by_g = int(np.ceil(g_floor_i / max(r, 1e-12) - 1e-9)) if g_floor_i > 0 else 0
            by_l = int(np.ceil(l_floor_i / max(1.0 - r, 1e-12) - 1e-9)) if l_floor_i > 0 else 0
            return int(max(base, by_g, by_l))

        def _np_band_penalty(n_exec_try: int) -> tuple[float, float]:
            p_raw = max(int(len(self.bounds)), 1)
            np_ratio = float(n_exec_try) / float(p_raw)
            if np_ratio < 5.0:
                return float(5.0 - np_ratio), np_ratio
            if np_ratio > 7.0:
                return float(np_ratio - 7.0), np_ratio
            return 0.0, np_ratio

        def _candidate_key(c: dict) -> tuple[float, ...]:
            stage_ge3_pen = 0.0 if int(c["stages_est"]) >= 3 else 1.0
            stage5_pen = float(abs(int(c["stages_est"]) - 5))
            np_pen = float(c["np_penalty"])
            np_center_pen = float(abs(float(c["np_ratio"]) - 6.0))
            return (
                stage_ge3_pen,
                stage5_pen,
                np_pen,
                np_center_pen,
                float(int(c["stages_est"])),
                float(int(c["n_exec"])),
            )

        remaining_i = int(max(remaining, 0))
        if remaining_i <= 0:
            return 0, {
                "alloc_source": "none",
                "global_floor": 0, "local_floor": 0, "exec_floor": 0,
                "stages_est": 0, "absorbed": False, "min_feasible": True,
            }

        minima = self._resolve_bucket_minima(phase=phase)
        global_floor = int(sum(minima.values()))
        local_floor = int(self.local_exec_min_anchors)
        exec_floor = int(max(global_floor + local_floor, self.exec_min))
        strict_floor = int(_strict_exec_floor())
        stages_remaining_cap = max(1, int(self.max_additional_stages) - int(round_idx))
        if remaining_i < exec_floor:
            return 0, {
                "alloc_source": "min_infeasible",
                "global_floor": int(global_floor),
                "local_floor": int(local_floor),
                "exec_floor": int(exec_floor),
                "exec_floor_strict": int(strict_floor),
                "stages_est": int(stages_remaining_cap),
                "absorbed": False,
                "bucket_minima_sum": int(global_floor),
                "min_feasible": False,
                "ratio_strict_selected": False,
                "n_exec_global_pref": 0,
                "n_exec_local_pref": 0,
            }

        strict_candidates: list[dict] = []
        relaxed_candidates: list[dict] = []
        for stages_est in range(1, int(stages_remaining_cap) + 1):
            n_base = int(np.ceil(float(remaining_i) / float(stages_est)))

            n_relaxed = int(max(n_base, exec_floor))
            if n_relaxed <= remaining_i:
                ok_relaxed, g_relaxed, l_relaxed, g_ratio_relaxed = _split_exec(
                    n_exec_try=n_relaxed,
                    strict_ratio=False,
                )
                if ok_relaxed:
                    np_pen, np_ratio = _np_band_penalty(n_relaxed)
                    relaxed_candidates.append(
                        {
                            "stages_est": int(stages_est),
                            "n_exec": int(n_relaxed),
                            "n_exec_global": int(g_relaxed),
                            "n_exec_local": int(l_relaxed),
                            "g_ratio": float(g_ratio_relaxed),
                            "np_penalty": float(np_pen),
                            "np_ratio": float(np_ratio),
                        }
                    )

            n_strict = int(max(n_base, strict_floor))
            if n_strict <= remaining_i:
                ok_strict, g_strict, l_strict, g_ratio_strict = _split_exec(
                    n_exec_try=n_strict,
                    strict_ratio=True,
                )
                if ok_strict:
                    np_pen, np_ratio = _np_band_penalty(n_strict)
                    strict_candidates.append(
                        {
                            "stages_est": int(stages_est),
                            "n_exec": int(n_strict),
                            "n_exec_global": int(g_strict),
                            "n_exec_local": int(l_strict),
                            "g_ratio": float(g_ratio_strict),
                            "np_penalty": float(np_pen),
                            "np_ratio": float(np_ratio),
                        }
                    )

        if not relaxed_candidates:
            return 0, {
                "alloc_source": "min_infeasible",
                "global_floor": int(global_floor),
                "local_floor": int(local_floor),
                "exec_floor": int(exec_floor),
                "exec_floor_strict": int(strict_floor),
                "stages_est": int(stages_remaining_cap),
                "absorbed": False,
                "bucket_minima_sum": int(global_floor),
                "min_feasible": False,
                "ratio_strict_selected": False,
                "n_exec_global_pref": 0,
                "n_exec_local_pref": 0,
            }

        use_strict = bool(len(strict_candidates) > 0)
        pool = strict_candidates if use_strict else relaxed_candidates
        chosen = sorted(pool, key=_candidate_key)[0]

        n_exec = int(chosen["n_exec"])
        stages_est = int(chosen["stages_est"])
        n_exec_global_pref = int(chosen["n_exec_global"])
        n_exec_local_pref = int(chosen["n_exec_local"])
        chosen_g_ratio = float(chosen["g_ratio"])

        # Tail absorb: if next remainder cannot satisfy min floor (or at hard stage cap), absorb now.
        remaining_after = remaining_i - n_exec
        is_last_allowed = (round_idx >= self.max_additional_stages - 1)
        absorbed = False
        if remaining_after > 0 and (remaining_after < exec_floor or is_last_allowed):
            n_exec = remaining_i
            ok_absorb, g_absorb, l_absorb, g_ratio_absorb = _split_exec(
                n_exec_try=n_exec,
                strict_ratio=use_strict,
            )
            if not ok_absorb:
                ok_absorb, g_absorb, l_absorb, g_ratio_absorb = _split_exec(
                    n_exec_try=n_exec,
                    strict_ratio=False,
                )
                use_strict = False
            if ok_absorb:
                n_exec_global_pref = int(g_absorb)
                n_exec_local_pref = int(l_absorb)
                chosen_g_ratio = float(g_ratio_absorb)
            absorbed = True

        return n_exec, {
            "alloc_source": "dynamic_priority_ratio_stage_np",
            "global_floor": int(global_floor),
            "local_floor": int(local_floor),
            "exec_floor": int(exec_floor),
            "exec_floor_strict": int(strict_floor),
            "stages_est": int(stages_est),
            "absorbed": bool(absorbed),
            "bucket_minima_sum": int(global_floor),
            "min_feasible": True,
            "ratio_strict_selected": bool(use_strict),
            "n_exec_global_pref": int(n_exec_global_pref),
            "n_exec_local_pref": int(n_exec_local_pref),
            "n_exec_global_ratio_pref": float(chosen_g_ratio),
        }

    def _compute_n_plan(self, *, remaining: int, round_idx: int) -> int:
        dim = len(self.bounds)
        base = self.plan_base_k * float(dim)
        base = min(base, self.plan_remaining_cap * float(remaining))
        n_plan = int(base * (self.plan_decay ** float(round_idx)))
        return max(n_plan, 1)

    def _compute_anchor_counts(self, *, round_idx: int) -> tuple[int, int, int]:
        base = float(self.local_anchor_max_base) * (self.local_anchor_max_decay ** float(round_idx))
        max_k = int(np.ceil(base))
        max_k = max(max_k, 1)

        best_k = int(np.ceil(max_k * self.local_anchor_best_ratio))
        best_k = min(max(best_k, 0), self.local_anchor_best_k, max_k)

        small_k = int(np.ceil(max_k * self.local_anchor_small_ratio))
        small_k = min(max(small_k, 0), self.local_anchor_small_k, max(max_k - best_k, 0))

        # ensure at least one slot for big clusters when possible
        if max_k >= 2 and (max_k - best_k - small_k) < 1:
            if best_k > 1:
                best_k -= 1
            elif small_k > 1:
                small_k -= 1

        return max_k, best_k, small_k

    def _fit_probe_gp(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[GaussianProcessRegressor | None, bool]:
        return fit_gp_with_fallback(
            X=X, y=y,
            include_white=self.local_gp_use_white_kernel,
            random_state=self.local_gp_seed,
        )

    def _dedup_candidates_norm(
        self,
        *,
        X: np.ndarray,
        spans: np.ndarray,
        dedup_tol: float,
    ) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.size == 0:
            return X
        kept: list[np.ndarray] = []
        for x in X:
            if not kept:
                kept.append(x)
                continue
            ref = np.vstack(kept)
            d = np.linalg.norm((ref - x.reshape(1, -1)) / spans.reshape(1, -1), axis=1)
            if bool(np.all(d > dedup_tol)):
                kept.append(x)
        return np.vstack(kept) if kept else np.empty((0, X.shape[1]), dtype=float)

    def _run_probe_stage(
        self,
        *,
        round_idx: int,
        objective_sense: str,
    ) -> bool:
        remaining = int(self.budget.remaining)
        if remaining <= 0:
            return True
        max_exec = min(int(self.probe_max_points), remaining)
        if max_exec <= 0:
            return True

        X_hist = np.asarray(self.store.X_success, dtype=float)
        y_hist = np.asarray(self.store.y_success, dtype=float).reshape(-1)
        if X_hist.ndim != 2 or X_hist.shape[0] < 2 or X_hist.shape[0] != y_hist.shape[0]:
            print(f"[ProbeStage] stage={round_idx} skipped: insufficient history")
            return True

        gp_model, fallback_used = self._fit_probe_gp(X=X_hist, y=y_hist)
        if gp_model is None:
            print(f"[ProbeStage] stage={round_idx} skipped: gp_fit_failed")
            return True

        sense = str(objective_sense).strip().lower()
        n_hist = X_hist.shape[0]
        n_top = min(max(2, int(np.ceil(n_hist * self.probe_top_ratio))), n_hist)
        if sense == "min":
            top_idx = np.argsort(y_hist)[:n_top]
            best_idx = int(np.argmin(y_hist))
        else:
            top_idx = np.argsort(-y_hist)[:n_top]
            best_idx = int(np.argmax(y_hist))
        top_points = X_hist[top_idx]
        center = np.mean(top_points, axis=0)
        stds = np.std(top_points, axis=0)

        spans, lbs = compute_spans_lbs(self.bounds)
        ubs = lbs + spans
        range_half = np.maximum(
            stds * float(self.probe_std_scale),
            spans * float(self.probe_min_range_ratio),
        )
        lb_probe = np.maximum(center - range_half, lbs)
        ub_probe = np.minimum(center + range_half, ubs)
        too_small = (ub_probe - lb_probe) < 1e-12
        if np.any(too_small):
            best_x = X_hist[best_idx]
            eps = 0.5 * spans * max(float(self.probe_perturb_ratio), 1e-3)
            lb_probe[too_small] = np.maximum(best_x[too_small] - eps[too_small], lbs[too_small])
            ub_probe[too_small] = np.minimum(best_x[too_small] + eps[too_small], ubs[too_small])

        best_x = X_hist[best_idx]
        best_y = float(y_hist[best_idx])
        probe_span = np.maximum(ub_probe - lb_probe, 1e-12)
        starts: list[np.ndarray] = [np.clip(best_x.copy(), lb_probe, ub_probe)]
        for _ in range(2):
            starts.append(
                np.clip(
                    best_x + self.rng.normal(0.0, float(self.probe_perturb_ratio) * probe_span),
                    lb_probe,
                    ub_probe,
                )
            )
        for _ in range(2):
            starts.append(self.rng.uniform(lb_probe, ub_probe))

        use_post_penalty = bool(self.post_use_penalty and self.has_post_constraints)
        post_lambda = float(self.post_lambda_current if use_post_penalty else 0.0)
        candidates: list[np.ndarray] = []
        for x0 in starts:
            x_opt = self._probe_acq.optimize(
                model=gp_model,
                y_best=best_y,
                lb=lb_probe,
                ub=ub_probe,
                starts=np.asarray(x0, dtype=float).reshape(1, -1),
                objective_sense=sense,
                acq_type="LCB",
                kappa=float(self.local_phase2_kappa),
                pre_feasible_fn=self._is_pre_feasible,
                post_feasible_prob_fn=self._post_feasible_prob if use_post_penalty else None,
                post_penalty_lambda=post_lambda,
            )
            if x_opt is not None:
                candidates.append(np.asarray(x_opt, dtype=float).reshape(-1))

        if not candidates:
            print(f"[ProbeStage] stage={round_idx} skipped: no_opt_candidates")
            return True

        min_exec = max(int(self.total_budget * self.exec_ratio), self.exec_min)
        n_exec = min(min_exec, remaining)
        n_plan_target = self._compute_n_plan(remaining=remaining, round_idx=round_idx)
        if n_plan_target < 3 * n_exec:
            n_plan_target = 3 * n_exec
        n_plan = self._plan_generation_count(target_count=n_plan_target)
        stage = round_idx + 1
        n_divisions = int(n_plan * (2 ** (stage // 2)))
        tol_probe = 0.5 * np.sqrt(len(self.bounds)) * (1.0 / max(n_divisions, 1))

        X_probe = self._dedup_candidates_norm(
            X=np.vstack(candidates),
            spans=spans,
            dedup_tol=tol_probe,
        )
        if X_probe.size == 0:
            print(f"[ProbeStage] stage={round_idx} skipped: dedup_internal_empty")
            return True

        keep_hist = self._dedup_mask_against_history_norm(
            X_new=X_probe,
            X_old=self.store.X,
            spans=spans,
            dedup_tol=tol_probe,
        )
        X_probe = X_probe[keep_hist]
        if X_probe.size == 0:
            print(f"[ProbeStage] stage={round_idx} skipped: dedup_history_empty")
            return True

        X_probe_f, c_probe_f, m_probe_f, _ = self._filter_by_constraints(
            X_probe,
            update_ratio=False,
        )
        if X_probe_f.size == 0:
            print(f"[ProbeStage] stage={round_idx} skipped: pre_constraint_empty")
            return True

        X_exec_probe = X_probe_f[:max_exec]
        c_exec_probe = c_probe_f[:max_exec]
        m_exec_probe = np.asarray(m_probe_f[:max_exec], dtype=float)
        print(
            "[ProbeStage] "
            f"stage={round_idx} candidates={len(candidates)} "
            f"kept={X_exec_probe.shape[0]} fallback_kernel={fallback_used}"
        )
        executed = self._execute_points(
            X_exec_probe,
            source="additional",
            round_idx=round_idx,
            exec_scope="probe",
            constraints_payloads=c_exec_probe,
            constraint_margins=m_exec_probe,
        )
        if not executed and self.failure_reason is None:
            self._set_failure("FAILED_BUDGET", stage=round_idx)
        return bool(executed)

    def _resolve_bucket_minima(self, *, phase: int) -> dict[str, int]:
        """phase/제약/crate 상태에 따라 bucket별 최소 개수 반환.

        Drop priority (유지 우선순위): top > margin > boundary_* > random
        (random은 min 보장 없음 — 부족 시 먼저 포기됨)

        Returns:
            dict mapping bucket name → minimum count. 활성 버킷만 포함.
        """
        if not self.global_bucket_minima_enabled:
            return {}
        has_c = bool(self.has_pre_constraints)
        is_hc = bool(
            has_c
            and float(self.constraint_rate_hat) < float(self.global_boundary_high_crate_threshold)
        )
        minima: dict[str, int] = {}
        if int(phase) == 2:
            # phase2: boundary는 phase2_disable_boundary_sampling=True로 이미 off
            # top + (constraint 있으면 margin) 만 보장
            if self.phase2_min_top > 0:
                minima["top"] = int(self.phase2_min_top)
            if has_c and self.phase2_min_margin > 0:
                minima["margin"] = int(self.phase2_min_margin)
            return minima
        # phase == 1
        if self.phase1_min_top > 0:
            minima["top"] = int(self.phase1_min_top)
        if has_c and self.phase1_min_margin > 0:
            minima["margin"] = int(self.phase1_min_margin)
        if has_c and is_hc:
            # high_crate: boundary_classic + boundary_cross 분리
            if self.phase1_min_boundary_classic_hc > 0:
                minima["boundary_classic"] = int(self.phase1_min_boundary_classic_hc)
            if self.phase1_min_boundary_cross_hc > 0:
                minima["boundary_cross"] = int(self.phase1_min_boundary_cross_hc)
        else:
            # normal constraint or no constraint: boundary_any 하나만
            if self.phase1_min_boundary_any > 0:
                minima["boundary_any"] = int(self.phase1_min_boundary_any)
        return minima

    def _bucket_minima_boundary_sum(self, minima: dict[str, int]) -> int:
        """boundary 계열 minima 합계 (planning pool floor 계산용)."""
        return int(
            minima.get("boundary_any", 0)
            + minima.get("boundary_classic", 0)
            + minima.get("boundary_cross", 0)
        )

    def _update_constraint_ratio(self, *, n_generated: int, n_feasible: int) -> None:
        if n_generated <= 0:
            return
        self._constraint_gen_total += int(n_generated)
        self._constraint_feas_total += int(n_feasible)
        raw_ratio = self._constraint_feas_total / max(self._constraint_gen_total, 1)
        self.constraint_rate_hat = clamp_ratio(raw_ratio, floor=self.plan_filter_r_floor)

    def _filter_by_constraints(
        self,
        X: np.ndarray,
        *,
        update_ratio: bool = False,
    ) -> tuple[np.ndarray, list[dict], np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        if X.size == 0:
            return (
                np.empty((0, len(self.bounds)), dtype=float),
                [],
                np.empty((0,), dtype=float),
                np.empty((0,), dtype=bool),
            )

        if not self.has_pre_constraints:
            mask = np.ones((X.shape[0],), dtype=bool)
            payloads = [{} for _ in range(X.shape[0])]
            margins = np.full((X.shape[0],), float("inf"), dtype=float)
        else:
            mask, payloads, margins = evaluate_constraints_batch(
                X=X,
                var_names=self.var_names,
                constraint_defs=self.constraint_defs,
                scope="pre",
            )

        if update_ratio:
            self._update_constraint_ratio(
                n_generated=X.shape[0],
                n_feasible=int(mask.sum()),
            )

        idx = np.where(mask)[0]
        if idx.size == 0:
            return (
                np.empty((0, X.shape[1]), dtype=float),
                [],
                np.empty((0,), dtype=float),
                mask,
            )

        return (
            X[idx],
            [payloads[i] for i in idx],
            margins[idx],
            mask,
        )

    def _plan_generation_count(self, *, target_count: int) -> int:
        target_count = max(int(target_count), 1)
        if not self.has_pre_constraints:
            return target_count
        r_used = max(float(self.constraint_rate_hat), float(self.plan_filter_r_floor))
        inv = max(int(np.ceil(1.0 / r_used)), 1)
        n_gen = int(np.ceil(target_count * self.plan_filter_safety * inv))
        return max(n_gen, target_count)

    def _update_post_rate(self, *, feasible: bool) -> None:
        if not self.has_post_constraints:
            return
        self._post_eval_total += 1
        if bool(feasible):
            self._post_feas_total += 1
        raw = self._post_feas_total / max(self._post_eval_total, 1)
        self.post_feasible_rate_hat = clamp_ratio(raw, floor=self.post_feasible_rate_floor)

    def _update_post_lambda(self) -> None:
        if not (self.post_use_penalty and self.has_post_constraints):
            self.post_lambda_current = 0.0
            return
        r = clamp_ratio(self.post_feasible_rate_hat, floor=self.post_feasible_rate_floor)
        scale = ((1.0 - r) / r) ** self.post_lambda_power
        lam = self.post_lambda_init * scale
        self.post_lambda_current = float(
            np.clip(lam, self.post_lambda_min, self.post_lambda_max)
        )

    def _fit_post_feasibility_model(self) -> None:
        self._post_model = None
        if not (self.post_use_penalty and self.has_post_constraints):
            return
        if self.store.size < self.post_clf_min_samples:
            return

        X = self.store.X
        if X.size == 0:
            return
        y = np.asarray([1 if r.feasible else 0 for r in self.store.rows], dtype=int)
        if y.shape[0] != X.shape[0]:
            return

        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        if n_pos < self.post_clf_min_pos or n_neg < self.post_clf_min_neg:
            return

        try:
            from xgboost import XGBClassifier

            clf = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.0,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
            )
            clf.fit(X.astype(float), y.astype(int))
            self._post_model = clf
        except Exception as exc:
            print(f"[AdditionalDOE] post feasibility model training skipped: {exc}")
            self._post_model = None

    def _predict_post_feasible_prob(self, X_candidate: np.ndarray) -> np.ndarray:
        X_candidate = np.asarray(X_candidate, dtype=float)
        if X_candidate.size == 0:
            return np.empty((0,), dtype=float)
        if self._post_model is None:
            return np.ones((X_candidate.shape[0],), dtype=float)
        try:
            proba = self._post_model.predict_proba(X_candidate)[:, 1]
            proba = np.asarray(proba, dtype=float)
            return np.clip(proba, 0.0, 1.0)
        except Exception as exc:
            print(f"[AdditionalDOE] post feasibility model prediction skipped: {exc}")
            return np.ones((X_candidate.shape[0],), dtype=float)

    def _is_pre_feasible(self, x: np.ndarray) -> bool:
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        if not self.has_pre_constraints:
            return True
        _payload, feasible, _margin = evaluate_constraints_point(
            x=x_arr,
            var_names=self.var_names,
            constraint_defs=self.constraint_defs,
            scope="pre",
        )
        return bool(feasible)

    def _post_feasible_prob(self, x: np.ndarray) -> float:
        x_arr = np.asarray(x, dtype=float).reshape(1, -1)
        if not (self.post_use_penalty and self.has_post_constraints):
            return 1.0
        p = self._predict_post_feasible_prob(x_arr)
        if p.size == 0:
            return 1.0
        return float(np.clip(p.reshape(-1)[0], 0.0, 1.0))

    def _make_penalized_models(self, *, models: list, objective_sense: str) -> list:
        if not (self.post_use_penalty and self.has_post_constraints and self._post_model is not None):
            return models
        if self.post_lambda_current <= 0.0:
            return models

        parent = self
        sense = str(objective_sense).strip().lower()

        class _PenaltyWrappedModel:
            def __init__(self, base_model):
                self._base_model = base_model

            def predict(self, X):
                X_arr = np.asarray(X, dtype=float)
                y = np.asarray(self._base_model.predict(X_arr), dtype=float).reshape(-1)
                p_feas = parent._predict_post_feasible_prob(X_arr)
                penalty = parent.post_lambda_current * (1.0 - p_feas)
                if sense == "min":
                    return y + penalty
                return y - penalty

        return [_PenaltyWrappedModel(m) for m in models]

    # ------------------------------------------------------------------
    # Unified dedup: spans=None → raw distance, spans 제공 → 정규화 거리
    # ------------------------------------------------------------------
    def _dedup_mask(
        self,
        *,
        X_new: np.ndarray,
        X_ref: np.ndarray,
        dedup_tol: float,
        spans: np.ndarray | None = None,
    ) -> np.ndarray:
        if X_new.size == 0:
            return np.empty((0,), dtype=bool)
        if X_ref.size == 0:
            return np.ones((X_new.shape[0],), dtype=bool)
        from scipy.spatial import cKDTree

        if spans is not None:
            safe_spans = np.where(spans > 1e-12, spans, 1.0)
            X_new_n = X_new / safe_spans
            X_ref_n = X_ref / safe_spans
        else:
            X_new_n = np.asarray(X_new, dtype=float)
            X_ref_n = np.asarray(X_ref, dtype=float)

        tree = cKDTree(X_ref_n)
        mask = np.ones(X_new_n.shape[0], dtype=bool)
        for i in range(X_new_n.shape[0]):
            if tree.query_ball_point(X_new_n[i], r=float(dedup_tol)):
                mask[i] = False
        return mask

    # backward-compatible wrappers
    def _dedup_mask_against_history_norm(self, *, X_new, X_old, spans, dedup_tol):
        return self._dedup_mask(X_new=X_new, X_ref=X_old, dedup_tol=dedup_tol, spans=spans)

    def _dedup_mask_against_history_raw(self, *, X_new, X_old, dedup_tol):
        return self._dedup_mask(X_new=X_new, X_ref=X_old, dedup_tol=dedup_tol, spans=None)

    def _dedup_mask_against_other_norm(self, *, X_new, X_ref, spans, dedup_tol):
        return self._dedup_mask(X_new=X_new, X_ref=X_ref, dedup_tol=dedup_tol, spans=spans)

    # -------------------------------------------------
    # Main entry
    # -------------------------------------------------

    def run(
        self,
        *,
        baseline: np.ndarray,
        problem_name: str,  # reserved for logs / future HPO keys
        base_seed: int,     # reserved for deterministic pipelines
        objective_sense: str,
    ):
        self.raw_objective_sense = str(objective_sense or "min").strip().lower()
        if self.raw_objective_sense not in {"min", "max"}:
            self.raw_objective_sense = "min"
        objective_sense = "min"
        # --------------------------------------------
        # Phase 0: Initial DOE (user-selected sampler)
        # --------------------------------------------
        n_init = max(int(self.total_budget * self.init_ratio), 1)
        n_baseline_reserved = 1 if (self.force_baseline and n_init > 0) else 0
        n_init_regular = n_init - n_baseline_reserved
        n_corner_target = int(round(float(n_init_regular) * self.initial_corner_ratio))
        n_corner_target = min(max(n_corner_target, 0), max(n_init_regular, 0))

        X_corner, corner_constraints, corner_margins = self._sample_initial_corners(
            n_target=n_corner_target,
            update_ratio=self.has_pre_constraints,
        )
        n_corner_ok = int(X_corner.shape[0])
        n_regular_target = max(n_init_regular - n_corner_ok, 0)

        if self.has_pre_constraints:
            if n_regular_target > 0:
                probe_size = max(1, int(np.ceil(n_regular_target * self.initial_probe_multiplier)))
                max_attempts = min(max(2, int(np.ceil(1.0 / max(self.plan_filter_r_floor, 1e-6)))), 50)
                pool_X: list[np.ndarray] = []
                pool_constraints: list[dict] = []
                pool_margins: list[float] = []

                for _attempt in range(max_attempts):
                    X_probe = self._sample_initial(n_samples=probe_size)

                    X_probe_f, constraints_f, margins_f, _ = self._filter_by_constraints(
                        X_probe,
                        update_ratio=True,
                    )
                    for i in range(X_probe_f.shape[0]):
                        pool_X.append(X_probe_f[i])
                        pool_constraints.append(constraints_f[i])
                        pool_margins.append(float(margins_f[i]))

                    if len(pool_X) >= n_regular_target:
                        break

                if len(pool_X) < n_regular_target:
                    self._set_failure("FAILED_FILTER_MIN", stage=0)
                    return self._export_results()

                selected_idx: list[int] = []
                selected_set: set[int] = set()

                def _take(candidates: list[int], k: int) -> list[int]:
                    if k <= 0:
                        return []
                    out: list[int] = []
                    for cid in candidates:
                        if cid in selected_set:
                            continue
                        out.append(int(cid))
                        if len(out) >= k:
                            break
                    return out

                n_margin_init = int(round(n_regular_target * self.global_margin_ratio))
                n_margin_init = min(max(n_margin_init, 0), n_regular_target)
                margin_subset_specs: list[tuple[tuple[str, ...], float]] = []
                if self.global_margin_subset_enabled and n_margin_init > 0:
                    pre_constraint_ids = self._extract_pre_constraint_ids(pool_constraints)
                    if len(pre_constraint_ids) >= 2:
                        margin_subset_specs = self._build_margin_subset_specs(pre_constraint_ids)

                if n_margin_init > 0:
                    if margin_subset_specs:
                        n_constraints = len(self._extract_pre_constraint_ids(pool_constraints))
                        edge_k = {1, int(n_constraints)}
                        subset_priority = sorted(
                            range(len(margin_subset_specs)),
                            key=lambda gi: (
                                0 if len(margin_subset_specs[gi][0]) in edge_k else 1,
                                -len(margin_subset_specs[gi][0]),
                                margin_subset_specs[gi][0],
                            ),
                        )
                        weights = [float(margin_subset_specs[gi][1]) for gi in range(len(margin_subset_specs))]
                        enforce_min_one = n_margin_init >= len(margin_subset_specs)
                        quotas = self._allocate_weighted_subset_quotas(
                            n_total=int(n_margin_init),
                            weights=weights,
                            enforce_min_one=bool(enforce_min_one),
                        )
                        candidate_margin_maps: list[dict[str, float]] = []
                        for payload in pool_constraints:
                            cur: dict[str, float] = {}
                            if isinstance(payload, dict):
                                for key, cinfo in payload.items():
                                    if not isinstance(cinfo, dict):
                                        continue
                                    scope = str(cinfo.get("scope", "pre")).strip().lower()
                                    if scope != "pre":
                                        continue
                                    cid = str(cinfo.get("id", key)).strip()
                                    mv = cinfo.get("margin")
                                    if cid:
                                        try:
                                            mval = float(mv)
                                            if np.isfinite(mval):
                                                cur[cid] = mval
                                        except Exception:
                                            continue
                            candidate_margin_maps.append(cur)

                        subset_orders: list[list[int]] = []
                        for subset_ids, _w in margin_subset_specs:
                            ranked: list[tuple[float, int]] = []
                            for idx in range(len(pool_X)):
                                m_map = candidate_margin_maps[idx] if idx < len(candidate_margin_maps) else {}
                                vals = [m_map.get(cid) for cid in subset_ids]
                                if any(v is None for v in vals):
                                    continue
                                s_margin = float(np.min(np.asarray(vals, dtype=float)))
                                ranked.append((s_margin, int(idx)))
                            ranked.sort(key=lambda t: (t[0], t[1]))
                            subset_orders.append([idx for _, idx in ranked])

                        allocated = [0 for _ in quotas]
                        if enforce_min_one:
                            for gi in subset_priority:
                                picked = _take(subset_orders[gi], 1)
                                if picked:
                                    selected_idx.extend(picked)
                                    selected_set.update(picked)
                                    allocated[gi] += len(picked)

                        for gi in subset_priority:
                            need = max(int(quotas[gi]) - int(allocated[gi]), 0)
                            if need <= 0:
                                continue
                            picked = _take(subset_orders[gi], need)
                            if picked:
                                selected_idx.extend(picked)
                                selected_set.update(picked)
                                allocated[gi] += len(picked)
                    else:
                        margin_order = np.argsort(np.asarray(pool_margins, dtype=float)).astype(int).tolist()
                        picked = _take(margin_order, n_margin_init)
                        selected_idx.extend(picked)
                        selected_set.update(picked)

                remaining_slots = max(int(n_regular_target - len(selected_idx)), 0)
                if remaining_slots > 0:
                    remain_candidates = [idx for idx in range(len(pool_X)) if idx not in selected_set]
                    if remain_candidates:
                        n_fill = min(remaining_slots, len(remain_candidates))
                        fill = self.rng.choice(
                            np.asarray(remain_candidates, dtype=int),
                            size=n_fill,
                            replace=False,
                        ).astype(int).tolist()
                        selected_idx.extend(fill)
                        selected_set.update(fill)

                if len(selected_idx) < n_regular_target:
                    fallback_candidates = [idx for idx in range(len(pool_X)) if idx not in selected_set]
                    selected_idx.extend(fallback_candidates[: max(n_regular_target - len(selected_idx), 0)])

                pick_idx = np.asarray(selected_idx[:n_regular_target], dtype=int)
                X_regular = np.vstack([pool_X[i].reshape(1, -1) for i in pick_idx]).astype(float)
                regular_constraints = [pool_constraints[i] for i in pick_idx]
                regular_margins = [float(pool_margins[i]) for i in pick_idx]
            else:
                X_regular = np.empty((0, len(self.bounds)), dtype=float)
                regular_constraints = []
                regular_margins = []
        else:
            self.constraint_rate_hat = 1.0
            if n_regular_target > 0:
                X_regular = self._sample_initial(n_samples=n_regular_target)
            else:
                X_regular = np.empty((0, len(self.bounds)), dtype=float)
            regular_constraints = [{} for _ in range(X_regular.shape[0])]
            regular_margins = [float("inf")] * X_regular.shape[0]

        X_init_parts: list[np.ndarray] = []
        init_constraints: list[dict] = []
        init_margin_list: list[float] = []

        if n_baseline_reserved > 0:
            baseline_x = np.asarray(baseline, dtype=float).reshape(1, -1)
            if self.has_pre_constraints:
                baseline_constraints, _, baseline_margin = evaluate_constraints_point(
                    x=np.asarray(baseline, dtype=float),
                    var_names=self.var_names,
                    constraint_defs=self.constraint_defs,
                    scope="pre",
                )
            else:
                baseline_constraints = {}
                baseline_margin = float("inf")
            X_init_parts.append(baseline_x)
            init_constraints.append(baseline_constraints)
            init_margin_list.append(float(baseline_margin))

        if X_corner.shape[0] > 0:
            X_init_parts.append(np.asarray(X_corner, dtype=float))
            init_constraints.extend(corner_constraints)
            init_margin_list.extend(float(v) for v in corner_margins)

        if X_regular.shape[0] > 0:
            X_init_parts.append(np.asarray(X_regular, dtype=float))
            init_constraints.extend(regular_constraints)
            init_margin_list.extend(float(v) for v in regular_margins)

        X_init = np.vstack(X_init_parts) if X_init_parts else np.empty((0, len(self.bounds)), dtype=float)
        init_margins = np.asarray(init_margin_list, dtype=float)

        executed = self._execute_points(
            X_init,
            source="initial",
            round_idx=None,
            exec_scope="initial",
            constraints_payloads=init_constraints,
            constraint_margins=init_margins,
        )
        if not executed:
            if self.failure_reason is None:
                self._set_failure("FAILED_BUDGET", stage=0)
            return self._export_results()

        # Guardrail: if CAE success ratio is too low, stop before surrogate build.
        n_success, n_total, success_ratio = self._success_stats()
        if success_ratio < float(self.success_rate_floor):
            self._set_failure("FAILED_SUCCESS_RATE_MIN", stage=0)
            print(
                "[AdditionalDOE] "
                f"success={n_success}/{n_total} "
                f"(ratio={success_ratio:.4f}) < floor={self.success_rate_floor:.4f}. "
                "Likely CAE runtime/output instability "
                "(explicit success=False, invalid payload, non-numeric/non-finite objective)."
            )
            return self._export_results()

        self._fit_post_feasibility_model()
        self._update_post_lambda()
        self._initial_budget_used = int(self.budget.used)
        self._additional_budget_total = int(max(self.total_budget - self._initial_budget_used, 0))
        print(
            f"[AdditionalDOE] dynamic_priority_ratio_stage_np: "
            f"additional_budget={self._additional_budget_total} "
            f"max_stages={self.max_additional_stages} "
            f"bucket_minima_enabled={self.global_bucket_minima_enabled}"
        )

        # --------------------------------------------
        # Additional DOE loop
        # --------------------------------------------
        round_idx = 0
        phase = 1
        self._probe_stage_done = False
        self._ever_phase2 = False
        self._collapse_diversity_pending = False
        self._gate_eval_count = 0
        self._gate1_ema = None
        self._gate2_ema = None

        while not self.budget.exhausted() and (
            self.budget_policy == "consume_all" or round_idx < self.max_additional_stages
        ):
            remaining = self.budget.remaining
            if remaining <= 0:
                break

            self._maybe_run_hpo(problem_name=problem_name, base_seed=base_seed)
            self._fit_post_feasibility_model()
            self._update_post_lambda()
            if phase == 2:
                self._ever_phase2 = True
            used_budget_ratio_loop = float(self.budget.used) / float(max(self.total_budget, 1))
            probe_gate_ready = bool(
                used_budget_ratio_loop >= self.phase2_min_used_budget_ratio
            ) if self.budget_policy == "consume_all" else bool(round_idx + 1 >= self.min_additional_rounds)
            probe_ready = bool(
                self.probe_stage_enabled
                and (not self._probe_stage_done)
                and self._ever_phase2
                and probe_gate_ready
            )
            if probe_ready:
                probe_ok = self._run_probe_stage(
                    round_idx=round_idx,
                    objective_sense=objective_sense,
                )
                self._probe_stage_done = True
                if not probe_ok:
                    break
                continue
            self.post_policy_log.append(
                {
                    "stage": int(round_idx),
                    "post_feasible_rate_hat": float(self.post_feasible_rate_hat),
                    "post_lambda": float(self.post_lambda_current),
                    "post_model_active": bool(self._post_model is not None),
                }
            )
            print(
                "[PostPolicy] "
                f"stage={round_idx} rate_hat={self.post_feasible_rate_hat:.4f} "
                f"lambda={self.post_lambda_current:.4f} model_active={self._post_model is not None}"
            )

            # -----------------------------
            # 2) Build X_plan (once per stage)
            # -----------------------------
            _bm_minima_stage = self._resolve_bucket_minima(phase=phase) if self.global_bucket_minima_enabled else {}
            n_exec, alloc_info = self._compute_n_exec_for_stage(
                round_idx=round_idx,
                remaining=remaining,
                phase=phase,
            )
            if not bool(alloc_info.get("min_feasible", True)):
                self._set_failure("FAILED_MIN_EXEC_REQUIREMENT", stage=round_idx)
                break
            _bm_n_exec_floor = int(alloc_info.get("exec_floor", 0))
            _bm_minima_boundary = self._bucket_minima_boundary_sum(_bm_minima_stage)

            # -----------------------------
            # 1) Build surrogates (bundle)
            # -----------------------------
            df = self.store.to_dataframe()
            bundle = self.surrogate_factory.build(
                df=df,
                round_idx=round_idx,
                exec_size=n_exec,
            )

            gate1_models = bundle["gate1"]
            gate2_models = bundle["gate2"]
            gate1_eval_models = self._make_penalized_models(
                models=gate1_models,
                objective_sense=objective_sense,
            )
            gate2_eval_models = self._make_penalized_models(
                models=gate2_models,
                objective_sense=objective_sense,
            )

            n_plan_target = self._compute_n_plan(remaining=remaining, round_idx=round_idx)
            if n_plan_target < 3 * n_exec:
                n_plan_target = 3 * n_exec
            n_plan = self._plan_generation_count(target_count=n_plan_target)
            stage = round_idx + 1
            base_divisions = n_plan
            n_divisions = int(base_divisions * (2 ** (stage // 2)))
            # Tighten dedup radius to keep more local samples.
            tol = 0.5 * np.sqrt(len(self.bounds)) * (1.0 / n_divisions)
            diversity_boost_active = bool(self._collapse_diversity_pending)
            collapse_detected = False
            collapse_trigger_span = None
            collapse_anchor_streak = int(self._anchor_spread_zero_streak)
            diversity_quota = 0
            diversity_sample_count = 0
            diversity_hit_rate = None

            ratio_global = self.phase1_global_ratio if phase == 1 else self.phase2_global_ratio
            n_global = int(round(n_plan * ratio_global))
            n_global = min(max(n_global, 0), n_plan)
            n_local = n_plan - n_global

            boundary_ratio = max(0.0, min(1.0, self.global_boundary_ratio))
            # constraint 문제에서 boundary 예산 확대
            if self.has_pre_constraints:
                boundary_ratio *= float(self.global_boundary_constraint_boost)
                boundary_ratio = min(boundary_ratio, 1.0)
            boundary_ratio *= float(0.9 ** round_idx)
            # zero_streak — 원래 동작(수렴 신호) 그대로 유지. minima와 orthogonal.
            if self._anchor_spread_zero_streak >= self.stop_anchor_spread_streak and (not diversity_boost_active):
                boundary_ratio = 0.0
            if diversity_boost_active:
                boundary_ratio = max(boundary_ratio, float(self.diversity_boundary_floor_ratio))
            if int(phase) == 2 and self.phase2_disable_boundary_sampling:
                # Phase-2에서는 boundary 버킷을 비활성화하고 global 예산을
                # margin/top/random 쪽으로만 사용한다.
                boundary_ratio = 0.0

            n_boundary = int(round(n_global * boundary_ratio))
            if n_boundary > 0 and n_global >= 2 and n_boundary < 2:
                n_boundary = 2
            # Bucket minima — 매 stage 독립 floor (boundary_ratio/zero_streak과 무관)
            # boundary 후보 candidate 존재 여부는 generation 후 exec 단계에서 graceful skip
            _bm_minima_boundary = self._bucket_minima_boundary_sum(_bm_minima_stage)
            if (
                self.global_bucket_minima_enabled
                and _bm_minima_boundary > 0
                and int(phase) == 1
                and n_global >= 2
            ):
                _plan_required = int(_bm_minima_boundary) * int(self.global_bucket_plan_multiplier)
                _plan_cap = max(1, int(n_global - 1))
                _plan_target = int(min(_plan_required, _plan_cap))
                if n_boundary < _plan_target:
                    n_boundary = int(_plan_target)
                    # zero_streak으로 ratio=0이어도 minima floor가 우선
                    # 최소 ratio 복원 (generation 코드가 ratio>0을 요구할 수 있음)
                    if boundary_ratio == 0.0:
                        boundary_ratio = float(_plan_target) / float(max(n_global, 1))
            spans, _ = compute_spans_lbs(self.bounds)
            offset = spans * tol

            boundary_flags = []
            if n_boundary > 0:
                # constraint 문제: 절반을 margin+boundary 교차용으로 예약
                _cross_ratio = float(self.global_boundary_margin_cross_ratio) if self.has_pre_constraints else 0.0
                n_margin_cross = int(round(n_boundary * _cross_ratio))
                n_classic_boundary = n_boundary - n_margin_cross

                corner_ratio = max(0.0, min(1.0, self.global_boundary_corner_ratio))
                corner_ratio *= float(0.9 ** round_idx)
                n_corner = int(round(n_classic_boundary * corner_ratio))
                n_partial = n_classic_boundary - n_corner
                if n_classic_boundary >= 2:
                    n_corner = min(max(n_corner, 1), n_classic_boundary - 1)
                    n_partial = n_classic_boundary - n_corner

                boundary_candidates = sample_boundary_corners(self.bounds, offset=offset)
                if n_corner > 0:
                    if boundary_candidates.shape[0] <= n_corner:
                        X_corner = boundary_candidates
                    else:
                        idx = self.rng.choice(boundary_candidates.shape[0], size=n_corner, replace=False)
                        X_corner = boundary_candidates[idx]
                else:
                    X_corner = np.empty((0, len(self.bounds)), dtype=float)

                if n_partial > 0:
                    X_base = self.plan_builder.build(n_plan=max(n_partial, 1), n_divisions=n_divisions)
                    boundary_dims_k = max(1, len(self.bounds) - 2)
                    X_partial = sample_boundary_partial(
                        self.bounds,
                        offset=offset,
                        base_points=X_base,
                        n_samples=n_partial,
                        n_boundary_dims=boundary_dims_k,
                        rng=self.rng,
                    )
                else:
                    X_partial = np.empty((0, len(self.bounds)), dtype=float)

                # margin+boundary 교차 bucket: LHC 기반 partial boundary 추가 생성
                # (constraint filter 후 margin이 작은 것만 _select_global_exec에서 선별)
                if n_margin_cross > 0:
                    X_mc_base = self.plan_builder.build(n_plan=max(n_margin_cross * 3, 6), n_divisions=n_divisions)
                    boundary_dims_k_mc = max(1, min(2, len(self.bounds) - 1))
                    X_margin_cross = sample_boundary_partial(
                        self.bounds,
                        offset=offset,
                        base_points=X_mc_base,
                        n_samples=max(n_margin_cross * 3, 6),
                        n_boundary_dims=boundary_dims_k_mc,
                        rng=self.rng,
                    )
                else:
                    X_margin_cross = np.empty((0, len(self.bounds)), dtype=float)

                parts = [p for p in [X_corner, X_partial, X_margin_cross] if p.size > 0]
                Xg_boundary = np.vstack(parts) if parts else np.empty((0, len(self.bounds)), dtype=float)
                boundary_flags = [True] * Xg_boundary.shape[0]
            else:
                Xg_boundary = np.empty((0, len(self.bounds)), dtype=float)

            n_lhc = max(n_global - Xg_boundary.shape[0], 0)
            Xg_lhc = self.plan_builder.build(n_plan=n_lhc, n_divisions=n_divisions) if n_lhc > 0 else np.empty((0, len(self.bounds)), dtype=float)

            Xg_boundary_f, constraints_boundary_f, margins_boundary_f, _ = self._filter_by_constraints(
                Xg_boundary,
                update_ratio=False,
            )
            Xg_lhc_f, constraints_lhc_f, margins_lhc_f, _ = self._filter_by_constraints(
                Xg_lhc,
                update_ratio=False,
            )
            Xg = np.vstack([Xg_boundary_f, Xg_lhc_f]) if Xg_boundary_f.size or Xg_lhc_f.size else np.empty((0, len(self.bounds)), dtype=float)
            constraints_g = constraints_boundary_f + constraints_lhc_f
            margins_g = np.concatenate([margins_boundary_f, margins_lhc_f]) if (margins_boundary_f.size or margins_lhc_f.size) else np.empty((0,), dtype=float)
            boundary_flags = np.concatenate(
                [
                    np.ones((Xg_boundary_f.shape[0],), dtype=bool),
                    np.zeros((Xg_lhc_f.shape[0],), dtype=bool),
                ]
            ) if (Xg_boundary_f.shape[0] + Xg_lhc_f.shape[0]) > 0 else np.zeros((0,), dtype=bool)

            if n_local > 0:
                # --- build local plan
                X_exec_success = self.store.X_success
                y_exec_success = self.store.y_success
                if phase == 1:
                    X_candidate = Xg
                    y_candidate = None
                else:
                    if X_exec_success.shape[0] > 0:
                        X_candidate = X_exec_success
                        y_candidate = y_exec_success
                    else:
                        X_candidate = Xg
                        y_candidate = None

                if X_candidate.shape[0] > 0:
                    radius_ratio = (
                        self.local_radius_ratio_phase1
                        if phase == 1
                        else self.local_radius_ratio_phase2
                    )
                    anchor_max, best_k, small_k = self._compute_anchor_counts(round_idx=round_idx)
                    debug_best_x = None
                    if phase == 2 and y_candidate is not None and y_candidate.size == X_candidate.shape[0]:
                        if objective_sense == "min":
                            best_idx = int(np.argmin(y_candidate))
                        else:
                            best_idx = int(np.argmax(y_candidate))
                        debug_best_x = X_candidate[best_idx]
                    Xl_raw, metrics, local_aux = self.local_sampler.build_local_plan(
                        models=gate1_models,
                        X_candidate=X_candidate,
                        y_candidate=y_candidate,
                        n_samples=n_local,
                        objective_sense=objective_sense,
                        phase=phase,
                        stage_eps=tol,
                        use_post_penalty=bool(self.post_use_penalty and self.has_post_constraints),
                        post_penalty_lambda=float(self.post_lambda_current),
                        local_radius_ratio=radius_ratio,
                        debug_best_x=debug_best_x,
                        anchor_max=anchor_max,
                        best_k=best_k,
                        small_k=small_k,
                    )
                    self._log_local_metrics(stage=round_idx, metrics=metrics)
                    self._update_convergence_streak(metrics)
                    collapse_trigger_span = (
                        float(metrics.get("local_span_ratio_mean"))
                        if metrics.get("local_span_ratio_mean") is not None
                        else None
                    )
                    collapse_anchor_streak = int(self._anchor_spread_zero_streak)
                    collapse_detected = self._is_local_collapse(
                        stage=stage,
                        span_ratio=collapse_trigger_span,
                        anchor_streak=collapse_anchor_streak,
                    )
                    Xl_anchors = np.asarray(local_aux.get("anchors", np.empty((0, Xg.shape[1]))), dtype=float)
                    Xl_anchor_ids = np.asarray(local_aux.get("anchor_ids", np.array([], dtype=int)))
                    tol_local = float(local_aux.get("tol_local", 0.0))
                    Xl, constraints_l, margins_l, mask_l = self._filter_by_constraints(
                        Xl_raw,
                        update_ratio=False,
                    )
                    Xl_anchor_ids = Xl_anchor_ids[mask_l] if Xl_anchor_ids.size == Xl_raw.shape[0] else np.array([], dtype=int)
                else:
                    Xl = np.empty((0, Xg.shape[1]), dtype=float)
                    constraints_l = []
                    margins_l = np.empty((0,), dtype=float)
                    Xl_anchors = np.empty((0, Xg.shape[1]), dtype=float)
                    Xl_anchor_ids = np.array([], dtype=int)
                    tol_local = 0.0
                    collapse_trigger_span = None
                    collapse_anchor_streak = int(self._anchor_spread_zero_streak)
                    collapse_detected = False

                X_plan = np.vstack([Xg, Xl]) if Xl.size else Xg
            else:
                Xl = np.empty((0, Xg.shape[1]), dtype=float)
                constraints_l = []
                margins_l = np.empty((0,), dtype=float)
                Xl_anchors = np.empty((0, Xg.shape[1]), dtype=float)
                Xl_anchor_ids = np.array([], dtype=int)
                tol_local = 0.0
                X_plan = Xg
                collapse_trigger_span = None
                collapse_anchor_streak = int(self._anchor_spread_zero_streak)
                collapse_detected = False

            # -----------------------------
            # 3) Gate evaluation (on X_plan, once per stage)
            # -----------------------------
            p_dim = max(int(len(self.bounds)), 1)
            usable_n = int(self._usable_sample_count())
            usable_np_ratio = float(usable_n) / float(p_dim)
            used_budget_ratio = float(self.budget.used) / float(max(self.total_budget, 1))
            budget_np_ratio = float(self.total_budget) / float(p_dim)
            phase2_np_gate = float(
                min(
                    self.phase2_min_usable_np_ratio,
                    self.phase2_np_ratio_cap_scale * budget_np_ratio,
                )
            )
            gate1_score = None  # raw score (backward-compatible key)
            gate2_score = None  # raw score (backward-compatible key)
            gate1_score_raw = None
            gate2_score_raw = None
            gate1_score_ema = None
            gate2_score_ema = None
            gate1_score_used = None
            gate2_score_used = None
            gate1_passed = None
            gate2_passed = None
            gate1_pass_smoothed = None
            gate2_pass_smoothed = None
            gate_stop_raw = None
            gate_stop_smoothed = None
            gate_decision_source = "raw"
            phase2_decision_source = "raw"
            gate_evaluated = False
            gate_eval_skipped_reason = None
            can_stop_by_data = None
            can_stop_by_round = None
            can_stop_by_budget = None
            converged_for_stop = None
            can_enter_phase2_by_data = None
            can_enter_phase2_by_budget = None
            can_enter_phase2_by_gate1 = None
            can_enter_phase2_by_gate2 = None
            gate2_phase_threshold = None
            phase2_gate1_rescue_triggered = False
            phase2_gate1_rescue_band_lower = None
            phase2_gate1_rescue_gate2_min_used = None
            if n_exec < self.exec_min:
                print(
                    f"[AdditionalDOE] stage={round_idx} n_exec={n_exec} < exec_min={self.exec_min}, "
                    "skip gate evaluation and run tail budget only."
                )
                gate_eval_skipped_reason = "exec_below_min"
                gate_stop = False
                next_phase = phase
            else:
                if X_plan.shape[0] < 3 * n_exec:
                    self._set_failure("FAILED_INSUFFICIENT_PLAN", stage=round_idx)
                    break
                g2 = self.gate2.evaluate(models=gate2_eval_models, X_candidate=X_plan)
                g1 = self.gate1.evaluate(
                    models=gate1_eval_models,
                    X_candidate=X_plan,
                    bounds=self.bounds,
                    eps=0.5 * (1.0 / n_divisions),
                )
                print(
                    f"[Gate1] stage={round_idx} score={g1.get('score')} passed={g1.get('passed')}"
                )
                print(
                    f"[Gate2] stage={round_idx} score={g2.get('score')} passed={g2.get('passed')}"
                )

                decision = self.gate_manager.evaluate(
                    gate1_result=g1,
                    gate2_result=g2,
                )
                gate_evaluated = True
                gate1_score_raw = float(g1.get("score")) if g1.get("score") is not None else None
                gate2_score_raw = float(g2.get("score")) if g2.get("score") is not None else None
                gate1_score = gate1_score_raw
                gate2_score = gate2_score_raw
                gate1_passed = bool(decision.get("gate1_passed", False))
                gate2_passed = bool(decision.get("gate2_passed", False))
                self._gate_eval_count += 1
                gate1_score_ema = self._update_ema(
                    prev=self._gate1_ema,
                    value=gate1_score_raw,
                    alpha=self.gate_ema_alpha,
                )
                gate2_score_ema = self._update_ema(
                    prev=self._gate2_ema,
                    value=gate2_score_raw,
                    alpha=self.gate_ema_alpha,
                )
                self._gate1_ema = gate1_score_ema
                self._gate2_ema = gate2_score_ema
                gate1_pass_threshold = float(getattr(self.gate1, "pass_ratio", 0.5))
                gate2_pass_threshold = float(getattr(self.gate2, "ratio_threshold", 0.8))
                gate1_pass_smoothed = bool(
                    gate1_score_ema is not None and gate1_score_ema >= gate1_pass_threshold
                )
                gate2_pass_smoothed = bool(
                    gate2_score_ema is not None and gate2_score_ema >= gate2_pass_threshold
                )
                gate_stop_smoothed = bool(gate1_pass_smoothed and gate2_pass_smoothed)
                use_ema = bool(
                    self.gate_smoothing_enabled
                    and self._gate_eval_count > self.gate_ema_warmup_stages
                )

                # Gate stop (strict) is used only in adaptive_early_stop mode.
                gate_stop_raw = bool(decision.get("gate_stop", False))
                stop_np_gate = float(max(self.stop_min_usable_np_ratio, self.early_stop_min_usable_np_ratio))
                can_stop_by_data = bool(usable_np_ratio >= stop_np_gate)
                can_stop_by_round = bool(round_idx + 1 >= self.min_additional_rounds)
                can_stop_by_budget = bool(used_budget_ratio >= self.early_stop_min_used_budget_ratio)
                converged_for_stop = bool(self._should_stop_by_convergence())
                gate_stop_signal = gate_stop_raw
                if use_ema and self.gate_smoothing_use_for_stop:
                    gate_stop_signal = bool(gate_stop_smoothed)
                    gate_decision_source = "ema"
                else:
                    gate_decision_source = "raw"

                if self.budget_policy == "adaptive_early_stop":
                    gate_stop = bool(
                        gate_stop_signal
                        and can_stop_by_round
                        and can_stop_by_data
                        and can_stop_by_budget
                        and converged_for_stop
                    )
                else:
                    gate_stop = False
                if gate_stop:
                    print(f"[AdditionalDOE] gate_stop=True, executing final batch (stage={round_idx})")

                can_enter_phase2_by_data = bool(usable_np_ratio > phase2_np_gate)
                # DSE-C: 고제약(crate_hat < high_crate_threshold) 시 phase2 진입 최소 예산 상향
                _eff_phase2_min_used_budget = (
                    self.phase2_min_used_budget_ratio_high_crate
                    if (
                        self.has_pre_constraints
                        and float(self.constraint_rate_hat) < self.phase2_high_crate_threshold
                    )
                    else self.phase2_min_used_budget_ratio
                )
                can_enter_phase2_by_budget = bool(used_budget_ratio >= _eff_phase2_min_used_budget)
                gate1_score_for_phase2 = gate1_score_raw
                gate2_score_for_phase2 = gate2_score_raw
                if use_ema and self.gate_smoothing_use_for_phase2:
                    gate1_score_for_phase2 = gate1_score_ema
                    gate2_score_for_phase2 = gate2_score_ema
                    phase2_decision_source = "ema"
                else:
                    phase2_decision_source = "raw"
                gate1_score_used = gate1_score_for_phase2
                gate2_score_used = gate2_score_for_phase2
                can_enter_phase2_by_gate1 = bool(
                    gate1_score_for_phase2 is not None
                    and gate1_score_for_phase2 >= self.phase2_gate1_score_min
                )
                if self.phase2_gate1_rescue_enabled:
                    phase2_gate1_rescue_band_lower = float(
                        max(0.0, self.phase2_gate1_score_min - self.phase2_gate1_rescue_band)
                    )
                    phase2_gate1_rescue_gate2_min_used = float(self.phase2_gate1_rescue_gate2_min)
                if (
                    (not can_enter_phase2_by_gate1)
                    and self.phase2_gate1_rescue_enabled
                    and gate1_score_for_phase2 is not None
                    and gate2_score_for_phase2 is not None
                ):
                    rescue_in_band = bool(
                        phase2_gate1_rescue_band_lower is not None
                        and phase2_gate1_rescue_band_lower
                        <= float(gate1_score_for_phase2)
                        < float(self.phase2_gate1_score_min)
                    )
                    rescue_gate2_strong = bool(
                        float(gate2_score_for_phase2) >= float(self.phase2_gate1_rescue_gate2_min)
                    )
                    if rescue_in_band and rescue_gate2_strong:
                        can_enter_phase2_by_gate1 = True
                        phase2_gate1_rescue_triggered = True
                        phase2_decision_source = f"{phase2_decision_source}_rescue"
                        print(
                            "[AdditionalDOE] phase2_gate1_rescue "
                            f"stage={round_idx} source={phase2_decision_source} "
                            f"g1={float(gate1_score_for_phase2):.3f} "
                            f"band=[{phase2_gate1_rescue_band_lower:.3f}, {float(self.phase2_gate1_score_min):.3f}) "
                            f"g2={float(gate2_score_for_phase2):.3f} "
                            f"g2_min={float(self.phase2_gate1_rescue_gate2_min):.3f}"
                        )
                gate2_phase_threshold = float(
                    self.phase2_gate2_score_sticky_min if int(phase) == 2 else self.phase2_gate2_score_min
                )
                can_enter_phase2_by_gate2 = bool(
                    gate2_score_for_phase2 is not None
                    and gate2_score_for_phase2 >= gate2_phase_threshold
                )
                next_phase = 2 if (
                    can_enter_phase2_by_gate1
                    and can_enter_phase2_by_gate2
                    and can_enter_phase2_by_data
                    and can_enter_phase2_by_budget
                ) else 1
                if next_phase == 2:
                    self._ever_phase2 = True

            # -----------------------------
            # 4) Filter X_plan then select X_exec
            # -----------------------------
            _global_floor = int(alloc_info.get("global_floor", 0))
            _local_floor = int(alloc_info.get("local_floor", 0))
            n_exec_global = int(alloc_info.get("n_exec_global_pref", 0))
            n_exec_local = int(alloc_info.get("n_exec_local_pref", int(max(n_exec - n_exec_global, 0))))
            if n_exec_global < 0:
                n_exec_global = 0
            if n_exec_local < 0:
                n_exec_local = 0
            if (n_exec_global + n_exec_local) != int(n_exec):
                n_exec_global = int(min(max(n_exec_global, 0), n_exec))
                n_exec_local = int(max(n_exec - n_exec_global, 0))
            # hard floor (best-effort)
            if n_exec >= (_global_floor + _local_floor):
                if n_exec_global < _global_floor:
                    n_exec_global = int(_global_floor)
                    n_exec_local = int(max(n_exec - n_exec_global, 0))
                if n_exec_local < _local_floor:
                    n_exec_local = int(_local_floor)
                    n_exec_global = int(max(n_exec - n_exec_local, 0))
            if gate_stop:
                # Final stage policy: exploit locally as much as possible.
                n_exec_global = 0
                n_exec_local = n_exec
            if diversity_boost_active and (not gate_stop):
                diversity_quota = self._resolve_diversity_quota(n_exec=n_exec)
                if diversity_quota > 0:
                    max_shift = max(n_exec_local - 1, 0)
                    shifted = min(diversity_quota, max_shift)
                    n_exec_local = int(max(n_exec_local - shifted, 0))
                    n_exec_global = int(min(n_exec_global + shifted, n_exec))
                    diversity_quota = int(shifted)
                if diversity_quota > 0:
                    print(
                        "[AdditionalDOE] diversity_injection "
                        f"stage={round_idx} quota={diversity_quota} "
                        f"exec_global/local={n_exec_global}/{n_exec_local}"
                    )

            print(
                f"[AdditionalDOE] stage={round_idx} phase={phase} "
                f"n_plan={n_plan} n_exec={n_exec} "
                f"global/local={n_global}/{n_local} "
                f"exec_global/local={n_exec_global}/{n_exec_local} "
                f"used_ratio={used_budget_ratio:.3f} policy={self.budget_policy} "
                f"alloc={alloc_info.get('alloc_source')} "
                f"floor(g/l/t)={alloc_info.get('global_floor')}/{alloc_info.get('local_floor')}/{alloc_info.get('exec_floor')} "
                f"stages_est={alloc_info.get('stages_est')} "
                f"absorbed={alloc_info.get('absorbed')}"
            )

            spans, lbs = compute_spans_lbs(self.bounds)
            mask_g = self._dedup_mask_against_history_norm(
                X_new=Xg,
                X_old=self.store.X,
                spans=spans,
                dedup_tol=tol,
            )
            Xg_f = Xg[mask_g]
            constraints_g_f = [constraints_g[i] for i in np.where(mask_g)[0]]
            margins_g_f = margins_g[mask_g] if margins_g.size == Xg.shape[0] else np.empty((Xg_f.shape[0],), dtype=float)
            boundary_flags_f = boundary_flags[mask_g] if boundary_flags.size == Xg.shape[0] else np.zeros((Xg_f.shape[0],), dtype=bool)

            # Internal Xg dedup after history dedup:
            # keep boundary points and drop only LHC points overlapping boundary.
            if Xg_f.size > 0 and boundary_flags_f.size == Xg_f.shape[0]:
                idx_boundary = np.where(boundary_flags_f)[0]
                idx_lhc = np.where(~boundary_flags_f)[0]
                if idx_boundary.size > 0 and idx_lhc.size > 0:
                    keep_lhc = self._dedup_mask_against_other_norm(
                        X_new=Xg_f[idx_lhc],
                        X_ref=Xg_f[idx_boundary],
                        spans=spans,
                        dedup_tol=tol,
                    )
                    keep_xg_internal = np.ones((Xg_f.shape[0],), dtype=bool)
                    keep_xg_internal[idx_lhc] = keep_lhc
                    Xg_f = Xg_f[keep_xg_internal]
                    constraints_g_f = [constraints_g_f[i] for i in np.where(keep_xg_internal)[0]]
                    margins_g_f = margins_g_f[keep_xg_internal] if margins_g_f.size == keep_xg_internal.shape[0] else margins_g_f
                    boundary_flags_f = boundary_flags_f[keep_xg_internal]

            if tol_local > 0.0:
                mask_l = self._dedup_mask_against_history_raw(
                    X_new=Xl,
                    X_old=self.store.X,
                    dedup_tol=tol_local,
                )
            else:
                mask_l = self._dedup_mask_against_history_norm(
                    X_new=Xl,
                    X_old=self.store.X,
                    spans=spans,
                    dedup_tol=tol,
                )
            Xl_f = Xl[mask_l]
            constraints_l_f = [constraints_l[i] for i in np.where(mask_l)[0]]
            margins_l_f = margins_l[mask_l] if margins_l.size == Xl.shape[0] else np.empty((Xl_f.shape[0],), dtype=float)
            Xl_anchor_ids_f = Xl_anchor_ids[mask_l] if Xl_anchor_ids.size == Xl.shape[0] else np.array([], dtype=int)

            if n_exec_local > 0 and n_exec_global > 0 and Xg_f.size > 0 and Xl_f.size > 0:
                mask_g_other = self._dedup_mask_against_other_norm(
                    X_new=Xg_f,
                    X_ref=Xl_f,
                    spans=spans,
                    dedup_tol=tol,
                )
                Xg_f = Xg_f[mask_g_other]
                constraints_g_f = [constraints_g_f[i] for i in np.where(mask_g_other)[0]]
                margins_g_f = margins_g_f[mask_g_other] if margins_g_f.size == mask_g_other.shape[0] else margins_g_f
                boundary_flags_f = boundary_flags_f[mask_g_other] if boundary_flags_f.size == mask_g_other.shape[0] else boundary_flags_f

            # local anchor는 local 실행에서 우선 포함될 수 있으므로,
            # global 실행 후보에서는 anchor와 좌표가 완전히 같은 점만 제거한다.
            if n_exec_local > 0 and Xg_f.size > 0 and Xl_anchors.size > 0:
                anchor_set = {tuple(row.tolist()) for row in np.asarray(Xl_anchors, dtype=float)}
                mask_g_anchor = np.asarray(
                    [tuple(row.tolist()) not in anchor_set for row in np.asarray(Xg_f, dtype=float)],
                    dtype=bool,
                )
                Xg_f = Xg_f[mask_g_anchor]
                constraints_g_f = [constraints_g_f[i] for i in np.where(mask_g_anchor)[0]]
                margins_g_f = margins_g_f[mask_g_anchor] if margins_g_f.size == mask_g_anchor.shape[0] else margins_g_f
                boundary_flags_f = boundary_flags_f[mask_g_anchor] if boundary_flags_f.size == mask_g_anchor.shape[0] else boundary_flags_f

            def _select_global_exec(
                *,
                n_pick_override: int | None = None,
                X_exclude: np.ndarray | None = None,
            ) -> tuple[np.ndarray, list[dict], np.ndarray]:
                n_pick = int(n_exec_global if n_pick_override is None else n_pick_override)
                if n_pick <= 0 or Xg_f.size == 0:
                    return np.empty((0, Xg_f.shape[1]), dtype=float), [], np.empty((0,), dtype=float)

                Xg_work = np.asarray(Xg_f, dtype=float)
                constraints_work = list(constraints_g_f)
                margins_work = np.asarray(margins_g_f, dtype=float)
                boundary_work = np.asarray(boundary_flags_f, dtype=bool)

                if X_exclude is not None and np.asarray(X_exclude).size > 0:
                    mask_ex = self._dedup_mask_against_other_norm(
                        X_new=Xg_work,
                        X_ref=np.asarray(X_exclude, dtype=float),
                        spans=spans,
                        dedup_tol=tol,
                    )
                    Xg_work = Xg_work[mask_ex]
                    constraints_work = [constraints_work[i] for i in np.where(mask_ex)[0]]
                    margins_work = (
                        margins_work[mask_ex]
                        if margins_work.size == mask_ex.shape[0]
                        else margins_work
                    )
                    boundary_work = (
                        boundary_work[mask_ex]
                        if boundary_work.size == mask_ex.shape[0]
                        else boundary_work
                    )

                if Xg_work.size == 0:
                    return np.empty((0, Xg_f.shape[1]), dtype=float), [], np.empty((0,), dtype=float)

                order_g = self._rank_by_pred(
                    models=gate1_models,
                    X_candidate=Xg_work,
                    objective_sense=objective_sense,
                )
                n_boundary_exec = int(round(n_pick * boundary_ratio))
                n_boundary_exec = min(max(n_boundary_exec, 0), n_pick)
                n_margin_exec = int(round(n_pick * self.global_margin_ratio))
                n_margin_exec = min(max(n_margin_exec, 0), n_pick)
                if not self.has_pre_constraints:
                    n_margin_exec = 0
                n_top_exec = int(round(n_pick * self.global_top_ratio))
                n_top_exec = min(max(n_top_exec, 0), n_pick)

                # ── Bucket minima 적용 (drop priority: top>margin>boundary>random) ──
                # actual feasible 후보가 있는 경우에만 minima 적용 (graceful degrade)
                _bm_achieved: dict[str, int] = {}
                _bm_dropped: list[str] = []
                if self.global_bucket_minima_enabled and n_pick >= 1 and _bm_minima_stage:
                    _avail_boundary = int(np.asarray(boundary_work, dtype=bool).sum()) if boundary_work.size == Xg_work.shape[0] else 0
                    _avail_margin = int(margins_work.size) if (self.has_pre_constraints and margins_work.size > 0) else 0
                    _avail_top = int(Xg_work.shape[0])
                    _remaining_slots = int(n_pick)
                    # Priority 1: top
                    _req_top = int(_bm_minima_stage.get("top", 0))
                    _take_top = min(_req_top, _avail_top, _remaining_slots)
                    if _req_top > 0 and _take_top == 0:
                        _bm_dropped.append("top")
                    if _take_top > 0:
                        _bm_achieved["top"] = _take_top
                        if n_top_exec < _take_top:
                            n_top_exec = int(_take_top)
                        _remaining_slots -= _take_top
                    # Priority 2: margin (constraint 있을 때만)
                    _req_margin = int(_bm_minima_stage.get("margin", 0))
                    _take_margin = min(_req_margin, _avail_margin, _remaining_slots)
                    if _req_margin > 0 and _take_margin == 0:
                        _bm_dropped.append("margin")
                    if _take_margin > 0:
                        _bm_achieved["margin"] = _take_margin
                        if n_margin_exec < _take_margin:
                            n_margin_exec = int(_take_margin)
                        _remaining_slots -= _take_margin
                    # Priority 3: boundary (classic + cross + any 합산, 세부 구분은 planning에서)
                    _req_boundary = int(
                        _bm_minima_stage.get("boundary_any", 0)
                        + _bm_minima_stage.get("boundary_classic", 0)
                        + _bm_minima_stage.get("boundary_cross", 0)
                    )
                    _take_boundary = min(_req_boundary, _avail_boundary, _remaining_slots)
                    if _req_boundary > 0 and _take_boundary == 0:
                        _bm_dropped.append("boundary")
                    if _take_boundary > 0:
                        _bm_achieved["boundary"] = _take_boundary
                        if n_boundary_exec < _take_boundary:
                            n_boundary_exec = int(_take_boundary)
                        _remaining_slots -= _take_boundary
                    # random은 min 보장 없음 — drop priority 최하위
                    if _bm_dropped:
                        self._bucket_min_relaxed_count += 1
                        if self.global_bucket_minima_strict:
                            self._set_failure(
                                f"FAILED_BUCKET_MINIMA:{','.join(_bm_dropped)}",
                                stage=round_idx,
                            )
                        else:
                            print(
                                f"[AdditionalDOE] bucket_min_relaxed stage={round_idx} "
                                f"dropped={_bm_dropped} req={_bm_minima_stage} "
                                f"avail(top={_avail_top},margin={_avail_margin},boundary={_avail_boundary})"
                            )
                margin_subset_specs: list[tuple[tuple[str, ...], float]] = []
                if self.global_margin_subset_enabled and n_margin_exec > 0:
                    pre_constraint_ids = self._extract_pre_constraint_ids(constraints_work)
                    if len(pre_constraint_ids) >= 2:
                        margin_subset_specs = self._build_margin_subset_specs(pre_constraint_ids)
                diversity_reserve = 0
                if (
                    diversity_boost_active
                    and diversity_quota > 0
                    and n_pick_override is None
                ):
                    diversity_reserve = min(int(diversity_quota), n_pick)
                    n_boundary_exec = max(n_boundary_exec, diversity_reserve)
                if n_pick > 0 and n_top_exec < 1:
                    n_top_exec = 1

                # top-k 최소 1개를 보장하기 위해 boundary/margin을 먼저 캡한다.
                max_for_pre_buckets = max(n_pick - n_top_exec, 0)
                if margin_subset_specs and max_for_pre_buckets > 0:
                    # ratio를 최소선으로 유지하되, subset별 최소 1개 보장을 위해 margin quota를 상향
                    n_margin_exec = max(
                        int(n_margin_exec),
                        int(min(len(margin_subset_specs), max_for_pre_buckets)),
                    )
                if (n_boundary_exec + n_margin_exec) > max_for_pre_buckets:
                    overflow = (n_boundary_exec + n_margin_exec) - max_for_pre_buckets
                    if margin_subset_specs:
                        # subset margin 모드에서는 margin quota를 우선 보존
                        reduce_boundary = min(overflow, n_boundary_exec)
                        n_boundary_exec -= reduce_boundary
                        overflow -= reduce_boundary
                        if overflow > 0:
                            n_margin_exec = max(n_margin_exec - overflow, 0)
                    else:
                        reduce_margin = min(overflow, n_margin_exec)
                        n_margin_exec -= reduce_margin
                        overflow -= reduce_margin
                        if overflow > 0:
                            n_boundary_exec = max(n_boundary_exec - overflow, 0)

                selected: list[int] = []
                selected_set: set[int] = set()

                def _take(candidates: list[int], k: int) -> list[int]:
                    if k <= 0:
                        return []
                    out: list[int] = []
                    for cid in candidates:
                        if cid in selected_set:
                            continue
                        out.append(int(cid))
                        if len(out) >= k:
                            break
                    return out

                # 0) collapse 후 diversity reserve를 경계 점에서 우선 확보
                if diversity_reserve > 0 and boundary_work.size == Xg_work.shape[0]:
                    boundary_order = [idx for idx in order_g.tolist() if boundary_work[idx]]
                    picked = _take(boundary_order, diversity_reserve)
                    selected.extend(picked)
                    selected_set.update(picked)

                # 1) margin 작은 순(제약 경계 근처)
                if n_margin_exec > 0 and margins_work.size == Xg_work.shape[0]:
                    if margin_subset_specs:
                        # subset별 margin(min) 기준으로 가중 배분
                        n_constraints = len(self._extract_pre_constraint_ids(constraints_work))
                        edge_k = {1, int(n_constraints)}
                        subset_priority = sorted(
                            range(len(margin_subset_specs)),
                            key=lambda gi: (
                                0 if len(margin_subset_specs[gi][0]) in edge_k else 1,
                                -len(margin_subset_specs[gi][0]),
                                margin_subset_specs[gi][0],
                            ),
                        )
                        weights = [float(margin_subset_specs[gi][1]) for gi in range(len(margin_subset_specs))]
                        enforce_min_one = n_margin_exec >= len(margin_subset_specs)
                        quotas = self._allocate_weighted_subset_quotas(
                            n_total=int(n_margin_exec),
                            weights=weights,
                            enforce_min_one=bool(enforce_min_one),
                        )
                        rank_pos = {
                            int(idx): int(pos)
                            for pos, idx in enumerate(order_g.astype(int).tolist())
                        }
                        # candidate별 제약 margin map 구성
                        candidate_margin_maps: list[dict[str, float]] = []
                        for payload in constraints_work:
                            cur: dict[str, float] = {}
                            if isinstance(payload, dict):
                                for key, cinfo in payload.items():
                                    if not isinstance(cinfo, dict):
                                        continue
                                    scope = str(cinfo.get("scope", "pre")).strip().lower()
                                    if scope != "pre":
                                        continue
                                    cid = str(cinfo.get("id", key)).strip()
                                    mv = cinfo.get("margin")
                                    if cid:
                                        try:
                                            mval = float(mv)
                                            if np.isfinite(mval):
                                                cur[cid] = mval
                                        except Exception:
                                            continue
                            candidate_margin_maps.append(cur)

                        subset_orders: list[list[int]] = []
                        for subset_ids, _w in margin_subset_specs:
                            ranked: list[tuple[float, int, int]] = []
                            for idx in range(Xg_work.shape[0]):
                                m_map = candidate_margin_maps[idx] if idx < len(candidate_margin_maps) else {}
                                vals = [m_map.get(cid) for cid in subset_ids]
                                if any(v is None for v in vals):
                                    continue
                                s_margin = float(np.min(np.asarray(vals, dtype=float)))
                                ranked.append((s_margin, rank_pos.get(int(idx), 10**9), int(idx)))
                            ranked.sort(key=lambda t: (t[0], t[1]))
                            subset_orders.append([idx for _, __, idx in ranked])

                        allocated = [0 for _ in quotas]
                        if enforce_min_one:
                            for gi in subset_priority:
                                picked = _take(subset_orders[gi], 1)
                                if picked:
                                    selected.extend(picked)
                                    selected_set.update(picked)
                                    allocated[gi] += len(picked)

                        for gi in subset_priority:
                            need = max(int(quotas[gi]) - int(allocated[gi]), 0)
                            if need <= 0:
                                continue
                            picked = _take(subset_orders[gi], need)
                            if picked:
                                selected.extend(picked)
                                selected_set.update(picked)
                                allocated[gi] += len(picked)
                    else:
                        # objective-aware margin 정렬: margin + objective 복합 순위
                        _alpha = float(self.global_margin_obj_alpha)
                        if _alpha < 1.0 and order_g.size == Xg_work.shape[0]:
                            from scipy.stats import rankdata as _rankdata
                            _m_rank = _rankdata(margins_work, method="ordinal") / max(float(Xg_work.shape[0]), 1.0)
                            _o_rank = np.zeros_like(_m_rank)
                            for _ri, _oi in enumerate(order_g.astype(int).tolist()):
                                if 0 <= _oi < _o_rank.shape[0]:
                                    _o_rank[_oi] = float(_ri) / max(float(Xg_work.shape[0]), 1.0)
                            _combined = _alpha * _m_rank + (1.0 - _alpha) * _o_rank
                            margin_order = np.argsort(_combined).astype(int).tolist()
                        else:
                            margin_order = np.argsort(margins_work).astype(int).tolist()
                        picked = _take(margin_order, n_margin_exec)
                        selected.extend(picked)
                        selected_set.update(picked)

                # 2) top-k
                if n_top_exec > 0:
                    picked = _take(order_g.astype(int).tolist(), n_top_exec)
                    selected.extend(picked)
                    selected_set.update(picked)

                # 3) boundary quota — constraint 문제에서는 margin+boundary 교차 우선
                if n_boundary_exec > 0 and boundary_work.size == Xg_work.shape[0]:
                    if (
                        self.has_pre_constraints
                        and margins_work.size == Xg_work.shape[0]
                        and self.global_boundary_margin_cross_ratio > 0.0
                    ):
                        # margin+boundary 교차: boundary 점 중 margin이 가장 작은 것 우선
                        _near_tol = float(self.global_boundary_margin_near_tol)
                        _cross_quota = int(round(n_boundary_exec * self.global_boundary_margin_cross_ratio))
                        _classic_quota = n_boundary_exec - _cross_quota

                        # 교차 조건: boundary flag AND 변수 1개+ 경계 근처
                        _cross_candidates: list[tuple[float, int]] = []
                        for idx in range(Xg_work.shape[0]):
                            if not boundary_work[idx]:
                                continue
                            x = Xg_work[idx]
                            near_any = False
                            for j, (lb, ub) in enumerate(self.bounds):
                                sp = max(float(ub) - float(lb), 1e-12)
                                if (float(x[j]) - float(lb)) < _near_tol * sp or (float(ub) - float(x[j])) < _near_tol * sp:
                                    near_any = True
                                    break
                            if near_any and np.isfinite(margins_work[idx]):
                                _cross_candidates.append((float(margins_work[idx]), int(idx)))
                        _cross_candidates.sort(key=lambda t: t[0])  # margin 오름차순
                        _cross_order = [idx for _, idx in _cross_candidates]
                        picked_cross = _take(_cross_order, _cross_quota)
                        selected.extend(picked_cross)
                        selected_set.update(picked_cross)

                        # 나머지: 기존 boundary 순서
                        boundary_order = [idx for idx in order_g.tolist() if boundary_work[idx]]
                        picked_classic = _take(boundary_order, _classic_quota)
                        selected.extend(picked_classic)
                        selected_set.update(picked_classic)
                    else:
                        boundary_order = [idx for idx in order_g.tolist() if boundary_work[idx]]
                        picked = _take(boundary_order, n_boundary_exec)
                        selected.extend(picked)
                        selected_set.update(picked)

                # 4) random(남은 슬롯)
                remaining_slots = max(n_pick - len(selected), 0)
                remain_order = [idx for idx in order_g.astype(int).tolist() if idx not in selected_set]
                if remaining_slots > 0 and remain_order:
                    rand_pool = np.asarray(remain_order, dtype=int)
                    n_take = min(remaining_slots, rand_pool.shape[0])
                    if n_take > 0:
                        rand_idx = self.rng.choice(rand_pool, size=n_take, replace=False).astype(int).tolist()
                        selected.extend(rand_idx)
                        selected_set.update(rand_idx)

                # 여전히 부족하면 예측 순으로 채움
                if len(selected) < n_pick:
                    remain = [idx for idx in order_g.astype(int).tolist() if idx not in selected_set]
                    fill = remain[: (n_pick - len(selected))]
                    selected.extend(fill)
                    selected_set.update(fill)

                selected = selected[:n_pick]
                if not selected:
                    return np.empty((0, Xg_f.shape[1]), dtype=float), [], np.empty((0,), dtype=float)
                idx = np.asarray(selected, dtype=int)
                if margins_work.size == Xg_work.shape[0]:
                    picked_margins = margins_work[idx]
                else:
                    picked_margins = np.full((idx.shape[0],), float("inf"), dtype=float)
                return Xg_work[idx], [constraints_work[i] for i in idx], picked_margins

            def _select_local_exec_equal() -> tuple[np.ndarray, list[dict], np.ndarray]:
                if n_exec_local <= 0 or Xl_f.size == 0:
                    return np.empty((0, Xl_f.shape[1]), dtype=float), [], np.empty((0,), dtype=float)

                anchors = Xl_anchors
                if anchors is None or anchors.size == 0:
                    return np.empty((0, Xl_f.shape[1]), dtype=float), [], np.empty((0,), dtype=float)

                if tol_local > 0.0:
                    anchor_keep = self._dedup_mask_against_history_raw(
                        X_new=anchors,
                        X_old=self.store.X,
                        dedup_tol=tol_local,
                    )
                else:
                    anchor_keep = self._dedup_mask_against_history_norm(
                        X_new=anchors,
                        X_old=self.store.X,
                        spans=spans,
                        dedup_tol=tol,
                    )

                anchor_count = anchors.shape[0]
                Xl_work = np.asarray(Xl_f, dtype=float).copy()
                constraints_work = list(constraints_l_f)
                margins_work = np.asarray(margins_l_f, dtype=float).copy()
                groups = {i: [] for i in range(anchor_count)}
                if Xl_anchor_ids_f.size == Xl_work.shape[0]:
                    for idx, aid in enumerate(Xl_anchor_ids_f.tolist()):
                        if 0 <= int(aid) < anchor_count:
                            groups[int(aid)].append(idx)

                active = []
                for i in range(anchor_count):
                    # Keep anchor active when either anchor itself survives dedup
                    # or it still has local candidate points assigned.
                    available = len(groups[i]) + (1 if anchor_keep[i] else 0)
                    if available > 0:
                        active.append(i)
                if not active:
                    return np.empty((0, Xl_f.shape[1]), dtype=float), [], np.empty((0,), dtype=float)

                per = n_exec_local // len(active)
                remainder = n_exec_local - per * len(active)
                quotas = {i: per + (1 if j < remainder else 0) for j, i in enumerate(active)}
                base_quotas = dict(quotas)
                deficits = 0

                # 제약 부족 anchor: 반경 축소 1회 재시도 후 실패 시 anchor 1개만 유지
                for i in active:
                    q_i = quotas[i]
                    if q_i <= 0:
                        continue
                    required_feasible = int(np.ceil(self.local_constraint_min_factor * (q_i + 1)))
                    f_i = len(groups[i])
                    if self.has_pre_constraints and f_i < required_feasible:
                        for retry_idx in range(self.local_constraint_retry_count):
                            shrink = self.local_constraint_shrink_factor ** float(retry_idx + 1)
                            X_retry = clamp_to_bounds(
                                anchors[i].reshape(1, -1)
                                + self.rng.normal(
                                    loc=0.0,
                                    scale=np.maximum(spans * radius_ratio * shrink, spans * self.local_min_radius_ratio),
                                    size=(max(4, required_feasible), Xl_work.shape[1]),
                                ),
                                self.bounds,
                            )
                            X_retry_f, c_retry_f, m_retry_f, _ = self._filter_by_constraints(X_retry, update_ratio=False)
                            if tol_local > 0.0:
                                mask_retry_hist = self._dedup_mask_against_history_raw(
                                    X_new=X_retry_f,
                                    X_old=self.store.X,
                                    dedup_tol=tol_local,
                                )
                            else:
                                mask_retry_hist = self._dedup_mask_against_history_norm(
                                    X_new=X_retry_f,
                                    X_old=self.store.X,
                                    spans=spans,
                                    dedup_tol=tol,
                                )
                            idx_retry = np.where(mask_retry_hist)[0]
                            for k in idx_retry.tolist():
                                if Xl_work.size > 0:
                                    if tol_local > 0.0:
                                        d_dup = np.linalg.norm(Xl_work - X_retry_f[k].reshape(1, -1), axis=1)
                                        if np.any(d_dup <= tol_local):
                                            continue
                                    else:
                                        d_dup = np.linalg.norm((Xl_work - X_retry_f[k].reshape(1, -1)) / spans, axis=1)
                                        if np.any(d_dup <= tol):
                                            continue
                                groups[i].append(len(Xl_work))
                                Xl_work = np.vstack([Xl_work, X_retry_f[k].reshape(1, -1)])
                                constraints_work.append(c_retry_f[k])
                                margins_work = np.concatenate([margins_work, np.asarray([m_retry_f[k]], dtype=float)])
                            f_i = len(groups[i])
                            if f_i >= required_feasible:
                                break
                    if self.has_pre_constraints and f_i < required_feasible:
                        deficits += max(q_i - 1, 0)
                        quotas[i] = 1

                if deficits > 0:
                    donors = [i for i in active if quotas[i] > 0]
                    for j in donors:
                        if deficits <= 0:
                            break
                        available_j = len(groups[j]) + (1 if anchor_keep[j] else 0)
                        start_q = max(base_quotas[j] - 1, 0)
                        spare = available_j - max(quotas[j], start_q)
                        if spare <= 0:
                            continue
                        take = min(spare, deficits)
                        quotas[j] += take
                        deficits -= take
                if deficits > 0:
                    return np.empty((0, Xl_work.shape[1]), dtype=float), [], np.empty((0,), dtype=float)

                # Capacity-based quota rebalance (constraint-agnostic)
                # so one weak anchor does not fail the entire local allocation.
                capacity_deficits = 0
                for i in active:
                    capacity_i = len(groups[i]) + (1 if anchor_keep[i] else 0)
                    if quotas[i] > capacity_i:
                        capacity_deficits += int(quotas[i] - capacity_i)
                        quotas[i] = int(capacity_i)
                if capacity_deficits > 0:
                    for j in active:
                        if capacity_deficits <= 0:
                            break
                        capacity_j = len(groups[j]) + (1 if anchor_keep[j] else 0)
                        spare = int(capacity_j - quotas[j])
                        if spare <= 0:
                            continue
                        take = min(spare, capacity_deficits)
                        quotas[j] += take
                        capacity_deficits -= take

                picks = []
                picks_constraints = []
                picks_margins = []
                local_shortage = 0
                for i in active:
                    required = quotas[i]
                    if required <= 0:
                        continue
                    already = 0
                    if anchor_keep[i]:
                        picks.append(anchors[i].reshape(1, -1))
                        c_anchor, f_anchor, m_anchor = evaluate_constraints_point(
                            x=anchors[i],
                            var_names=self.var_names,
                            constraint_defs=self.constraint_defs,
                            scope="pre",
                        )
                        if not self.has_pre_constraints or f_anchor:
                            picks_constraints.append(c_anchor if self.has_pre_constraints else {})
                            picks_margins.append(float(m_anchor if self.has_pre_constraints else float("inf")))
                            already = 1
                    need = required - already
                    if need <= 0:
                        continue
                    cand_idx = groups[i]
                    if len(cand_idx) < need:
                        local_shortage += int(need - len(cand_idx))
                        need = int(len(cand_idx))
                    if need <= 0:
                        continue
                    if self.local_exec_pick_mode == "distance":
                        pts = Xl_work[cand_idx]
                        dists = np.linalg.norm(pts - anchors[i].reshape(1, -1), axis=1)
                        order = np.argsort(dists)[:need]
                        chosen_idx = [cand_idx[int(k)] for k in order.tolist()]
                    else:
                        chosen_rel = self.rng.choice(len(cand_idx), size=need, replace=False)
                        chosen_idx = [cand_idx[int(k)] for k in np.asarray(chosen_rel, dtype=int).tolist()]
                    picks.append(Xl_work[chosen_idx])
                    picks_constraints.extend([constraints_work[k] for k in chosen_idx])
                    picks_margins.extend([float(margins_work[k]) for k in chosen_idx])

                if not picks:
                    return np.empty((0, Xl_work.shape[1]), dtype=float), [], np.empty((0,), dtype=float)
                X_pick = np.vstack(picks).astype(float)
                if local_shortage > 0:
                    print(
                        "[AdditionalDOE] local quota shortage "
                        f"stage={round_idx} requested={n_exec_local} selected={X_pick.shape[0]}"
                    )
                return X_pick[:n_exec_local], picks_constraints[:n_exec_local], np.asarray(picks_margins[:n_exec_local], dtype=float)

            if n_exec_local > 0:
                X_exec_g, constraints_exec_g, margins_exec_g = _select_global_exec()
                X_exec_l, constraints_exec_l, margins_exec_l = _select_local_exec_equal()
                if X_exec_g.shape[0] < n_exec_global:
                    self._set_failure("FAILED_DEDUP", stage=round_idx)
                    break
                if X_exec_l.shape[0] < n_exec_local:
                    shortage = int(n_exec_local - X_exec_l.shape[0])
                    if self.budget_policy == "consume_all":
                        early_round_relax = bool(used_budget_ratio < self.phase2_min_used_budget_ratio)
                    else:
                        early_round_relax = bool((round_idx + 1) < int(self.min_additional_rounds))
                    allow_global_backfill = bool(early_round_relax or (gate_stop and n_exec_global == 0))

                    filled = 0
                    if allow_global_backfill and shortage > 0:
                        if X_exec_l.shape[0] > 0 and X_exec_g.shape[0] > 0:
                            x_exclude = np.vstack([X_exec_l, X_exec_g])
                        elif X_exec_l.shape[0] > 0:
                            x_exclude = X_exec_l
                        elif X_exec_g.shape[0] > 0:
                            x_exclude = X_exec_g
                        else:
                            x_exclude = None

                        X_fill_g, c_fill_g, m_fill_g = _select_global_exec(
                            n_pick_override=shortage,
                            X_exclude=x_exclude,
                        )
                        filled = int(X_fill_g.shape[0])
                        if filled > 0:
                            if X_exec_g.shape[0] > 0:
                                X_exec_g = np.vstack([X_exec_g, X_fill_g])
                                constraints_exec_g.extend(c_fill_g)
                                if margins_exec_g.size > 0:
                                    margins_exec_g = np.concatenate([margins_exec_g, m_fill_g])
                                else:
                                    margins_exec_g = np.asarray(m_fill_g, dtype=float)
                            else:
                                X_exec_g = X_fill_g
                                constraints_exec_g = c_fill_g
                                margins_exec_g = np.asarray(m_fill_g, dtype=float)

                    remaining_shortage = max(shortage - filled, 0)
                    if remaining_shortage > 0:
                        if early_round_relax:
                            print(
                                "[AdditionalDOE] WARN_LOCAL_SHORTAGE_EARLY "
                                f"stage={round_idx} shortage={remaining_shortage} "
                                f"(min_rounds={self.min_additional_rounds})"
                            )
                        else:
                            self._set_failure("FAILED_LOCAL_FEASIBILITY", stage=round_idx)
                            break
                    else:
                        if gate_stop and n_exec_global == 0 and not early_round_relax:
                            print(
                                f"[AdditionalDOE] final-stage local shortage={shortage}, "
                                "backfilled from global."
                            )
                        elif shortage > 0:
                            print(
                                "[AdditionalDOE] local shortage backfilled "
                                f"stage={round_idx} shortage={shortage}"
                            )
            else:
                X_exec_g, constraints_exec_g, margins_exec_g = _select_global_exec()
                if X_exec_g.shape[0] < n_exec_global:
                    self._set_failure("FAILED_DEDUP", stage=round_idx)
                    break
            selected_exec_global = int(X_exec_g.shape[0])
            selected_exec_local = int(X_exec_l.shape[0]) if n_exec_local > 0 else 0
            if diversity_boost_active:
                diversity_sample_count = int(min(max(diversity_quota, 0), selected_exec_global))
            else:
                diversity_sample_count = 0
            self.gate_history.append(
                {
                    "stage": int(round_idx),
                    "phase": int(phase),
                    "next_phase": int(next_phase),
                    "n_plan": int(n_plan),
                    "n_exec": int(n_exec),
                    "n_exec_global_floor": int(alloc_info.get("global_floor", 0)),
                    "n_exec_local_floor": int(alloc_info.get("local_floor", 0)),
                    "n_exec_floor": int(alloc_info.get("exec_floor", 0)),
                    "n_exec_preclip": int(alloc_info.get("n_exec_preclip", n_exec)),
                    "alloc_source": str(alloc_info.get("alloc_source", "")),
                    "alloc_stages_est": alloc_info.get("stages_est"),
                    "alloc_absorbed": alloc_info.get("absorbed"),
                    "alloc_min_feasible": bool(alloc_info.get("min_feasible", True)),
                    "alloc_ratio_strict": bool(alloc_info.get("ratio_strict_selected", False)),
                    "alloc_exec_floor_strict": alloc_info.get("exec_floor_strict"),
                    "alloc_bucket_minima_sum": alloc_info.get("bucket_minima_sum"),
                    "n_global_plan": int(n_global),
                    "n_local_plan": int(n_local),
                    "n_exec_global_target": int(n_exec_global),
                    "n_exec_local_target": int(n_exec_local),
                    "n_exec_global_selected": int(selected_exec_global),
                    "n_exec_local_selected": int(selected_exec_local),
                    "usable_n": int(usable_n),
                    "p_dim": int(p_dim),
                    "usable_n_over_p": float(usable_np_ratio),
                    "budget_total": int(self.total_budget),
                    "budget_used_before_exec": int(self.budget.used),
                    "used_budget_ratio_before_exec": float(used_budget_ratio),
                    "budget_np_ratio": float(budget_np_ratio),
                    "budget_policy": str(self.budget_policy),
                    "gate_evaluated": bool(gate_evaluated),
                    "gate_eval_skipped_reason": gate_eval_skipped_reason,
                    "gate1_score": gate1_score,
                    "gate1_score_raw": gate1_score_raw,
                    "gate1_score_ema": gate1_score_ema,
                    "gate1_score_used": gate1_score_used,
                    "gate1_pass": gate1_passed,
                    "gate1_pass_smoothed": gate1_pass_smoothed,
                    "gate2_score": gate2_score,
                    "gate2_score_raw": gate2_score_raw,
                    "gate2_score_ema": gate2_score_ema,
                    "gate2_score_used": gate2_score_used,
                    "gate2_pass": gate2_passed,
                    "gate2_pass_smoothed": gate2_pass_smoothed,
                    "gate_decision_source": gate_decision_source,
                    "phase2_decision_source": phase2_decision_source,
                    "gate_stop_raw": gate_stop_raw,
                    "gate_stop_smoothed": gate_stop_smoothed,
                    "gate_stop_final": bool(gate_stop),
                    "can_stop_by_data": can_stop_by_data,
                    "can_stop_by_round": can_stop_by_round,
                    "can_stop_by_budget": can_stop_by_budget,
                    "converged_for_stop": converged_for_stop,
                    "collapse_detected": bool(collapse_detected),
                    "collapse_span_ratio": collapse_trigger_span,
                    "collapse_anchor_streak": int(collapse_anchor_streak),
                    "diversity_boost_active": bool(diversity_boost_active),
                    "diversity_quota": int(diversity_quota),
                    "diversity_sample_count": int(diversity_sample_count),
                    "diversity_hit_rate": diversity_hit_rate,
                    "phase2_gate1_score_min": float(self.phase2_gate1_score_min),
                    "phase2_gate1_rescue_enabled": bool(self.phase2_gate1_rescue_enabled),
                    "phase2_gate1_rescue_band": float(self.phase2_gate1_rescue_band),
                    "phase2_gate1_rescue_band_lower": phase2_gate1_rescue_band_lower,
                    "phase2_gate1_rescue_gate2_min": phase2_gate1_rescue_gate2_min_used,
                    "phase2_gate1_rescue_triggered": bool(phase2_gate1_rescue_triggered),
                    "phase2_gate2_score_min": float(self.phase2_gate2_score_min),
                    "phase2_gate2_score_sticky_min": float(self.phase2_gate2_score_sticky_min),
                    "phase2_np_gate": float(phase2_np_gate),
                    "phase2_min_used_budget_ratio": float(self.phase2_min_used_budget_ratio),
                    "phase2_gate2_threshold_used": gate2_phase_threshold,
                    "can_enter_phase2_by_data": can_enter_phase2_by_data,
                    "can_enter_phase2_by_budget": can_enter_phase2_by_budget,
                    "can_enter_phase2_by_gate1": can_enter_phase2_by_gate1,
                    "can_enter_phase2_by_gate2": can_enter_phase2_by_gate2,
                    "remaining_budget_before_exec": int(remaining),
                    "bucket_min_required": dict(_bm_minima_stage),
                    "bucket_min_boundary_sum": int(_bm_minima_boundary),
                    "bucket_min_high_crate": bool(
                        self.has_pre_constraints
                        and float(self.constraint_rate_hat) < float(self.global_boundary_high_crate_threshold)
                    ),
                    "n_exec_floor": int(_bm_n_exec_floor),
                }
            )

            if X_exec_g.shape[0] == 0 and (n_exec_local <= 0 or X_exec_l.shape[0] == 0):
                self._set_failure("FAILED_DEDUP", stage=round_idx)
                break

            # -----------------------------
            # 5) Execute CAE
            # -----------------------------
            executed_ok = True
            if X_exec_g.shape[0] > 0:
                n_div = int(min(max(diversity_sample_count, 0), X_exec_g.shape[0]))
                if n_div > 0:
                    X_exec_g_div = X_exec_g[:n_div]
                    c_exec_g_div = constraints_exec_g[:n_div]
                    m_exec_g_div = (
                        margins_exec_g[:n_div]
                        if np.asarray(margins_exec_g).size >= n_div
                        else np.empty((0,), dtype=float)
                    )
                    pre_rows = int(len(self.store.rows))
                    executed_g_div = self._execute_points(
                        X_exec_g_div,
                        source="additional",
                        round_idx=round_idx,
                        exec_scope="global_diversity",
                        constraints_payloads=c_exec_g_div,
                        constraint_margins=m_exec_g_div,
                    )
                    executed_ok = executed_ok and bool(executed_g_div)
                    post_rows = int(len(self.store.rows))
                    if post_rows > pre_rows:
                        div_rows = self.store.rows[pre_rows:post_rows]
                        div_hits = int(
                            sum(
                                1
                                for rr in div_rows
                                if bool(getattr(rr, "success", False)) and bool(getattr(rr, "feasible", False))
                            )
                        )
                        diversity_hit_rate = float(div_hits / max(len(div_rows), 1))
                        diversity_sample_count = int(len(div_rows))
                    else:
                        diversity_hit_rate = None
                        diversity_sample_count = 0
                    X_exec_g_main = X_exec_g[n_div:]
                    c_exec_g_main = constraints_exec_g[n_div:]
                    m_exec_g_main = (
                        margins_exec_g[n_div:]
                        if np.asarray(margins_exec_g).size >= n_div
                        else np.empty((0,), dtype=float)
                    )
                else:
                    X_exec_g_main = X_exec_g
                    c_exec_g_main = constraints_exec_g
                    m_exec_g_main = margins_exec_g
                if X_exec_g_main.shape[0] > 0:
                    executed_g = self._execute_points(
                        X_exec_g_main,
                        source="additional",
                        round_idx=round_idx,
                        exec_scope="global",
                        constraints_payloads=c_exec_g_main,
                        constraint_margins=m_exec_g_main,
                    )
                    executed_ok = executed_ok and bool(executed_g)
            if n_exec_local > 0 and X_exec_l.shape[0] > 0:
                executed_l = self._execute_points(
                    X_exec_l,
                    source="additional",
                    round_idx=round_idx,
                    exec_scope="local",
                    constraints_payloads=constraints_exec_l,
                    constraint_margins=margins_exec_l,
                )
                executed_ok = executed_ok and bool(executed_l)

            if not executed_ok:
                if self.failure_reason is None:
                    self._set_failure("FAILED_BUDGET", stage=round_idx)
                break
            if self.gate_history:
                self.gate_history[-1]["diversity_sample_count"] = int(diversity_sample_count)
                self.gate_history[-1]["diversity_hit_rate"] = diversity_hit_rate

            if gate_stop:
                break

            # DOE-2: gate2 saturation 가드 — phase2 진입 후 g2_ema 포화 + 고제약 +
            # 예산 잔여 충분 시 다음 stage에 diversity injection 강제
            _doe2_trigger = False
            try:
                _g2_ema_val = float(self._gate2_ema) if self._gate2_ema is not None else None
            except Exception:
                _g2_ema_val = None
            _next_is_phase2 = bool(next_phase == 2 or self._ever_phase2)
            _used_ratio_now = float(self.budget.used) / float(max(self.total_budget, 1))
            _remaining_ratio_now = max(0.0, 1.0 - _used_ratio_now)
            _is_high_crate_now = bool(
                self.has_pre_constraints
                and float(self.constraint_rate_hat) < self.phase2_high_crate_threshold
            )
            if (
                _next_is_phase2
                and _g2_ema_val is not None
                and _g2_ema_val >= self.phase2_g2_saturation_threshold
                and _is_high_crate_now
                and _remaining_ratio_now >= self.phase2_g2_saturation_min_remaining
            ):
                _doe2_trigger = True
                self._g2_saturation_trigger_count += 1
                print(
                    f"[AdditionalDOE] DOE-2 g2 saturation: g2_ema={_g2_ema_val:.3f} "
                    f"crate={self.constraint_rate_hat:.3f} remaining={_remaining_ratio_now:.2f} "
                    f"→ diversity_boost forced for stage {round_idx+1}"
                )

            # increase round only when actual execution happened
            self._collapse_diversity_pending = bool(collapse_detected or _doe2_trigger)
            round_idx += 1
            phase = next_phase

        if (
            self.budget_policy != "consume_all"
            and round_idx >= self.max_additional_stages
            and not self.budget.exhausted()
            and self.failure_reason is None
        ):
            self._set_failure("STOP_MAX_ADDITIONAL_STAGES", stage=round_idx)

        exec_used = int(self.budget.used)
        exec_total = int(self.budget.total)
        if self.failure_reason is not None:
            print(
                f"[AdditionalDOE] terminated: {self.failure_reason} "
                f"(executed={exec_used}/{exec_total})"
            )
        else:
            print(
                "[AdditionalDOE] terminated: COMPLETED "
                f"(executed={exec_used}/{exec_total})"
            )
        return self._export_results()

    def get_diagnostics(self) -> dict:
        n_success, n_total, success_ratio = self._success_stats()
        p_dim = max(int(len(self.bounds)), 1)
        usable_n = int(self._usable_sample_count())
        usable_n_over_p = float(usable_n) / float(p_dim)
        additional_rows = [
            r for r in self.store.rows
            if str(getattr(r, "source", "")).strip().lower() == "additional"
        ]
        gate_rows = [g for g in self.gate_history if bool(g.get("gate_evaluated", False))]
        gate1_pass_rate = None
        gate2_pass_rate = None
        gate1_score_mean = None
        gate2_score_mean = None
        gate1_score_last = None
        gate2_score_last = None
        gate1_score_ema_mean = None
        gate2_score_ema_mean = None
        gate1_score_ema_last = None
        gate2_score_ema_last = None
        gate_decision_source_last = None
        if gate_rows:
            g1_pass_vals = [1.0 if bool(g.get("gate1_pass", False)) else 0.0 for g in gate_rows]
            g2_pass_vals = [1.0 if bool(g.get("gate2_pass", False)) else 0.0 for g in gate_rows]
            gate1_pass_rate = float(np.mean(g1_pass_vals))
            gate2_pass_rate = float(np.mean(g2_pass_vals))

            g1_scores = [float(g["gate1_score"]) for g in gate_rows if g.get("gate1_score") is not None]
            g2_scores = [float(g["gate2_score"]) for g in gate_rows if g.get("gate2_score") is not None]
            if g1_scores:
                gate1_score_mean = float(np.mean(g1_scores))
                gate1_score_last = float(g1_scores[-1])
            if g2_scores:
                gate2_score_mean = float(np.mean(g2_scores))
                gate2_score_last = float(g2_scores[-1])
            g1_ema_scores = [float(g["gate1_score_ema"]) for g in gate_rows if g.get("gate1_score_ema") is not None]
            g2_ema_scores = [float(g["gate2_score_ema"]) for g in gate_rows if g.get("gate2_score_ema") is not None]
            if g1_ema_scores:
                gate1_score_ema_mean = float(np.mean(g1_ema_scores))
                gate1_score_ema_last = float(g1_ema_scores[-1])
            if g2_ema_scores:
                gate2_score_ema_mean = float(np.mean(g2_ema_scores))
                gate2_score_ema_last = float(g2_ema_scores[-1])
            gate_decision_source_last = gate_rows[-1].get("gate_decision_source")

        local_span_ratio_mean_last = None
        if self.local_metrics:
            local_span_ratio_mean_last = self.local_metrics[-1].get("local_span_ratio_mean")
        used_budget_ratio_final = float(self.budget.used) / float(max(self.budget.total, 1))
        phase2_np_gate_last = None
        gate2_phase_threshold_used_last = None
        stage_count_total = int(len(self.gate_history))
        stage_gate_eval_count = int(
            sum(1 for g in self.gate_history if bool(g.get("gate_evaluated", False)))
        )
        gate_eval_skipped_count = int(stage_count_total - stage_gate_eval_count)
        phase2_stage_indices = [
            int(g.get("stage"))
            for g in self.gate_history
            if int(g.get("next_phase", 1)) == 2
        ]
        phase2_first_stage = phase2_stage_indices[0] if phase2_stage_indices else None
        phase2_stage_count = int(len(phase2_stage_indices))
        phase2_transition_count = int(
            sum(
                1
                for g in self.gate_history
                if int(g.get("phase", 1)) == 1 and int(g.get("next_phase", 1)) == 2
            )
        )
        phase2_gate1_rescue_trigger_count = int(
            sum(1 for g in self.gate_history if bool(g.get("phase2_gate1_rescue_triggered", False)))
        )
        gate_stop_raw_count = int(
            sum(1 for g in self.gate_history if bool(g.get("gate_stop_raw", False)))
        )
        gate_stop_final_count = int(
            sum(1 for g in self.gate_history if bool(g.get("gate_stop_final", False)))
        )
        gate_stop_blocked_count = int(
            sum(
                1
                for g in self.gate_history
                if bool(g.get("gate_stop_raw", False)) and (not bool(g.get("gate_stop_final", False)))
            )
        )
        collapse_detected_count = int(
            sum(1 for g in self.gate_history if bool(g.get("collapse_detected", False)))
        )
        diversity_injection_applied_count = int(
            sum(1 for g in self.gate_history if int(g.get("diversity_quota", 0) or 0) > 0)
        )
        diversity_sample_count_total = int(
            sum(int(g.get("diversity_sample_count", 0) or 0) for g in self.gate_history)
        )
        diversity_hit_rate_vals = [
            float(g.get("diversity_hit_rate"))
            for g in self.gate_history
            if g.get("diversity_hit_rate") is not None
        ]
        diversity_hit_rate_mean = (
            float(np.mean(diversity_hit_rate_vals)) if diversity_hit_rate_vals else None
        )
        used_budget_ratio_phase2_first = None
        if phase2_first_stage is not None:
            row = next(
                (g for g in self.gate_history if int(g.get("stage", -1)) == int(phase2_first_stage)),
                None,
            )
            if isinstance(row, dict):
                used_budget_ratio_phase2_first = row.get("used_budget_ratio_before_exec")
        if self.gate_history:
            phase2_np_gate_last = self.gate_history[-1].get("phase2_np_gate")
            gate2_phase_threshold_used_last = self.gate_history[-1].get("phase2_gate2_threshold_used")
        alloc_rows = [g for g in self.gate_history if g.get("alloc_source") is not None]
        budget_exhausted = bool(self.budget.exhausted())
        terminated_by = (
            str(self.failure_reason)
            if self.failure_reason is not None
            else ("budget_exhausted" if budget_exhausted else "completed_without_exhaustion")
        )
        out = {
            "failure_reason": self.failure_reason,
            "local_metrics": self.local_metrics,
            "gate_history": self.gate_history,
            "budget_policy": str(self.budget_policy),
            "constraint_rate_hat": float(self.constraint_rate_hat),
            "post_feasible_rate_hat": float(self.post_feasible_rate_hat),
            "post_lambda": float(self.post_lambda_current),
            "post_model_active": bool(self._post_model is not None),
            "post_policy_log": self.post_policy_log,
            "success_rate_floor": float(self.success_rate_floor),
            "success_count": int(n_success),
            "success_total": int(n_total),
            "success_ratio": float(success_ratio),
            "p_dim": int(p_dim),
            "usable_n": int(usable_n),
            "usable_n_over_p": float(usable_n_over_p),
            "used_budget_ratio_final": float(used_budget_ratio_final),
            "stage_alloc_mode": "dynamic_priority_ratio_stage_np",
            "global_margin_subset_enabled": bool(self.global_margin_subset_enabled),
            "global_margin_subset_random_k_count": int(self.global_margin_subset_random_k_count),
            "alloc_stages_total": len(alloc_rows),
            "phase2_gate1_score_min": float(self.phase2_gate1_score_min),
            "phase2_gate1_rescue_enabled": bool(self.phase2_gate1_rescue_enabled),
            "phase2_gate1_rescue_band": float(self.phase2_gate1_rescue_band),
            "phase2_gate1_rescue_gate2_min": float(self.phase2_gate1_rescue_gate2_min),
            "phase2_gate1_rescue_trigger_count": int(phase2_gate1_rescue_trigger_count),
            "phase2_gate2_score_min": float(self.phase2_gate2_score_min),
            "phase2_gate2_score_sticky_min": float(self.phase2_gate2_score_sticky_min),
            "phase2_disable_boundary_sampling": bool(self.phase2_disable_boundary_sampling),
            "gate_smoothing_enabled": bool(self.gate_smoothing_enabled),
            "gate_ema_alpha": float(self.gate_ema_alpha),
            "gate_ema_warmup_stages": int(self.gate_ema_warmup_stages),
            "gate_smoothing_use_for_phase2": bool(self.gate_smoothing_use_for_phase2),
            "gate_smoothing_use_for_stop": bool(self.gate_smoothing_use_for_stop),
            "collapse_span_ratio_threshold": float(self.collapse_span_ratio_threshold),
            "collapse_anchor_streak_threshold": int(self.collapse_anchor_streak_threshold),
            "collapse_min_stage": int(self.collapse_min_stage),
            "diversity_injection_ratio": float(self.diversity_injection_ratio),
            "diversity_injection_min_points": int(self.diversity_injection_min_points),
            "diversity_injection_max_ratio": float(self.diversity_injection_max_ratio),
            "diversity_boundary_floor_ratio": float(self.diversity_boundary_floor_ratio),
            "phase2_min_usable_np_ratio": float(self.phase2_min_usable_np_ratio),
            "phase2_np_ratio_cap_scale": float(self.phase2_np_ratio_cap_scale),
            "phase2_min_used_budget_ratio": float(self.phase2_min_used_budget_ratio),
            "phase2_min_used_budget_ratio_high_crate": float(self.phase2_min_used_budget_ratio_high_crate),
            "phase2_high_crate_threshold": float(self.phase2_high_crate_threshold),
            "phase2_g2_saturation_threshold": float(self.phase2_g2_saturation_threshold),
            "phase2_g2_saturation_min_remaining": float(self.phase2_g2_saturation_min_remaining),
            "g2_saturation_trigger_count": int(self._g2_saturation_trigger_count),
            # Bucket minima 진단 (DOE-3 대체)
            "global_bucket_minima_enabled": bool(self.global_bucket_minima_enabled),
            "global_bucket_minima_strict": bool(self.global_bucket_minima_strict),
            "global_boundary_high_crate_threshold": float(self.global_boundary_high_crate_threshold),
            "phase1_min_top": int(self.phase1_min_top),
            "phase1_min_margin": int(self.phase1_min_margin),
            "phase1_min_boundary_any": int(self.phase1_min_boundary_any),
            "phase1_min_boundary_classic_hc": int(self.phase1_min_boundary_classic_hc),
            "phase1_min_boundary_cross_hc": int(self.phase1_min_boundary_cross_hc),
            "phase2_min_top": int(self.phase2_min_top),
            "phase2_min_margin": int(self.phase2_min_margin),
            "global_bucket_plan_multiplier": int(self.global_bucket_plan_multiplier),
            "local_exec_min_anchors": int(self.local_exec_min_anchors),
            "bucket_min_relaxed_count": int(self._bucket_min_relaxed_count),
            "phase2_np_gate_last": phase2_np_gate_last,
            "gate2_phase_threshold_used_last": gate2_phase_threshold_used_last,
            "early_stop_min_used_budget_ratio": float(self.early_stop_min_used_budget_ratio),
            "early_stop_min_usable_np_ratio": float(self.early_stop_min_usable_np_ratio),
            "budget_used": int(self.budget.used),
            "budget_total": int(self.budget.total),
            "budget_exhausted": bool(budget_exhausted),
            "stage_count_total": stage_count_total,
            "stage_gate_eval_count": stage_gate_eval_count,
            "gate_eval_skipped_count": gate_eval_skipped_count,
            "phase2_first_stage": phase2_first_stage,
            "phase2_stage_count": phase2_stage_count,
            "phase2_transition_count": phase2_transition_count,
            "gate_stop_raw_count": gate_stop_raw_count,
            "gate_stop_final_count": gate_stop_final_count,
            "gate_stop_blocked_count": gate_stop_blocked_count,
            "collapse_detected_count": collapse_detected_count,
            "diversity_injection_applied_count": diversity_injection_applied_count,
            "diversity_sample_count_total": diversity_sample_count_total,
            "diversity_hit_rate_mean": diversity_hit_rate_mean,
            "used_budget_ratio_phase2_first": used_budget_ratio_phase2_first,
            "terminated_by": terminated_by,
            "doe_additional_triggered": bool(len(additional_rows) > 0),
            "doe_added_samples": int(len(additional_rows)),
            "doe_gate1_pass_rate": gate1_pass_rate,
            "doe_gate2_pass_rate": gate2_pass_rate,
            "doe_gate1_score_mean": gate1_score_mean,
            "doe_gate2_score_mean": gate2_score_mean,
            "doe_gate1_score_last": gate1_score_last,
            "doe_gate2_score_last": gate2_score_last,
            "doe_gate1_score_ema_mean": gate1_score_ema_mean,
            "doe_gate2_score_ema_mean": gate2_score_ema_mean,
            "doe_gate1_score_ema_last": gate1_score_ema_last,
            "doe_gate2_score_ema_last": gate2_score_ema_last,
            "doe_gate_decision_source_last": gate_decision_source_last,
            "doe_collapse_detected_count": collapse_detected_count,
            "doe_diversity_injection_applied_count": diversity_injection_applied_count,
            "doe_diversity_sample_count_total": diversity_sample_count_total,
            "doe_diversity_hit_rate_mean": diversity_hit_rate_mean,
            "phase2_entered": bool(self._ever_phase2),
            "has_pre_constraints": bool(self.has_pre_constraints),
            "has_post_constraints": bool(self.has_post_constraints),
            "local_span_ratio_mean_last": local_span_ratio_mean_last,
            "anchor_spread_zero_streak_max": int(self._anchor_spread_zero_streak_max),
        }
        if self.has_pre_constraints:
            out["constraint_stats"] = {
                "generated": int(self._constraint_gen_total),
                "feasible": int(self._constraint_feas_total),
            }
        return out

    # -------------------------------------------------
    # Internal helpers
    # -------------------------------------------------
    def _update_convergence_streak(self, metrics: dict) -> None:
        anchor_spread = metrics.get("anchor_spread_mean")
        if anchor_spread == 0:
            self._anchor_spread_zero_streak += 1
        else:
            self._anchor_spread_zero_streak = 0
        if self._anchor_spread_zero_streak > self._anchor_spread_zero_streak_max:
            self._anchor_spread_zero_streak_max = int(self._anchor_spread_zero_streak)

    @staticmethod
    def _update_ema(*, prev: float | None, value: float | None, alpha: float) -> float | None:
        if value is None:
            return prev
        if prev is None:
            return float(value)
        return float(alpha * float(value) + (1.0 - alpha) * float(prev))

    def _is_local_collapse(
        self,
        *,
        stage: int,
        span_ratio: float | None,
        anchor_streak: int,
    ) -> bool:
        if int(stage) < int(self.collapse_min_stage):
            return False
        if span_ratio is None:
            return False
        if float(span_ratio) > float(self.collapse_span_ratio_threshold):
            return False
        return bool(int(anchor_streak) >= int(self.collapse_anchor_streak_threshold))

    def _resolve_diversity_quota(self, *, n_exec: int) -> int:
        if n_exec <= 0:
            return 0
        if self.diversity_injection_ratio <= 0.0:
            return 0
        quota = int(round(float(n_exec) * float(self.diversity_injection_ratio)))
        quota = max(quota, int(self.diversity_injection_min_points))
        max_quota = int(np.floor(float(n_exec) * float(self.diversity_injection_max_ratio)))
        if max_quota <= 0:
            max_quota = 1
        quota = min(quota, max_quota, int(n_exec))
        return int(max(quota, 0))

    def _success_stats(self) -> tuple[int, int, float]:
        n_total = int(self.store.size)
        n_success = int(len(self.store.successful_rows))
        ratio = float(n_success / max(n_total, 1))
        return n_success, n_total, ratio

    def _usable_sample_count(self) -> int:
        return int(
            sum(
                1
                for r in self.store.rows
                if bool(getattr(r, "success", False)) and bool(getattr(r, "feasible", False))
            )
        )

    def _should_stop_by_convergence(self) -> bool:
        if not self.local_metrics:
            return False
        latest = self.local_metrics[-1]
        span_ratio = latest.get("local_span_ratio_mean")
        if span_ratio is not None and span_ratio < self.stop_span_ratio_threshold:
            return True
        if self._anchor_spread_zero_streak >= self.stop_anchor_spread_streak:
            return True
        return False

    def _sample_initial_corners(
        self,
        *,
        n_target: int,
        update_ratio: bool = False,
    ) -> tuple[np.ndarray, list[dict], list[float]]:
        n_target = int(max(n_target, 0))
        if n_target <= 0:
            return np.empty((0, len(self.bounds)), dtype=float), [], []

        offset = np.zeros((len(self.bounds),), dtype=float)
        n_corner_pool = n_target if not self.has_pre_constraints else max((2 * n_target), (n_target + 4))
        corner_pool = sample_boundary_corners_random(
            self.bounds,
            offset=offset,
            n_samples=n_corner_pool,
            rng=self.rng,
        )
        if corner_pool.size == 0:
            return np.empty((0, len(self.bounds)), dtype=float), [], []
        corner_pool = np.unique(np.asarray(corner_pool, dtype=float), axis=0)
        if corner_pool.shape[0] == 0:
            return np.empty((0, len(self.bounds)), dtype=float), [], []

        if not self.has_pre_constraints:
            n_take = min(n_target, corner_pool.shape[0])
            if n_take <= 0:
                return np.empty((0, len(self.bounds)), dtype=float), [], []
            pick_idx = self.rng.choice(corner_pool.shape[0], size=n_take, replace=False)
            X_corner = corner_pool[pick_idx].astype(float)
            return (
                X_corner,
                [{} for _ in range(X_corner.shape[0])],
                [float("inf")] * X_corner.shape[0],
            )

        remaining_idx = np.arange(corner_pool.shape[0], dtype=int)
        picked_rows: list[np.ndarray] = []
        picked_constraints: list[dict] = []
        picked_margins: list[float] = []

        # First draw + one additional draw from unused corners.
        for _attempt in range(2):
            need = n_target - len(picked_rows)
            if need <= 0 or remaining_idx.size == 0:
                break
            n_take = min(need, remaining_idx.size)
            pick_pos = self.rng.choice(remaining_idx.size, size=n_take, replace=False)
            pick_idx = remaining_idx[pick_pos]
            keep_mask = np.ones((remaining_idx.size,), dtype=bool)
            keep_mask[pick_pos] = False
            remaining_idx = remaining_idx[keep_mask]

            X_try = corner_pool[pick_idx]
            X_feas, constraints_feas, margins_feas, _ = self._filter_by_constraints(
                X_try,
                update_ratio=update_ratio,
            )
            if X_feas.size == 0:
                continue
            for i in range(X_feas.shape[0]):
                if len(picked_rows) >= n_target:
                    break
                picked_rows.append(X_feas[i].reshape(1, -1))
                picked_constraints.append(constraints_feas[i])
                picked_margins.append(float(margins_feas[i]))

        if not picked_rows:
            return np.empty((0, len(self.bounds)), dtype=float), [], []
        X_corner = np.vstack(picked_rows).astype(float)
        return X_corner, picked_constraints, picked_margins

    def _sample_initial(self, *, n_samples: int) -> np.ndarray:
        """
        Call user-selected sampler with compatible kwargs.
        """
        sig = inspect.signature(self.sampler)
        kwargs = {"n_samples": n_samples, "bounds": self.bounds, "rng": self.rng}
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        X = self.sampler(**accepted)
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.bounds):
            raise ValueError(f"Sampler returned invalid shape: {X.shape}")
        return X

    def _rank_by_pred(
        self,
        *,
        models,
        X_candidate: np.ndarray,
        objective_sense: str,
    ) -> np.ndarray:
        X_candidate = np.asarray(X_candidate, dtype=float)
        if X_candidate.ndim != 2 or X_candidate.shape[0] == 0:
            return np.array([], dtype=int)
        Yhat = np.vstack([m.predict(X_candidate) for m in models])
        y_mean = np.mean(Yhat, axis=0)
        if self.post_use_penalty and self.has_post_constraints and self.post_lambda_current > 0.0:
            p_feas = self._predict_post_feasible_prob(X_candidate)
            penalty = self.post_lambda_current * (1.0 - p_feas)
            if objective_sense == "min":
                score = y_mean + penalty
                order = np.argsort(score)
            else:
                score = y_mean - penalty
                order = np.argsort(-score)
        else:
            if objective_sense == "min":
                order = np.argsort(y_mean)
            else:
                order = np.argsort(-y_mean)
        return order.astype(int)

    def _execute_points(
        self,
        X: np.ndarray,
        *,
        source: str,
        round_idx: Optional[int],
        exec_scope: Optional[str] = None,
        constraints_payloads: list[dict] | None = None,
        constraint_margins: np.ndarray | None = None,
    ) -> bool:
        """
        Execute CAE on points.
        Returns False if budget exhausted, else True.
        """

        X = np.asarray(X, dtype=float)

        for i, x in enumerate(X):
            # 🔴 budget 먼저 확인
            if self.budget.remaining <= 0:
                return False

            out = self.evaluate_func(x)
            success, objective_raw, outputs, invalid_reason, raw_obj_repr = sanitize_evaluate_output(out)
            objective = canonical_objective_for_result(
                raw_objective=float(objective_raw),
                objective_sense=getattr(self, "raw_objective_sense", "min"),
                success=bool(success),
            )
            if invalid_reason is not None:
                print(
                    "[AdditionalDOE][INVALID_OBJECTIVE] "
                    f"stage={round_idx} exec_scope={exec_scope} idx={i} "
                    f"reason={invalid_reason} raw={raw_obj_repr} "
                    "-> success=False objective=inf"
                )

            if constraints_payloads is not None and i < len(constraints_payloads):
                constraints_pre = constraints_payloads[i] if constraints_payloads[i] is not None else {}
                feasible_pre = bool(self.feasibility_func(constraints_pre)) if constraints_pre is not None else True
                if constraint_margins is not None and i < len(constraint_margins):
                    margin_pre = float(constraint_margins[i])
                else:
                    margin_pre = float("inf")
            elif self.has_pre_constraints:
                constraints_pre, feasible_pre, margin_pre = evaluate_constraints_point(
                    x=x,
                    var_names=self.var_names,
                    constraint_defs=self.constraint_defs,
                    scope="pre",
                )
            else:
                constraints_pre = {}
                feasible_pre = True
                margin_pre = float("inf")

            constraints_post: dict = {}
            feasible_post = True
            margin_post = float("inf")
            if self.has_post_constraints:
                try:
                    constraints_post, feasible_post, margin_post = evaluate_constraints_point(
                        x=x,
                        var_names=self.var_names,
                        constraint_defs=self.constraint_defs,
                        scope="post",
                        env_extra={**outputs, "objective": objective_raw, "objective_raw": objective_raw},
                        fail_fast_output_missing=True,
                    )
                except Exception as exc:
                    self._set_failure("FAILED_POST_CONSTRAINT_OUTPUT", stage=int(round_idx or 0))
                    print(f"[AdditionalDOE] post-constraint evaluation failed: {exc}")
                    return False

            constraints = {**constraints_pre, **constraints_post}
            feasible = bool(success and feasible_pre and feasible_post)
            self._update_post_rate(feasible=bool(feasible))

            # store에는 기록 (실패도 기록 가능)
            self.store.add(
                x=x,
                objective=objective,
                objective_raw=float(objective_raw),
                constraints=constraints,
                feasible_pre=bool(feasible_pre),
                feasible_post=bool(feasible_post),
                feasible=bool(feasible),
                success=success,
                margin_pre=float(margin_pre),
                margin_post=float(margin_post),
                source=source,
                round_idx=round_idx,
                exec_scope=exec_scope,
            )

            consumed = self.budget.consume(1)
            if consumed < 1:
                return False

        return True

    def _set_failure(self, reason: str, *, stage: int) -> None:
        if self.failure_reason is None:
            self.failure_reason = reason
        print(f"[AdditionalDOE] stop_reason={reason} stage={stage}")

    def _maybe_run_hpo(self, *, problem_name: str, base_seed: int) -> None:
        if self.hpo_runner is None or self._hpo_done:
            return
        X_hpo = np.asarray(self.store.X_success, dtype=float)
        y_hpo = np.asarray(self.store.y_success, dtype=float)
        n_rows = int(X_hpo.shape[0])
        if n_rows < int(self.hpo_min_samples):
            return
        kfold_splits = min(5, n_rows)
        if kfold_splits < 3:
            return
        try:
            print(
                "[AdditionalDOE] run HPO "
                f"(n={n_rows}, kfold={kfold_splits}, seed={base_seed})"
            )
            result = self.hpo_runner.run_xgb(
                X=X_hpo,
                y=y_hpo,
                base_random_seed=base_seed,
                problem_name=problem_name,
                kfold_splits=kfold_splits,
            )
            best_params = result["best_params"]
            self.surrogate_factory.xgb_params = {
                **self.surrogate_factory.xgb_params,
                **best_params,
            }
            self._hpo_done = True
            print("[AdditionalDOE] HPO applied to surrogate xgb_params")
        except Exception as exc:
            print(f"[AdditionalDOE] HPO skipped due to error: {exc}")

    def _log_local_metrics(self, *, stage: int, metrics: dict) -> None:
        payload = {"stage": stage, **metrics}
        self.local_metrics.append(payload)
        span_ratio = metrics.get("local_span_ratio_mean")
        pairwise_mean = metrics.get("exec_pairwise_dist_mean")
        anchor_spread = metrics.get("anchor_spread_mean")
        if span_ratio is None:
            print(f"[LocalMetrics] stage={stage} local empty")
            return
        print(
            "[LocalMetrics] "
            f"stage={stage} span_ratio_mean={span_ratio:.4f} "
            f"pairwise_mean={pairwise_mean} anchor_spread_mean={anchor_spread}"
        )
        best_in_topk = metrics.get("best_in_topk")
        best_anchor_dist = metrics.get("best_anchor_dist")
        best_local_min_dist = metrics.get("best_local_min_dist")
        if best_in_topk is not None or best_anchor_dist is not None or best_local_min_dist is not None:
            print(
                "[LocalDebug] "
                f"stage={stage} best_in_topk={best_in_topk} "
                f"best_anchor_dist={best_anchor_dist} best_local_min_dist={best_local_min_dist}"
            )




    def _export_results(self):
        results = []
        rows = self.store.rows  

        for i, r in enumerate(rows):
            pre = float(r.margin_pre)
            post = float(r.margin_post)
            pre_ok = bool(np.isfinite(pre))
            post_ok = bool(np.isfinite(post))
            if pre_ok and not post_ok:
                constraint_margin = pre
            elif post_ok and not pre_ok:
                constraint_margin = post
            elif pre_ok and post_ok:
                constraint_margin = float(min(pre, post))
            else:
                constraint_margin = float("inf")

            results.append(
                {
                    "id": i,
                    "x": r.x.tolist(),
                    "objective": float(r.objective),
                    "objective_raw": float(r.objective_raw),
                    "constraints": r.constraints,
                    "margin_pre": float(r.margin_pre),
                    "margin_post": float(r.margin_post),
                    "constraint_margin": float(constraint_margin),
                    "feasible_pre": bool(r.feasible_pre),
                    "feasible_post": bool(r.feasible_post),
                    "feasible": bool(r.feasible),
                    "success": bool(r.success),
                    "source": r.source,
                    "round": r.round_idx,
                    "exec_scope": r.exec_scope,
                }
            )
        return results
