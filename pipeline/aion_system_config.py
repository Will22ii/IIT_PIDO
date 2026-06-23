from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from Explorer.strategy_presets import apply_explorer_strategy_preset
from pipeline.config import PipelineConfig

if TYPE_CHECKING:
    from Optimizer.config import OptimizerSystemConfig


def _aion_optimizer_overrides() -> dict[str, object]:
    return {
        "focus3_profile_id": "aion_trusted_bounds_v1",
        # Modeler에서 탈락한 변수는 Optimizer 탐색 좌표로 되살리지 않는다.
        # AION 기본값은 Optimizer 입력/archive CSV의 best feasible full row를
        # omitted-feature context로 사용한다. 임시 실험은 run_pipelines.py에서
        # user_value=0.0으로 override할 수 있다.
        "omitted_feature_freeze_mode": "auto",
        "omitted_feature_freeze_value": 0.0,
        # AION Explorer bounds는 신뢰도 높은 근거로 본다. 일반 boundary/random 탐색보다
        # 해당 bounds 내부의 구조적 exploitation에 Focus3 예산을 더 쓴다.
        "focus3_ultra_low_source_topk_prob": 0.78,
        "focus3_ultra_low_source_boundary_prob": 0.05,
        "focus3_ultra_low_source_random_prob": 0.17,
        "focus3_low_source_topk_prob": 0.82,
        "focus3_low_source_boundary_prob": 0.04,
        "focus3_low_source_random_prob": 0.14,
        "focus3_normal_source_topk_prob": 0.86,
        "focus3_normal_source_boundary_prob": 0.03,
        "focus3_normal_source_random_prob": 0.11,
        "focus3_rich_source_topk_prob": 0.88,
        "focus3_rich_source_boundary_prob": 0.02,
        "focus3_rich_source_random_prob": 0.10,
        "focus3_low_reliability_boundary_bonus": 0.0,
        "focus3_low_reliability_random_bonus": 0.06,
        "focus3_mid_reliability_random_bonus": 0.02,
        "focus3_gp_fallback_boundary_bonus": 0.0,
        "focus3_gp_fallback_random_bonus": 0.08,
        "source_stagnation_boundary_bonus": 0.0,
        "source_stagnation_random_bonus": 0.01,
        # 무제약 benchmark에서는 boundary/random 과선택이 관측됐다.
        # 작은 coverage만 유지하고, 명확한 acquisition 우위가 있을 때만 허용한다.
        "focus3_no_constraint_boundary_max": 0.02,
        "focus3_no_constraint_topk_min": 0.72,
        "focus3_no_constraint_random_min": 0.03,
        "focus3_no_constraint_boundary_gate_margin_ratio": 0.16,
        "focus3_no_constraint_boundary_score_penalty_ratio": 0.45,
        "focus3_no_constraint_random_pool_ratio": 0.25,
        "focus3_no_constraint_random_refine_gate_margin_ratio": 0.16,
        "focus3_no_constraint_random_discrete_gate_margin_ratio": 0.16,
        "focus3_recover_random_discrete_gate_margin_ratio": 0.12,
        "focus3_recover_random_discrete_gate_min_no_improve": 80,
        # source-performance 전환을 더 일찍 켠다. 이전 run에서는 작동했지만,
        # plateau에 빠진 Focus3 trajectory를 구하기에는 너무 늦었다.
        "focus3_source_performance_recover_only": False,
        "focus3_source_performance_window": 80,
        "focus3_source_performance_min_focus3_evals": 30,
        "focus3_source_performance_min_source_count": 8,
        "focus3_source_performance_poor_improve_rate": 0.006,
        "focus3_source_performance_random_penalty_fraction": 0.90,
        "focus3_source_performance_random_min_quota_fraction": 0.01,
        "focus3_source_performance_boundary_min_count": 8,
        "focus3_source_performance_boundary_poor_improve_rate": 0.003,
        "focus3_source_performance_boundary_penalty_fraction": 0.95,
        "focus3_source_performance_boundary_min_quota_fraction": 0.0,
        "focus3_source_performance_local_probe_min_count": 8,
        "focus3_source_performance_local_probe_min_quota_fraction": 0.03,
        "focus3_source_performance_best_plan_filter_near_goal_preferred_source": "performance",
        "focus3_source_performance_best_plan_filter_near_goal_random_fallback_enabled": True,
        "focus3_source_performance_best_plan_filter_near_goal_random_min_advantage": 0.005,
        # 다중 incumbent local exploitation: 상위 archive basin에 quota를 더 주되,
        # 모든 후보가 현재 단일 best에만 묶이지 않게 한다.
        "focus3_best_local_prob": 0.45,
        "focus3_best_local_max_prob": 0.68,
        "focus3_best_local_min_focus3_evals": 1,
        "focus3_best_local_min_data_ratio": 3.0,
        "focus3_best_local_sigma": 0.018,
        "focus3_best_local_sigma_mid_eval": 70,
        "focus3_best_local_sigma_mid": 0.008,
        "focus3_best_local_sigma_late_eval": 220,
        "focus3_best_local_sigma_late": 0.003,
        "focus3_best_local_sigma_recover_strong_multiplier": 0.90,
        "focus3_best_local_top_count": 8,
        "focus3_best_local_pool_ratio": 1.0,
        "focus3_best_local_elite_std_scale": 0.65,
        "focus3_best_local_max_sigma": 0.10,
        "focus3_best_local_anchor_best_prob": 0.28,
        # AION recovery는 넓은 random 탐색으로 돌아가기 전에
        # 구조적 local search를 먼저 강화한다.
        "focus3_recover_window": 40,
        "focus3_recover_min_history": 45,
        "focus3_recover_boundary_bonus": 0.05,
        "focus3_recover_random_bonus": 0.03,
        "focus3_recover_mild_boundary_scale": 0.0,
        "focus3_recover_mild_random_scale": 0.25,
        "focus3_recover_mild_kappa_multiplier": 1.03,
        "focus3_recover_kappa_multiplier": 1.12,
        "focus3_recover_max_kappa": 2.20,
        "focus3_no_constraint_recover_boundary_scale": 0.0,
        "focus3_no_constraint_recover_random_scale": 0.05,
        "focus3_no_constraint_recover_topk_bonus": 0.22,
        "focus3_no_constraint_recover_strong_no_improve": 260,
        "focus3_no_constraint_recover_best_local_mild_bonus": 0.16,
        "focus3_no_constraint_recover_best_local_strong_bonus": 0.30,
        "focus3_no_constraint_recover_best_local_max": 0.72,
        "focus3_no_constraint_recover_best_local_late_no_improve": 220,
        "focus3_no_constraint_recover_best_local_late_bonus": 0.10,
        "focus3_no_constraint_recover_best_local_late_max": 0.80,
        "focus3_local_probe_min_no_improve": 35,
        "focus3_local_probe_prob": 0.18,
        "focus3_local_probe_max_prob": 0.30,
        "focus3_local_probe_refine_floor_fraction": 0.16,
        "focus3_local_probe_late_no_improve": 160,
        "focus3_local_probe_late_refine_floor_fraction": 0.22,
        "focus3_local_probe_pool_max": 256,
        "focus3_local_probe_step_ratio": 0.025,
        "focus3_local_probe_min_step_ratio": 0.001,
        "focus3_local_probe_scales": "1.0,0.5,0.25,0.1,0.05,0.025,0.01,0.005,1.75,2.5,3.5",
        # 신뢰 가능한 좁은 AION bounds는 exploitation으로 조금 더 빨리 진입시킨다.
        "focus3_auto_mean_min_data_ratio": 12.0,
        "focus3_auto_mean_max_volume_ratio": 0.35,
        "focus3_auto_ei_max_volume_ratio": 0.35,
        "focus3_auto_ei_max_mean_width_ratio": 0.85,
        # refine scheduling은 adaptive로 유지하되,
        # discrete/refine 성능 차이에 더 일찍 반응한다.
        "focus3_refine_adaptive_window": 60,
        "focus3_refine_adaptive_min_samples": 5,
        "focus3_refine_adaptive_max_every": 3,
        "focus3_refine_adaptive_worse_every": 3,
        "focus3_refine_cooldown_min_focus3_evals": 50,
        "focus3_refine_cooldown_window": 40,
        # 제약 문제가 feasible border 근처에서도 동작할 수 있게 유지한다.
        "focus3_constraint_boundary_prob": 0.25,
    }


@dataclass
class AIONSystemConfig:
    explorer_strategy_id: str = "S4_dual"
    optimizer_focus_planner_profile: str = "aion"
    optimizer_focus3_profile_id: str = "aion_trusted_bounds_v1"
    optimizer_system_overrides: dict[str, object] = field(default_factory=_aion_optimizer_overrides)
    # AION 전용 Explorer 연계. DOE metadata를 standalone Explorer 입력으로 만들지 않으면서
    # router decision에 필요한 DOE diagnostic signal만 복원한다.
    enable_doe_router_signals: bool = True


AION_SYSTEM_CONFIG = AIONSystemConfig()


def apply_aion_optimizer_system_config(system: "OptimizerSystemConfig") -> "OptimizerSystemConfig":
    system.focus_planner_profile = str(AION_SYSTEM_CONFIG.optimizer_focus_planner_profile)
    for key, value in dict(AION_SYSTEM_CONFIG.optimizer_system_overrides).items():
        if not hasattr(system, key):
            raise RuntimeError(f"Unknown AION optimizer system override: {key}")
        setattr(system, key, value)
    if hasattr(system, "focus3_profile_id"):
        system.focus3_profile_id = str(AION_SYSTEM_CONFIG.optimizer_focus3_profile_id)
    return system


def apply_aion_system_config(config: PipelineConfig) -> PipelineConfig:
    if not bool(getattr(config, "aion_mode", False)):
        return config

    if config.explorer is not None:
        apply_explorer_strategy_preset(
            config.explorer.system,
            AION_SYSTEM_CONFIG.explorer_strategy_id,
        )
        config.explorer.system.enable_doe_router_signals = bool(
            AION_SYSTEM_CONFIG.enable_doe_router_signals
        )

    if config.optimizer is not None:
        apply_aion_optimizer_system_config(config.optimizer.system)

    return config
