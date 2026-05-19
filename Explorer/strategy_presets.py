from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from Explorer.config import ExplorerSystemConfig


@dataclass(frozen=True)
class ExplorerStrategy:
    strategy_id: str
    overrides: dict[str, Any]


def _strategy_overrides(
    *,
    strategy_params: dict[str, Any],
    quantile_threshold: float,
    bounds_margin_ratio: float,
    dbscan_eps_quantile: float,
    bounds_expansion_mode: str = "fi_aware",
) -> dict[str, Any]:
    return {
        "strategy_params": dict(strategy_params),
        "bounds_expansion_mode": bounds_expansion_mode,
        "quantile_threshold": quantile_threshold,
        "bounds_margin_ratio": bounds_margin_ratio,
        "dbscan_eps_quantile": dbscan_eps_quantile,
    }


_DUAL_SHARED_PARAMS: dict[str, Any] = {
    "max_volume_ratio_target": 0.249,
    "dual_policy_mode": "routed_v2",
    "dual_total_starts": 40,
    "dual_np_ratio_low": 12.0,
    "dual_np_ratio_high": 24.0,
    "dual_obj_ratio_low_np": 0.62,
    "dual_obj_ratio_mid_np": 0.50,
    "dual_obj_ratio_high_np": 0.42,
    "dual_high_dim_threshold": 6,
    "dual_high_dim_obj_bonus": 0.08,
    "dual_disagree_l1_threshold": 0.35,
    "dual_disagree_iou_threshold": 0.20,
    "dual_disagree_obj_bonus": 0.12,
    "dual_obj_ratio_min": 0.25,
    "dual_obj_ratio_max": 0.75,
    "dual_center_tilt_strength": 0.45,
    "dual_center_tilt_aniso_gamma": 0.6,
    "dual_center_bias_obj_ratio_weight": 0.50,
    "pred_cluster_beta": 0.20,
    "pred_refine_bounds_scale": 1.30,
    "pred_multistart_det_fraction": 0.35,
    "pred_obj_disjoint_iou": 0.10,
    "pred_obj_fallback_conf_high": 0.45,
    "pred_obj_fallback_iou_low": 0.15,
    "pred_obj_fallback_center_blend": 0.50,
    "pred_conf_danger_low": 0.40,
    "pred_conf_danger_high": 0.60,
    "pred_conf_danger_iou_max": 0.15,
    "obj_diversity_extra_clusters": 2,
    "obj_diversity_weight": 0.35,
    "obj_diversity_min_distance": 0.22,
    "obj_diversity_close_penalty": 0.80,
    "obj_diversity_min_dim": 4,
    "mode": "dual_refine_ei",
}


_OBJ_SHARED_PARAMS: dict[str, Any] = {
    "max_volume_ratio_target": 0.249,
    "dual_policy_mode": "routed_v2",
    "obj_refine_bounds_scale": 1.45,
    "obj_diversity_extra_clusters": 2,
    "obj_diversity_weight": 0.35,
    "obj_diversity_min_distance": 0.22,
    "obj_diversity_close_penalty": 0.80,
    "obj_diversity_min_dim": 4,
    "mode": "obj_refine_ei",
}


EXPLORER_STRATEGIES: tuple[ExplorerStrategy, ...] = (
    ExplorerStrategy(
        "S4_obj",
        _strategy_overrides(
            strategy_params=_OBJ_SHARED_PARAMS,
            quantile_threshold=0.88,
            bounds_margin_ratio=0.03,
            dbscan_eps_quantile=0.90,
        ),
    ),
    ExplorerStrategy(
        "S4_dual",
        _strategy_overrides(
            strategy_params=_DUAL_SHARED_PARAMS,
            quantile_threshold=0.89,
            bounds_margin_ratio=0.02,
            dbscan_eps_quantile=0.88,
        ),
    ),
)


VALID_EXPLORER_STRATEGY_IDS: set[str] = {s.strategy_id for s in EXPLORER_STRATEGIES}


def strategy_map() -> dict[str, ExplorerStrategy]:
    return {s.strategy_id: s for s in EXPLORER_STRATEGIES}


def normalize_explorer_strategy_id(strategy_id: str | None) -> str:
    raw = str(strategy_id or "S4_obj").strip()
    for valid in VALID_EXPLORER_STRATEGY_IDS:
        if raw.lower() == valid.lower():
            return valid
    raise ValueError(
        f"Unsupported Explorer strategy_id: {strategy_id}. "
        f"Valid strategies: {sorted(VALID_EXPLORER_STRATEGY_IDS)}"
    )


def apply_explorer_strategy_preset(
    system_cfg: ExplorerSystemConfig,
    strategy_id: str | None = None,
) -> ExplorerSystemConfig:
    normalized = normalize_explorer_strategy_id(strategy_id or system_cfg.strategy_id)
    preset = strategy_map()[normalized]
    existing_params = dict(system_cfg.strategy_params or {})

    system_cfg.strategy_id = normalized
    for key, value in preset.overrides.items():
        if key == "strategy_params" and isinstance(value, dict):
            merged = copy.deepcopy(value)
            merged.update(existing_params)
            system_cfg.strategy_params = merged
            continue
        if hasattr(system_cfg, key):
            setattr(system_cfg, key, copy.deepcopy(value))
    return system_cfg

