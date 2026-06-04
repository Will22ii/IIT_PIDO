from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from Explorer.config import ExplorerSystemConfig
from pipeline.run_context import RunContext


@dataclass
class ExplorerReportPayload:
    inputs: dict[str, Any]
    resolved_params: dict[str, Any]
    analysis_metadata: dict[str, Any]
    results_summary: dict[str, Any]


def _rel_or_abs(path: str | None, *, task_dir: str) -> str | None:
    if not path:
        return None
    try:
        return os.path.relpath(path, task_dir)
    except ValueError:
        return path


def build_explorer_report_payload(
    *,
    context: Mapping[str, Any],
    task_dir: str,
    run_context: RunContext,
    system_config: ExplorerSystemConfig,
) -> ExplorerReportPayload:
    """Build read-only metadata/report payload after Explorer scoring is done."""
    selected_features = context["selected_features"]
    base_n = context["base_n"]
    strategy_id = context["strategy_id"]
    strategy_alias = context["strategy_alias"]
    strategy_alias_requested = context["strategy_alias_requested"]
    strategy_mode = context["strategy_mode"]
    vol_ratio = context["vol_ratio"]
    raw_objective_sense = str(context.get("raw_objective_sense", context["objective_sense"]))
    internal_objective_sense = str(
        context.get("internal_objective_sense", context["objective_sense"])
    )
    objective_transform = "negate" if raw_objective_sense == "max" else "identity"

    previous: dict[str, str] = {}
    data_csv_ref = _rel_or_abs(context.get("doe_csv_path"), task_dir=task_dir)
    model_ref = _rel_or_abs(context.get("modeler_pkl_path"), task_dir=task_dir)
    selected_features_ref = _rel_or_abs(
        context.get("selected_features_csv_path"),
        task_dir=task_dir,
    )
    if data_csv_ref:
        previous["data_csv"] = data_csv_ref
    if model_ref:
        previous["model_bundle"] = model_ref
    if selected_features_ref:
        previous["selected_features_csv"] = selected_features_ref

    inputs = {
        "user_config": os.path.relpath(
            run_context.user_config_snapshot_path,
            task_dir,
        ),
        "system_config_snapshot": {
            "n_samples": context["n_samples"],
            "sample_multiplier": context["sample_multiplier"],
            "quantile_threshold": context["quantile_threshold"],
            "min_topk_count": int(context["min_topk_count"]),
            "max_topk_count": (
                None
                if getattr(system_config, "max_topk_count", None) is None
                else int(getattr(system_config, "max_topk_count"))
            ),
            "max_topk_count_used": (
                int(context["max_topk_count_used"])
                if context["max_topk_count_used"] is not None
                else None
            ),
            "dbscan_min_samples": context["dbscan_min_samples"],
            "debug_level": context["debug_level"],
            "strategy_id": strategy_id,
            "probe_multistart": int(system_config.probe_multistart),
            "dual_policy_mode_default": str(
                getattr(system_config, "dual_policy_mode_default", "routed_v2")
            ),
            "dual_tilt_disagree_ref": float(
                getattr(system_config, "dual_tilt_disagree_ref", 0.10)
            ),
            "constraint_aware_enabled": bool(
                getattr(system_config, "constraint_aware_enabled", True)
            ),
            "constraint_aware_edge_ratio": float(
                getattr(system_config, "constraint_aware_edge_ratio", 0.05)
            ),
            "constraint_aware_probe_anchor_max": int(
                getattr(system_config, "constraint_aware_probe_anchor_max", 16)
            ),
            "constraint_aware_probe_steps": int(
                getattr(system_config, "constraint_aware_probe_steps", 3)
            ),
            "constraint_aware_min_side_samples": int(
                getattr(system_config, "constraint_aware_min_side_samples", 24)
            ),
            "constraint_aware_side_rate_min": float(
                getattr(system_config, "constraint_aware_side_rate_min", 0.55)
            ),
            "constraint_aware_side_rate_gap": float(
                getattr(system_config, "constraint_aware_side_rate_gap", 0.12)
            ),
            "constraint_aware_pre_shift_enabled": bool(
                getattr(system_config, "constraint_aware_pre_shift_enabled", True)
            ),
            "constraint_aware_use_for_volume_cap": bool(
                getattr(system_config, "constraint_aware_use_for_volume_cap", True)
            ),
            "volume_cap_boundary_pin_tol_ratio": float(
                getattr(system_config, "volume_cap_boundary_pin_tol_ratio", 0.03)
            ),
            "strategy_params": dict(system_config.strategy_params or {}),
        },
        "previous": previous,
        "selected_features": selected_features,
        "model_path_input": context["modeler_pkl_path"],
        "selected_features_csv_input": context["selected_features_csv_path"],
        "doe_csv_input": context["doe_csv_path"],
    }

    usable_n_over_p = (
        float(base_n) / float(max(len(selected_features), 1))
        if len(selected_features) > 0
        else None
    )
    resolved_params = {
        "p_dim": int(len(selected_features)),
        "usable_n": int(base_n),
        "usable_n_over_p": usable_n_over_p,
        "has_pre_constraints": bool(context["has_pre_constraints"]),
        "has_post_constraints": bool(context["has_post_constraints"]),
        "selected_bounds_volume_ratio": vol_ratio,
        "strategy_id": strategy_id,
        "strategy_alias": strategy_alias,
        "strategy_alias_requested": strategy_alias_requested,
        "strategy_mode": strategy_mode,
        "dual_obj_equivalent_forced": bool(context["dual_obj_equivalent_forced"]),
        "dual_volume_cap_applied": bool(context["dual_volume_cap_applied"]),
        "dse_cap_boundary_touch_dim_count": int(
            len(context["dse_cap_boundary_touch_dims"])
        ),
        "dse_cap_center_realign_applied": bool(
            context["dse_cap_center_realign_applied"]
        ),
        "dse_cap_force_pin_applied": bool(context["dse_cap_force_pin_applied"]),
        "constraint_aware_enabled": bool(context["constraint_aware_enabled"]),
        "constraint_aware_policy_applied": bool(
            context["constraint_aware_policy_applied"]
        ),
        "constraint_aware_probe_count": context["constraint_aware_probe_count"],
        "constraint_aware_anchor_count": context["constraint_aware_anchor_count"],
        "constraint_aware_shifted_dims": context["constraint_aware_shifted_dims"],
        "objective_sense": raw_objective_sense,
        "raw_objective_sense": raw_objective_sense,
        "canonical_objective_sense": internal_objective_sense,
        "objective_transform": objective_transform,
    }

    routed_v2_decision = context["routed_v2_decision"]
    analysis_metadata = {
        "run_id": run_context.run_id if run_context is not None else None,
        "problem": context["doe_problem_name"],
        "seed": int(context["rng_seed"]),
        "objective_sense": raw_objective_sense,
        "raw_objective_sense": raw_objective_sense,
        "canonical_objective_sense": internal_objective_sense,
        "internal_objective_sense": internal_objective_sense,
        "objective_transform": objective_transform,
        "strategy_id": strategy_id,
        "strategy_alias": strategy_alias,
        "strategy_alias_requested": strategy_alias_requested,
        "strategy_mode": strategy_mode,
        "dual_obj_equivalent_forced": bool(context["dual_obj_equivalent_forced"]),
        "p_dim": int(len(selected_features)),
        "usable_n": int(base_n),
        "usable_n_over_p": usable_n_over_p,
        "has_pre_constraints": bool(context["has_pre_constraints"]),
        "has_post_constraints": bool(context["has_post_constraints"]),
        "selected_bounds_volume_ratio": vol_ratio,
        "selected_bounds_volume_ratio_before_cap": (
            float(context["selected_bounds_volume_ratio_before_cap"])
            if context["selected_bounds_volume_ratio_before_cap"] is not None
            else None
        ),
        "pred_bounds_volume_ratio": (
            float(context["pred_bounds_volume_ratio"])
            if context["pred_bounds_volume_ratio"] is not None
            else None
        ),
        "obj_bounds_volume_ratio": (
            float(context["obj_bounds_volume_ratio"])
            if context["obj_bounds_volume_ratio"] is not None
            else None
        ),
        "selected_bounds_center_l1_to_design_center": (
            float(context["selected_bounds_center_l1_to_design_center"])
            if context["selected_bounds_center_l1_to_design_center"] is not None
            else None
        ),
        "selected_bounds_widths_normalized": (
            list(context["selected_bounds_widths_normalized"])
            if context["selected_bounds_widths_normalized"] is not None
            else None
        ),
        "pred_bounds_widths_normalized": (
            list(context["pred_bounds_widths_normalized"])
            if context["pred_bounds_widths_normalized"] is not None
            else None
        ),
        "obj_bounds_widths_normalized": (
            list(context["obj_bounds_widths_normalized"])
            if context["obj_bounds_widths_normalized"] is not None
            else None
        ),
        "selected_bounds_volume_ratio_pre_l2_expand": (
            float(context["selected_bounds_volume_ratio_pre_l2_expand"])
            if context["selected_bounds_volume_ratio_pre_l2_expand"] is not None
            else None
        ),
        "pre_cap_l2_expand_applied": bool(context["pre_cap_l2_expand_applied"]),
        "dual_volume_cap_applied": bool(context["dual_volume_cap_applied"]),
        "dse_cap_boundary_touch_dims": context["dse_cap_boundary_touch_dims"],
        "dse_cap_boundary_touch_dim_count": int(
            len(context["dse_cap_boundary_touch_dims"])
        ),
        "dse_cap_boundary_touch_eps_ratio": context["dse_cap_boundary_touch_eps_ratio"],
        "dse_cap_center_realign_applied": bool(
            context["dse_cap_center_realign_applied"]
        ),
        "dse_cap_center_blend_used": context["dse_cap_center_blend_used"],
        "dse_cap_force_pin_applied": bool(context["dse_cap_force_pin_applied"]),
        "constraint_aware_enabled": bool(context["constraint_aware_enabled"]),
        "constraint_aware_policy_applied": bool(
            context["constraint_aware_policy_applied"]
        ),
        "constraint_aware_probe_count": context["constraint_aware_probe_count"],
        "constraint_aware_anchor_count": context["constraint_aware_anchor_count"],
        "constraint_aware_shifted_dims": context["constraint_aware_shifted_dims"],
        "constraint_aware_top_obj_veto_count": int(
            context["constraint_aware_top_obj_veto_count"]
        ),
        "constraint_aware_top_obj_confirmed_count": int(
            context["constraint_aware_top_obj_confirmed_count"]
        ),
        "constraint_aware_obj_centroid_blend_applied": bool(
            context["constraint_aware_obj_centroid_blend_applied"]
        ),
        "constraint_aware_shift_boost_applied": bool(
            context["constraint_aware_shift_boost_applied"]
        ),
        "constraint_aware_shift_fraction_used": (
            float(context["constraint_aware_shift_fraction_used"])
            if context["constraint_aware_shift_fraction_used"] is not None
            else None
        ),
        "constraint_aware_side_scores": (
            context["constraint_aware_side_scores"]
            if isinstance(context["constraint_aware_side_scores"], list)
            else []
        ),
        "post_penalty_active": bool(context["has_post_penalty"]),
        "post_lambda": float(context["post_lambda"]),
        "generated_total_raw": int(context["pre_generated"]),
        "generated_total_after_pre": int(context["X"].shape[0]),
        "generated_boundary": int(context["X_boundary_raw"].shape[0]),
        "r_used": float(context["r_used"]),
        "min_topk_count": int(context["min_topk_count"]),
        "max_topk_count_used": (
            int(context["max_topk_count_used"])
            if context["max_topk_count_used"] is not None
            else None
        ),
        "pred_cluster_signal_mode": context["pred_cluster_signal_mode"],
        "pred_cluster_beta_used": context["pred_cluster_beta_used"],
        "pred_cluster_confidence": context["pred_cluster_confidence"],
        "pred_cluster_selected_count": context["pred_cluster_selected_count"],
        "pred_cluster_sigma_mean_selected": context[
            "pred_cluster_sigma_mean_selected"
        ],
        "pred_refine_shift_norm": context["pred_refine_shift_norm"],
        "pred_obj_miss_case": context["pred_obj_miss_case"],
        "pred_obj_iou": context["pred_obj_iou"],
        "pred_obj_center_l1_norm": context["pred_obj_center_l1_norm"],
        "pred_multistart_det_fraction_used": context[
            "pred_multistart_det_fraction_used"
        ],
        "pred_refine_bounds_scale_used": context["pred_refine_bounds_scale_used"],
        "pred_n_starts_used": context["pred_n_starts_used"],
        "dual_policy_mode": context["dual_policy_mode"],
        "dual_total_starts_target": context["dual_total_starts_target"],
        "dual_total_starts_used": context["dual_total_starts_used"],
        "dual_np_ratio_used": context["dual_np_ratio_used"],
        "dual_pred_ratio_used": context["dual_pred_ratio_used"],
        "dual_obj_ratio_used": context["dual_obj_ratio_used"],
        "dual_n_pred_starts": context["dual_n_pred_starts"],
        "dual_n_obj_starts": context["dual_n_obj_starts"],
        "dual_disagreement_center_l1_norm": context[
            "dual_disagreement_center_l1_norm"
        ],
        "dual_disagreement_iou": context["dual_disagreement_iou"],
        "dual_disagreement_triggered": bool(context["dual_disagreement_triggered"]),
        "dual_center_bias_used": context["dual_center_bias_used"],
        "dual_center_tilt_strength_base": context["dual_center_tilt_strength_base"],
        "dual_center_tilt_strength_used_mean": context[
            "dual_center_tilt_strength_used_mean"
        ],
        "dual_center_tilt_applied": bool(context["dual_center_tilt_applied"]),
        "dual_volume_cap_target": context["dual_volume_cap_target"],
        "dual_volume_ratio_before_cap": context["dual_volume_ratio_before_cap"],
        "router_decision": (
            routed_v2_decision.to_dict() if routed_v2_decision is not None else None
        ),
    }

    pred_stats = context["pred_stats"]
    n_clusters = int(pred_stats.get("n_clusters", 0)) if isinstance(pred_stats, dict) else 0
    results_summary = {
        "n_candidates_total": int(len(context["df"])),
        "n_q_samples": context["n_q_samples"],
        "n_clusters": n_clusters,
        "cluster_sizes": [],
        "pred_mean_volume": context["pred_mean_vol"],
        "obj_mean_volume": context["obj_mean_vol"],
        "strategy_id": strategy_id,
    }

    return ExplorerReportPayload(
        inputs=inputs,
        resolved_params=resolved_params,
        analysis_metadata=analysis_metadata,
        results_summary=results_summary,
    )
