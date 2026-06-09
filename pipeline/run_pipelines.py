from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from CAE_tool_interface.config import CAEConfig, CAEUserConfig, CAESystemConfig
from DOE.config import DOEConfig, DOESystemConfig, DOEUserConfig
from Explorer.config import ExplorerConfig, ExplorerSystemConfig, ExplorerUserConfig
from Explorer.executor.explorer_orchestrator import ExplorerOrchestrator
from Explorer.strategy_presets import EXPLORER_STRATEGIES, ExplorerStrategy
from Explorer.strategy_presets import apply_explorer_strategy_preset
from Explorer.strategy_presets import strategy_map
from Modeler.config import ModelerConfig, ModelerSystemConfig, ModelerUserConfig
from pipeline.aion_system_config import AION_SYSTEM_CONFIG, apply_aion_system_config
from pipeline.config import PipelineConfig, PipelineTasks
from pipeline.run_pipeline import run_pipeline


@dataclass(frozen=True)
class ProblemCase:
    problem_name: str
    known_optimum: Any
    # DOE budget. Optimizer budget is configured separately below.
    n_samples: int
    optimizer_n_samples: int | None = None
    objective_sense: str = "min"
    repeats: int = 1
    optimizer_goal: float | None = None


PROBLEM_CASE_PRESETS: dict[str, ProblemCase] = {
    "rosenbrock": ProblemCase(
        problem_name="rosenbrock",
        known_optimum={"x1": 1.0, "x2": 1.0, "x3": 1.0, "x4": 1.0, "x5": 1.0},
        optimizer_goal=1.2,
        n_samples=450,
        optimizer_n_samples=1050,
        repeats=10,
    ),
    "cantilever_beam": ProblemCase(
        problem_name="cantilever_beam",
        known_optimum={"H": 7.0, "h1": 0.1, "b1": 9.48482, "b2": 0.1},
        optimizer_goal=114.66,
        n_samples=90,
        optimizer_n_samples=60,
        repeats=25,
    ),
    "goldstein_price": ProblemCase(
        problem_name="goldstein_price",
        known_optimum={"x1": 0.0, "x2": -1.0},
        optimizer_goal=3.6,
        n_samples=150,
        optimizer_n_samples=350,
        repeats=50,
    ),
    "six_hump_camel": ProblemCase(
        problem_name="six_hump_camel",
        known_optimum=[
            {"x1": 0.0898, "x2": -0.7126},
            {"x1": -0.0898, "x2": 0.7126},
        ],
        optimizer_goal=-0.86,
        n_samples=50,
        optimizer_n_samples=50,
        repeats=25,
    ),
    "rosenbrock_nodummy": ProblemCase(
        problem_name="rosenbrock_nodummy",
        known_optimum={"x1": 1.0, "x2": 1.0, "x3": 1.0, "x4": 1.0, "x5": 1.0},
        optimizer_goal=1.2,
        n_samples=450,
        optimizer_n_samples=1500,
        repeats=10,
    ),
    "cantilever_beam_nodummy": ProblemCase(
        problem_name="cantilever_beam_nodummy",
        known_optimum={"H": 7.0, "h1": 0.1, "b1": 9.48482, "b2": 0.1},
        optimizer_goal=114.66,
        n_samples=90,
        optimizer_n_samples=150,
        repeats=25,
    ),
    "goldstein_price_nodummy": ProblemCase(
        problem_name="goldstein_price_nodummy",
        known_optimum={"x1": 0.0, "x2": -1.0},
        optimizer_goal=3.6,
        n_samples=150,
        optimizer_n_samples=500,
        repeats=50,
    ),
    "six_hump_camel_nodummy": ProblemCase(
        problem_name="six_hump_camel_nodummy",
        known_optimum=[
            {"x1": 0.0898, "x2": -0.7126},
            {"x1": -0.0898, "x2": 0.7126},
        ],
        optimizer_goal=-0.86,
        n_samples=40,
        optimizer_n_samples=40,
        repeats=25,
    ),
}

# Activate only the cases used in this run.
ACTIVE_PROBLEM_CASES: list[str] = [
    "cantilever_beam",
    "rosenbrock",
    "goldstein_price",
    "six_hump_camel",
    # "cantilever_beam_nodummy",
    # "rosenbrock_nodummy",
    # "goldstein_price_nodummy",
    # "six_hump_camel_nodummy",
]

PROBLEM_SUITE: list[ProblemCase] = [PROBLEM_CASE_PRESETS[name] for name in ACTIVE_PROBLEM_CASES]

# Batch runner local config. This file is an analysis runner, so the default
# execution policy lives here instead of requiring CLI flags.
BATCH_TASKS: dict[str, bool] = {
    "doe": True,
    "modeler": True,
    "explorer": True,
    "optimizer": True,
}
BATCH_DEFAULT_OPTIMIZER_N_SAMPLES = 80
BATCH_USE_ADDITIONAL = True
BATCH_USE_HPO = True
BATCH_USE_PRIMARY_SELECTION = True
BATCH_USE_TIMESTAMP = True
BATCH_DEBUG_LEVEL = "on"
BATCH_CONTINUE_ON_ERROR = False
BATCH_AION_MODE = True

# Keep the per-problem n_samples/repeats fixed for score experiments. These are
# the target budgets, not a convenience runtime knob.
BATCH_FAST_OPTIMIZER_MODE = False
BATCH_FAST_REPEATS: int | None = None
BATCH_FAST_OPTIMIZER_N_SAMPLES_BY_PROBLEM: dict[str, int] = {}
BATCH_OPTIMIZER_SYSTEM_OVERRIDES: dict[str, Any] = {
    # Score experiments should preserve optimizer defaults. Put temporary
    # profiling-only overrides here only when explicitly testing runtime tradeoffs.
}


def _strategy_map() -> dict[str, ExplorerStrategy]:
    return strategy_map()


def _default_explorer_strategy_id(*, aion_mode: bool) -> str:
    if bool(aion_mode):
        return str(AION_SYSTEM_CONFIG.explorer_strategy_id)
    return str(ExplorerSystemConfig().strategy_id)


def _resolve_case_repeats(case: ProblemCase) -> int:
    if BATCH_FAST_OPTIMIZER_MODE and BATCH_FAST_REPEATS is not None:
        return int(max(int(BATCH_FAST_REPEATS), 1))
    return int(max(int(case.repeats), 1))


def _resolve_case_optimizer_samples(case: ProblemCase, default_value: int) -> int:
    if BATCH_FAST_OPTIMIZER_MODE and case.problem_name in BATCH_FAST_OPTIMIZER_N_SAMPLES_BY_PROBLEM:
        return int(max(int(BATCH_FAST_OPTIMIZER_N_SAMPLES_BY_PROBLEM[case.problem_name]), 0))
    if case.optimizer_n_samples is not None:
        return int(max(int(case.optimizer_n_samples), 0))
    return int(max(int(default_value), 0))


def _cli_option_present(name: str) -> bool:
    token = str(name)
    prefix = token + "="
    return any(arg == token or arg.startswith(prefix) for arg in sys.argv[1:])


def _case_has_pre_constraints(problem_name: str) -> bool:
    try:
        from CAE_tool_interface.executor.configurator import select_cae_by_name

        cae_entry = select_cae_by_name(str(problem_name))
        spec = cae_entry[0]() if isinstance(cae_entry, tuple) and callable(cae_entry[0]) else cae_entry
        for item in spec.get("constraint_defs", []) if isinstance(spec, dict) else []:
            if isinstance(item, dict) and str(item.get("scope", "pre")).strip().lower() == "pre":
                return True
    except Exception:
        return False
    return False


def _build_optimizer_system_config(*, debug_level: str, problem_name: str | None = None):
    from Optimizer.config import OptimizerSystemConfig

    system_cfg = OptimizerSystemConfig(debug_level=str(debug_level))
    if problem_name and _case_has_pre_constraints(str(problem_name)):
        system_cfg.enforce_pre_constraints = True
    for key, value in BATCH_OPTIMIZER_SYSTEM_OVERRIDES.items():
        if not hasattr(system_cfg, key):
            raise RuntimeError(f"Unknown optimizer system override in run_pipelines.py: {key}")
        setattr(system_cfg, key, value)
    return system_cfg


def _case_real_variables(case: ProblemCase) -> list[str]:
    ko = case.known_optimum
    if isinstance(ko, dict):
        return sorted([str(k) for k in ko.keys()])
    if isinstance(ko, list):
        keys: set[str] = set()
        for item in ko:
            if isinstance(item, dict):
                keys.update(str(k) for k in item.keys())
        return sorted(keys)
    return []


def _load_modeler_selected_features(modeler_metadata_path: str | None) -> list[str]:
    if not modeler_metadata_path or not os.path.exists(modeler_metadata_path):
        return []
    try:
        with open(modeler_metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        resolved = meta.get("resolved_params", {}) if isinstance(meta, dict) else {}
        selected = resolved.get("selected_features", []) if isinstance(resolved, dict) else []
        if isinstance(selected, list):
            return [str(v) for v in selected]
    except Exception:
        return []
    return []


def _choose_strategies_by_dim(
    *,
    selected_feature_count: int,
    requested: list[ExplorerStrategy],
) -> tuple[str, list[ExplorerStrategy]]:
    requested_map = {s.strategy_id: s for s in requested}
    policy = "low_dim_active" if selected_feature_count <= 3 else "high_dim_active"
    # Preserve a stable catalog order when multiple strategies are explicitly
    # requested. Default ownership lives in ExplorerSystemConfig/AIONSystemConfig.
    ordered = ["S4_dual", "S4_obj"]
    chosen = [requested_map[sid] for sid in ordered if sid in requested_map]
    if not chosen:
        chosen = requested[:]
    return policy, chosen


def _build_pipeline_config(
    *,
    case: ProblemCase,
    seed: int,
    run_doe: bool,
    run_modeler: bool,
    run_explorer: bool,
    run_optimizer: bool,
    optimizer_n_samples: int,
    use_additional: bool,
    use_hpo: bool,
    use_primary_selection: bool,
    use_timestamp: bool,
    debug_level: str,
    aion_mode: bool,
) -> PipelineConfig:
    cae_cfg = CAEConfig(
        user=CAEUserConfig(
            problem_name=case.problem_name,
            seed=int(seed),
            objective_sense=str(case.objective_sense),
        ),
        system=CAESystemConfig(
            use_timestamp=bool(use_timestamp),
            allow_latest_fallback=False,
        ),
    )

    doe_cfg = DOEConfig(
        cae=cae_cfg,
        cae_user=None,
        user=DOEUserConfig(algo_name="lhs", use_additional=bool(use_additional)),
        system=DOESystemConfig(n_samples=int(case.n_samples), debug_level=str(debug_level)),
    )

    modeler_cfg = ModelerConfig(
        user=ModelerUserConfig(
            model_name="xgb",
            use_hpo=bool(use_hpo),
            use_secondary_selection=False,
        ),
        system=ModelerSystemConfig(
            use_primary_selection=bool(use_primary_selection),
            debug_level=str(debug_level),
        ),
        cae=cae_cfg,
        doe_csv_path=None,
        doe_metadata_path=None,
    )

    explorer_cfg = ExplorerConfig(
        user=ExplorerUserConfig(known_optimum=case.known_optimum),
        system=ExplorerSystemConfig(debug_level=str(debug_level)),
        cae=cae_cfg,
        doe_csv_path=None,
        model_pkl_path=None,
    )

    optimizer_cfg = None
    if run_optimizer:
        from Optimizer.config import OptimizerConfig, OptimizerUserConfig

        optimizer_cfg = OptimizerConfig(
            user=OptimizerUserConfig(
                n_samples=int(max(int(optimizer_n_samples), 0)),
                known_optimum=case.known_optimum,
                goal=case.optimizer_goal,
            ),
            system=_build_optimizer_system_config(debug_level=str(debug_level), problem_name=case.problem_name),
            cae=cae_cfg,
            cae_metadata_path=None,
            doe_metadata_path=None,
            explorer_metadata_path=None,
            modeler_metadata_path=None,
        )

    pipeline_cfg = PipelineConfig(
        cae=cae_cfg,
        doe=doe_cfg if run_doe else None,
        modeler=modeler_cfg if run_modeler else None,
        explorer=explorer_cfg,
        optimizer=optimizer_cfg,
        tasks=PipelineTasks(
            run_doe=bool(run_doe),
            run_modeler=bool(run_modeler),
            run_explorer=bool(run_explorer),
            run_optimizer=bool(run_optimizer),
        ),
        aion_mode=bool(aion_mode),
    )
    return apply_aion_system_config(pipeline_cfg)


def _resolve_requested_strategies(raw: str) -> list[ExplorerStrategy]:
    catalog = _strategy_map()
    wanted = [tok.strip() for tok in str(raw).split(",") if tok.strip()]
    if not wanted or (len(wanted) == 1 and wanted[0].lower() == "all"):
        wanted = [s.strategy_id for s in EXPLORER_STRATEGIES]
    catalog_by_lower = {key.lower(): value for key, value in catalog.items()}
    out: list[ExplorerStrategy] = []
    for sid in wanted:
        strategy = catalog.get(sid) or catalog_by_lower.get(sid.lower())
        if strategy is None:
            raise ValueError(
                f"Unknown explorer strategy: {sid}. "
                f"Valid choices: {sorted(catalog.keys())}"
            )
        out.append(strategy)
    return out


def _build_strategy_explorer_config(
    *,
    base_explorer: ExplorerConfig,
    strategy: ExplorerStrategy,
) -> ExplorerConfig:
    system_cfg = apply_explorer_strategy_preset(
        copy.deepcopy(base_explorer.system),
        strategy.strategy_id,
    )

    return ExplorerConfig(
        user=copy.deepcopy(base_explorer.user),
        system=system_cfg,
        cae=base_explorer.cae,
        cae_metadata_path=base_explorer.cae_metadata_path,
        doe_csv_path=base_explorer.doe_csv_path,
        selected_features_csv_path=base_explorer.selected_features_csv_path,
        model_pkl_path=base_explorer.model_pkl_path,
        fi_scores_path=base_explorer.fi_scores_path,
    )


def _parse_selected_bounds(bounds_path: str | None) -> dict[str, tuple[float, float]]:
    if not bounds_path or (not os.path.exists(bounds_path)):
        return {}
    with open(bounds_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    selected = payload.get("selected_bounds")
    if isinstance(selected, dict):
        out = {}
        for name, item in selected.items():
            if not isinstance(item, dict):
                continue
            try:
                out[str(name)] = (float(item["lb"]), float(item["ub"]))
            except Exception:
                continue
        return out
    if isinstance(selected, list):
        order = payload.get("bounds_order", [])
        out = {}
        for idx, item in enumerate(selected):
            if idx >= len(order) or not isinstance(item, dict):
                continue
            try:
                out[str(order[idx])] = (float(item["lb"]), float(item["ub"]))
            except Exception:
                continue
        return out
    return {}


def _known_optimum_list(known_optimum: Any) -> list[dict[str, float]]:
    if isinstance(known_optimum, dict):
        return [{str(k): float(v) for k, v in known_optimum.items()}]
    if isinstance(known_optimum, list):
        out = []
        for item in known_optimum:
            if not isinstance(item, dict):
                continue
            out.append({str(k): float(v) for k, v in item.items()})
        return out
    return []


def _read_json_file(path: str | None) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _read_task_analysis_metadata(
    task_metadata_path: str | None,
    task_metadata: dict[str, Any],
) -> dict[str, Any]:
    if not task_metadata_path or not isinstance(task_metadata, dict):
        return {}
    artifacts = task_metadata.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return {}
    meta_artifacts = artifacts.get("meta", {})
    if not isinstance(meta_artifacts, dict):
        return {}

    analysis_ref = None
    if isinstance(meta_artifacts.get("analysis"), str):
        analysis_ref = meta_artifacts.get("analysis")
    else:
        for k, v in meta_artifacts.items():
            if isinstance(k, str) and k.startswith("analysis") and isinstance(v, str):
                analysis_ref = v
                break

    if not isinstance(analysis_ref, str) or not analysis_ref.strip():
        return {}

    base_dir = os.path.dirname(task_metadata_path)
    analysis_path = (
        analysis_ref
        if os.path.isabs(analysis_ref)
        else os.path.normpath(os.path.join(base_dir, analysis_ref))
    )
    return _read_json_file(analysis_path)


def _extract_doe_router_metadata(doe_metadata_path: str | None) -> dict[str, Any]:
    meta = _read_json_file(doe_metadata_path)
    resolved = meta.get("resolved_params", {}) if isinstance(meta.get("resolved_params"), dict) else {}
    results = meta.get("results", {}) if isinstance(meta.get("results"), dict) else {}
    extra = resolved.get("extra_context", {}) if isinstance(resolved.get("extra_context"), dict) else {}
    analysis = _read_task_analysis_metadata(doe_metadata_path, meta)

    def _pick(key: str) -> Any:
        if key in resolved:
            return resolved.get(key)
        if key in extra:
            return extra.get(key)
        if key in analysis:
            return analysis.get(key)
        if key in results:
            return results.get(key)
        return None

    return {
        "doe_failure_reason": _pick("failure_reason"),
        "doe_config_hash": _pick("doe_config_hash"),
        "doe_budget_policy": _pick("budget_policy"),
        "doe_initial_corner_ratio": _pick("initial_corner_ratio"),
        "doe_initial_corner_ratio_base": _pick("initial_corner_ratio_base"),
        "doe_initial_corner_ratio_policy": _pick("initial_corner_ratio_policy"),
        "doe_initial_corner_np_ratio": _pick("initial_corner_np_ratio"),
        "doe_initial_corner_p_dim": _pick("initial_corner_p_dim"),
        "doe_has_pre_constraints": _pick("has_pre_constraints"),
        "doe_has_post_constraints": _pick("has_post_constraints"),
        "doe_additional_triggered": _pick("doe_additional_triggered"),
        "doe_added_samples": _pick("doe_added_samples"),
        "doe_gate1_pass_rate": _pick("doe_gate1_pass_rate"),
        "doe_gate2_pass_rate": _pick("doe_gate2_pass_rate"),
        "doe_gate1_score_mean": _pick("doe_gate1_score_mean"),
        "doe_gate2_score_mean": _pick("doe_gate2_score_mean"),
        "doe_gate1_score_last": _pick("doe_gate1_score_last"),
        "doe_gate2_score_last": _pick("doe_gate2_score_last"),
        "doe_gate1_score_ema_mean": _pick("doe_gate1_score_ema_mean"),
        "doe_gate2_score_ema_mean": _pick("doe_gate2_score_ema_mean"),
        "doe_gate1_score_ema_last": _pick("doe_gate1_score_ema_last"),
        "doe_gate2_score_ema_last": _pick("doe_gate2_score_ema_last"),
        "doe_gate_decision_source_last": _pick("doe_gate_decision_source_last"),
        "doe_phase2_entered": _pick("phase2_entered"),
        "doe_used_budget_ratio_final": _pick("used_budget_ratio_final"),
        "doe_phase2_gate1_score_min": _pick("phase2_gate1_score_min"),
        "doe_phase2_gate2_score_min": _pick("phase2_gate2_score_min"),
        "doe_phase2_gate2_score_sticky_min": _pick("phase2_gate2_score_sticky_min"),
        "doe_gate_smoothing_enabled": _pick("gate_smoothing_enabled"),
        "doe_gate_ema_alpha": _pick("gate_ema_alpha"),
        "doe_gate_ema_warmup_stages": _pick("gate_ema_warmup_stages"),
        "doe_gate_smoothing_use_for_phase2": _pick("gate_smoothing_use_for_phase2"),
        "doe_gate_smoothing_use_for_stop": _pick("gate_smoothing_use_for_stop"),
        "doe_collapse_span_ratio_threshold": _pick("collapse_span_ratio_threshold"),
        "doe_collapse_anchor_streak_threshold": _pick("collapse_anchor_streak_threshold"),
        "doe_collapse_min_stage": _pick("collapse_min_stage"),
        "doe_diversity_injection_ratio": _pick("diversity_injection_ratio"),
        "doe_diversity_injection_min_points": _pick("diversity_injection_min_points"),
        "doe_diversity_injection_max_ratio": _pick("diversity_injection_max_ratio"),
        "doe_diversity_boundary_floor_ratio": _pick("diversity_boundary_floor_ratio"),
        "doe_phase2_min_used_budget_ratio": _pick("phase2_min_used_budget_ratio"),
        "doe_phase2_np_gate_last": _pick("phase2_np_gate_last"),
        "doe_early_stop_min_used_budget_ratio": _pick("early_stop_min_used_budget_ratio"),
        "doe_early_stop_min_usable_np_ratio": _pick("early_stop_min_usable_np_ratio"),
        "doe_budget_used": _pick("budget_used"),
        "doe_budget_total": _pick("budget_total"),
        "doe_budget_exhausted": _pick("budget_exhausted"),
        "doe_stage_count_total": _pick("stage_count_total"),
        "doe_stage_gate_eval_count": _pick("stage_gate_eval_count"),
        "doe_gate_eval_skipped_count": _pick("gate_eval_skipped_count"),
        "doe_phase2_first_stage": _pick("phase2_first_stage"),
        "doe_phase2_stage_count": _pick("phase2_stage_count"),
        "doe_phase2_transition_count": _pick("phase2_transition_count"),
        "doe_gate_stop_raw_count": _pick("gate_stop_raw_count"),
        "doe_gate_stop_final_count": _pick("gate_stop_final_count"),
        "doe_gate_stop_blocked_count": _pick("gate_stop_blocked_count"),
        "doe_collapse_detected_count": _pick("doe_collapse_detected_count"),
        "doe_diversity_injection_applied_count": _pick("doe_diversity_injection_applied_count"),
        "doe_diversity_sample_count_total": _pick("doe_diversity_sample_count_total"),
        "doe_diversity_hit_rate_mean": _pick("doe_diversity_hit_rate_mean"),
        "doe_dynamic_exec_allocation_enabled": _pick("dynamic_exec_allocation_enabled"),
        "doe_exec_round_shares": _pick("exec_round_shares"),
        "doe_alloc_multiplier_last": _pick("alloc_multiplier_last"),
        "doe_alloc_multiplier_mean": _pick("alloc_multiplier_mean"),
        "doe_alloc_eff_prev_last": _pick("alloc_eff_prev_last"),
        "doe_alloc_eff_ref_last": _pick("alloc_eff_ref_last"),
        "doe_alloc_dynamic_applied_count": _pick("alloc_dynamic_applied_count"),
        "doe_used_budget_ratio_phase2_first": _pick("used_budget_ratio_phase2_first"),
        "doe_terminated_by": _pick("terminated_by"),
        "doe_p_dim": _pick("p_dim"),
        "doe_usable_n": _pick("usable_n"),
        "doe_usable_n_over_p": _pick("usable_n_over_p"),
        "doe_local_span_ratio_mean_last": _pick("local_span_ratio_mean_last"),
        "doe_anchor_spread_zero_streak_max": _pick("anchor_spread_zero_streak_max"),
        "doe_metadata": doe_metadata_path,
    }


def _extract_explorer_router_metadata(explorer_metadata_path: str | None) -> dict[str, Any]:
    meta = _read_json_file(explorer_metadata_path)
    resolved = meta.get("resolved_params", {}) if isinstance(meta.get("resolved_params"), dict) else {}
    analysis = _read_task_analysis_metadata(explorer_metadata_path, meta)

    def _pick(key: str) -> Any:
        if key in resolved:
            return resolved.get(key)
        if key in analysis:
            return analysis.get(key)
        return None

    return {
        "explorer_strategy_alias": _pick("strategy_alias"),
        "explorer_strategy_mode": _pick("strategy_mode"),
        "explorer_p_dim": _pick("p_dim"),
        "explorer_usable_n": _pick("usable_n"),
        "explorer_usable_n_over_p": _pick("usable_n_over_p"),
        "explorer_selected_bounds_volume_ratio": _pick("selected_bounds_volume_ratio"),
        "explorer_pred_cluster_signal_mode": _pick("pred_cluster_signal_mode"),
        "explorer_pred_cluster_beta_used": _pick("pred_cluster_beta_used"),
        "explorer_pred_cluster_confidence": _pick("pred_cluster_confidence"),
        "explorer_pred_cluster_selected_count": _pick("pred_cluster_selected_count"),
        "explorer_pred_cluster_sigma_mean_selected": _pick("pred_cluster_sigma_mean_selected"),
        "explorer_pred_refine_shift_norm": _pick("pred_refine_shift_norm"),
        "explorer_pred_obj_miss_case": _pick("pred_obj_miss_case"),
        "explorer_pred_obj_iou": _pick("pred_obj_iou"),
        "explorer_pred_obj_center_l1_norm": _pick("pred_obj_center_l1_norm"),
        "explorer_pred_multistart_det_fraction_used": _pick("pred_multistart_det_fraction_used"),
        "explorer_pred_refine_bounds_scale_used": _pick("pred_refine_bounds_scale_used"),
        "explorer_pred_n_starts_used": _pick("pred_n_starts_used"),
        "explorer_dual_policy_mode": _pick("dual_policy_mode"),
        "explorer_dual_total_starts_target": _pick("dual_total_starts_target"),
        "explorer_dual_total_starts_used": _pick("dual_total_starts_used"),
        "explorer_dual_np_ratio_used": _pick("dual_np_ratio_used"),
        "explorer_dual_pred_ratio_used": _pick("dual_pred_ratio_used"),
        "explorer_dual_obj_ratio_used": _pick("dual_obj_ratio_used"),
        "explorer_dual_n_pred_starts": _pick("dual_n_pred_starts"),
        "explorer_dual_n_obj_starts": _pick("dual_n_obj_starts"),
        "explorer_dual_disagreement_center_l1_norm": _pick("dual_disagreement_center_l1_norm"),
        "explorer_dual_disagreement_iou": _pick("dual_disagreement_iou"),
        "explorer_dual_disagreement_triggered": _pick("dual_disagreement_triggered"),
        "explorer_dual_center_bias_used": _pick("dual_center_bias_used"),
        "explorer_dual_center_tilt_strength_base": _pick("dual_center_tilt_strength_base"),
        "explorer_dual_center_tilt_strength_used_mean": _pick("dual_center_tilt_strength_used_mean"),
        "explorer_dual_center_tilt_applied": _pick("dual_center_tilt_applied"),
        "explorer_dual_volume_cap_target": _pick("dual_volume_cap_target"),
        "explorer_dual_volume_ratio_before_cap": _pick("dual_volume_ratio_before_cap"),
        "explorer_dual_volume_cap_applied": _pick("dual_volume_cap_applied"),
    }


def _is_optimum_included(
    *,
    known_optimum: Any,
    selected_bounds: dict[str, tuple[float, float]],
) -> tuple[bool, int, int]:
    opts = _known_optimum_list(known_optimum)
    if not opts or not selected_bounds:
        return False, 0, len(opts)
    hit_count = 0
    for opt in opts:
        required_keys = list(opt.keys())
        if not required_keys:
            continue
        # strict mode: known optimum의 key가 selected_bounds에 하나라도 없으면 미포함 처리
        if any(k not in selected_bounds for k in required_keys):
            continue
        included = True
        for key in required_keys:
            lb, ub = selected_bounds[key]
            val = float(opt[key])
            if val < lb or val > ub:
                included = False
                break
        if included:
            hit_count += 1
    return hit_count > 0, hit_count, len(opts)


def _save_explorer_stats_csv(
    *,
    detail_rows: list[dict[str, Any]],
) -> tuple[str, str]:
    stats_root = os.path.join(PROJECT_ROOT, "result", "explorer_strategy_stats")
    os.makedirs(stats_root, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        detail_df = pd.DataFrame(
            columns=[
                "run",
                "repeat",
                "problem",
                "seed",
                "strategy",
                "executed_strategy",
                "fallback_from",
                "fallback_applied",
                "survivor_optimum_included",
                "optimum_hit_count",
                "optimum_total_count",
                "selected_feature_count",
                "strategy_policy",
                "modeler_selected_features",
                "modeler_selected_real_count",
                "modeler_selected_dummy_count",
                "modeler_real_coverage_pct",
                "modeler_all_real_included",
                "modeler_all_real_only",
                "volume_ratio",
                "volume_ratio_pct",
                "volume_cap_pass",
                "explorer_bounds_pass",
                "joint_pass",
                "fail_type",
                "explorer_metadata",
                "selected_bounds_path",
                "run_root",
                "p_dim",
                "usable_n",
                "usable_n_over_p",
                "doe_failure_reason",
                "doe_config_hash",
                "doe_budget_policy",
                "doe_initial_corner_ratio",
                "doe_initial_corner_ratio_base",
                "doe_initial_corner_ratio_policy",
                "doe_initial_corner_np_ratio",
                "doe_initial_corner_p_dim",
                "doe_has_pre_constraints",
                "doe_has_post_constraints",
                "doe_additional_triggered",
                "doe_added_samples",
                "doe_gate1_pass_rate",
                "doe_gate2_pass_rate",
                "doe_gate1_score_mean",
                "doe_gate2_score_mean",
                "doe_gate1_score_last",
                "doe_gate2_score_last",
                "doe_gate1_score_ema_mean",
                "doe_gate2_score_ema_mean",
                "doe_gate1_score_ema_last",
                "doe_gate2_score_ema_last",
                "doe_gate_decision_source_last",
                "doe_phase2_entered",
                "doe_used_budget_ratio_final",
                "doe_phase2_gate1_score_min",
                "doe_phase2_gate2_score_min",
                "doe_phase2_gate2_score_sticky_min",
                "doe_gate_smoothing_enabled",
                "doe_gate_ema_alpha",
                "doe_gate_ema_warmup_stages",
                "doe_gate_smoothing_use_for_phase2",
                "doe_gate_smoothing_use_for_stop",
                "doe_collapse_span_ratio_threshold",
                "doe_collapse_anchor_streak_threshold",
                "doe_collapse_min_stage",
                "doe_diversity_injection_ratio",
                "doe_diversity_injection_min_points",
                "doe_diversity_injection_max_ratio",
                "doe_diversity_boundary_floor_ratio",
                "doe_phase2_min_used_budget_ratio",
                "doe_phase2_np_gate_last",
                "doe_early_stop_min_used_budget_ratio",
                "doe_early_stop_min_usable_np_ratio",
                "doe_budget_used",
                "doe_budget_total",
                "doe_budget_exhausted",
                "doe_stage_count_total",
                "doe_stage_gate_eval_count",
                "doe_gate_eval_skipped_count",
                "doe_phase2_first_stage",
                "doe_phase2_stage_count",
                "doe_phase2_transition_count",
                "doe_gate_stop_raw_count",
                "doe_gate_stop_final_count",
                "doe_gate_stop_blocked_count",
                "doe_collapse_detected_count",
                "doe_diversity_injection_applied_count",
                "doe_diversity_sample_count_total",
                "doe_diversity_hit_rate_mean",
                "doe_dynamic_exec_allocation_enabled",
                "doe_exec_round_shares",
                "doe_alloc_multiplier_last",
                "doe_alloc_multiplier_mean",
                "doe_alloc_eff_prev_last",
                "doe_alloc_eff_ref_last",
                "doe_alloc_dynamic_applied_count",
                "doe_used_budget_ratio_phase2_first",
                "doe_terminated_by",
                "doe_p_dim",
                "doe_usable_n",
                "doe_usable_n_over_p",
                "doe_local_span_ratio_mean_last",
                "doe_anchor_spread_zero_streak_max",
                "doe_metadata",
                "explorer_strategy_alias",
                "explorer_strategy_mode",
                "explorer_p_dim",
                "explorer_usable_n",
                "explorer_usable_n_over_p",
                "explorer_selected_bounds_volume_ratio",
                "explorer_pred_cluster_signal_mode",
                "explorer_pred_cluster_beta_used",
                "explorer_pred_cluster_confidence",
                "explorer_pred_cluster_selected_count",
                "explorer_pred_cluster_sigma_mean_selected",
                "explorer_pred_refine_shift_norm",
                "explorer_pred_obj_miss_case",
                "explorer_pred_obj_iou",
                "explorer_pred_obj_center_l1_norm",
                "explorer_pred_multistart_det_fraction_used",
                "explorer_pred_refine_bounds_scale_used",
                "explorer_pred_n_starts_used",
                "explorer_dual_policy_mode",
                "explorer_dual_total_starts_target",
                "explorer_dual_total_starts_used",
                "explorer_dual_np_ratio_used",
                "explorer_dual_pred_ratio_used",
                "explorer_dual_obj_ratio_used",
                "explorer_dual_n_pred_starts",
                "explorer_dual_n_obj_starts",
                "explorer_dual_disagreement_center_l1_norm",
                "explorer_dual_disagreement_iou",
                "explorer_dual_disagreement_triggered",
                "explorer_dual_center_bias_used",
                "explorer_dual_center_tilt_strength_base",
                "explorer_dual_center_tilt_strength_used_mean",
                "explorer_dual_center_tilt_applied",
                "explorer_dual_volume_cap_target",
                "explorer_dual_volume_ratio_before_cap",
                "explorer_dual_volume_cap_applied",
            ]
        )

    # --- derive DSE metrics per row ---
    if not detail_df.empty:
        _opt = detail_df["survivor_optimum_included"].astype(bool)
        if "modeler_all_real_included" in detail_df.columns:
            _feature = detail_df["modeler_all_real_included"].astype(bool)
        else:
            _feature = detail_df["modeler_all_real_only"].astype(bool)
        _vr = pd.to_numeric(detail_df["volume_ratio"], errors="coerce").fillna(1.0)
        detail_df["volume_cap_pass"] = (_vr <= 0.25).astype(int)
        detail_df["explorer_bounds_pass"] = (_opt & (_vr <= 0.25)).astype(int)
        detail_df["joint_pass"] = (_feature & _opt & (_vr <= 0.25)).astype(int)
        detail_df["fail_type"] = "both_fail"
        detail_df.loc[~_feature, "fail_type"] = "feature_fail"
        detail_df.loc[_feature & _opt & (_vr <= 0.25), "fail_type"] = "pass"
        detail_df.loc[_feature & ~_opt & (_vr <= 0.25), "fail_type"] = "over_shrink_fail"
        detail_df.loc[_feature & _opt & (_vr > 0.25), "fail_type"] = "over_wide_fail"

    detail_path = os.path.join(stats_root, f"explorer_strategy_try_stats_{ts}.csv")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    summary_df = detail_df.copy()
    if not summary_df.empty:
        summary_df["optimum_included_num"] = summary_df["survivor_optimum_included"].astype(bool).astype(int)
        if "modeler_all_real_included" not in summary_df.columns:
            summary_df["modeler_all_real_included"] = summary_df["modeler_all_real_only"]
        summary_df["modeler_all_real_included_num"] = summary_df["modeler_all_real_included"].astype(bool).astype(int)
        summary_df["modeler_all_real_only_num"] = summary_df["modeler_all_real_only"].astype(bool).astype(int)
        summary_df["explorer_bounds_pass_num"] = summary_df["explorer_bounds_pass"].astype(bool).astype(int)
        summary_df["feature_fail"] = (summary_df["fail_type"] == "feature_fail").astype(int)
        summary_df["over_shrink_fail"] = (summary_df["fail_type"] == "over_shrink_fail").astype(int)
        summary_df["over_wide_fail"] = (summary_df["fail_type"] == "over_wide_fail").astype(int)
        summary_df["both_fail"] = (summary_df["fail_type"] == "both_fail").astype(int)
        grouped = (
            summary_df
            .groupby(["strategy", "problem"], as_index=False)
            .agg(
                tries=("strategy", "count"),
                survivor_optimum_included_pct=("optimum_included_num", "mean"),
                modeler_all_real_included_pct=("modeler_all_real_included_num", "mean"),
                modeler_all_real_only_pct=("modeler_all_real_only_num", "mean"),
                modeler_real_coverage_pct_mean=("modeler_real_coverage_pct", "mean"),
                volume_ratio_pct_mean=("volume_ratio_pct", "mean"),
                joint_pass_pct=("joint_pass", "mean"),
                explorer_bounds_pass_pct=("explorer_bounds_pass_num", "mean"),
                volume_cap_pass_pct=("volume_cap_pass", "mean"),
                feature_fail_pct=("feature_fail", "mean"),
                over_shrink_fail_pct=("over_shrink_fail", "mean"),
                over_wide_fail_pct=("over_wide_fail", "mean"),
                both_fail_pct=("both_fail", "mean"),
            )
        )
        grouped["survivor_optimum_included_pct"] = grouped["survivor_optimum_included_pct"] * 100.0
        grouped["modeler_all_real_included_pct"] = grouped["modeler_all_real_included_pct"] * 100.0
        grouped["modeler_all_real_only_pct"] = grouped["modeler_all_real_only_pct"] * 100.0
        grouped["joint_pass_pct"] = grouped["joint_pass_pct"] * 100.0
        grouped["explorer_bounds_pass_pct"] = grouped["explorer_bounds_pass_pct"] * 100.0
        grouped["volume_cap_pass_pct"] = grouped["volume_cap_pass_pct"] * 100.0
        grouped["feature_fail_pct"] = grouped["feature_fail_pct"] * 100.0
        grouped["over_shrink_fail_pct"] = grouped["over_shrink_fail_pct"] * 100.0
        grouped["over_wide_fail_pct"] = grouped["over_wide_fail_pct"] * 100.0
        grouped["both_fail_pct"] = grouped["both_fail_pct"] * 100.0
        grouped = grouped[
            [
                "strategy",
                "problem",
                "tries",
                "joint_pass_pct",
                "explorer_bounds_pass_pct",
                "survivor_optimum_included_pct",
                "modeler_all_real_included_pct",
                "modeler_all_real_only_pct",
                "modeler_real_coverage_pct_mean",
                "volume_ratio_pct_mean",
                "volume_cap_pass_pct",
                "feature_fail_pct",
                "over_shrink_fail_pct",
                "over_wide_fail_pct",
                "both_fail_pct",
            ]
        ]
    else:
        grouped = pd.DataFrame(
            columns=[
                "strategy",
                "problem",
                "tries",
                "joint_pass_pct",
                "explorer_bounds_pass_pct",
                "survivor_optimum_included_pct",
                "modeler_all_real_included_pct",
                "modeler_all_real_only_pct",
                "modeler_real_coverage_pct_mean",
                "volume_ratio_pct_mean",
                "volume_cap_pass_pct",
                "feature_fail_pct",
                "over_shrink_fail_pct",
                "over_wide_fail_pct",
                "both_fail_pct",
            ]
        )

    summary_path = os.path.join(stats_root, f"explorer_strategy_problem_summary_{ts}.csv")
    grouped.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return detail_path, summary_path


def _save_fi_stats_csv(
    *,
    detail_rows: list[dict[str, Any]],
) -> tuple[str, str]:
    stats_root = os.path.join(PROJECT_ROOT, "result", "explorer_strategy_stats")
    os.makedirs(stats_root, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        detail_df = pd.DataFrame(
            columns=[
                "run",
                "repeat",
                "problem",
                "seed",
                "selected_feature_count",
                "modeler_selected_features",
                "modeler_selected_real_count",
                "modeler_selected_dummy_count",
                "modeler_real_coverage_pct",
                "fi_all_real_included",
                "fi_real_only_success",
                "modeler_metadata",
                "run_root",
            ]
        )

    detail_path = os.path.join(stats_root, f"fi_primary_try_stats_{ts}.csv")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    summary_df = detail_df.copy()
    if not summary_df.empty:
        summary_df["fi_all_real_included_num"] = summary_df["fi_all_real_included"].astype(bool).astype(int)
        summary_df["fi_real_only_success_num"] = summary_df["fi_real_only_success"].astype(bool).astype(int)
        grouped = (
            summary_df
            .groupby(["problem"], as_index=False)
            .agg(
                tries=("problem", "count"),
                fi_all_real_included_pct=("fi_all_real_included_num", "mean"),
                fi_real_only_success_pct=("fi_real_only_success_num", "mean"),
                modeler_real_coverage_pct_mean=("modeler_real_coverage_pct", "mean"),
                modeler_real_coverage_pct_std=("modeler_real_coverage_pct", "std"),
                selected_feature_count_mean=("selected_feature_count", "mean"),
                selected_feature_count_std=("selected_feature_count", "std"),
                modeler_selected_dummy_count_mean=("modeler_selected_dummy_count", "mean"),
            )
        )
        grouped["fi_all_real_included_pct"] = grouped["fi_all_real_included_pct"] * 100.0
        grouped["fi_real_only_success_pct"] = grouped["fi_real_only_success_pct"] * 100.0
    else:
        grouped = pd.DataFrame(
            columns=[
                "problem",
                "tries",
                "fi_all_real_included_pct",
                "fi_real_only_success_pct",
                "modeler_real_coverage_pct_mean",
                "modeler_real_coverage_pct_std",
                "selected_feature_count_mean",
                "selected_feature_count_std",
                "modeler_selected_dummy_count_mean",
            ]
        )

    summary_path = os.path.join(stats_root, f"fi_primary_problem_summary_{ts}.csv")
    grouped.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return detail_path, summary_path


def _optimizer_goal_hit(*, best_objective: float, goal: float | None, objective_sense: str) -> bool:
    if goal is None:
        return False
    try:
        best = float(best_objective)
        target = float(goal)
    except Exception:
        return False
    if not pd.notna(best) or not pd.notna(target):
        return False
    sense = str(objective_sense or "min").strip().lower()
    if sense == "max":
        return bool(best >= target)
    return bool(best <= target)


def _read_focus3_goal_diagnostics(
    *,
    optimizer_metadata_path: str | None,
    goal: float | None,
    objective_sense: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "focus3_best_objective": float("nan"),
        "focus3_goal_hit": False,
        "focus3_eval_count": 0,
        "focus3_improved_count": 0,
    }
    if not optimizer_metadata_path:
        return out
    opt_dir = os.path.dirname(os.path.abspath(str(optimizer_metadata_path)))
    focus_summary_path = os.path.join(opt_dir, "artifacts", "debug", "optimizer_focus_summary.csv")
    if not os.path.exists(focus_summary_path):
        return out
    try:
        focus_df = pd.read_csv(focus_summary_path)
    except Exception:
        return out
    if focus_df.empty or "segment" not in focus_df.columns:
        return out
    rows = focus_df[focus_df["segment"].astype(str).str.lower() == "focus3"]
    if rows.empty:
        return out
    row = rows.iloc[-1]
    sense = str(objective_sense or "min").strip().lower()
    best_col = "objective_raw_max" if sense == "max" else "objective_raw_min"
    try:
        focus3_best = float(row.get(best_col, float("nan")))
    except Exception:
        focus3_best = float("nan")
    try:
        focus3_count = int(float(row.get("count", 0) or 0))
    except Exception:
        focus3_count = 0
    try:
        focus3_improved_count = int(float(row.get("improved_count", 0) or 0))
    except Exception:
        focus3_improved_count = 0
    out.update(
        {
            "focus3_best_objective": focus3_best,
            "focus3_goal_hit": _optimizer_goal_hit(
                best_objective=focus3_best,
                goal=goal,
                objective_sense=objective_sense,
            ),
            "focus3_eval_count": focus3_count,
            "focus3_improved_count": focus3_improved_count,
        }
    )
    return out


def _save_optimizer_stats_csv(
    *,
    detail_rows: list[dict[str, Any]],
) -> tuple[str, str]:
    stats_root = os.path.join(PROJECT_ROOT, "result", "explorer_strategy_stats")
    os.makedirs(stats_root, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    detail_columns = [
        "run",
        "repeat",
        "problem",
        "seed",
        "strategy",
        "executed_strategy",
        "objective_sense",
        "optimizer_goal",
        "optimizer_best_objective",
        "optimizer_best_objective_raw",
        "optimizer_goal_hit",
        "focus3_best_objective",
        "focus3_goal_hit",
        "archive_only_goal_hit",
        "focus3_eval_count",
        "focus3_improved_count",
        "optimizer_n_iterations",
        "modeler_all_real_included",
        "modeler_all_real_only",
        "modeler_selected_features",
        "selected_feature_count",
        "modeler_selected_real_count",
        "modeler_selected_dummy_count",
        "modeler_real_coverage_pct",
        "survivor_optimum_included",
        "volume_ratio",
        "volume_ratio_pct",
        "volume_cap_pass",
        "explorer_bounds_pass",
        "explorer_joint_pass",
        "optimizer_final_pass",
        "optimizer_csv",
        "optimizer_metadata",
        "explorer_metadata",
        "selected_bounds_path",
        "run_root",
    ]
    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        detail_df = pd.DataFrame(columns=detail_columns)
    else:
        for col in detail_columns:
            if col not in detail_df.columns:
                detail_df[col] = None
        detail_df = detail_df[detail_columns]

    detail_path = os.path.join(stats_root, f"optimizer_try_stats_{ts}.csv")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    summary_df = detail_df.copy()
    if not summary_df.empty:
        for col in [
            "optimizer_goal_hit",
            "focus3_goal_hit",
            "archive_only_goal_hit",
            "modeler_all_real_included",
            "modeler_all_real_only",
            "survivor_optimum_included",
            "volume_cap_pass",
            "explorer_bounds_pass",
            "explorer_joint_pass",
            "optimizer_final_pass",
        ]:
            summary_df[col + "_num"] = summary_df[col].astype(bool).astype(int)

        grouped = (
            summary_df
            .groupby(["strategy", "problem"], as_index=False)
            .agg(
                tries=("strategy", "count"),
                optimizer_final_pass_pct=("optimizer_final_pass_num", "mean"),
                optimizer_goal_hit_pct=("optimizer_goal_hit_num", "mean"),
                focus3_goal_hit_pct=("focus3_goal_hit_num", "mean"),
                archive_only_goal_hit_pct=("archive_only_goal_hit_num", "mean"),
                explorer_joint_pass_pct=("explorer_joint_pass_num", "mean"),
                explorer_bounds_pass_pct=("explorer_bounds_pass_num", "mean"),
                modeler_all_real_included_pct=("modeler_all_real_included_num", "mean"),
                modeler_all_real_only_pct=("modeler_all_real_only_num", "mean"),
                survivor_optimum_included_pct=("survivor_optimum_included_num", "mean"),
                volume_cap_pass_pct=("volume_cap_pass_num", "mean"),
                optimizer_best_objective_mean=("optimizer_best_objective", "mean"),
                optimizer_best_objective_median=("optimizer_best_objective", "median"),
                focus3_best_objective_mean=("focus3_best_objective", "mean"),
                focus3_best_objective_median=("focus3_best_objective", "median"),
                optimizer_n_iterations_mean=("optimizer_n_iterations", "mean"),
                focus3_eval_count_mean=("focus3_eval_count", "mean"),
                focus3_improved_count_mean=("focus3_improved_count", "mean"),
            )
        )
        overall = (
            summary_df
            .groupby(["strategy"], as_index=False)
            .agg(
                tries=("strategy", "count"),
                optimizer_final_pass_pct=("optimizer_final_pass_num", "mean"),
                optimizer_goal_hit_pct=("optimizer_goal_hit_num", "mean"),
                focus3_goal_hit_pct=("focus3_goal_hit_num", "mean"),
                archive_only_goal_hit_pct=("archive_only_goal_hit_num", "mean"),
                explorer_joint_pass_pct=("explorer_joint_pass_num", "mean"),
                explorer_bounds_pass_pct=("explorer_bounds_pass_num", "mean"),
                modeler_all_real_included_pct=("modeler_all_real_included_num", "mean"),
                modeler_all_real_only_pct=("modeler_all_real_only_num", "mean"),
                survivor_optimum_included_pct=("survivor_optimum_included_num", "mean"),
                volume_cap_pass_pct=("volume_cap_pass_num", "mean"),
                optimizer_best_objective_mean=("optimizer_best_objective", "mean"),
                optimizer_best_objective_median=("optimizer_best_objective", "median"),
                focus3_best_objective_mean=("focus3_best_objective", "mean"),
                focus3_best_objective_median=("focus3_best_objective", "median"),
                optimizer_n_iterations_mean=("optimizer_n_iterations", "mean"),
                focus3_eval_count_mean=("focus3_eval_count", "mean"),
                focus3_improved_count_mean=("focus3_improved_count", "mean"),
            )
        )
        overall.insert(1, "problem", "__overall__")
        grouped = pd.concat([grouped, overall], ignore_index=True)
        pct_cols = [
            "optimizer_final_pass_pct",
            "optimizer_goal_hit_pct",
            "focus3_goal_hit_pct",
            "archive_only_goal_hit_pct",
            "explorer_joint_pass_pct",
            "explorer_bounds_pass_pct",
            "modeler_all_real_included_pct",
            "modeler_all_real_only_pct",
            "survivor_optimum_included_pct",
            "volume_cap_pass_pct",
        ]
        for col in pct_cols:
            grouped[col] = grouped[col] * 100.0
    else:
        grouped = pd.DataFrame(
            columns=[
                "strategy",
                "problem",
                "tries",
                "optimizer_final_pass_pct",
                "optimizer_goal_hit_pct",
                "focus3_goal_hit_pct",
                "archive_only_goal_hit_pct",
                "explorer_joint_pass_pct",
                "explorer_bounds_pass_pct",
                "modeler_all_real_included_pct",
                "modeler_all_real_only_pct",
                "survivor_optimum_included_pct",
                "volume_cap_pass_pct",
                "optimizer_best_objective_mean",
                "optimizer_best_objective_median",
                "focus3_best_objective_mean",
                "focus3_best_objective_median",
                "optimizer_n_iterations_mean",
                "focus3_eval_count_mean",
                "focus3_improved_count_mean",
            ]
        )

    summary_path = os.path.join(stats_root, f"optimizer_problem_summary_{ts}.csv")
    grouped.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return detail_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch runner: execute multiple problems with repeated pipeline runs.",
    )
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed.")
    parser.add_argument(
        "--repeat-seed-step",
        type=int,
        default=1000,
        help="Seed offset added per repeat.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining runs even if one run fails.",
    )
    parser.add_argument("--no-additional", action="store_true", help="Disable DOE additional mode.")
    parser.add_argument("--no-hpo", action="store_true", help="Disable Modeler HPO.")
    parser.add_argument(
        "--no-primary-selection",
        action="store_true",
        help="Disable Modeler primary selection (FI) and use all features.",
    )
    parser.add_argument("--no-timestamp", action="store_true", help="Disable timestamp in CAE/system config.")
    parser.add_argument("--skip-doe", action="store_true", help="Skip DOE stage.")
    parser.add_argument("--skip-modeler", action="store_true", help="Skip Modeler stage.")
    parser.add_argument("--skip-explorer", action="store_true", help="Skip Explorer stage.")
    parser.add_argument("--run-optimizer", action="store_true", help="Enable Optimizer stage.")
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug artifacts and plots for enabled tasks.",
    )
    parser.add_argument(
        "--optimizer-samples",
        type=int,
        default=30,
        help="Optimizer user n_samples.",
    )
    # Strategy policy is owned by aion_system_config when AION mode is enabled.
    # Stand-alone defaults remain owned by ExplorerSystemConfig.
    _explorer_system_default = ExplorerSystemConfig().strategy_id
    _pipeline_explorer_default = _default_explorer_strategy_id(aion_mode=BATCH_AION_MODE)
    parser.add_argument(
        "--explorer-strategies",
        type=str,
        default=_pipeline_explorer_default,
        help=(
            "Comma-separated Explorer strategy IDs. "
            f"pipeline default = {_pipeline_explorer_default} "
            f"(AION defaults come from pipeline/aion_system_config.py). "
            f"Explorer system default = {_explorer_system_default} (stand-alone safe — "
            f"Modeler 미동반 환경). stand-alone Explorer 호출 시 system default 적용."
        ),
    )
    args = parser.parse_args()

    base_seed = int(args.base_seed)
    repeat_seed_step = int(args.repeat_seed_step)
    continue_on_error = bool(BATCH_CONTINUE_ON_ERROR or args.continue_on_error)

    run_doe = bool(BATCH_TASKS.get("doe", True))
    run_modeler = bool(BATCH_TASKS.get("modeler", True))
    run_explorer = bool(BATCH_TASKS.get("explorer", True))
    run_optimizer = bool(BATCH_TASKS.get("optimizer", False))
    if bool(args.skip_doe):
        run_doe = False
    if bool(args.skip_modeler):
        run_modeler = False
    if bool(args.skip_explorer):
        run_explorer = False
    if bool(args.run_optimizer):
        run_optimizer = True

    optimizer_n_samples_default = int(max(int(BATCH_DEFAULT_OPTIMIZER_N_SAMPLES), 0))
    if _cli_option_present("--optimizer-samples"):
        optimizer_n_samples_default = int(max(int(args.optimizer_samples), 0))

    use_additional = bool(BATCH_USE_ADDITIONAL)
    if bool(args.no_additional):
        use_additional = False
    use_hpo = bool(BATCH_USE_HPO)
    if bool(args.no_hpo):
        use_hpo = False
    use_primary_selection = bool(BATCH_USE_PRIMARY_SELECTION)
    if bool(args.no_primary_selection):
        use_primary_selection = False
    use_timestamp = bool(BATCH_USE_TIMESTAMP)
    if bool(args.no_timestamp):
        use_timestamp = False
    debug_level = str(BATCH_DEBUG_LEVEL).strip().lower() or "on"
    if bool(args.no_debug):
        debug_level = "off"
    explorer_strategies = _resolve_requested_strategies(args.explorer_strategies)

    total_runs = sum(_resolve_case_repeats(case) for case in PROBLEM_SUITE)
    run_counter = 0
    failures: list[dict[str, Any]] = []
    explorer_failures: list[dict[str, Any]] = []
    optimizer_failures: list[dict[str, Any]] = []
    explorer_detail_rows: list[dict[str, Any]] = []
    fi_detail_rows: list[dict[str, Any]] = []
    optimizer_detail_rows: list[dict[str, Any]] = []

    print("===================================")
    print(" Batch Pipeline 실행 시작")
    print("===================================")
    print(
        f"- total_runs={total_runs} "
        f"tasks(doe/modeler/explorer/optimizer)="
        f"{run_doe}/{run_modeler}/{run_explorer}/{run_optimizer} "
        f"modeler(primary_selection)={use_primary_selection} "
        f"debug={debug_level}"
    )
    if run_optimizer:
        print(f"- default_optimizer_n_samples={optimizer_n_samples_default}")
        print(
            "- optimizer_goals="
            + ",".join(
                [
                    f"{case.problem_name}:{case.optimizer_goal}"
                    for case in PROBLEM_SUITE
                    if case.optimizer_goal is not None
                ]
            )
        )
    print(
        "- problem_repeats="
        + ",".join(
            [
                f"{case.problem_name}:repeat={_resolve_case_repeats(case)},"
                f"doe_n={case.n_samples},"
                f"opt_n={_resolve_case_optimizer_samples(case, optimizer_n_samples_default)}"
                for case in PROBLEM_SUITE
            ]
        )
    )
    if run_explorer:
        print(f"- explorer_strategies={','.join([s.strategy_id for s in explorer_strategies])}")

    for idx, case in enumerate(PROBLEM_SUITE):
        case_repeats = _resolve_case_repeats(case)
        case_optimizer_n_samples = _resolve_case_optimizer_samples(
            case,
            optimizer_n_samples_default,
        )
        for rep in range(case_repeats):
            run_counter += 1
            seed = base_seed + rep * repeat_seed_step + idx
            print(
                "[Batch] "
                f"run={run_counter}/{total_runs} repeat={rep + 1}/{case_repeats} "
                f"problem={case.problem_name} seed={seed} "
                f"doe_n_samples={case.n_samples} optimizer_n_samples={case_optimizer_n_samples}"
            )

            cfg = _build_pipeline_config(
                case=case,
                seed=seed,
                run_doe=run_doe,
                run_modeler=run_modeler,
                run_explorer=False,
                run_optimizer=bool(run_optimizer and (not run_explorer)),
                optimizer_n_samples=case_optimizer_n_samples,
                use_additional=use_additional,
                use_hpo=use_hpo,
                use_primary_selection=use_primary_selection,
                use_timestamp=use_timestamp,
                debug_level=debug_level,
                aion_mode=BATCH_AION_MODE,
            )

            try:
                pipe_out = run_pipeline(config=cfg)
                run_context = pipe_out["run_context"]
                doe_meta_features = _extract_doe_router_metadata(pipe_out.get("doe_metadata"))
                modeler_metadata_path = pipe_out.get("modeler_metadata")
                modeler_selected_features = _load_modeler_selected_features(modeler_metadata_path)
                selected_feature_count = int(len(modeler_selected_features))
                real_variables = set(_case_real_variables(case))
                selected_set = set(modeler_selected_features)
                modeler_selected_real_count = int(len(selected_set.intersection(real_variables)))
                modeler_selected_dummy_count = int(len(selected_set.difference(real_variables)))
                modeler_real_coverage_pct = (
                    float(modeler_selected_real_count) / float(len(real_variables)) * 100.0
                    if len(real_variables) > 0 else 0.0
                )
                # strict success: selected features exactly match real variables
                modeler_all_real_only = bool(selected_set == real_variables and len(real_variables) > 0)
                modeler_all_real_included = bool(
                    len(real_variables) > 0 and real_variables.issubset(selected_set)
                )

                fi_detail_rows.append(
                    {
                        "run": run_counter,
                        "repeat": rep + 1,
                        "problem": case.problem_name,
                        "seed": seed,
                        "selected_feature_count": selected_feature_count,
                        "modeler_selected_features": json.dumps(modeler_selected_features, ensure_ascii=False),
                        "modeler_selected_real_count": modeler_selected_real_count,
                        "modeler_selected_dummy_count": modeler_selected_dummy_count,
                        "modeler_real_coverage_pct": modeler_real_coverage_pct,
                        "fi_all_real_included": bool(modeler_all_real_included),
                        "fi_real_only_success": bool(modeler_all_real_only),
                        "modeler_metadata": modeler_metadata_path,
                        "run_root": run_context.run_root,
                    }
                )

                if run_explorer and cfg.explorer is not None:
                    if selected_feature_count <= 0:
                        strategy_policy, active_strategies = _choose_strategies_by_dim(
                            selected_feature_count=4,
                            requested=explorer_strategies,
                        )
                    else:
                        strategy_policy, active_strategies = _choose_strategies_by_dim(
                            selected_feature_count=selected_feature_count,
                            requested=explorer_strategies,
                        )

                    print(
                        f"[Explorer][Policy] problem={case.problem_name} "
                        f"selected_feature_count={selected_feature_count} "
                        f"strategy_policy={strategy_policy} "
                        f"selected_features={modeler_selected_features}"
                    )

                    for strategy in active_strategies:
                        requested_id = strategy.strategy_id

                        try:
                            explorer_cfg = _build_strategy_explorer_config(
                                base_explorer=cfg.explorer,
                                strategy=strategy,
                            )
                            exp_out = ExplorerOrchestrator(explorer_cfg, run_context=run_context).run()
                            selected_bounds = _parse_selected_bounds(exp_out.get("selected_bounds_path"))
                            vol_ratio = exp_out.get("selected_bounds_volume_ratio")

                            included, hit_count, total_count = _is_optimum_included(
                                known_optimum=case.known_optimum,
                                selected_bounds=selected_bounds,
                            )
                            vol_ratio_float = float(vol_ratio) if vol_ratio is not None else None
                            vol_pct = (vol_ratio_float * 100.0) if vol_ratio_float is not None else None
                            explorer_meta_features = _extract_explorer_router_metadata(exp_out.get("metadata"))
                            print(f"[Explorer][{requested_id}] executed={strategy.strategy_id}")
                            if vol_pct is not None:
                                print(
                                    f"[Explorer][{requested_id}] "
                                    f"selected_bounds volume_ratio={vol_ratio_float:.4f} ({vol_pct:.2f}%)"
                                )
                            else:
                                print(f"[Explorer][{requested_id}] selected_bounds volume_ratio=none")
                            print(
                                f"[Explorer][{requested_id}] optimum_included={included} "
                                f"(hits={hit_count}/{total_count})"
                            )

                            volume_cap_pass = bool(vol_ratio_float is not None and vol_ratio_float <= 0.25)
                            explorer_bounds_pass = bool(included and volume_cap_pass)
                            explorer_joint_pass = bool(modeler_all_real_included and explorer_bounds_pass)

                            if run_optimizer:
                                try:
                                    from Optimizer.run_Optimizer import run_optimizer as _run_optimizer
                                    from Optimizer.config import (
                                        OptimizerConfig,
                                        OptimizerUserConfig,
                                    )

                                    opt_cfg = OptimizerConfig(
                                        user=OptimizerUserConfig(
                                            n_samples=case_optimizer_n_samples,
                                            known_optimum=case.known_optimum,
                                            goal=case.optimizer_goal,
                                        ),
                                        system=_build_optimizer_system_config(debug_level=debug_level, problem_name=case.problem_name),
                                        cae=cfg.cae,
                                        cae_metadata_path=None,
                                        doe_metadata_path=None,
                                        explorer_metadata_path=None,
                                        modeler_metadata_path=None,
                                    )
                                    opt_out = _run_optimizer(
                                        config=opt_cfg,
                                        run_context=run_context,
                                    )
                                    opt_best = float(opt_out.get("best_objective", float("nan")))
                                    opt_goal_hit = _optimizer_goal_hit(
                                        best_objective=opt_best,
                                        goal=case.optimizer_goal,
                                        objective_sense=case.objective_sense,
                                    )
                                    focus3_diag = _read_focus3_goal_diagnostics(
                                        optimizer_metadata_path=opt_out.get("metadata"),
                                        goal=case.optimizer_goal,
                                        objective_sense=case.objective_sense,
                                    )
                                    optimizer_final_pass = bool(explorer_joint_pass and opt_goal_hit)
                                    optimizer_detail_rows.append(
                                        {
                                            "run": run_counter,
                                            "repeat": rep + 1,
                                            "problem": case.problem_name,
                                            "seed": seed,
                                            "strategy": requested_id,
                                            "executed_strategy": strategy.strategy_id,
                                            "objective_sense": str(case.objective_sense),
                                            "optimizer_goal": case.optimizer_goal,
                                            "optimizer_best_objective": opt_best,
                                            "optimizer_best_objective_raw": float(opt_out.get("best_objective_raw", float("nan"))),
                                            "optimizer_goal_hit": bool(opt_goal_hit),
                                            "focus3_best_objective": focus3_diag["focus3_best_objective"],
                                            "focus3_goal_hit": bool(focus3_diag["focus3_goal_hit"]),
                                            "archive_only_goal_hit": bool(opt_goal_hit and not bool(focus3_diag["focus3_goal_hit"])),
                                            "focus3_eval_count": int(focus3_diag["focus3_eval_count"]),
                                            "focus3_improved_count": int(focus3_diag["focus3_improved_count"]),
                                            "optimizer_n_iterations": int(opt_out.get("n_iterations", 0)),
                                            "modeler_all_real_included": bool(modeler_all_real_included),
                                            "modeler_all_real_only": bool(modeler_all_real_only),
                                            "modeler_selected_features": json.dumps(modeler_selected_features, ensure_ascii=False),
                                            "selected_feature_count": selected_feature_count,
                                            "modeler_selected_real_count": modeler_selected_real_count,
                                            "modeler_selected_dummy_count": modeler_selected_dummy_count,
                                            "modeler_real_coverage_pct": modeler_real_coverage_pct,
                                            "survivor_optimum_included": bool(included),
                                            "volume_ratio": vol_ratio_float,
                                            "volume_ratio_pct": vol_pct,
                                            "volume_cap_pass": bool(volume_cap_pass),
                                            "explorer_bounds_pass": bool(explorer_bounds_pass),
                                            "explorer_joint_pass": bool(explorer_joint_pass),
                                            "optimizer_final_pass": bool(optimizer_final_pass),
                                            "optimizer_csv": opt_out.get("csv"),
                                            "optimizer_metadata": opt_out.get("metadata"),
                                            "explorer_metadata": exp_out.get("metadata"),
                                            "selected_bounds_path": exp_out.get("selected_bounds_path"),
                                            "run_root": run_context.run_root,
                                        }
                                    )
                                    print(
                                        f"[Optimizer][{requested_id}] "
                                        f"best={opt_best:.6f} goal_hit={opt_goal_hit} "
                                        f"focus3_goal_hit={focus3_diag['focus3_goal_hit']} "
                                        f"final_pass={optimizer_final_pass}"
                                    )
                                except Exception as opt_exc:
                                    payload = {
                                        "run": run_counter,
                                        "repeat": rep + 1,
                                        "problem": case.problem_name,
                                        "seed": seed,
                                        "strategy": requested_id,
                                        "error": str(opt_exc),
                                    }
                                    optimizer_failures.append(payload)
                                    print(
                                        "[Batch][Optimizer] FAILED "
                                        f"problem={case.problem_name} repeat={rep + 1} "
                                        f"seed={seed} strategy={requested_id} error={opt_exc}"
                                    )
                                    if not continue_on_error:
                                        raise

                            explorer_detail_rows.append(
                                {
                                    "run": run_counter,
                                    "repeat": rep + 1,
                                    "problem": case.problem_name,
                                    "seed": seed,
                                    "strategy": requested_id,
                                    "executed_strategy": strategy.strategy_id,
                                    "fallback_from": None,
                                    "fallback_applied": False,
                                    "survivor_optimum_included": bool(included),
                                    "optimum_hit_count": int(hit_count),
                                    "optimum_total_count": int(total_count),
                                    "selected_feature_count": selected_feature_count,
                                    "strategy_policy": strategy_policy,
                                    "modeler_selected_features": json.dumps(modeler_selected_features, ensure_ascii=False),
                                    "modeler_selected_real_count": modeler_selected_real_count,
                                    "modeler_selected_dummy_count": modeler_selected_dummy_count,
                                    "modeler_real_coverage_pct": modeler_real_coverage_pct,
                                    "modeler_all_real_included": bool(modeler_all_real_included),
                                    "modeler_all_real_only": bool(modeler_all_real_only),
                                    "volume_ratio": vol_ratio_float,
                                    "volume_ratio_pct": vol_pct,
                                    "explorer_bounds_pass": bool(explorer_bounds_pass),
                                    "explorer_joint_pass": bool(explorer_joint_pass),
                                    "explorer_metadata": exp_out.get("metadata"),
                                    "selected_bounds_path": exp_out.get("selected_bounds_path"),
                                    "run_root": run_context.run_root,
                                    "p_dim": int(selected_feature_count),
                                    "usable_n": doe_meta_features.get("doe_usable_n"),
                                    "usable_n_over_p": doe_meta_features.get("doe_usable_n_over_p"),
                                    "doe_failure_reason": doe_meta_features.get("doe_failure_reason"),
                                    "doe_config_hash": doe_meta_features.get("doe_config_hash"),
                                    "doe_budget_policy": doe_meta_features.get("doe_budget_policy"),
                                    "doe_initial_corner_ratio": doe_meta_features.get("doe_initial_corner_ratio"),
                                    "doe_initial_corner_ratio_base": doe_meta_features.get("doe_initial_corner_ratio_base"),
                                    "doe_initial_corner_ratio_policy": doe_meta_features.get("doe_initial_corner_ratio_policy"),
                                    "doe_initial_corner_np_ratio": doe_meta_features.get("doe_initial_corner_np_ratio"),
                                    "doe_initial_corner_p_dim": doe_meta_features.get("doe_initial_corner_p_dim"),
                                    "doe_has_pre_constraints": doe_meta_features.get("doe_has_pre_constraints"),
                                    "doe_has_post_constraints": doe_meta_features.get("doe_has_post_constraints"),
                                    "doe_additional_triggered": doe_meta_features.get("doe_additional_triggered"),
                                    "doe_added_samples": doe_meta_features.get("doe_added_samples"),
                                    "doe_gate1_pass_rate": doe_meta_features.get("doe_gate1_pass_rate"),
                                    "doe_gate2_pass_rate": doe_meta_features.get("doe_gate2_pass_rate"),
                                    "doe_gate1_score_mean": doe_meta_features.get("doe_gate1_score_mean"),
                                    "doe_gate2_score_mean": doe_meta_features.get("doe_gate2_score_mean"),
                                    "doe_gate1_score_last": doe_meta_features.get("doe_gate1_score_last"),
                                    "doe_gate2_score_last": doe_meta_features.get("doe_gate2_score_last"),
                                    "doe_gate1_score_ema_mean": doe_meta_features.get("doe_gate1_score_ema_mean"),
                                    "doe_gate2_score_ema_mean": doe_meta_features.get("doe_gate2_score_ema_mean"),
                                    "doe_gate1_score_ema_last": doe_meta_features.get("doe_gate1_score_ema_last"),
                                    "doe_gate2_score_ema_last": doe_meta_features.get("doe_gate2_score_ema_last"),
                                    "doe_gate_decision_source_last": doe_meta_features.get("doe_gate_decision_source_last"),
                                    "doe_phase2_entered": doe_meta_features.get("doe_phase2_entered"),
                                    "doe_used_budget_ratio_final": doe_meta_features.get("doe_used_budget_ratio_final"),
                                    "doe_phase2_gate1_score_min": doe_meta_features.get("doe_phase2_gate1_score_min"),
                                    "doe_phase2_gate2_score_min": doe_meta_features.get("doe_phase2_gate2_score_min"),
                                    "doe_phase2_gate2_score_sticky_min": doe_meta_features.get("doe_phase2_gate2_score_sticky_min"),
                                    "doe_gate_smoothing_enabled": doe_meta_features.get("doe_gate_smoothing_enabled"),
                                    "doe_gate_ema_alpha": doe_meta_features.get("doe_gate_ema_alpha"),
                                    "doe_gate_ema_warmup_stages": doe_meta_features.get("doe_gate_ema_warmup_stages"),
                                    "doe_gate_smoothing_use_for_phase2": doe_meta_features.get("doe_gate_smoothing_use_for_phase2"),
                                    "doe_gate_smoothing_use_for_stop": doe_meta_features.get("doe_gate_smoothing_use_for_stop"),
                                    "doe_collapse_span_ratio_threshold": doe_meta_features.get("doe_collapse_span_ratio_threshold"),
                                    "doe_collapse_anchor_streak_threshold": doe_meta_features.get("doe_collapse_anchor_streak_threshold"),
                                    "doe_collapse_min_stage": doe_meta_features.get("doe_collapse_min_stage"),
                                    "doe_diversity_injection_ratio": doe_meta_features.get("doe_diversity_injection_ratio"),
                                    "doe_diversity_injection_min_points": doe_meta_features.get("doe_diversity_injection_min_points"),
                                    "doe_diversity_injection_max_ratio": doe_meta_features.get("doe_diversity_injection_max_ratio"),
                                    "doe_diversity_boundary_floor_ratio": doe_meta_features.get("doe_diversity_boundary_floor_ratio"),
                                    "doe_phase2_min_used_budget_ratio": doe_meta_features.get("doe_phase2_min_used_budget_ratio"),
                                    "doe_phase2_np_gate_last": doe_meta_features.get("doe_phase2_np_gate_last"),
                                    "doe_early_stop_min_used_budget_ratio": doe_meta_features.get("doe_early_stop_min_used_budget_ratio"),
                                    "doe_early_stop_min_usable_np_ratio": doe_meta_features.get("doe_early_stop_min_usable_np_ratio"),
                                    "doe_budget_used": doe_meta_features.get("doe_budget_used"),
                                    "doe_budget_total": doe_meta_features.get("doe_budget_total"),
                                    "doe_budget_exhausted": doe_meta_features.get("doe_budget_exhausted"),
                                    "doe_stage_count_total": doe_meta_features.get("doe_stage_count_total"),
                                    "doe_stage_gate_eval_count": doe_meta_features.get("doe_stage_gate_eval_count"),
                                    "doe_gate_eval_skipped_count": doe_meta_features.get("doe_gate_eval_skipped_count"),
                                    "doe_phase2_first_stage": doe_meta_features.get("doe_phase2_first_stage"),
                                    "doe_phase2_stage_count": doe_meta_features.get("doe_phase2_stage_count"),
                                    "doe_phase2_transition_count": doe_meta_features.get("doe_phase2_transition_count"),
                                    "doe_gate_stop_raw_count": doe_meta_features.get("doe_gate_stop_raw_count"),
                                    "doe_gate_stop_final_count": doe_meta_features.get("doe_gate_stop_final_count"),
                                    "doe_gate_stop_blocked_count": doe_meta_features.get("doe_gate_stop_blocked_count"),
                                    "doe_collapse_detected_count": doe_meta_features.get("doe_collapse_detected_count"),
                                    "doe_diversity_injection_applied_count": doe_meta_features.get("doe_diversity_injection_applied_count"),
                                    "doe_diversity_sample_count_total": doe_meta_features.get("doe_diversity_sample_count_total"),
                                    "doe_diversity_hit_rate_mean": doe_meta_features.get("doe_diversity_hit_rate_mean"),
                                    "doe_dynamic_exec_allocation_enabled": doe_meta_features.get("doe_dynamic_exec_allocation_enabled"),
                                    "doe_exec_round_shares": doe_meta_features.get("doe_exec_round_shares"),
                                    "doe_alloc_multiplier_last": doe_meta_features.get("doe_alloc_multiplier_last"),
                                    "doe_alloc_multiplier_mean": doe_meta_features.get("doe_alloc_multiplier_mean"),
                                    "doe_alloc_eff_prev_last": doe_meta_features.get("doe_alloc_eff_prev_last"),
                                    "doe_alloc_eff_ref_last": doe_meta_features.get("doe_alloc_eff_ref_last"),
                                    "doe_alloc_dynamic_applied_count": doe_meta_features.get("doe_alloc_dynamic_applied_count"),
                                    "doe_used_budget_ratio_phase2_first": doe_meta_features.get("doe_used_budget_ratio_phase2_first"),
                                    "doe_terminated_by": doe_meta_features.get("doe_terminated_by"),
                                    "doe_p_dim": doe_meta_features.get("doe_p_dim"),
                                    "doe_usable_n": doe_meta_features.get("doe_usable_n"),
                                    "doe_usable_n_over_p": doe_meta_features.get("doe_usable_n_over_p"),
                                    "doe_local_span_ratio_mean_last": doe_meta_features.get("doe_local_span_ratio_mean_last"),
                                    "doe_anchor_spread_zero_streak_max": doe_meta_features.get("doe_anchor_spread_zero_streak_max"),
                                    "doe_metadata": doe_meta_features.get("doe_metadata"),
                                    "explorer_strategy_alias": explorer_meta_features.get("explorer_strategy_alias"),
                                    "explorer_strategy_mode": explorer_meta_features.get("explorer_strategy_mode"),
                                    "explorer_p_dim": explorer_meta_features.get("explorer_p_dim"),
                                    "explorer_usable_n": explorer_meta_features.get("explorer_usable_n"),
                                    "explorer_usable_n_over_p": explorer_meta_features.get("explorer_usable_n_over_p"),
                                    "explorer_selected_bounds_volume_ratio": explorer_meta_features.get("explorer_selected_bounds_volume_ratio"),
                                    "explorer_pred_cluster_signal_mode": explorer_meta_features.get("explorer_pred_cluster_signal_mode"),
                                    "explorer_pred_cluster_beta_used": explorer_meta_features.get("explorer_pred_cluster_beta_used"),
                                    "explorer_pred_cluster_confidence": explorer_meta_features.get("explorer_pred_cluster_confidence"),
                                    "explorer_pred_cluster_selected_count": explorer_meta_features.get("explorer_pred_cluster_selected_count"),
                                    "explorer_pred_cluster_sigma_mean_selected": explorer_meta_features.get("explorer_pred_cluster_sigma_mean_selected"),
                                    "explorer_pred_refine_shift_norm": explorer_meta_features.get("explorer_pred_refine_shift_norm"),
                                    "explorer_pred_obj_miss_case": explorer_meta_features.get("explorer_pred_obj_miss_case"),
                                    "explorer_pred_obj_iou": explorer_meta_features.get("explorer_pred_obj_iou"),
                                    "explorer_pred_obj_center_l1_norm": explorer_meta_features.get("explorer_pred_obj_center_l1_norm"),
                                    "explorer_pred_multistart_det_fraction_used": explorer_meta_features.get("explorer_pred_multistart_det_fraction_used"),
                                    "explorer_pred_refine_bounds_scale_used": explorer_meta_features.get("explorer_pred_refine_bounds_scale_used"),
                                    "explorer_pred_n_starts_used": explorer_meta_features.get("explorer_pred_n_starts_used"),
                                    "explorer_dual_policy_mode": explorer_meta_features.get("explorer_dual_policy_mode"),
                                    "explorer_dual_total_starts_target": explorer_meta_features.get("explorer_dual_total_starts_target"),
                                    "explorer_dual_total_starts_used": explorer_meta_features.get("explorer_dual_total_starts_used"),
                                    "explorer_dual_np_ratio_used": explorer_meta_features.get("explorer_dual_np_ratio_used"),
                                    "explorer_dual_pred_ratio_used": explorer_meta_features.get("explorer_dual_pred_ratio_used"),
                                    "explorer_dual_obj_ratio_used": explorer_meta_features.get("explorer_dual_obj_ratio_used"),
                                    "explorer_dual_n_pred_starts": explorer_meta_features.get("explorer_dual_n_pred_starts"),
                                    "explorer_dual_n_obj_starts": explorer_meta_features.get("explorer_dual_n_obj_starts"),
                                    "explorer_dual_disagreement_center_l1_norm": explorer_meta_features.get("explorer_dual_disagreement_center_l1_norm"),
                                    "explorer_dual_disagreement_iou": explorer_meta_features.get("explorer_dual_disagreement_iou"),
                                    "explorer_dual_disagreement_triggered": explorer_meta_features.get("explorer_dual_disagreement_triggered"),
                                    "explorer_dual_center_bias_used": explorer_meta_features.get("explorer_dual_center_bias_used"),
                                    "explorer_dual_center_tilt_strength_base": explorer_meta_features.get("explorer_dual_center_tilt_strength_base"),
                                    "explorer_dual_center_tilt_strength_used_mean": explorer_meta_features.get("explorer_dual_center_tilt_strength_used_mean"),
                                    "explorer_dual_center_tilt_applied": explorer_meta_features.get("explorer_dual_center_tilt_applied"),
                                    "explorer_dual_volume_cap_target": explorer_meta_features.get("explorer_dual_volume_cap_target"),
                                    "explorer_dual_volume_ratio_before_cap": explorer_meta_features.get("explorer_dual_volume_ratio_before_cap"),
                                    "explorer_dual_volume_cap_applied": explorer_meta_features.get("explorer_dual_volume_cap_applied"),
                                }
                            )
                        except Exception as exp_exc:
                            payload = {
                                "run": run_counter,
                                "repeat": rep + 1,
                                "problem": case.problem_name,
                                "seed": seed,
                                "strategy": requested_id,
                                "error": str(exp_exc),
                            }
                            explorer_failures.append(payload)
                            print(
                                "[Batch][Explorer] FAILED "
                                f"problem={case.problem_name} repeat={rep + 1} "
                                f"seed={seed} strategy={requested_id} error={exp_exc}"
                            )
                            if not continue_on_error:
                                raise

            except Exception as exc:
                payload = {
                    "run": run_counter,
                    "repeat": rep + 1,
                    "problem": case.problem_name,
                    "seed": seed,
                    "error": str(exc),
                }
                failures.append(payload)
                print(
                    "[Batch] FAILED "
                    f"problem={case.problem_name} repeat={rep + 1} seed={seed} error={exc}"
                )
                if not continue_on_error:
                    raise

    fi_detail_csv, fi_summary_csv = _save_fi_stats_csv(detail_rows=fi_detail_rows)
    detail_csv, summary_csv = _save_explorer_stats_csv(detail_rows=explorer_detail_rows)
    optimizer_detail_csv, optimizer_summary_csv = _save_optimizer_stats_csv(detail_rows=optimizer_detail_rows)

    print("===================================")
    print(" Batch Pipeline 실행 완료")
    print("===================================")
    print(f"- total_runs={total_runs}")
    print(f"- base_pipeline_success_runs={total_runs - len(failures)}")
    print(f"- base_pipeline_failed_runs={len(failures)}")
    print(f"- explorer_runtime_failed_runs={len(explorer_failures)}")
    print(f"- optimizer_runtime_failed_runs={len(optimizer_failures)}")
    print(f"- fi_primary_try_stats_csv={fi_detail_csv}")
    print(f"- fi_primary_problem_summary_csv={fi_summary_csv}")
    print(f"- explorer_try_stats_csv={detail_csv}")
    print(f"- explorer_problem_summary_csv={summary_csv}")
    print(f"- optimizer_try_stats_csv={optimizer_detail_csv}")
    print(f"- optimizer_problem_summary_csv={optimizer_summary_csv}")
    if failures:
        print("- failure_details:")
        for item in failures:
            problem = item.get("problem")
            repeat = item.get("repeat")
            seed = item.get("seed")
            run = item.get("run")
            strategy = item.get("strategy")
            err = item.get("error")
            if strategy:
                print(
                    f"  run={run} repeat={repeat} problem={problem} "
                    f"seed={seed} strategy={strategy} error={err}"
                )
            else:
                print(
                    f"  run={run} repeat={repeat} problem={problem} "
                    f"seed={seed} error={err}"
                )
    if explorer_failures:
        print("- explorer_failure_details:")
        for item in explorer_failures:
            print(
                f"  run={item.get('run')} repeat={item.get('repeat')} "
                f"problem={item.get('problem')} seed={item.get('seed')} "
                f"strategy={item.get('strategy')} error={item.get('error')}"
            )
    if optimizer_failures:
        print("- optimizer_failure_details:")
        for item in optimizer_failures:
            print(
                f"  run={item.get('run')} repeat={item.get('repeat')} "
                f"problem={item.get('problem')} seed={item.get('seed')} "
                f"strategy={item.get('strategy')} error={item.get('error')}"
            )


if __name__ == "__main__":
    main()
