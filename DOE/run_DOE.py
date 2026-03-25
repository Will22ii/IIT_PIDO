import json
import os

from CAE_tool_interface.config import CAEConfig, CAEUserConfig, CAESystemConfig
from CAE_tool_interface.executor.configurator import select_cae_by_name
from DOE.config import DOEConfig, DOESystemConfig, DOEUserConfig
from DOE.executor.doe_orchestrator import run_doe_orchestrator
from pipeline.run_context import (
    RunContext,
    create_run_context,
    get_task_metadata_path,
    update_run_index,
)
from DOE.doe_algorithm.registry import get_doe_algorithm


def _normalize_debug_level(value: str | None) -> str:
    level = str(value or "off").strip().lower()
    if level not in {"off", "full"}:
        raise ValueError("DOE debug_level must be one of: off, full")
    return level


def _resolve_existing_cae_metadata_path(
    *,
    config: DOEConfig,
    run_context: RunContext | None,
) -> str:
    if run_context is not None:
        path = get_task_metadata_path(run_context, "CAE")
        if path and os.path.exists(path):
            return path
        raise RuntimeError(
            "DOE requires existing CAE metadata in run context. "
            "Run CAE task first and then execute DOE."
        )

    raw = str(config.cae_metadata_path or "").strip()
    if raw:
        candidates = [raw]
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates.append(os.path.join(project_root, raw))
        for p in candidates:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"CAE metadata not found: {raw}")

    raise RuntimeError(
        "DOE requires existing CAE metadata. "
        "Provide DOEConfig.cae_metadata_path or run via pipeline run_context."
    )


def _load_cae_metadata(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid CAE metadata payload: {path}")
    return payload


def _extract_seed_from_cae_metadata(*, cae_meta: dict, cae_meta_path: str) -> int:
    resolved = cae_meta.get("resolved_params", {}) if isinstance(cae_meta.get("resolved_params", {}), dict) else {}
    direct_candidates = [
        resolved.get("seed"),
        cae_meta.get("seed"),
        (cae_meta.get("inputs", {}) if isinstance(cae_meta.get("inputs", {}), dict) else {}).get("seed"),
    ]
    for cand in direct_candidates:
        try:
            if cand is not None:
                return int(cand)
        except Exception:
            pass

    inputs = cae_meta.get("inputs", {}) if isinstance(cae_meta.get("inputs", {}), dict) else {}
    user_ref = str(inputs.get("user_config", "")).strip()
    if user_ref:
        user_cfg_path = user_ref
        if not os.path.isabs(user_cfg_path):
            user_cfg_path = os.path.join(os.path.dirname(cae_meta_path), user_cfg_path)
        if os.path.exists(user_cfg_path):
            with open(user_cfg_path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            if isinstance(user_cfg, dict) and ("seed" in user_cfg):
                return int(user_cfg["seed"])

    raise RuntimeError(
        "CAE metadata missing seed information. "
        "Expected one of: resolved_params.seed, inputs.seed, or inputs.user_config(seed)."
    )


def _extract_cae_inputs(cae_meta: dict) -> tuple[str, list[dict], list[dict], str]:
    problem_name = str(cae_meta.get("problem", "")).strip()
    inputs = cae_meta.get("inputs", {}) if isinstance(cae_meta.get("inputs", {}), dict) else {}
    variables = inputs.get("variables", [])
    if not isinstance(variables, list):
        variables = []
    constraint_defs = inputs.get("constraint_defs", [])
    if not isinstance(constraint_defs, list):
        constraint_defs = []
    resolved = cae_meta.get("resolved_params", {}) if isinstance(cae_meta.get("resolved_params", {}), dict) else {}
    objective_sense = str(resolved.get("objective_sense", "min")).strip().lower()
    if objective_sense not in {"min", "max"}:
        objective_sense = "min"
    if not problem_name:
        raise RuntimeError("CAE metadata missing required field: problem")
    if len(variables) == 0:
        raise RuntimeError("CAE metadata missing required field: inputs.variables")
    return problem_name, variables, constraint_defs, objective_sense


def run_doe(*, config: DOEConfig, run_context: RunContext | None = None) -> dict:
    print("===================================")
    print(" DOE 실행 시작")
    print("===================================")

    cae_meta_path = _resolve_existing_cae_metadata_path(
        config=config,
        run_context=run_context,
    )
    cae_meta = _load_cae_metadata(cae_meta_path)
    (
        problem_name,
        variables,
        constraint_defs,
        objective_sense,
    ) = _extract_cae_inputs(cae_meta)
    cae_seed = _extract_seed_from_cae_metadata(cae_meta=cae_meta, cae_meta_path=cae_meta_path)
    cae_user = config.cae.user
    configured_problem = str(cae_user.problem_name).strip()
    if configured_problem and configured_problem != problem_name:
        raise RuntimeError(
            "Problem mismatch between DOE config and CAE metadata: "
            f"config={configured_problem}, cae_metadata={problem_name}"
        )

    problem_spec = {"name": problem_name, "constraint_defs": constraint_defs}
    if isinstance(config.cae_output, dict):
        upstream_spec = config.cae_output.get("problem_spec", {})
        if isinstance(upstream_spec, dict):
            upstream_name = str(upstream_spec.get("name", "")).strip()
            if upstream_name and upstream_name != problem_name:
                raise RuntimeError(
                    "Problem mismatch between cae_output.problem_spec and CAE metadata: "
                    f"cae_output={upstream_name}, cae_metadata={problem_name}"
                )
            merged_spec = dict(upstream_spec)
            merged_spec["name"] = problem_name
            merged_spec["constraint_defs"] = constraint_defs
            problem_spec = merged_spec

    if isinstance(config.cae_output, dict) and callable(config.cae_output.get("evaluate_func")):
        evaluate_func = config.cae_output["evaluate_func"]
    else:
        _, evaluate_func = select_cae_by_name(problem_name)

    # -------------------------------------------------
    # 4. DOE 알고리즘 선택
    # -------------------------------------------------
    algo_name = config.user.algo_name
    sampler = get_doe_algorithm(algo_name)

    # -------------------------------------------------
    # 5. DOE 실행 설정
    # -------------------------------------------------
    n_samples = int(config.system.n_samples)
    if n_samples <= 5:
        msg = (
            "DOE n_samples는 5 이하로 설정할 수 없습니다. "
            "Modeler KFold(5) 학습을 위해 6 이상으로 설정하세요."
        )
        print(f"- ERROR: {msg}")
        raise ValueError(msg)

    use_timestamp = (
        config.cae.system.use_timestamp if config.cae is not None else False
    )
    run_cfg = {
        "n_samples": n_samples,
        "seed": int(cae_seed),
        "use_timestamp": use_timestamp,
        "debug_level": _normalize_debug_level(config.system.debug_level),
        "success_rate_floor": config.system.success_rate_floor,
        "force_baseline_initial": config.system.force_baseline_initial,
        "initial_corner_ratio": config.system.initial_corner_ratio,
        "initial_probe_multiplier": config.system.additional_initial_probe_multiplier,
        "plan_filter_safety": config.system.plan_filter_safety,
        "plan_filter_r_floor": config.system.plan_filter_r_floor,
    }

    additional_cfg = None
    if config.user.use_additional:
        additional_cfg = {
            "init_ratio": config.system.additional_init_ratio,
            "exec_ratio": config.system.additional_exec_ratio,
            "initial_corner_ratio": config.system.initial_corner_ratio,
            "initial_probe_multiplier": config.system.additional_initial_probe_multiplier,
            "success_rate_floor": config.system.success_rate_floor,
            "global_boundary_ratio": config.system.global_boundary_ratio,
            "global_margin_ratio": config.system.global_margin_ratio,
            "global_top_ratio": config.system.global_top_ratio,
            "global_boundary_corner_ratio": config.system.global_boundary_corner_ratio,
            "plan_base_k": config.system.plan_base_k,
            "plan_remaining_cap": config.system.plan_remaining_cap,
            "plan_decay": config.system.plan_decay,
            "plan_filter_safety": config.system.plan_filter_safety,
            "plan_filter_r_floor": config.system.plan_filter_r_floor,
            "max_additional_stages": config.system.max_additional_stages,
            "gate1_ratio": config.system.gate1_ratio,
            "gate1_pass_ratio": config.system.gate1_pass_ratio,
            "local_anchor_max_base": config.system.local_anchor_max_base,
            "local_anchor_max_decay": config.system.local_anchor_max_decay,
            "local_anchor_best_ratio": config.system.local_anchor_best_ratio,
            "local_anchor_small_ratio": config.system.local_anchor_small_ratio,
            "local_radius_ratio_phase1": config.system.local_radius_ratio_phase1,
            "local_radius_ratio_phase2": config.system.local_radius_ratio_phase2,
            "local_top_p": config.system.local_top_p,
            "local_dbscan_min_samples": config.system.local_dbscan_min_samples,
            "local_dbscan_q_eps": config.system.local_dbscan_q_eps,
            "local_dbscan_eps_max": config.system.local_dbscan_eps_max,
            "local_min_radius_ratio": config.system.local_min_radius_ratio,
            "local_tol_ratio": config.system.local_tol_ratio,
            "local_refine_min_points": config.system.local_refine_min_points,
            "local_cluster_delta_ratio": config.system.local_cluster_delta_ratio,
            "local_singleton_box_ratio": config.system.local_singleton_box_ratio,
            "local_phase1_kappa": config.system.local_phase1_kappa,
            "local_phase2_kappa": config.system.local_phase2_kappa,
            "local_base_perturb_ratio": config.system.local_base_perturb_ratio,
            "local_gp_use_white_kernel": config.system.local_gp_use_white_kernel,
            "local_constraint_retry_count": config.system.local_constraint_retry_count,
            "local_constraint_shrink_factor": config.system.local_constraint_shrink_factor,
            "local_constraint_min_factor": config.system.local_constraint_min_factor,
            "local_exec_pick_mode": config.system.local_exec_pick_mode,
            "post_use_penalty": config.system.post_use_penalty,
            "post_lambda_init": config.system.post_lambda_init,
            "post_lambda_min": config.system.post_lambda_min,
            "post_lambda_max": config.system.post_lambda_max,
            "post_lambda_power": config.system.post_lambda_power,
            "post_feasible_rate_floor": config.system.post_feasible_rate_floor,
            "post_clf_min_samples": config.system.post_clf_min_samples,
            "post_clf_min_pos": config.system.post_clf_min_pos,
            "post_clf_min_neg": config.system.post_clf_min_neg,
            "gate2_k": config.system.gate2_k,
            "gate2_cdf_level": config.system.gate2_cdf_level,
            "gate2_ratio_threshold": config.system.gate2_ratio_threshold,
            "gate2_relax_factor": config.system.gate2_relax_factor,
            "phase1_global_ratio": config.system.phase1_global_ratio,
            "phase2_global_ratio": config.system.phase2_global_ratio,
            "min_additional_rounds": config.system.min_additional_rounds,
            "phase2_min_usable_np_ratio": config.system.phase2_min_usable_np_ratio,
            "stop_span_ratio_threshold": config.system.stop_span_ratio_threshold,
            "stop_anchor_spread_streak": config.system.stop_anchor_spread_streak,
            "stop_min_usable_np_ratio": config.system.stop_min_usable_np_ratio,
            "probe_stage_enabled": config.system.probe_stage_enabled,
            "probe_top_ratio": config.system.probe_top_ratio,
            "probe_max_points": config.system.probe_max_points,
            "probe_min_range_ratio": config.system.probe_min_range_ratio,
            "probe_std_scale": config.system.probe_std_scale,
            "probe_perturb_ratio": config.system.probe_perturb_ratio,
        }
        if config.system.additional_cfg:
            additional_cfg.update(config.system.additional_cfg)

    if run_context is None:
        design_bounds = {v["name"]: [v["lb"], v["ub"]] for v in variables}
        user_snapshot = {
            "problem": problem_spec["name"],
            "seed": int(cae_seed),
            "objective_sense": objective_sense,
            "design_bounds": design_bounds,
            "total_budget": n_samples,
            "task": "DOE",
        }
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        run_context = create_run_context(
            project_root=project_root,
            user_config_snapshot=user_snapshot,
        )
    if get_task_metadata_path(run_context, "CAE") is None:
        update_run_index(run_context, "CAE", os.path.abspath(cae_meta_path))

    return run_doe_orchestrator(
        problem_spec=problem_spec,
        evaluate_func=evaluate_func,
        variables=variables,
        algo_name=algo_name,
        sampler=sampler,
        run_cfg=run_cfg,
        objective_sense=objective_sense,
        use_additional=config.user.use_additional,
        additional_cfg=additional_cfg,
        run_context=run_context,
    )


if __name__ == "__main__":
    # Standalone 실행은 CAE metadata 경로가 필요합니다.
    # 예: "result/run_<id>/CAE/metadata.json"
    cfg = DOEConfig(
        cae=CAEConfig(
            user=CAEUserConfig(problem_name="goldstein_price", seed=42),
            system=CAESystemConfig(use_timestamp=True),
        ),
        cae_user=None,
        user=DOEUserConfig(algo_name="lhs", use_additional=False),
        system=DOESystemConfig(n_samples=100),
        cae_metadata_path=None,
    )
    run_doe(config=cfg)
