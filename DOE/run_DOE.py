import json
import os

from CAE_tool_interface.executor.configurator import select_cae_by_name
from DOE.config import (
    DOEConfig,
    DOESystemConfig,
    build_additional_cfg_from_system,
)
from DOE.executor.doe_orchestrator import run_doe_orchestrator
from pipeline.run_context import (
    RunContext,
    create_run_context,
    get_task_metadata_path,
    update_run_index,
)
from DOE.doe_algorithm.registry import get_doe_algorithm


def _normalize_debug_level(value: str | None) -> str:
    level = str(value or "on").strip().lower()
    if level == "full":
        level = "on"
    if level not in {"off", "on"}:
        raise ValueError("DOE debug_level must be one of: off, on")
    return level


def _resolve_initial_corner_ratio(
    *,
    system: DOESystemConfig,
    n_samples: int,
    p_dim: int,
) -> tuple[float, str, dict]:
    base_ratio = float(system.initial_corner_ratio)
    effective_ratio = float(max(0.0, min(1.0, base_ratio)))
    policy = "fixed"

    p_dim_eff = max(int(p_dim), 1)
    np_ratio = float(n_samples) / float(p_dim_eff)
    context = {
        "base_ratio": float(base_ratio),
        "effective_ratio": float(effective_ratio),
        "np_ratio": float(np_ratio),
        "p_dim": int(p_dim_eff),
    }

    if not bool(getattr(system, "initial_corner_adaptive_enabled", False)):
        return effective_ratio, policy, context

    np_ratio_max = float(getattr(system, "initial_corner_adaptive_np_ratio_max", 10.0))
    low_dim_max = int(getattr(system, "initial_corner_adaptive_low_dim_max", 6))
    low_dim_ratio = float(getattr(system, "initial_corner_adaptive_ratio_low_dim", 0.05))

    if np_ratio <= np_ratio_max:
        if p_dim_eff <= low_dim_max:
            effective_ratio = min(effective_ratio, low_dim_ratio)
            policy = "adaptive_p_le_6_low_np"
        else:
            policy = "adaptive_p_gt_6_keep"
    else:
        policy = "adaptive_np_high_keep"

    effective_ratio = float(max(0.0, min(1.0, effective_ratio)))
    context["effective_ratio"] = float(effective_ratio)
    return effective_ratio, policy, context


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
    p_dim = int(len(variables))
    initial_corner_ratio_eff, initial_corner_policy, initial_corner_context = _resolve_initial_corner_ratio(
        system=config.system,
        n_samples=n_samples,
        p_dim=p_dim,
    )

    use_timestamp = (
        config.cae.system.use_timestamp if config.cae is not None else False
    )
    run_cfg = {
        "n_samples": n_samples,
        "seed": int(cae_seed),
        "use_timestamp": use_timestamp,
        "debug_level": _normalize_debug_level(config.system.debug_level),
        "success_rate_floor": config.system.success_rate_floor,
        "initial_corner_ratio": initial_corner_ratio_eff,
        "initial_corner_ratio_base": float(config.system.initial_corner_ratio),
        "initial_corner_ratio_policy": initial_corner_policy,
        "initial_corner_np_ratio": initial_corner_context["np_ratio"],
        "initial_corner_p_dim": p_dim,
        "initial_probe_multiplier": config.system.additional_initial_probe_multiplier,
        "plan_filter_safety": config.system.plan_filter_safety,
        "plan_filter_r_floor": config.system.plan_filter_r_floor,
        "projection_pool_multiplier": config.system.projection_pool_multiplier,
        "projection_pool_min": config.system.projection_pool_min,
    }

    additional_cfg = None
    if config.user.use_additional:
        additional_cfg = build_additional_cfg_from_system(
            config.system,
            initial_corner_ratio=initial_corner_ratio_eff,
            initial_corner_ratio_base=float(config.system.initial_corner_ratio),
            initial_corner_ratio_policy=initial_corner_policy,
            initial_corner_np_ratio=initial_corner_context["np_ratio"],
            initial_corner_p_dim=p_dim,
        )
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
    raise SystemExit(
        "DOE/run_DOE.py is an internal task runner. "
        "Use pipeline/run_pipeline.py as the execution entrypoint."
    )
