# run_DOE.py

from CAE_tool_interface.run_CAE import run_cae
from DOE.config import DOEConfig, DOESystemConfig, DOEUserConfig
from DOE.executor.doe_orchestrator import run_doe_orchestrator
from pipeline.run_context import RunContext
from DOE.doe_algorithm.registry import get_doe_algorithm


def run_doe(*, config: DOEConfig, run_context: RunContext | None = None) -> list[dict]:
    print("===================================")
    print(" DOE 실행 시작")
    print("===================================")

    # -------------------------------------------------
    # 1. CAE 선택
    # -------------------------------------------------
    if config.cae_output:
        cae_out = config.cae_output
    else:
        cae_out = run_cae(config=config.cae)
    cae_user = None
    cae_user = config.cae.user
    problem_spec = cae_out["problem_spec"]
    evaluate_func = cae_out["evaluate_func"]
    variables = cae_out["variables"]
    objective_sense = cae_user.objective_sense

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
        "seed": cae_user.seed,
        "use_timestamp": use_timestamp,
        "force_baseline_initial": config.system.force_baseline_initial,
        "initial_probe_multiplier": config.system.additional_initial_probe_multiplier,
        "plan_filter_safety": config.system.plan_filter_safety,
        "plan_filter_r_floor": config.system.plan_filter_r_floor,
    }

    additional_cfg = None
    if config.user.use_additional:
        additional_cfg = {
            "init_ratio": config.system.additional_init_ratio,
            "exec_ratio": config.system.additional_exec_ratio,
            "initial_probe_multiplier": config.system.additional_initial_probe_multiplier,
            "global_random_ratio": config.system.global_random_ratio,
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
            "local_anchor_best_k": config.system.local_anchor_best_k,
            "local_anchor_small_k": config.system.local_anchor_small_k,
            "local_anchor_best_ratio": config.system.local_anchor_best_ratio,
            "local_anchor_small_ratio": config.system.local_anchor_small_ratio,
            "local_radius_ratio_phase1": config.system.local_radius_ratio_phase1,
            "local_radius_ratio_phase2": config.system.local_radius_ratio_phase2,
            "local_top_p": config.system.local_top_p,
            "local_top_k_min": config.system.local_top_k_min,
            "local_dbscan_min_samples": config.system.local_dbscan_min_samples,
            "local_dbscan_q_eps": config.system.local_dbscan_q_eps,
            "local_dbscan_eps_max": config.system.local_dbscan_eps_max,
            "local_min_radius_ratio": config.system.local_min_radius_ratio,
            "local_tol_ratio": config.system.local_tol_ratio,
            "local_constraint_retry_count": config.system.local_constraint_retry_count,
            "local_constraint_shrink_factor": config.system.local_constraint_shrink_factor,
            "local_constraint_min_factor": config.system.local_constraint_min_factor,
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
            "stop_span_ratio_threshold": config.system.stop_span_ratio_threshold,
            "stop_anchor_spread_streak": config.system.stop_anchor_spread_streak,
        }
        if config.system.additional_cfg:
            additional_cfg.update(config.system.additional_cfg)

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
    cfg = DOEConfig(
        cae=CAEConfig(
            user=CAEUserConfig(problem_name="goldstein_price", seed=42),
            system=CAESystemConfig(use_timestamp=True),
        ),
        cae_user=None,
        user=DOEUserConfig(algo_name="lhs", use_additional=False),
        system=DOESystemConfig(n_samples=100),
    )
    run_doe(config=cfg)
