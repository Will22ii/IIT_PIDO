# DOE/executor/doe_orchestrator.py

import json
import os
import numpy as np
import pandas as pd

from utils.feasibility import evaluate_feasibility
from utils.result_saver import ResultSaver
from DOE.executor.constraint_filter import (
    clamp_ratio,
    evaluate_constraints_batch,
    evaluate_constraints_point,
    validate_constraint_defs,
)
from DOE.executor.additional_orchestrator import AdditionalDOEOrchestrator
from DOE.executor.doe_report_builder import DOEReportBuilder
from DOE.executor.surrogate_factory import SurrogateFactory
from DOE.gate.gate1_topk_stability import Gate1TopKStability
from DOE.gate.gate2_uncertainty import Gate2Uncertainty
from DOE.gate.gate_manager import GateManager
from Modeler.executor.hpo_runner import HPORunner
from pipeline.run_context import RunContext, update_run_index


DEFAULT_ADDITIONAL_CFG = {
    "init_ratio": 0.4,
    "exec_ratio": 0.1,
    "global_random_ratio": 0.4,
    "global_boundary_ratio": 0.2,
    "global_margin_ratio": 0.2,
    "global_top_ratio": 0.2,
    "global_boundary_corner_ratio": 0.4,
    "plan_base_k": 200.0,
    "plan_remaining_cap": 3.0,
    "plan_decay": 0.85,
    "gate1_ratio": 0.5,
    "gate1_pass_ratio": 0.4,
    "local_anchor_max_base": 8,
    "local_anchor_max_decay": 0.9,
    "local_anchor_best_k": 3,
    "local_anchor_small_k": 2,
    "local_anchor_best_ratio": 0.35,
    "local_anchor_small_ratio": 0.2,
    "local_radius_ratio_phase1": 0.15,
    "local_radius_ratio_phase2": 0.05,
    "local_top_p": 0.5,
    "local_top_k_min": 10,
    "local_dbscan_min_samples": 2,
    "local_dbscan_q_eps": 0.8,
    "local_dbscan_eps_max": 0.3,
    "local_min_radius_ratio": 0.01,
    "local_tol_ratio": 0.2,
    "gate2_k": 2,
    "gate2_cdf_level": 0.9,
    "gate2_ratio_threshold": 0.9,
    "gate2_relax_factor": 1.1,
    "phase1_global_ratio": 0.8,
    "phase2_global_ratio": 0.2,
    "min_additional_rounds": 3,
    "stop_span_ratio_threshold": 0.3,
    "stop_anchor_spread_streak": 2,
    "initial_probe_multiplier": 2.0,
    "plan_filter_safety": 1.2,
    "plan_filter_r_floor": 0.02,
    "max_additional_stages": 10,
    "local_constraint_retry_count": 1,
    "local_constraint_shrink_factor": 0.5,
    "local_constraint_min_factor": 2.0,
    "post_use_penalty": True,
    "post_lambda_init": 2.0,
    "post_lambda_min": 0.25,
    "post_lambda_max": 8.0,
    "post_lambda_power": 1.0,
    "post_feasible_rate_floor": 0.05,
    "post_clf_min_samples": 30,
    "post_clf_min_pos": 5,
    "post_clf_min_neg": 5,
}


def _save_doe_results(
    *,
    results: list[dict],
    variables: list[dict],
    problem_name: str,
    workflow_info: dict,
    objective_sense: str,
    seed: int,
    saver: ResultSaver,
    additional_doe: bool = False,
    run_context: RunContext | None = None,
    system_config_snapshot: dict | None = None,
    resolved_params_extra: dict | None = None,
    results_extra: dict | None = None,
    extra_metadata: dict | None = None,
    stage_name: str = "DOE",
    constraint_defs: list[dict] | None = None,
) -> dict:
    var_names = [v["name"] for v in variables]
    rows = []
    constraint_rows = []
    for r in results:
        row = {
            "id": r["id"],
            "objective": r["objective"],
            "feasible_pre": r.get("feasible_pre", True),
            "feasible_post": r.get("feasible_post", True),
            "feasible_final": r.get("feasible_final", r.get("feasible", True)),
            "feasible": r.get("feasible", r.get("feasible_final", True)),
            "success": r["success"],
            "margin_pre": r.get("margin_pre", float("inf")),
            "margin_post": r.get("margin_post", float("inf")),
            "constraint_margin": r.get("constraint_margin", float("inf")),
        }
        row["source"] = r.get("source", "basic")
        row["round"] = r.get("round")
        row["exec_scope"] = r.get("exec_scope", "basic")
        for name, v in zip(var_names, r["x"]):
            row[name] = v
        rows.append(row)

        constraints = r.get("constraints") or {}
        if isinstance(constraints, dict):
            for cid, cinfo in constraints.items():
                if not isinstance(cinfo, dict):
                    continue
                constraint_rows.append(
                    {
                        "sample_id": r["id"],
                        "constraint_id": cinfo.get("id", cid),
                        "constraint_name": cinfo.get("name", cid),
                        "scope": cinfo.get("scope"),
                        "type": cinfo.get("type"),
                        "limit": cinfo.get("limit"),
                        "value": cinfo.get("value"),
                        "margin": cinfo.get("margin"),
                        "g": cinfo.get("g"),
                        "ok": cinfo.get("ok"),
                        "expr_error": cinfo.get("expr_error"),
                        "source": r.get("source", "basic"),
                        "round": r.get("round"),
                        "exec_scope": r.get("exec_scope", "basic"),
                    }
                )

    df = pd.DataFrame(rows)
    df_constraints = pd.DataFrame(constraint_rows)

    if run_context:
        inputs = {
            "user_config": os.path.relpath(
                run_context.user_config_snapshot_path,
                os.path.join(run_context.run_root, stage_name),
            ),
            "system_config_snapshot": system_config_snapshot or {},
            "previous": {},
            "variables": variables,
            "constraint_defs": constraint_defs or [],
        }
        resolved_params = {
            "n_samples": len(df),
        }
        if resolved_params_extra:
            resolved_params.update(resolved_params_extra)
        results_summary = {
            "n_samples_total": len(df),
            "n_success": int(df["success"].sum()) if "success" in df.columns else 0,
            "n_feasible": int(df["feasible"].sum()) if "feasible" in df.columns else 0,
        }
        if not df_constraints.empty:
            results_summary["n_constraint_rows"] = int(len(df_constraints))
        if results_extra:
            results_summary.update(results_extra)
        artifacts = {}
        stage_out = saver.save_stage_v2(
            run_root=run_context.run_root,
            stage=stage_name,
            problem_name=problem_name,
            df=df,
            inputs=inputs,
            resolved_params=resolved_params,
            results=results_summary,
            artifacts=artifacts,
        )
        post_policy_log = None
        if results_extra and isinstance(results_extra.get("post_policy_log"), list):
            post_policy_log = results_extra.get("post_policy_log")
        if post_policy_log:
            post_policy_csv = os.path.join(stage_out["artifacts_dir"], "doe_post_policy_log.csv")
            pd.DataFrame(post_policy_log).to_csv(post_policy_csv, index=False)
            with open(stage_out["metadata"], "r") as f:
                meta = json.load(f)
            meta.setdefault("artifacts", {})
            meta["artifacts"]["post_policy_log_csv"] = os.path.relpath(
                post_policy_csv,
                stage_out["stage_dir"],
            )
            with open(stage_out["metadata"], "w") as f:
                json.dump(meta, f, indent=2)
        if not df_constraints.empty:
            constraints_csv = os.path.join(stage_out["artifacts_dir"], "doe_constraints.csv")
            df_constraints.to_csv(constraints_csv, index=False)
            with open(stage_out["metadata"], "r") as f:
                meta = json.load(f)
            meta.setdefault("artifacts", {})
            meta["artifacts"]["constraints_csv"] = os.path.relpath(
                constraints_csv,
                stage_out["stage_dir"],
            )
            with open(stage_out["metadata"], "w") as f:
                json.dump(meta, f, indent=2)
        update_run_index(run_context, stage_name, stage_out["metadata"])
    else:
        stage_out = saver.save_stage_csv(
            stage="DOE",
            df=df,
            problem_name=problem_name,
            extra_metadata={
                "seed": seed,
                "n_samples": len(df),
                "workflow_info": workflow_info,
                "objective_sense": objective_sense,
                "variables": variables,
                "additional_doe": additional_doe,
                "constraint_defs": constraint_defs or [],
                **(extra_metadata or {}),
            },
        )
        if not df_constraints.empty:
            constraints_csv = os.path.join(
                os.path.dirname(stage_out["csv"]),
                f"doe_constraints_{problem_name}.csv",
            )
            df_constraints.to_csv(constraints_csv, index=False)
            with open(stage_out["metadata"], "r") as f:
                meta = json.load(f)
            meta["constraints_csv_path"] = constraints_csv
            with open(stage_out["metadata"], "w") as f:
                json.dump(meta, f, indent=2)
        post_policy_log = None
        if results_extra and isinstance(results_extra.get("post_policy_log"), list):
            post_policy_log = results_extra.get("post_policy_log")
        if post_policy_log:
            post_policy_csv = os.path.join(
                os.path.dirname(stage_out["csv"]),
                f"doe_post_policy_log_{problem_name}.csv",
            )
            pd.DataFrame(post_policy_log).to_csv(post_policy_csv, index=False)
            with open(stage_out["metadata"], "r") as f:
                meta = json.load(f)
            meta["post_policy_log_csv_path"] = post_policy_csv
            with open(stage_out["metadata"], "w") as f:
                json.dump(meta, f, indent=2)

    print(f"\nDOE CSV saved to: {stage_out['csv']}")
    print(f"DOE metadata saved to: {stage_out['metadata']}")

    return {
        "df": df,
        "stage_out": stage_out,
    }


def _save_final_report(
    *,
    problem_name: str,
    workflow_info: dict,
    results: list[dict],
    n_samples: int,
    dimension: int,
    objective_sense: str,
    dump_table: bool,
    output_path: str | None = None,
    use_timestamp: bool = False,
):
    report_lines = DOEReportBuilder.build(
        problem_name=problem_name,
        workflow_info=workflow_info,
        results=results,
        n_samples=n_samples,
        dimension=dimension,
        objective_sense=objective_sense,
        dump_table=dump_table,
    )

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("========================================\n")
            f.write(f"{problem_name} DOE RESULT\n")
            f.write("========================================\n\n")
            for line in report_lines:
                f.write(line + "\n")
    else:
        saver_txt = ResultSaver(use_timestamp=use_timestamp)
        saver_txt.save_final_txt(
            content_lines=report_lines,
            problem_name=problem_name,
            workflow_info=workflow_info,
        )


def run_doe_orchestrator(
    *,
    problem_spec: dict,
    evaluate_func,
    variables: list[dict],
    algo_name: str,
    sampler,
    run_cfg: dict,
    objective_sense: str,
    use_additional: bool = False,
    additional_cfg: dict | None = None,
    run_context: RunContext | None = None,
    stage_name: str = "DOE",
) -> list[dict]:
    dim = len(variables)
    seed = run_cfg["seed"]
    n_samples_total = int(run_cfg["n_samples"])
    if n_samples_total <= 5:
        raise ValueError(
            "DOE n_samples must be greater than 5 for downstream Modeler KFold(5)."
        )
    force_baseline = bool(run_cfg.get("force_baseline_initial", False))
    var_names = [v["name"] for v in variables]
    constraint_defs = validate_constraint_defs(problem_spec.get("constraint_defs", []) or [])
    has_pre_constraints = any(
        str(c.get("scope", "x_only")).strip().lower() == "x_only"
        for c in constraint_defs
    )
    filter_safety = float(run_cfg.get("plan_filter_safety", 1.2))
    filter_r_floor = float(run_cfg.get("plan_filter_r_floor", 0.02))

    workflow_info = {
        "DOE": algo_name,
        "MODELER": None,
        "EXPLORER": None,
        "OPT": None,
    }

    if not use_additional:
        n_samples = int(run_cfg["n_samples"])

        print("\nDOE 설정 요약")
        print(f"- 문제명           : {problem_spec['name']}")
        print(f"- 초기 DOE 알고리즘 : {algo_name}")
        print(f"- 차원 수           : {dim}")
        print(f"- 샘플 수           : {n_samples}")
        print(f"- Seed              : {seed}")

        rng = np.random.default_rng(seed)
        saver = ResultSaver(use_timestamp=bool(run_cfg.get("use_timestamp", False)))

        bounds = [(v["lb"], v["ub"]) for v in variables]
        baseline = np.array([v["baseline"] for v in variables])

        probe_multiplier = float(run_cfg.get("initial_probe_multiplier", 2.0))
        if has_pre_constraints:
            n_probe = max(n_samples, int(np.ceil(n_samples * probe_multiplier)))
            n_probe = int(np.ceil(n_probe * filter_safety))
            X_probe = sampler(
                n_samples=n_probe,
                bounds=bounds,
                rng=rng,
                n_divisions=max(n_probe, 1),
            )
            if force_baseline and X_probe.shape[0] > 0:
                X_probe[0, :] = baseline
            feas_mask, constraint_payloads, margins_pre = evaluate_constraints_batch(
                X=X_probe,
                var_names=var_names,
                constraint_defs=constraint_defs,
                scope="x_only",
            )
            n_feas = int(feas_mask.sum())
            r_hat = clamp_ratio(
                n_feas / max(X_probe.shape[0], 1),
                floor=filter_r_floor,
            )
            if n_feas < n_samples:
                raise RuntimeError(
                    f"FAILED_FILTER_MIN: initial feasible points {n_feas} < target {n_samples}"
                )
            feas_idx = np.where(feas_mask)[0]
            pick_idx = rng.choice(feas_idx, size=n_samples, replace=False)
            X = X_probe[pick_idx]
            picked_constraints = [constraint_payloads[i] for i in pick_idx]
            picked_margins = np.asarray([margins_pre[i] for i in pick_idx], dtype=float)
        else:
            r_hat = 1.0
            n_probe = n_samples
            X = sampler(
                n_samples=n_samples,
                bounds=bounds,
                rng=rng,
                n_divisions=max(n_samples, 1),
            )
            if force_baseline and X.shape[0] > 0:
                X[0, :] = baseline
            picked_constraints = [{} for _ in range(X.shape[0])]
            picked_margins = np.full((X.shape[0],), float("inf"), dtype=float)

        print("\nDOE 샘플 생성 완료")
        print("\nCAE 평가 시작...\n")
        results = []

        for i, x in enumerate(X):
            y = evaluate_func(x)
            success = bool(y.get("success", True))
            objective = float(y.get("objective", float("inf")))
            outputs = y.get("outputs", {})
            if not isinstance(outputs, dict):
                outputs = {}

            constraints_pre = picked_constraints[i] if i < len(picked_constraints) else {}
            feasible_pre = evaluate_feasibility(constraints_pre)
            margin_pre = float(picked_margins[i]) if i < len(picked_margins) else float("inf")

            constraints_post = {}
            feasible_post = True
            margin_post = float("inf")
            try:
                constraints_post, feasible_post, margin_post = evaluate_constraints_point(
                    x=np.asarray(x, dtype=float),
                    var_names=var_names,
                    constraint_defs=constraint_defs,
                    scope="cae_dependent",
                    env_extra={**outputs, "objective": objective},
                    fail_fast_output_missing=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"FAILED_POST_CONSTRAINT_OUTPUT: sample={i}, error={exc}"
                ) from exc

            constraints = {**constraints_pre, **constraints_post}
            feasible_final = bool(success and feasible_pre and feasible_post)
            margin_vals = [v for v in [margin_pre, margin_post] if np.isfinite(v)]
            constraint_margin = float(min(margin_vals) if margin_vals else float("inf"))

            results.append({
                "id": i,
                "x": x.tolist(),
                "objective": objective,
                "constraints": constraints,
                "margin_pre": float(margin_pre),
                "margin_post": float(margin_post),
                "constraint_margin": float(constraint_margin),
                "feasible_pre": bool(feasible_pre),
                "feasible_post": bool(feasible_post),
                "feasible_final": bool(feasible_final),
                "feasible": bool(feasible_final),
                "success": bool(success),
            })

            print(
                f"[{i+1:03d}/{n_samples}] "
                f"objective = {objective}, "
                f"feasible_final = {feasible_final}"
            )

        system_snapshot = {
            "n_samples": run_cfg.get("n_samples"),
            "use_timestamp": run_cfg.get("use_timestamp"),
            "initial_probe_multiplier": probe_multiplier,
            "plan_filter_safety": filter_safety,
            "plan_filter_r_floor": filter_r_floor,
            "constraint_r_hat": r_hat,
        }
        out = _save_doe_results(
            results=results,
            variables=variables,
            problem_name=problem_spec["name"],
            workflow_info=workflow_info,
            objective_sense=objective_sense,
            seed=seed,
            saver=saver,
            additional_doe=False,
            run_context=run_context,
            system_config_snapshot=system_snapshot,
            stage_name=stage_name,
            constraint_defs=constraint_defs,
        )

        report_path = None
        if run_context:
            report_path = os.path.join(
                out["stage_out"]["artifacts_dir"],
                "doe_report.txt",
            )
            out["stage_out"]["artifacts_dir"]
        _save_final_report(
            problem_name=problem_spec["name"],
            workflow_info=workflow_info,
            results=results,
            n_samples=n_samples,
            dimension=dim,
            objective_sense=objective_sense,
            dump_table=False,
            output_path=report_path,
            use_timestamp=bool(run_cfg.get("use_timestamp", False)),
        )
        if run_context and report_path:
            with open(out["stage_out"]["metadata"], "r") as f:
                meta = json.load(f)
            meta["artifacts"]["doe_report"] = os.path.relpath(
                report_path,
                out["stage_out"]["stage_dir"],
            )
            with open(out["stage_out"]["metadata"], "w") as f:
                json.dump(meta, f, indent=2)

        print("\n===================================")
        print(" DOE 실행 완료")
        print("===================================")
        return results

    # -------------------------------------------------
    # Additional DOE branch
    # -------------------------------------------------
    cfg = dict(DEFAULT_ADDITIONAL_CFG)
    if additional_cfg:
        cfg.update(additional_cfg)

    total_budget = int(run_cfg["n_samples"])

    print("\nDOE 설정 요약")
    print(f"- 문제명           : {problem_spec['name']}")
    print(f"- 초기 DOE 알고리즘 : {algo_name}")
    print(f"- 차원 수           : {dim}")
    print(f"- 총 예산           : {total_budget}")
    print(f"- Seed              : {seed}")
    print(f"- Additional DOE    : ENABLED")

    rng = np.random.default_rng(seed)
    saver = ResultSaver(use_timestamp=bool(run_cfg.get("use_timestamp", False)))

    gate1 = Gate1TopKStability(
        k_ratio=cfg["gate1_ratio"],
        pass_ratio=cfg["gate1_pass_ratio"],
        objective_sense=objective_sense,
    )

    gate2 = Gate2Uncertainty(
        k=cfg["gate2_k"],
        cdf_level=cfg["gate2_cdf_level"],
        ratio_threshold=cfg["gate2_ratio_threshold"],
        relax_factor=cfg["gate2_relax_factor"],
    )

    gate_manager = GateManager()

    gate2_bootstrap_seeds = tuple(seed + i for i in range(5))
    surrogate_factory = SurrogateFactory(
        n_models=5,
        gate1_fixed_seed=seed,
        gate1_fixed_model_seed=seed,
        gate2_bootstrap_seeds=gate2_bootstrap_seeds,
    )

    hpo_runner = HPORunner(n_trials=10)

    bounds = [(v["lb"], v["ub"]) for v in variables]
    baseline = np.array([v["baseline"] for v in variables], dtype=float)

    orchestrator = AdditionalDOEOrchestrator(
        bounds=bounds,
        sampler=sampler,
        evaluate_func=evaluate_func,
        feasibility_func=evaluate_feasibility,
        var_names=var_names,
        constraint_defs=constraint_defs,
        surrogate_factory=surrogate_factory,
        gate1=gate1,
        gate2=gate2,
        gate_manager=gate_manager,
        rng=rng,
        total_budget=total_budget,
        init_ratio=cfg["init_ratio"],
        exec_ratio=cfg["exec_ratio"],
        global_random_ratio=cfg["global_random_ratio"],
        global_boundary_ratio=cfg.get("global_boundary_ratio", 0.1),
        global_margin_ratio=cfg.get("global_margin_ratio", 0.2),
        global_top_ratio=cfg.get("global_top_ratio", 0.2),
        global_boundary_corner_ratio=cfg.get("global_boundary_corner_ratio", 0.5),
        plan_base_k=cfg["plan_base_k"],
        plan_remaining_cap=cfg["plan_remaining_cap"],
        plan_decay=cfg["plan_decay"],
        phase1_global_ratio=cfg["phase1_global_ratio"],
        phase2_global_ratio=cfg["phase2_global_ratio"],
        min_additional_rounds=cfg["min_additional_rounds"],
        stop_span_ratio_threshold=cfg["stop_span_ratio_threshold"],
        stop_anchor_spread_streak=cfg["stop_anchor_spread_streak"],
        initial_probe_multiplier=cfg.get("initial_probe_multiplier", 2.0),
        plan_filter_safety=cfg.get("plan_filter_safety", 1.2),
        plan_filter_r_floor=cfg.get("plan_filter_r_floor", 0.02),
        max_additional_stages=cfg.get("max_additional_stages", 10),
        local_anchor_max_base=cfg.get("local_anchor_max_base", 8),
        local_anchor_max_decay=cfg.get("local_anchor_max_decay", 0.9),
        local_anchor_best_k=cfg.get("local_anchor_best_k", 3),
        local_anchor_small_k=cfg.get("local_anchor_small_k", 2),
        local_anchor_best_ratio=cfg.get("local_anchor_best_ratio", 0.35),
        local_anchor_small_ratio=cfg.get("local_anchor_small_ratio", 0.2),
        local_radius_ratio_phase1=cfg["local_radius_ratio_phase1"],
        local_radius_ratio_phase2=cfg["local_radius_ratio_phase2"],
        local_top_p=cfg["local_top_p"],
        local_top_k_min=cfg["local_top_k_min"],
        local_dbscan_min_samples=cfg["local_dbscan_min_samples"],
        local_dbscan_q_eps=cfg["local_dbscan_q_eps"],
        local_dbscan_eps_max=cfg["local_dbscan_eps_max"],
        local_min_radius_ratio=cfg["local_min_radius_ratio"],
        local_tol_ratio=cfg["local_tol_ratio"],
        local_constraint_retry_count=cfg.get("local_constraint_retry_count", 1),
        local_constraint_shrink_factor=cfg.get("local_constraint_shrink_factor", 0.5),
        local_constraint_min_factor=cfg.get("local_constraint_min_factor", 2.0),
        post_use_penalty=cfg.get("post_use_penalty", True),
        post_lambda_init=cfg.get("post_lambda_init", 2.0),
        post_lambda_min=cfg.get("post_lambda_min", 0.25),
        post_lambda_max=cfg.get("post_lambda_max", 8.0),
        post_lambda_power=cfg.get("post_lambda_power", 1.0),
        post_feasible_rate_floor=cfg.get("post_feasible_rate_floor", 0.05),
        post_clf_min_samples=cfg.get("post_clf_min_samples", 30),
        post_clf_min_pos=cfg.get("post_clf_min_pos", 5),
        post_clf_min_neg=cfg.get("post_clf_min_neg", 5),
        hpo_runner=hpo_runner,
        force_baseline=force_baseline,
    )

    results = orchestrator.run(
        baseline=baseline,
        problem_name=problem_spec["name"],
        base_seed=seed,
        objective_sense=objective_sense,
    )
    diagnostics = orchestrator.get_diagnostics()

    system_snapshot = {
        "n_samples": run_cfg.get("n_samples"),
        "use_timestamp": run_cfg.get("use_timestamp"),
        "additional_cfg": cfg,
    }
    out = _save_doe_results(
        results=results,
        variables=variables,
        problem_name=problem_spec["name"],
        workflow_info=workflow_info,
        objective_sense=objective_sense,
        seed=seed,
        saver=saver,
        additional_doe=True,
        run_context=run_context,
        system_config_snapshot=system_snapshot,
        stage_name=stage_name,
        resolved_params_extra={
            "failure_reason": diagnostics.get("failure_reason"),
            "constraint_rate_hat": diagnostics.get("constraint_rate_hat"),
            "post_feasible_rate_hat": diagnostics.get("post_feasible_rate_hat"),
            "post_lambda": diagnostics.get("post_lambda"),
            "post_model_active": diagnostics.get("post_model_active"),
        },
        results_extra={
            "local_metrics": diagnostics.get("local_metrics"),
            "post_policy_log": diagnostics.get("post_policy_log"),
        },
        extra_metadata={
            "failure_reason": diagnostics.get("failure_reason"),
            "local_metrics": diagnostics.get("local_metrics"),
            "constraint_rate_hat": diagnostics.get("constraint_rate_hat"),
            "post_feasible_rate_hat": diagnostics.get("post_feasible_rate_hat"),
            "post_lambda": diagnostics.get("post_lambda"),
            "post_model_active": diagnostics.get("post_model_active"),
            "post_policy_log": diagnostics.get("post_policy_log"),
        },
        constraint_defs=constraint_defs,
    )

    report_path = None
    if run_context:
        report_path = os.path.join(
            out["stage_out"]["artifacts_dir"],
            "doe_report.txt",
        )
    _save_final_report(
        problem_name=problem_spec["name"],
        workflow_info=workflow_info,
        results=results,
        n_samples=len(results),
        dimension=dim,
        objective_sense=objective_sense,
        dump_table=True,
        output_path=report_path,
        use_timestamp=bool(run_cfg.get("use_timestamp", False)),
    )
    if run_context and report_path:
        with open(out["stage_out"]["metadata"], "r") as f:
            meta = json.load(f)
        meta["artifacts"]["doe_report"] = os.path.relpath(
            report_path,
            out["stage_out"]["stage_dir"],
        )
        with open(out["stage_out"]["metadata"], "w") as f:
            json.dump(meta, f, indent=2)

    print("\n===================================")
    print(" DOE + ADDITIONAL DOE 실행 완료")
    print("===================================")
    return results
