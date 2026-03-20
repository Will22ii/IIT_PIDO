# DOE/executor/doe_orchestrator.py

import json
import os
import re
import numpy as np
import pandas as pd

from utils.boundary_sampling import sample_boundary_corners_random
from utils.feasibility import evaluate_feasibility
from utils.result_saver import ResultSaver
from DOE.executor.constraint_filter import (
    clamp_ratio,
    evaluate_constraints_batch,
    evaluate_constraints_point,
    validate_constraint_defs,
)
from DOE.executor.additional_orchestrator import AdditionalDOEOrchestrator
from DOE.executor.eval_sanitizer import sanitize_evaluate_output
from DOE.executor.surrogate_factory import SurrogateFactory
from DOE.gate.gate1_topk_stability import Gate1TopKStability
from DOE.gate.gate2_uncertainty import Gate2Uncertainty
from DOE.gate.gate_manager import GateManager
from Modeler.executor.hpo_runner import HPORunner
from pipeline.run_context import RunContext, update_run_index


DEFAULT_ADDITIONAL_CFG = {
    "init_ratio": 0.4,
    "exec_ratio": 0.1,
    "initial_corner_ratio": 0.05,
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
    "local_refine_min_points": 15,
    "local_cluster_delta_ratio": 0.02,
    "local_singleton_box_ratio": 0.03,
    "local_phase1_kappa": 0.75,
    "local_phase2_kappa": 0.5,
    "local_base_perturb_ratio": 0.02,
    "gate2_k": 2,
    "gate2_cdf_level": 0.9,
    "gate2_ratio_threshold": 0.9,
    "gate2_relax_factor": 1.1,
    "phase1_global_ratio": 0.8,
    "phase2_global_ratio": 0.2,
    "min_additional_rounds": 3,
    "phase2_min_usable_np_ratio": 15.0,
    "stop_span_ratio_threshold": 0.3,
    "stop_anchor_spread_streak": 2,
    "stop_min_usable_np_ratio": 20.0,
    "probe_stage_enabled": True,
    "probe_top_ratio": 0.3,
    "probe_max_points": 5,
    "probe_min_range_ratio": 0.25,
    "probe_std_scale": 2.0,
    "probe_perturb_ratio": 0.02,
    "initial_probe_multiplier": 2.0,
    "success_rate_floor": 0.02,
    "plan_filter_safety": 1.2,
    "plan_filter_r_floor": 0.02,
    "max_additional_stages": 10,
    "local_constraint_retry_count": 1,
    "local_constraint_shrink_factor": 0.5,
    "local_constraint_min_factor": 2.0,
    "local_exec_pick_mode": "random",
    "post_use_penalty": True,
    "post_lambda_init": 2.0,
    "post_lambda_min": 0.25,
    "post_lambda_max": 8.0,
    "post_lambda_power": 1.0,
    "post_feasible_rate_floor": 0.05,
    "post_clf_min_samples": 30,
    "post_clf_min_pos": 5,
    "post_clf_min_neg": 5,
    "local_gp_use_white_kernel": False,
}


def _normalize_debug_level(value: str | None) -> str:
    level = str(value or "off").strip().lower()
    if level not in {"off", "full"}:
        raise ValueError("DOE debug_level must be one of: off, full")
    return level


def _sanitize_constraint_token(raw: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z_]+", "_", str(raw or "").strip().lower())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "constraint"


def _build_constraint_column_map(
    *,
    results: list[dict],
    constraint_defs: list[dict] | None,
) -> dict[str, str]:
    ordered_ids: list[str] = []
    seen_ids: set[str] = set()

    for item in constraint_defs or []:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id", "")).strip()
        if cid and cid not in seen_ids:
            ordered_ids.append(cid)
            seen_ids.add(cid)

    for r in results:
        constraints = r.get("constraints") or {}
        if not isinstance(constraints, dict):
            continue
        for key, cinfo in constraints.items():
            if not isinstance(cinfo, dict):
                continue
            cid = str(cinfo.get("id", key)).strip()
            if cid and cid not in seen_ids:
                ordered_ids.append(cid)
                seen_ids.add(cid)

    mapping: dict[str, str] = {}
    used_tokens: set[str] = set()
    for cid in ordered_ids:
        base = _sanitize_constraint_token(cid)
        token = base
        suffix = 2
        while token in used_tokens:
            token = f"{base}_{suffix}"
            suffix += 1
        mapping[cid] = token
        used_tokens.add(token)
    return mapping


def _combine_constraint_margin(*, margin_pre: float, margin_post: float) -> float:
    pre = float(margin_pre)
    post = float(margin_post)
    pre_ok = bool(np.isfinite(pre))
    post_ok = bool(np.isfinite(post))
    if pre_ok and not post_ok:
        return pre
    if post_ok and not pre_ok:
        return post
    if pre_ok and post_ok:
        return float(min(pre, post))
    return float("inf")


def _calc_success_stats(results: list[dict]) -> tuple[int, int, float]:
    total = int(len(results))
    success_count = int(
        sum(1 for r in results if bool((r or {}).get("success", False)))
    )
    ratio = float(success_count / max(total, 1))
    return success_count, total, ratio


def _raise_if_success_rate_below_floor(
    *,
    success_count: int,
    total_count: int,
    success_ratio: float,
    floor: float,
) -> None:
    if float(success_ratio) >= float(floor):
        return
    raise RuntimeError(
        "FAILED_SUCCESS_RATE_MIN: "
        f"success={int(success_count)}/{int(total_count)} "
        f"(ratio={float(success_ratio):.4f}) < floor={float(floor):.4f}. "
        "Likely CAE runtime/output instability "
        "(explicit success=False, invalid payload, non-numeric/non-finite objective)."
    )


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
    task_name: str = "DOE",
    constraint_defs: list[dict] | None = None,
    debug_level: str = "off",
) -> dict:
    debug_level = _normalize_debug_level(debug_level)
    keep_debug = debug_level == "full"
    var_names = [v["name"] for v in variables]
    constraint_column_map = _build_constraint_column_map(
        results=results,
        constraint_defs=constraint_defs,
    )
    rows_public = []
    rows_internal = []
    for r in results:
        row_public = {
            "id": r["id"],
            "objective": r["objective"],
            "feasible": r.get("feasible", True),
            "success": r["success"],
            "source": r.get("source", "basic"),
            "round": r.get("round"),
            "exec_scope": r.get("exec_scope", "basic"),
        }
        for name, v in zip(var_names, r["x"]):
            row_public[name] = v

        constraints = r.get("constraints") or {}
        if not isinstance(constraints, dict):
            constraints = {}
        constraint_by_id = {}
        for cid, cinfo in constraints.items():
            if not isinstance(cinfo, dict):
                continue
            resolved_id = str(cinfo.get("id", cid)).strip()
            if resolved_id:
                constraint_by_id[resolved_id] = cinfo

        for constraint_id, token in constraint_column_map.items():
            cinfo = constraint_by_id.get(constraint_id)
            row_public[f"constraint_{token}_value"] = (
                cinfo.get("value") if isinstance(cinfo, dict) else np.nan
            )
            ok = cinfo.get("ok") if isinstance(cinfo, dict) else None
            row_public[f"constraint_{token}_feasible"] = (
                bool(ok) if ok is not None else None
            )

        rows_public.append(row_public)

        row_internal = dict(row_public)
        row_internal["feasible_pre"] = r.get("feasible_pre", True)
        row_internal["feasible_post"] = r.get("feasible_post", True)
        row_internal["margin_pre"] = r.get("margin_pre", float("inf"))
        row_internal["margin_post"] = r.get("margin_post", float("inf"))
        row_internal["constraint_margin"] = r.get("constraint_margin", float("inf"))
        row_internal["source"] = r.get("source", "basic")
        row_internal["round"] = r.get("round")
        row_internal["exec_scope"] = r.get("exec_scope", "basic")
        row_internal["constraint_details_json"] = json.dumps(
            constraints,
            ensure_ascii=False,
        )
        rows_internal.append(row_internal)

    df_public = pd.DataFrame(rows_public)
    df_internal = pd.DataFrame(rows_internal)

    if run_context is None:
        raise RuntimeError("run_context is required for v3 task output.")

    inputs = {
        "user_config": os.path.relpath(
            run_context.user_config_snapshot_path,
            os.path.join(run_context.run_root, task_name),
        ),
        "system_config_snapshot": system_config_snapshot or {},
        "previous": {},
        "variables": variables,
        "constraint_defs": constraint_defs or [],
    }
    resolved_params = {
        "n_samples": len(df_public),
        "seed": seed,
        "objective_sense": objective_sense,
        "additional_doe": additional_doe,
        "constraint_scope_schema": "pre_post_only",
        "debug_level": debug_level,
        "constraint_column_schema": "constraint_<id>_{value|feasible}",
        "constraint_columns": [
            {
                "constraint_id": cid,
                "value_col": f"constraint_{token}_value",
                "feasible_col": f"constraint_{token}_feasible",
            }
            for cid, token in constraint_column_map.items()
        ],
        "workflow_info": workflow_info,
    }
    if resolved_params_extra:
        resolved_params.update(resolved_params_extra)
    if extra_metadata:
        extra_context = dict(extra_metadata)
        if extra_context:
            resolved_params["extra_context"] = extra_context

    results_summary = {
        "n_samples_total": len(df_public),
        "n_success": int(df_public["success"].sum()) if "success" in df_public.columns else 0,
        "n_feasible": int(df_public["feasible"].sum()) if "feasible" in df_public.columns else 0,
    }
    if results_extra:
        for key, value in results_extra.items():
            results_summary[key] = value
    public_artifacts = {}
    meta_artifacts = {}
    debug_artifacts = {}
    if keep_debug:
        debug_artifacts["results_internal_csv"] = os.path.join(
            "artifacts", "debug", "doe_results_internal.csv"
        )

    task_out = saver.save_task_v3(
        run_root=run_context.run_root,
        task=task_name,
        problem_name=problem_name,
        df=df_public,
        inputs=inputs,
        resolved_params=resolved_params,
        results=results_summary,
        public_artifacts=public_artifacts,
        meta_artifacts=meta_artifacts,
        debug_artifacts=debug_artifacts,
    )

    if keep_debug:
        internal_csv = os.path.join(task_out["debug_dir"], "doe_results_internal.csv")
        df_internal.to_csv(internal_csv, index=False)

    update_run_index(run_context, task_name, task_out["metadata"])

    print(f"\nDOE CSV saved to: {task_out['csv']}")
    print(f"DOE metadata saved to: {task_out['metadata']}")

    return {
        "df": df_public,
        "task_out": task_out,
    }


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
    task_name: str = "DOE",
) -> list[dict]:
    run_cfg["debug_level"] = _normalize_debug_level(run_cfg.get("debug_level", "off"))
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
        str(c.get("scope", "pre")).strip().lower() == "pre"
        for c in constraint_defs
    )
    filter_safety = float(run_cfg.get("plan_filter_safety", 1.2))
    filter_r_floor = float(run_cfg.get("plan_filter_r_floor", 0.02))
    success_rate_floor = float(run_cfg.get("success_rate_floor", 0.02))
    if not (0.0 <= success_rate_floor <= 1.0):
        raise ValueError("DOE success_rate_floor must be in [0, 1].")
    initial_corner_ratio = float(run_cfg.get("initial_corner_ratio", 0.05))
    if not (0.0 <= initial_corner_ratio <= 1.0):
        raise ValueError("DOE initial_corner_ratio must be in [0, 1].")

    workflow_info = {
        "DOE": algo_name,
        "MODELER": None,
        "EXPLORER": None,
        "OPT": None,
    }

    if not use_additional:
        n_samples = int(run_cfg["n_samples"])
        n_baseline_reserved = 1 if (force_baseline and n_samples > 0) else 0
        n_regular_samples = n_samples - n_baseline_reserved

        print("\nDOE 설정 요약")
        print(f"- 문제명           : {problem_spec['name']}")
        print(f"- 초기 DOE 알고리즘 : {algo_name}")
        print(f"- 차원 수           : {dim}")
        print(f"- 샘플 수           : {n_samples}")
        print(f"- Seed              : {seed}")

        rng = np.random.default_rng(seed)
        saver = ResultSaver(use_timestamp=bool(run_cfg.get("use_timestamp", False)))

        bounds = [(v["lb"], v["ub"]) for v in variables]
        baseline = np.array([v["baseline"] for v in variables], dtype=float)

        probe_multiplier = float(run_cfg.get("initial_probe_multiplier", 2.0))
        n_corner_target = int(round(float(n_regular_samples) * initial_corner_ratio))
        n_corner_target = min(max(n_corner_target, 0), max(n_regular_samples, 0))

        X_corner = np.empty((0, len(bounds)), dtype=float)
        corner_constraints: list[dict] = []
        corner_margins: list[float] = []
        filter_gen_total = 0
        filter_feas_total = 0

        if n_corner_target > 0:
            n_corner_pool = n_corner_target if not has_pre_constraints else max((2 * n_corner_target), (n_corner_target + 4))
            corner_pool = sample_boundary_corners_random(
                bounds,
                offset=np.zeros((len(bounds),), dtype=float),
                n_samples=n_corner_pool,
                rng=rng,
            )
            if corner_pool.size > 0:
                corner_pool = np.unique(np.asarray(corner_pool, dtype=float), axis=0)
                if has_pre_constraints:
                    remaining_idx = np.arange(corner_pool.shape[0], dtype=int)
                    corner_rows: list[np.ndarray] = []
                    for _attempt in range(2):
                        need = n_corner_target - len(corner_rows)
                        if need <= 0 or remaining_idx.size == 0:
                            break
                        n_take = min(need, remaining_idx.size)
                        pick_pos = rng.choice(remaining_idx.size, size=n_take, replace=False)
                        pick_idx = remaining_idx[pick_pos]
                        keep_mask = np.ones((remaining_idx.size,), dtype=bool)
                        keep_mask[pick_pos] = False
                        remaining_idx = remaining_idx[keep_mask]

                        X_try = corner_pool[pick_idx]
                        feas_mask, constraint_payloads, margins_pre = evaluate_constraints_batch(
                            X=X_try,
                            var_names=var_names,
                            constraint_defs=constraint_defs,
                            scope="pre",
                        )
                        n_feas_corner = int(np.sum(feas_mask))
                        filter_gen_total += int(X_try.shape[0])
                        filter_feas_total += n_feas_corner
                        if n_feas_corner <= 0:
                            continue
                        feas_idx = np.where(feas_mask)[0]
                        for j in feas_idx.tolist():
                            if len(corner_rows) >= n_corner_target:
                                break
                            corner_rows.append(X_try[j].reshape(1, -1))
                            corner_constraints.append(constraint_payloads[j])
                            corner_margins.append(float(margins_pre[j]))
                    if corner_rows:
                        X_corner = np.vstack(corner_rows).astype(float)
                else:
                    n_take = min(n_corner_target, corner_pool.shape[0])
                    if n_take > 0:
                        pick_idx = rng.choice(corner_pool.shape[0], size=n_take, replace=False)
                        X_corner = corner_pool[pick_idx].astype(float)
                        corner_constraints = [{} for _ in range(X_corner.shape[0])]
                        corner_margins = [float("inf")] * X_corner.shape[0]

        n_corner_ok = int(X_corner.shape[0])
        n_sampler_target = max(n_regular_samples - n_corner_ok, 0)

        if has_pre_constraints:
            if n_sampler_target > 0:
                n_probe = max(n_sampler_target, int(np.ceil(n_sampler_target * probe_multiplier)))
                n_probe = int(np.ceil(n_probe * filter_safety))
                X_probe = sampler(
                    n_samples=n_probe,
                    bounds=bounds,
                    rng=rng,
                    n_divisions=max(n_probe, 1),
                )
                feas_mask, constraint_payloads, margins_pre = evaluate_constraints_batch(
                    X=X_probe,
                    var_names=var_names,
                    constraint_defs=constraint_defs,
                    scope="pre",
                )
                n_feas = int(feas_mask.sum())
                filter_gen_total += int(X_probe.shape[0])
                filter_feas_total += int(n_feas)
                if n_feas < n_sampler_target:
                    raise RuntimeError(
                        f"FAILED_FILTER_MIN: initial feasible points {n_feas} < target {n_sampler_target}"
                    )
                feas_idx = np.where(feas_mask)[0]
                pick_idx = rng.choice(feas_idx, size=n_sampler_target, replace=False)
                X_regular = X_probe[pick_idx]
                regular_constraints = [constraint_payloads[i] for i in pick_idx]
                regular_margins = [float(margins_pre[i]) for i in pick_idx]
            else:
                n_probe = 0
                X_regular = np.empty((0, len(bounds)), dtype=float)
                regular_constraints = []
                regular_margins = []
            if filter_gen_total > 0:
                r_hat = clamp_ratio(
                    filter_feas_total / max(filter_gen_total, 1),
                    floor=filter_r_floor,
                )
            else:
                r_hat = 1.0
        else:
            r_hat = 1.0
            n_probe = n_sampler_target
            if n_sampler_target > 0:
                X_regular = sampler(
                    n_samples=n_sampler_target,
                    bounds=bounds,
                    rng=rng,
                    n_divisions=max(n_sampler_target, 1),
                )
            else:
                X_regular = np.empty((0, len(bounds)), dtype=float)
            regular_constraints = [{} for _ in range(X_regular.shape[0])]
            regular_margins = [float("inf")] * X_regular.shape[0]

        X_parts: list[np.ndarray] = []
        picked_constraints: list[dict] = []
        picked_margin_list: list[float] = []

        if n_baseline_reserved > 0:
            baseline_x = np.asarray(baseline, dtype=float).reshape(1, -1)
            if has_pre_constraints:
                baseline_constraints, _, baseline_margin = evaluate_constraints_point(
                    x=np.asarray(baseline, dtype=float),
                    var_names=var_names,
                    constraint_defs=constraint_defs,
                    scope="pre",
                )
            else:
                baseline_constraints = {}
                baseline_margin = float("inf")
            X_parts.append(baseline_x)
            picked_constraints.append(baseline_constraints)
            picked_margin_list.append(float(baseline_margin))

        if X_corner.shape[0] > 0:
            X_parts.append(np.asarray(X_corner, dtype=float))
            picked_constraints.extend(corner_constraints)
            picked_margin_list.extend(float(v) for v in corner_margins)

        if X_regular.shape[0] > 0:
            X_parts.append(np.asarray(X_regular, dtype=float))
            picked_constraints.extend(regular_constraints)
            picked_margin_list.extend(float(v) for v in regular_margins)

        X = np.vstack(X_parts) if X_parts else np.empty((0, len(bounds)), dtype=float)
        picked_margins = np.asarray(picked_margin_list, dtype=float)

        print("\nDOE 샘플 생성 완료")
        print("\nCAE 평가 시작...\n")
        results = []

        for i, x in enumerate(X):
            y = evaluate_func(x)
            success, objective, outputs, invalid_reason, raw_obj_repr = sanitize_evaluate_output(y)
            if invalid_reason is not None:
                print(
                    "[DOE][INVALID_OBJECTIVE] "
                    f"idx={i} reason={invalid_reason} raw={raw_obj_repr} "
                    "-> success=False objective=inf"
                )

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
                    scope="post",
                    env_extra={**outputs, "objective": objective},
                    fail_fast_output_missing=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"FAILED_POST_CONSTRAINT_OUTPUT: sample={i}, error={exc}"
                ) from exc

            constraints = {**constraints_pre, **constraints_post}
            feasible = bool(success and feasible_pre and feasible_post)
            constraint_margin = _combine_constraint_margin(
                margin_pre=margin_pre,
                margin_post=margin_post,
            )

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
                "feasible": bool(feasible),
                "success": bool(success),
            })

            if not bool(feasible):
                print(
                    f"[{i+1:03d}/{n_samples}] "
                    f"objective = {objective}, "
                    f"feasible = {feasible}"
                )

        n_success, n_total, success_ratio = _calc_success_stats(results)
        _raise_if_success_rate_below_floor(
            success_count=n_success,
            total_count=n_total,
            success_ratio=success_ratio,
            floor=success_rate_floor,
        )

        system_snapshot = {
            "n_samples": run_cfg.get("n_samples"),
            "use_timestamp": run_cfg.get("use_timestamp"),
            "initial_corner_ratio": initial_corner_ratio,
            "initial_probe_multiplier": probe_multiplier,
            "plan_filter_safety": filter_safety,
            "plan_filter_r_floor": filter_r_floor,
            "constraint_r_hat": r_hat,
            "success_rate_floor": success_rate_floor,
            "success_count": n_success,
            "success_ratio": success_ratio,
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
            task_name=task_name,
            constraint_defs=constraint_defs,
            debug_level=str(run_cfg.get("debug_level", "off")),
        )

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
        initial_corner_ratio=cfg.get("initial_corner_ratio", initial_corner_ratio),
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
        phase2_min_usable_np_ratio=cfg.get("phase2_min_usable_np_ratio", 15.0),
        stop_span_ratio_threshold=cfg["stop_span_ratio_threshold"],
        stop_anchor_spread_streak=cfg["stop_anchor_spread_streak"],
        stop_min_usable_np_ratio=cfg.get("stop_min_usable_np_ratio", 20.0),
        probe_stage_enabled=cfg.get("probe_stage_enabled", True),
        probe_top_ratio=cfg.get("probe_top_ratio", 0.3),
        probe_max_points=cfg.get("probe_max_points", 5),
        probe_min_range_ratio=cfg.get("probe_min_range_ratio", 0.25),
        probe_std_scale=cfg.get("probe_std_scale", 2.0),
        probe_perturb_ratio=cfg.get("probe_perturb_ratio", 0.02),
        initial_probe_multiplier=cfg.get("initial_probe_multiplier", 2.0),
        plan_filter_safety=cfg.get("plan_filter_safety", 1.2),
        plan_filter_r_floor=cfg.get("plan_filter_r_floor", 0.02),
        success_rate_floor=cfg.get("success_rate_floor", success_rate_floor),
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
        local_refine_min_points=cfg.get("local_refine_min_points", 15),
        local_cluster_delta_ratio=cfg.get("local_cluster_delta_ratio", 0.02),
        local_singleton_box_ratio=cfg.get("local_singleton_box_ratio", 0.03),
        local_phase1_kappa=cfg.get("local_phase1_kappa", 0.75),
        local_phase2_kappa=cfg.get("local_phase2_kappa", 0.5),
        local_base_perturb_ratio=cfg.get("local_base_perturb_ratio", 0.02),
        local_constraint_retry_count=cfg.get("local_constraint_retry_count", 1),
        local_constraint_shrink_factor=cfg.get("local_constraint_shrink_factor", 0.5),
        local_constraint_min_factor=cfg.get("local_constraint_min_factor", 2.0),
        local_exec_pick_mode=cfg.get("local_exec_pick_mode", "random"),
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
        local_gp_seed=seed,
        local_gp_use_white_kernel=cfg.get("local_gp_use_white_kernel", False),
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
        task_name=task_name,
        resolved_params_extra={
            "failure_reason": diagnostics.get("failure_reason"),
            "constraint_rate_hat": diagnostics.get("constraint_rate_hat"),
            "post_feasible_rate_hat": diagnostics.get("post_feasible_rate_hat"),
            "post_lambda": diagnostics.get("post_lambda"),
            "post_model_active": diagnostics.get("post_model_active"),
            "success_rate_floor": diagnostics.get("success_rate_floor"),
            "success_count": diagnostics.get("success_count"),
            "success_total": diagnostics.get("success_total"),
            "success_ratio": diagnostics.get("success_ratio"),
        },
        extra_metadata={
            "failure_reason": diagnostics.get("failure_reason"),
            "constraint_rate_hat": diagnostics.get("constraint_rate_hat"),
            "post_feasible_rate_hat": diagnostics.get("post_feasible_rate_hat"),
            "post_lambda": diagnostics.get("post_lambda"),
            "post_model_active": diagnostics.get("post_model_active"),
            "success_rate_floor": diagnostics.get("success_rate_floor"),
            "success_count": diagnostics.get("success_count"),
            "success_total": diagnostics.get("success_total"),
            "success_ratio": diagnostics.get("success_ratio"),
        },
        constraint_defs=constraint_defs,
        debug_level=str(run_cfg.get("debug_level", "off")),
    )

    print("\n===================================")
    print(" DOE + ADDITIONAL DOE 실행 완료")
    print("===================================")
    return results
