# DOE/executor/doe_orchestrator.py

import json
import os
import re
import hashlib
import inspect
import numpy as np
import pandas as pd

from DOE.config import DOESystemConfig, build_additional_cfg_from_system
from utils.boundary_sampling import sample_boundary_corners_random
from utils.feasibility import evaluate_feasibility
from utils.objective_sense import canonical_objective_for_result
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


DEFAULT_ADDITIONAL_CFG = build_additional_cfg_from_system(DOESystemConfig())


def _normalize_debug_level(value: str | None) -> str:
    level = str(value or "on").strip().lower()
    if level == "full":
        level = "on"
    if level not in {"off", "on"}:
        raise ValueError("DOE debug_level must be one of: off, on")
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


def _has_equality_constraints(constraint_defs: list[dict] | None) -> bool:
    return any(str(c.get("type", "")).strip() == "==" for c in constraint_defs or [])


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
    meta_artifacts_extra: dict | None = None,
    task_name: str = "DOE",
    constraint_defs: list[dict] | None = None,
    debug_level: str = "on",
) -> dict:
    debug_level = _normalize_debug_level(debug_level)
    keep_debug = debug_level == "on"
    var_names = [v["name"] for v in variables]
    constraint_column_map = _build_constraint_column_map(
        results=results,
        constraint_defs=constraint_defs,
    )
    rows_public = []
    rows_internal = []
    for r in results:
        objective_raw = float(r.get("objective_raw", r["objective"]))
        objective_internal = float(r["objective"])
        row_public = {
            "id": r["id"],
            "objective": objective_internal,
            "objective_raw": objective_raw,
            "feasible": r.get("feasible", True),
            "success": r["success"],
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

        constraint_value_cols = {}
        for constraint_id, token in constraint_column_map.items():
            cinfo = constraint_by_id.get(constraint_id)
            constraint_value_cols[f"constraint_{token}_value"] = (
                cinfo.get("value") if isinstance(cinfo, dict) else np.nan
            )
            ok = cinfo.get("ok") if isinstance(cinfo, dict) else None
            constraint_value_cols[f"constraint_{token}_feasible"] = (
                bool(ok) if ok is not None else None
            )

        rows_public.append(row_public)

        row_internal = dict(row_public)
        row_internal.update(constraint_value_cols)
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
        "seed": seed,
        "objective_sense": objective_sense,
        "raw_objective_sense": objective_sense,
        "canonical_objective_sense": "min",
        "objective_transform": "negate" if str(objective_sense).strip().lower() == "max" else "identity",
        "additional_doe": additional_doe,
        "workflow_info": workflow_info,
    }
    if resolved_params_extra:
        resolved_params.update(resolved_params_extra)

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
    if meta_artifacts_extra:
        meta_artifacts.update(meta_artifacts_extra)
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
    run_cfg["debug_level"] = _normalize_debug_level(run_cfg.get("debug_level", "on"))
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
        if _has_equality_constraints(constraint_defs):
            raise RuntimeError(
                "UNSUPPORTED_PLAIN_DOE_EQUALITY_CONSTRAINT: "
                "plain DOE supports pre inequality filtering only. "
                "Use additional DOE for equality constraints, or replace equality "
                "with an explicit relaxed inequality band."
            )

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
                max_attempts = min(
                    max(2, int(np.ceil(1.0 / max(filter_r_floor, 1e-6)))),
                    50,
                )
                pool_X: list[np.ndarray] = []
                pool_constraints: list[dict] = []
                pool_margins: list[float] = []
                n_probe = 0

                for _attempt in range(max_attempts):
                    need = n_sampler_target - len(pool_X)
                    if need <= 0:
                        break
                    n_probe_i = max(need, int(np.ceil(need * probe_multiplier)))
                    n_probe_i = int(np.ceil(n_probe_i * filter_safety))
                    n_probe += int(n_probe_i)
                    X_probe = sampler(
                        n_samples=n_probe_i,
                        bounds=bounds,
                        rng=rng,
                        n_divisions=max(n_probe_i, 1),
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
                    if n_feas <= 0:
                        continue
                    for j in np.where(feas_mask)[0].tolist():
                        if len(pool_X) >= n_sampler_target:
                            break
                        pool_X.append(np.asarray(X_probe[j], dtype=float))
                        pool_constraints.append(constraint_payloads[j])
                        pool_margins.append(float(margins_pre[j]))

                if len(pool_X) < n_sampler_target:
                    raise RuntimeError(
                        "FAILED_FILTER_MIN: plain DOE pre-constraint feasible "
                        f"points {len(pool_X)} < target {n_sampler_target} "
                        f"after {max_attempts} attempts and {filter_gen_total} generated candidates. "
                        "Use additional DOE, relax constraints, or increase n_samples/constraint tolerance."
                    )

                X_pool = np.vstack([x.reshape(1, -1) for x in pool_X]).astype(float)
                if X_pool.shape[0] > n_sampler_target:
                    pick_idx = rng.choice(X_pool.shape[0], size=n_sampler_target, replace=False)
                else:
                    pick_idx = np.arange(X_pool.shape[0], dtype=int)
                X_regular = X_pool[pick_idx]
                regular_constraints = [pool_constraints[i] for i in pick_idx]
                regular_margins = [float(pool_margins[i]) for i in pick_idx]
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
            success, objective_raw, outputs, invalid_reason, raw_obj_repr = sanitize_evaluate_output(y)
            objective = canonical_objective_for_result(
                raw_objective=float(objective_raw),
                objective_sense=objective_sense,
                success=bool(success),
            )
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
                    env_extra={**outputs, "objective": objective_raw, "objective_raw": objective_raw},
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
                "objective_raw": float(objective_raw),
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
            "initial_corner_ratio_base": run_cfg.get("initial_corner_ratio_base", initial_corner_ratio),
            "initial_corner_ratio_policy": run_cfg.get("initial_corner_ratio_policy", "fixed"),
            "initial_corner_np_ratio": run_cfg.get("initial_corner_np_ratio"),
            "initial_corner_p_dim": run_cfg.get("initial_corner_p_dim"),
            "initial_probe_multiplier": probe_multiplier,
            "plan_filter_safety": filter_safety,
            "plan_filter_r_floor": filter_r_floor,
            "constraint_r_hat": r_hat,
            "success_rate_floor": success_rate_floor,
            "success_count": n_success,
            "success_ratio": success_ratio,
        }
        save_out = _save_doe_results(
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
            resolved_params_extra={
                "constraint_rate_hat": float(r_hat),
                "constraint_r_hat": float(r_hat),
            },
            task_name=task_name,
            constraint_defs=constraint_defs,
            debug_level=str(run_cfg.get("debug_level", "on")),
        )

        print("\n===================================")
        print(" DOE 실행 완료")
        print("===================================")
        return {"results": results, "df": save_out["df"]}

    # -------------------------------------------------
    # Additional DOE branch
    # -------------------------------------------------
    cfg = dict(DEFAULT_ADDITIONAL_CFG)
    cfg["initial_corner_ratio"] = float(run_cfg.get("initial_corner_ratio", cfg["initial_corner_ratio"]))
    cfg["initial_probe_multiplier"] = float(run_cfg.get("initial_probe_multiplier", cfg["initial_probe_multiplier"]))
    cfg["plan_filter_safety"] = float(run_cfg.get("plan_filter_safety", cfg["plan_filter_safety"]))
    cfg["plan_filter_r_floor"] = float(run_cfg.get("plan_filter_r_floor", cfg["plan_filter_r_floor"]))
    cfg["success_rate_floor"] = float(run_cfg.get("success_rate_floor", cfg["success_rate_floor"]))
    if run_cfg.get("initial_corner_ratio_base") is not None:
        cfg["initial_corner_ratio_base"] = float(run_cfg["initial_corner_ratio_base"])
    if run_cfg.get("initial_corner_ratio_policy") is not None:
        cfg["initial_corner_ratio_policy"] = str(run_cfg["initial_corner_ratio_policy"])
    if run_cfg.get("initial_corner_np_ratio") is not None:
        cfg["initial_corner_np_ratio"] = float(run_cfg["initial_corner_np_ratio"])
    if run_cfg.get("initial_corner_p_dim") is not None:
        cfg["initial_corner_p_dim"] = int(run_cfg["initial_corner_p_dim"])
    if additional_cfg:
        cfg.update(additional_cfg)
    cfg_hash = hashlib.sha1(
        json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()

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
        objective_sense="min",
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

    orchestrator_kwargs = {
        "bounds": bounds,
        "sampler": sampler,
        "evaluate_func": evaluate_func,
        "feasibility_func": evaluate_feasibility,
        "var_names": var_names,
        "constraint_defs": constraint_defs,
        "surrogate_factory": surrogate_factory,
        "gate1": gate1,
        "gate2": gate2,
        "gate_manager": gate_manager,
        "rng": rng,
        "total_budget": total_budget,
        "hpo_runner": hpo_runner,
        "force_baseline": force_baseline,
        "local_gp_seed": seed,
    }
    valid_kwargs = set(inspect.signature(AdditionalDOEOrchestrator.__init__).parameters.keys())
    valid_kwargs.discard("self")
    for key, value in cfg.items():
        if key in valid_kwargs and key not in orchestrator_kwargs:
            orchestrator_kwargs[key] = value

    missing_required = [
        key for key in ("phase1_global_ratio", "phase2_global_ratio")
        if key not in orchestrator_kwargs or orchestrator_kwargs[key] is None
    ]
    if missing_required:
        raise ValueError(
            "Missing required Additional DOE config keys: "
            + ", ".join(missing_required)
        )

    orchestrator = AdditionalDOEOrchestrator(**orchestrator_kwargs)

    results = orchestrator.run(
        baseline=baseline,
        problem_name=problem_spec["name"],
        base_seed=seed,
        objective_sense=objective_sense,
    )
    diagnostics = orchestrator.get_diagnostics()
    # --- gate_history 축약: stage당 69→15 keys ---
    _gate_history_keep = {
        "stage", "phase", "next_phase", "n_exec",
        "n_exec_global_selected", "n_exec_local_selected",
        "gate1_score", "gate1_score_ema", "gate1_pass",
        "gate2_score", "gate2_score_ema", "gate2_pass",
        "usable_n", "collapse_detected", "diversity_boost_active",
        "phase2_gate1_rescue_triggered",
    }
    _raw_gh = diagnostics.get("gate_history") or []
    _slim_gh = [
        {k: v for k, v in entry.items() if k in _gate_history_keep}
        for entry in _raw_gh
    ] if isinstance(_raw_gh, list) else _raw_gh

    # --- Operational metadata: downstream task + try_stats 집계용 ---
    diagnostic_meta = {
        "p_dim": diagnostics.get("p_dim"),
        "usable_n": diagnostics.get("usable_n"),
        "usable_n_over_p": diagnostics.get("usable_n_over_p"),
        "constraint_rate_hat": diagnostics.get("constraint_rate_hat"),
        "constraint_r_hat": diagnostics.get("constraint_rate_hat"),
        "has_pre_constraints": diagnostics.get("has_pre_constraints"),
        "has_post_constraints": diagnostics.get("has_post_constraints"),
        "budget_used": diagnostics.get("budget_used"),
        "budget_total": diagnostics.get("budget_total"),
        "budget_policy": diagnostics.get("budget_policy"),
        "terminated_by": diagnostics.get("terminated_by"),
        "success_ratio": diagnostics.get("success_ratio"),
        "doe_additional_triggered": diagnostics.get("doe_additional_triggered"),
        "doe_added_samples": diagnostics.get("doe_added_samples"),
        "doe_gate1_pass_rate": diagnostics.get("doe_gate1_pass_rate"),
        "doe_gate2_pass_rate": diagnostics.get("doe_gate2_pass_rate"),
        "phase2_entered": diagnostics.get("phase2_entered"),
        "post_lambda": diagnostics.get("post_lambda"),
        "post_model_active": diagnostics.get("post_model_active"),
        "additional_gp_x_scaling_enabled": diagnostics.get("additional_gp_x_scaling_enabled"),
        "additional_gp_x_scaling_mode": diagnostics.get("additional_gp_x_scaling_mode"),
        "additional_gp_x_scaling_auto_span_ratio_threshold": diagnostics.get(
            "additional_gp_x_scaling_auto_span_ratio_threshold"
        ),
        "additional_gp_x_scaling_span_floor": diagnostics.get("additional_gp_x_scaling_span_floor"),
    }

    # --- Analysis metadata: 패턴 분석 + 개선안 도출용 (artifacts/meta/analysis.json) ---
    _doe_analysis_meta = {
        # Identity/context (analysis-only 파싱 지원)
        "run_id": run_context.run_id if run_context is not None else None,
        "problem": problem_spec.get("name"),
        "task": task_name,
        "seed": seed,
        "objective_sense": objective_sense,
        "doe_config_hash": cfg_hash,
        # Operational mirror (analysis-only에서 조인/상태 해석 가능하도록 포함)
        "p_dim": diagnostics.get("p_dim"),
        "usable_n": diagnostics.get("usable_n"),
        "usable_n_over_p": diagnostics.get("usable_n_over_p"),
        "has_pre_constraints": diagnostics.get("has_pre_constraints"),
        "has_post_constraints": diagnostics.get("has_post_constraints"),
        "budget_used": diagnostics.get("budget_used"),
        "budget_total": diagnostics.get("budget_total"),
        "budget_policy": diagnostics.get("budget_policy"),
        "budget_exhausted": diagnostics.get("budget_exhausted"),
        "terminated_by": diagnostics.get("terminated_by"),
        "success_ratio": diagnostics.get("success_ratio"),
        "doe_additional_triggered": diagnostics.get("doe_additional_triggered"),
        "doe_added_samples": diagnostics.get("doe_added_samples"),
        "doe_gate1_pass_rate": diagnostics.get("doe_gate1_pass_rate"),
        "doe_gate2_pass_rate": diagnostics.get("doe_gate2_pass_rate"),
        "phase2_entered": diagnostics.get("phase2_entered"),
        "post_lambda": diagnostics.get("post_lambda"),
        "post_model_active": diagnostics.get("post_model_active"),
        "failure_reason": diagnostics.get("failure_reason"),
        "additional_gp_x_scaling_enabled": diagnostics.get("additional_gp_x_scaling_enabled"),
        "additional_gp_x_scaling_mode": diagnostics.get("additional_gp_x_scaling_mode"),
        "additional_gp_x_scaling_auto_span_ratio_threshold": diagnostics.get(
            "additional_gp_x_scaling_auto_span_ratio_threshold"
        ),
        "additional_gp_x_scaling_span_floor": diagnostics.get("additional_gp_x_scaling_span_floor"),
        # Gate scores
        "doe_gate1_score_mean": diagnostics.get("doe_gate1_score_mean"),
        "doe_gate2_score_mean": diagnostics.get("doe_gate2_score_mean"),
        "doe_gate1_score_last": diagnostics.get("doe_gate1_score_last"),
        "doe_gate2_score_last": diagnostics.get("doe_gate2_score_last"),
        "doe_gate1_score_ema_mean": diagnostics.get("doe_gate1_score_ema_mean"),
        "doe_gate2_score_ema_mean": diagnostics.get("doe_gate2_score_ema_mean"),
        "doe_gate1_score_ema_last": diagnostics.get("doe_gate1_score_ema_last"),
        "doe_gate2_score_ema_last": diagnostics.get("doe_gate2_score_ema_last"),
        "doe_gate_decision_source_last": diagnostics.get("doe_gate_decision_source_last"),
        "gate_smoothing_enabled": diagnostics.get("gate_smoothing_enabled"),
        "gate_ema_alpha": diagnostics.get("gate_ema_alpha"),
        "gate_ema_warmup_stages": diagnostics.get("gate_ema_warmup_stages"),
        "gate_smoothing_use_for_phase2": diagnostics.get("gate_smoothing_use_for_phase2"),
        "gate_smoothing_use_for_stop": diagnostics.get("gate_smoothing_use_for_stop"),
        # Phase transition
        "phase2_first_stage": diagnostics.get("phase2_first_stage"),
        "phase2_stage_count": diagnostics.get("phase2_stage_count"),
        "phase2_transition_count": diagnostics.get("phase2_transition_count"),
        "phase2_np_gate_last": diagnostics.get("phase2_np_gate_last"),
        "phase2_gate1_score_min": diagnostics.get("phase2_gate1_score_min"),
        "phase2_gate1_rescue_enabled": diagnostics.get("phase2_gate1_rescue_enabled"),
        "phase2_gate1_rescue_band": diagnostics.get("phase2_gate1_rescue_band"),
        "phase2_gate1_rescue_gate2_min": diagnostics.get("phase2_gate1_rescue_gate2_min"),
        "phase2_gate1_rescue_trigger_count": diagnostics.get("phase2_gate1_rescue_trigger_count"),
        "phase2_gate2_score_min": diagnostics.get("phase2_gate2_score_min"),
        "phase2_gate2_score_sticky_min": diagnostics.get("phase2_gate2_score_sticky_min"),
        "phase2_min_used_budget_ratio": diagnostics.get("phase2_min_used_budget_ratio"),
        "used_budget_ratio_final": diagnostics.get("used_budget_ratio_final"),
        "used_budget_ratio_phase2_first": diagnostics.get("used_budget_ratio_phase2_first"),
        "early_stop_min_used_budget_ratio": diagnostics.get("early_stop_min_used_budget_ratio"),
        "early_stop_min_usable_np_ratio": diagnostics.get("early_stop_min_usable_np_ratio"),
        # Stage summary
        "stage_count_total": diagnostics.get("stage_count_total"),
        "stage_gate_eval_count": diagnostics.get("stage_gate_eval_count"),
        "gate_eval_skipped_count": diagnostics.get("gate_eval_skipped_count"),
        "gate_stop_raw_count": diagnostics.get("gate_stop_raw_count"),
        "gate_stop_final_count": diagnostics.get("gate_stop_final_count"),
        "gate_stop_blocked_count": diagnostics.get("gate_stop_blocked_count"),
        # Collapse/diversity
        "doe_collapse_detected_count": diagnostics.get("doe_collapse_detected_count"),
        "doe_diversity_injection_applied_count": diagnostics.get("doe_diversity_injection_applied_count"),
        "doe_diversity_sample_count_total": diagnostics.get("doe_diversity_sample_count_total"),
        "doe_diversity_hit_rate_mean": diagnostics.get("doe_diversity_hit_rate_mean"),
        "collapse_span_ratio_threshold": diagnostics.get("collapse_span_ratio_threshold"),
        "collapse_anchor_streak_threshold": diagnostics.get("collapse_anchor_streak_threshold"),
        "collapse_min_stage": diagnostics.get("collapse_min_stage"),
        "diversity_injection_ratio": diagnostics.get("diversity_injection_ratio"),
        "diversity_injection_min_points": diagnostics.get("diversity_injection_min_points"),
        "diversity_injection_max_ratio": diagnostics.get("diversity_injection_max_ratio"),
        "diversity_boundary_floor_ratio": diagnostics.get("diversity_boundary_floor_ratio"),
        "local_span_ratio_mean_last": diagnostics.get("local_span_ratio_mean_last"),
        "anchor_spread_zero_streak_max": diagnostics.get("anchor_spread_zero_streak_max"),
        # Alloc/constraint
        "alloc_multiplier_last": diagnostics.get("alloc_multiplier_last"),
        "alloc_multiplier_mean": diagnostics.get("alloc_multiplier_mean"),
        "alloc_eff_prev_last": diagnostics.get("alloc_eff_prev_last"),
        "alloc_eff_ref_last": diagnostics.get("alloc_eff_ref_last"),
        "alloc_dynamic_applied_count": diagnostics.get("alloc_dynamic_applied_count"),
        "constraint_rate_hat": diagnostics.get("constraint_rate_hat"),
        "initial_corner_ratio": cfg.get("initial_corner_ratio"),
        "initial_corner_ratio_base": cfg.get("initial_corner_ratio_base"),
        "initial_corner_ratio_policy": cfg.get("initial_corner_ratio_policy"),
        "initial_corner_np_ratio": cfg.get("initial_corner_np_ratio"),
        "initial_corner_p_dim": cfg.get("initial_corner_p_dim"),
        # Gate history (축약)
        "gate_history": _slim_gh,
    }

    system_snapshot = {
        "n_samples": run_cfg.get("n_samples"),
        "use_timestamp": run_cfg.get("use_timestamp"),
        "additional_cfg": cfg,
    }
    # Analysis metadata를 별도 파일로 저장하고 metadata.json에서 참조한다.
    _analysis_path = os.path.join(
        run_context.run_root, task_name, "artifacts", "meta", "analysis.json"
    )
    os.makedirs(os.path.dirname(_analysis_path), exist_ok=True)
    with open(_analysis_path, "w", encoding="utf-8") as _af:
        json.dump(_doe_analysis_meta, _af, ensure_ascii=False, indent=2, default=str)

    save_out = _save_doe_results(
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
        resolved_params_extra=diagnostic_meta,
        results_extra={
            "doe_additional_triggered": diagnostics.get("doe_additional_triggered"),
            "doe_added_samples": diagnostics.get("doe_added_samples"),
            "budget_used": diagnostics.get("budget_used"),
            "budget_total": diagnostics.get("budget_total"),
            "terminated_by": diagnostics.get("terminated_by"),
            "usable_n": diagnostics.get("usable_n"),
            "p_dim": diagnostics.get("p_dim"),
            "phase2_entered": diagnostics.get("phase2_entered"),
        },
        constraint_defs=constraint_defs,
        debug_level=str(run_cfg.get("debug_level", "on")),
        meta_artifacts_extra={"analysis": os.path.join("artifacts", "meta", "analysis.json")},
    )

    print("\n===================================")
    print(" DOE + ADDITIONAL DOE 실행 완료")
    print("===================================")
    return {"results": results, "df": save_out["df"]}
