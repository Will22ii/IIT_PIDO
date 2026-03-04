from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from pipeline.run_context import RunContext, update_run_index, get_task_metadata_path
from utils.result_saver import ResultSaver


def _normalize_variables(variables: list[dict] | None) -> list[dict]:
    out: list[dict] = []
    for item in variables or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        if "lb" not in item or "ub" not in item:
            continue
        out.append(
            {
                "name": name,
                "lb": float(item["lb"]),
                "ub": float(item["ub"]),
                "baseline": float(item.get("baseline", 0.0)),
            }
        )
    return out


def _build_variables_df(variables: list[dict]) -> pd.DataFrame:
    if not variables:
        return pd.DataFrame(columns=["name", "lb", "ub", "baseline"])
    rows = []
    for v in variables:
        rows.append(
            {
                "name": v["name"],
                "lb": float(v["lb"]),
                "ub": float(v["ub"]),
                "baseline": float(v.get("baseline", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def save_cae_task(
    *,
    run_context: RunContext,
    cae_out: dict[str, Any],
    use_timestamp: bool,
) -> str:
    problem_spec = cae_out.get("problem_spec", {}) if isinstance(cae_out, dict) else {}
    problem_name = str(problem_spec.get("name", "")).strip() or "unknown_problem"
    variables = _normalize_variables(cae_out.get("variables", []))
    objective_sense = str(cae_out.get("objective_sense", "min")).strip().lower()
    if objective_sense not in {"min", "max"}:
        objective_sense = "min"
    constraint_defs = problem_spec.get("constraint_defs", [])
    if not isinstance(constraint_defs, list):
        constraint_defs = []

    task_dir = os.path.join(run_context.run_root, "CAE")
    inputs = {
        "user_config": os.path.relpath(
            run_context.user_config_snapshot_path,
            task_dir,
        ),
        "system_config_snapshot": {"use_timestamp": bool(use_timestamp)},
        "previous": {},
        "variables": variables,
        "constraint_defs": constraint_defs,
    }
    resolved_params = {
        "objective_sense": objective_sense,
        "n_variables": int(len(variables)),
    }
    results = {
        "n_variables": int(len(variables)),
        "has_constraints": bool(len(constraint_defs) > 0),
    }

    saver = ResultSaver(use_timestamp=bool(use_timestamp))
    out = saver.save_task_v3(
        run_root=run_context.run_root,
        task="CAE",
        problem_name=problem_name,
        df=_build_variables_df(variables),
        inputs=inputs,
        resolved_params=resolved_params,
        results=results,
        public_artifacts={},
        meta_artifacts={},
        debug_artifacts={},
    )
    update_run_index(run_context, "CAE", out["metadata"])
    return out["metadata"]


def load_cae_metadata_from_context(*, run_context: RunContext) -> dict | None:
    meta_path = get_task_metadata_path(run_context, "CAE")
    if not meta_path or not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
        return None
    except Exception:
        return None

