from dataclasses import dataclass
import json
import os

import pandas as pd

from Modeler.config import ModelerConfig
from pipeline.run_context import (
    RunContext,
    create_run_context,
    get_task_metadata_path,
    update_run_index,
)
from utils.result_loader import ResultLoader


@dataclass
class ModelerInputResult:
    df: pd.DataFrame
    doe_meta: dict
    csv_path: str | None
    problem_name: str
    base_seed: int
    cae_problem_name: str
    cae_variables: list
    cae_constraint_defs: list
    cae_objective_sense: str
    cae_metadata_path: str


def _resolve_existing_cae_metadata_path(
    *,
    config: ModelerConfig,
    run_context: RunContext | None,
) -> str:
    if run_context is not None:
        path = get_task_metadata_path(run_context, "CAE")
        if path and os.path.exists(path):
            return path
        raise RuntimeError(
            "Modeler requires existing CAE metadata in run context. "
            "Run CAE task first and then execute Modeler."
        )

    raw = str(config.cae_metadata_path or "").strip()
    if raw:
        candidates = [raw]
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidates.append(os.path.join(project_root, raw))
        for p in candidates:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"CAE metadata not found: {raw}")

    raise RuntimeError(
        "Modeler requires existing CAE metadata. "
        "Provide ModelerConfig.cae_metadata_path or run via pipeline run_context."
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


def _extract_cae_fields(cae_meta: dict) -> tuple[str, list, list, str]:
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


def resolve_modeler_input(
    *,
    config: ModelerConfig,
    run_context: RunContext | None,
) -> ModelerInputResult:
    loader = ResultLoader()
    cae_meta_path = _resolve_existing_cae_metadata_path(config=config, run_context=run_context)
    cae_meta = _load_cae_metadata(cae_meta_path)
    (
        cae_problem_name,
        cae_variables,
        cae_constraint_defs,
        cae_objective_sense,
    ) = _extract_cae_fields(cae_meta)
    cae_seed = _extract_seed_from_cae_metadata(cae_meta=cae_meta, cae_meta_path=cae_meta_path)
    configured_problem = str(config.cae.user.problem_name).strip()
    if configured_problem and configured_problem != cae_problem_name:
        raise RuntimeError(
            "Problem mismatch between Modeler config and CAE metadata: "
            f"config={configured_problem}, cae_metadata={cae_problem_name}"
        )

    doe_meta = {}
    csv_path = None
    used_input_paths = bool(config.doe_metadata_path or config.doe_csv_path)

    if used_input_paths:
        if config.doe_metadata_path:
            print(f"- DOE metadata path provided: {config.doe_metadata_path}")
            doe_result = loader.load_task(
                task="DOE",
                metadata_path=config.doe_metadata_path,
                csv_path=config.doe_csv_path,
                allow_latest_fallback=False,
            )
            doe_meta = doe_result.metadata or {}
            csv_path = doe_result.csv_path
            df = doe_result.df
        elif config.doe_csv_path:
            print(f"- DOE CSV path provided: {config.doe_csv_path}")
            if not os.path.exists(config.doe_csv_path):
                raise FileNotFoundError(f"CSV not found: {config.doe_csv_path}")
            csv_path = config.doe_csv_path
            df = pd.read_csv(csv_path)
        else:
            raise RuntimeError("Modeler input CSV path not resolved.")
    elif run_context:
        doe_meta_path = get_task_metadata_path(run_context, "DOE")
        if not doe_meta_path:
            raise RuntimeError("DOE metadata not found in run context.")
        doe_result = loader.load_task(
            task="DOE",
            metadata_path=doe_meta_path,
            allow_latest_fallback=False,
        )
        doe_meta = doe_result.metadata or {}
        csv_path = doe_result.csv_path
        df = doe_result.df
    else:
        raise RuntimeError(
            "Modeler requires explicit DOE input. "
            "Provide doe_metadata_path/doe_csv_path or run via pipeline run_context."
        )

    problem_name = doe_meta.get("problem") or cae_problem_name or config.cae.user.problem_name
    if not problem_name:
        raise RuntimeError("DOE metadata에 problem 정보가 없습니다.")
    if doe_meta.get("problem") and str(doe_meta.get("problem")).strip() != str(cae_problem_name).strip():
        raise RuntimeError(
            "Problem mismatch between DOE metadata and CAE metadata: "
            f"doe={doe_meta.get('problem')}, cae={cae_problem_name}"
        )

    base_seed = int(cae_seed)

    if csv_path:
        print(f"- DOE CSV: {csv_path}")
    print(f"- Base seed: {base_seed}")
    print(f"- Problem name: {problem_name}")

    return ModelerInputResult(
        df=df,
        doe_meta=doe_meta,
        csv_path=csv_path,
        problem_name=str(problem_name),
        base_seed=int(base_seed),
        cae_problem_name=cae_problem_name,
        cae_variables=cae_variables if isinstance(cae_variables, list) else [],
        cae_constraint_defs=cae_constraint_defs if isinstance(cae_constraint_defs, list) else [],
        cae_objective_sense=str(cae_objective_sense),
        cae_metadata_path=str(cae_meta_path),
    )


def ensure_modeler_run_context(
    *,
    run_context: RunContext | None,
    project_root: str,
    problem_name: str,
    base_seed: int,
    objective_sense: str,
    variables: list | None,
    cae_metadata_path: str,
) -> RunContext:
    if run_context is None:
        design_bounds = None
        if variables:
            design_bounds = {
                v["name"]: [v["lb"], v["ub"]]
                for v in variables
                if isinstance(v, dict) and {"name", "lb", "ub"}.issubset(v.keys())
            }
        user_snapshot = {
            "problem": problem_name,
            "seed": int(base_seed),
            "objective_sense": objective_sense,
            "task": "Modeler",
        }
        if design_bounds:
            user_snapshot["design_bounds"] = design_bounds
        run_context = create_run_context(
            project_root=project_root,
            user_config_snapshot=user_snapshot,
        )

    if get_task_metadata_path(run_context, "CAE") is None:
        path = str(cae_metadata_path or "").strip()
        if not path:
            raise RuntimeError(
                "Modeler requires existing CAE metadata. "
                "Set ModelerConfig.cae_metadata_path or run via pipeline run_context."
            )
        if not os.path.exists(path):
            raise FileNotFoundError(f"CAE metadata not found: {path}")
        update_run_index(run_context, "CAE", os.path.abspath(path))

    return run_context
