import json
import os

from CAE_tool_interface.run_CAE import run_cae
from CAE_tool_interface.task_metadata import save_cae_task
from DOE.run_DOE import run_doe
from Explorer.executor.explorer_orchestrator import ExplorerOrchestrator
from Modeler.run_Modeler import run_modeler
from pipeline.aion_system_config import apply_aion_system_config
from pipeline.config import PipelineConfig
from pipeline.config_io import pipeline_config_from_dict, pipeline_config_from_json
from pipeline.run_context import RunContext
from pipeline.run_context import create_run_context
from pipeline.run_context import get_task_metadata_path
from pipeline.run_context import load_run_context


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid JSON object payload: {path}")
    return payload


def _get_public_artifact_path(run_root: str, task_dir: str, filename: str) -> str | None:
    path = os.path.join(run_root, task_dir, "artifacts", "public", filename)
    return path if os.path.exists(path) else None


def _validate_existing_cae_context(*, config: PipelineConfig, cae_metadata_path: str) -> None:
    cae_meta = _read_json(cae_metadata_path)
    cae_problem = str(cae_meta.get("problem", "")).strip()
    cfg_problem = str(config.cae.user.problem_name or "").strip()
    if cfg_problem and cae_problem and cfg_problem != cae_problem:
        raise RuntimeError(
            "Problem mismatch between PipelineConfig.cae and existing CAE metadata: "
            f"config={cfg_problem}, cae_metadata={cae_problem}"
        )

    resolved = (
        cae_meta.get("resolved_params", {})
        if isinstance(cae_meta.get("resolved_params", {}), dict)
        else {}
    )
    cae_sense = str(resolved.get("objective_sense", "")).strip().lower()
    cfg_sense = str(config.cae.user.objective_sense or "").strip().lower()
    if cfg_sense in {"min", "max"} and cae_sense in {"min", "max"} and cfg_sense != cae_sense:
        raise RuntimeError(
            "Objective sense mismatch between PipelineConfig.cae and existing CAE metadata: "
            f"config={cfg_sense}, cae_metadata={cae_sense}"
        )

    configured_variables = config.cae.user.variables or []
    if configured_variables:
        inputs = cae_meta.get("inputs", {}) if isinstance(cae_meta.get("inputs", {}), dict) else {}
        meta_variables = inputs.get("variables", [])
        if not isinstance(meta_variables, list):
            meta_variables = []
        meta_by_name = {
            str(v.get("name")): v
            for v in meta_variables
            if isinstance(v, dict) and str(v.get("name", "")).strip()
        }
        for var in configured_variables:
            if not isinstance(var, dict):
                continue
            name = str(var.get("name", "")).strip()
            if not name:
                continue
            if name not in meta_by_name:
                raise RuntimeError(
                    "Configured CAE variable not found in existing CAE metadata: "
                    f"{name}"
                )
            meta_var = meta_by_name[name]
            for key in ("lb", "ub"):
                if key in var and key in meta_var and float(var[key]) != float(meta_var[key]):
                    raise RuntimeError(
                        "Configured CAE variable bound differs from existing CAE metadata: "
                        f"{name}.{key} config={var[key]}, cae_metadata={meta_var[key]}"
                    )


def _create_new_run_context(*, config: PipelineConfig, project_root: str) -> tuple[RunContext, dict]:
    cae_out = run_cae(config=config.cae)
    design_bounds = {
        v["name"]: [v["lb"], v["ub"]] for v in cae_out["variables"]
    }
    user_snapshot = {
        "problem": config.cae.user.problem_name,
        "seed": config.cae.user.seed,
        "objective_sense": config.cae.user.objective_sense,
        "design_bounds": design_bounds,
        "run_policy": "new_run",
    }
    if config.doe is not None:
        user_snapshot["total_budget"] = config.doe.system.n_samples

    run_context = create_run_context(
        project_root=project_root,
        user_config_snapshot=user_snapshot,
    )
    save_cae_task(
        run_context=run_context,
        cae_out=cae_out,
        use_timestamp=bool(config.cae.system.use_timestamp),
    )
    return run_context, cae_out


def _load_existing_run_context(*, config: PipelineConfig) -> tuple[RunContext, None]:
    run_context = load_run_context(run_root=str(config.run_root))
    cae_metadata_path = get_task_metadata_path(run_context, "CAE")
    if not cae_metadata_path or not os.path.exists(cae_metadata_path):
        raise RuntimeError(
            "Existing run_root does not contain CAE metadata. "
            "Create a new run or provide a run_root initialized by CAE."
        )
    _validate_existing_cae_context(config=config, cae_metadata_path=cae_metadata_path)
    return run_context, None


def _resolve_existing_doe_csv(run_root: str) -> str | None:
    return _get_public_artifact_path(run_root, "DOE", "doe_results.csv")


def _resolve_configured_doe_csv(config: PipelineConfig) -> str | None:
    candidates: list[str] = []
    if config.modeler is not None and config.modeler.doe_csv_path:
        candidates.append(str(config.modeler.doe_csv_path))
    if config.explorer is not None and config.explorer.doe_csv_path:
        candidates.append(str(config.explorer.doe_csv_path))
    if config.optimizer is not None:
        if config.optimizer.user.doe_csv_path:
            candidates.append(str(config.optimizer.user.doe_csv_path))
        if config.optimizer.doe_csv_path:
            candidates.append(str(config.optimizer.doe_csv_path))

    normalized: dict[str, str] = {}
    for path in candidates:
        key = os.path.normcase(os.path.abspath(path))
        normalized.setdefault(key, path)
    if len(normalized) > 1:
        raise RuntimeError(
            "Multiple explicit DOE/input CSV paths were provided. "
            "Use one shared input CSV or set task inputs intentionally in a later policy. "
            f"paths={list(normalized.values())}"
        )
    return next(iter(normalized.values()), None)


def _resolve_existing_modeler_artifacts(run_root: str) -> dict[str, str | None]:
    return {
        "model_path": _get_public_artifact_path(
            run_root,
            "Modeler",
            "modeler_selected_models.pkl",
        ),
        "selected_features_csv_path": _get_public_artifact_path(
            run_root,
            "Modeler",
            "selected_features.csv",
        ),
        "feas_model_path": _get_public_artifact_path(
            run_root,
            "Modeler",
            "modeler_feas_models.pkl",
        ),
    }


def run_pipeline(*, config: PipelineConfig) -> dict:
    apply_aion_system_config(config)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if config.run_root:
        run_context, cae_out = _load_existing_run_context(config=config)
    else:
        run_context, cae_out = _create_new_run_context(
            config=config,
            project_root=project_root,
        )

    doe_df = None
    explicit_doe_csv_path = _resolve_configured_doe_csv(config)
    existing_doe_csv_path = (
        _resolve_existing_doe_csv(run_context.run_root)
        if bool(config.reuse.use_existing_doe_csv)
        else None
    )
    doe_csv_path = explicit_doe_csv_path or existing_doe_csv_path
    doe_dataframe_for_downstream = None
    if config.tasks.run_doe and config.doe is not None:
        if cae_out is not None:
            config.doe.cae_output = cae_out
        config.doe.cae_user = config.cae.user
        config.doe.cae = config.cae
        doe_out = run_doe(config=config.doe, run_context=run_context)
        if isinstance(doe_out, dict):
            doe_df = doe_out.get("df")
            task_out = doe_out.get("task_out")
            if isinstance(task_out, dict):
                if not explicit_doe_csv_path:
                    doe_csv_path = task_out.get("csv")
                    doe_dataframe_for_downstream = doe_df

    modeler_out = None
    if config.tasks.run_modeler and config.modeler is not None:
        config.modeler.cae_user = config.cae.user
        config.modeler.cae = config.cae
        if doe_csv_path and not config.modeler.doe_csv_path:
            config.modeler.doe_csv_path = doe_csv_path
        modeler_out = run_modeler(
            config=config.modeler,
            run_context=run_context,
            doe_dataframe=doe_dataframe_for_downstream,
        )

    if config.explorer is not None:
        if doe_csv_path and not config.explorer.doe_csv_path:
            config.explorer.doe_csv_path = doe_csv_path
        if isinstance(modeler_out, dict):
            if modeler_out.get("model_path") and not config.explorer.model_pkl_path:
                config.explorer.model_pkl_path = modeler_out.get("model_path")
            if (
                modeler_out.get("selected_features_csv_path")
                and not config.explorer.selected_features_csv_path
            ):
                config.explorer.selected_features_csv_path = modeler_out.get("selected_features_csv_path")
        else:
            existing_modeler = (
                _resolve_existing_modeler_artifacts(run_context.run_root)
                if bool(config.reuse.use_existing_modeler_artifacts)
                else {}
            )
            if existing_modeler.get("model_path") and not config.explorer.model_pkl_path:
                config.explorer.model_pkl_path = existing_modeler.get("model_path")
            if (
                existing_modeler.get("selected_features_csv_path")
                and not config.explorer.selected_features_csv_path
            ):
                config.explorer.selected_features_csv_path = existing_modeler.get(
                    "selected_features_csv_path"
                )

    if config.tasks.run_explorer and config.explorer is not None:
        config.explorer.cae = config.cae
        ExplorerOrchestrator(config.explorer, run_context=run_context).run()

    if config.tasks.run_optimizer and config.optimizer is not None:
        # 지연 import: optimizer 의존성(scipy 등)이 없는 파이프라인 실행 경로를 보호
        from Optimizer.run_Optimizer import run_optimizer

        config.optimizer.cae = config.cae
        if doe_csv_path and not config.optimizer.doe_csv_path:
            config.optimizer.doe_csv_path = doe_csv_path
        if doe_csv_path and not config.optimizer.user.doe_csv_path:
            config.optimizer.user.doe_csv_path = doe_csv_path
        run_optimizer(
            config=config.optimizer,
            run_context=run_context,
            doe_dataframe=doe_dataframe_for_downstream,
        )

    return {
        "run_context": run_context,
        "cae_metadata": get_task_metadata_path(run_context, "CAE"),
        "doe_metadata": get_task_metadata_path(run_context, "DOE"),
        "modeler_metadata": get_task_metadata_path(run_context, "Modeler"),
        "explorer_metadata": get_task_metadata_path(run_context, "Explorer"),
        "optimizer_metadata": get_task_metadata_path(run_context, "OPT"),
        "run_root": run_context.run_root,
        "reused_run": bool(config.run_root),
    }


def run_pipeline_from_dict(payload: dict) -> dict:
    return run_pipeline(config=pipeline_config_from_dict(payload))


def run_pipeline_from_json(config_path: str) -> dict:
    return run_pipeline(config=pipeline_config_from_json(config_path))


if __name__ == "__main__":
    raise SystemExit(
        "pipeline/run_pipeline.py exposes run_pipeline(config=...). "
        "CLI is disabled by policy; call run_pipeline_from_dict/json from backend/Python code."
    )
