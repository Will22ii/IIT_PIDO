from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from typing import Any

from CAE_tool_interface.config import CAEConfig, CAESystemConfig, CAEUserConfig
from DOE.config import DOEConfig, DOESystemConfig, DOEUserConfig
from Explorer.config import ExplorerConfig, ExplorerSystemConfig, ExplorerUserConfig
from Explorer.strategy_presets import apply_explorer_strategy_preset
from Modeler.config import ModelerConfig, ModelerSystemConfig, ModelerUserConfig
from Optimizer.config import OptimizerConfig, OptimizerSystemConfig, OptimizerUserConfig
from pipeline.config import PipelineConfig, PipelineReusePolicy, PipelineTasks


TOP_LEVEL_KEYS = {
    "problem",
    "run",
    "tasks",
    "reuse",
    "inputs",
    "doe",
    "modeler",
    "explorer",
    "optimizer",
}


def load_json_object(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Config JSON must contain an object: {path}")
    return payload


def _as_dict(value: Any, *, name: str, required: bool = False) -> dict[str, Any]:
    if value is None:
        if required:
            raise RuntimeError(f"Missing required config section: {name}")
        return {}
    if not isinstance(value, dict):
        raise RuntimeError(f"Config section must be an object: {name}")
    return value


def _pop_known(section: dict[str, Any], key: str, default: Any = None) -> Any:
    return section.pop(key) if key in section else default


def _apply_dataclass_overrides(target: Any, overrides: dict[str, Any], *, section_name: str) -> None:
    if not overrides:
        return
    if not is_dataclass(target):
        raise RuntimeError(f"Target is not a dataclass: {section_name}")
    valid = {f.name for f in fields(target)}
    unknown = sorted(str(k) for k in overrides.keys() if str(k) not in valid)
    if unknown:
        raise RuntimeError(
            f"Unknown config key(s) in {section_name}: {unknown}. "
            f"Valid keys: {sorted(valid)}"
        )
    for key, value in overrides.items():
        setattr(target, key, value)


def _task_enabled(tasks: dict[str, Any], key: str, default: bool) -> bool:
    value = tasks.get(key, default)
    return bool(value)


def pipeline_config_from_dict(payload: dict[str, Any]) -> PipelineConfig:
    if not isinstance(payload, dict):
        raise RuntimeError("Pipeline config payload must be a dict.")

    unknown_top = sorted(str(k) for k in payload.keys() if str(k) not in TOP_LEVEL_KEYS)
    if unknown_top:
        raise RuntimeError(
            f"Unknown top-level config key(s): {unknown_top}. "
            f"Valid keys: {sorted(TOP_LEVEL_KEYS)}"
        )

    problem = _as_dict(payload.get("problem"), name="problem", required=True).copy()
    run = _as_dict(payload.get("run"), name="run").copy()
    tasks_section = _as_dict(payload.get("tasks"), name="tasks").copy()
    reuse_section = _as_dict(payload.get("reuse", run.get("reuse")), name="reuse").copy()
    inputs = _as_dict(payload.get("inputs"), name="inputs").copy()

    problem_name = _pop_known(problem, "name", None)
    if problem_name is None:
        problem_name = _pop_known(problem, "problem_name", None)
    if not str(problem_name or "").strip():
        raise RuntimeError("Config problem.name is required.")
    seed = int(_pop_known(problem, "seed", 42))
    objective_sense = str(_pop_known(problem, "objective_sense", "min"))
    variables = _pop_known(problem, "variables", None)
    known_optimum = _pop_known(problem, "known_optimum", None)
    if problem:
        raise RuntimeError(f"Unknown config key(s) in problem: {sorted(problem.keys())}")

    debug_level = str(_pop_known(run, "debug_level", "on"))
    use_timestamp = bool(_pop_known(run, "use_timestamp", False))
    run_root = _pop_known(run, "run_root", None)
    if run:
        raise RuntimeError(f"Unknown config key(s) in run: {sorted(run.keys())}")

    reuse_policy = PipelineReusePolicy()
    _apply_dataclass_overrides(reuse_policy, reuse_section, section_name="reuse")

    run_doe = _task_enabled(tasks_section, "doe", True)
    run_modeler = _task_enabled(tasks_section, "modeler", True)
    run_explorer = _task_enabled(tasks_section, "explorer", True)
    run_optimizer = _task_enabled(tasks_section, "optimizer", False)
    valid_task_keys = {"doe", "modeler", "explorer", "optimizer"}
    unknown_task = sorted(str(k) for k in tasks_section.keys() if str(k) not in valid_task_keys)
    if unknown_task:
        raise RuntimeError(f"Unknown config key(s) in tasks: {unknown_task}")

    cae_cfg = CAEConfig(
        user=CAEUserConfig(
            problem_name=str(problem_name),
            seed=seed,
            objective_sense=objective_sense,
            variables=variables,
        ),
        system=CAESystemConfig(
            use_timestamp=use_timestamp,
            allow_latest_fallback=False,
        ),
    )

    doe_cfg = _build_doe_config(
        payload=payload,
        cae_cfg=cae_cfg,
        debug_level=debug_level,
        enabled=run_doe,
    )
    modeler_cfg = _build_modeler_config(
        payload=payload,
        cae_cfg=cae_cfg,
        debug_level=debug_level,
        enabled=run_modeler,
        inputs=inputs,
    )
    explorer_cfg = _build_explorer_config(
        payload=payload,
        cae_cfg=cae_cfg,
        debug_level=debug_level,
        enabled=run_explorer,
        inputs=inputs,
        known_optimum=known_optimum,
    )
    optimizer_cfg = _build_optimizer_config(
        payload=payload,
        cae_cfg=cae_cfg,
        debug_level=debug_level,
        enabled=run_optimizer,
        inputs=inputs,
    )

    allowed_inputs = {
        "doe_csv_path",
        "selected_features_csv_path",
        "model_pkl_path",
        "fi_scores_path",
        "explorer_bounds_path",
    }
    unknown_inputs = sorted(str(k) for k in inputs.keys() if str(k) not in allowed_inputs)
    if unknown_inputs:
        raise RuntimeError(f"Unknown config key(s) in inputs: {unknown_inputs}")

    return PipelineConfig(
        cae=cae_cfg,
        doe=doe_cfg,
        modeler=modeler_cfg,
        explorer=explorer_cfg,
        optimizer=optimizer_cfg,
        tasks=PipelineTasks(
            run_doe=run_doe,
            run_modeler=run_modeler,
            run_explorer=run_explorer,
            run_optimizer=run_optimizer,
        ),
        reuse=reuse_policy,
        run_root=run_root,
    )


def pipeline_config_from_json(path: str) -> PipelineConfig:
    return pipeline_config_from_dict(load_json_object(path))


def _split_user_system(section: dict[str, Any], *, section_name: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    section = section.copy()
    user = _as_dict(section.pop("user", {}), name=f"{section_name}.user").copy()
    system = _as_dict(section.pop("system", {}), name=f"{section_name}.system").copy()
    return section, user, system


def _build_doe_config(
    *,
    payload: dict[str, Any],
    cae_cfg: CAEConfig,
    debug_level: str,
    enabled: bool,
) -> DOEConfig | None:
    section, user_overrides, system_overrides = _split_user_system(
        _as_dict(payload.get("doe"), name="doe"),
        section_name="doe",
    )
    if not enabled and not section and not user_overrides and not system_overrides:
        return None

    user = DOEUserConfig(
        algo_name=str(section.pop("algo_name", user_overrides.pop("algo_name", "lhs"))),
        use_additional=bool(
            section.pop("use_additional", user_overrides.pop("use_additional", False))
        ),
    )
    _apply_dataclass_overrides(user, user_overrides, section_name="doe.user")

    system = DOESystemConfig(
        n_samples=int(section.pop("n_samples", system_overrides.pop("n_samples", 120))),
        debug_level=debug_level,
    )
    if "debug_level" in system_overrides:
        system_overrides.pop("debug_level")
    _apply_dataclass_overrides(system, system_overrides, section_name="doe.system")

    if section:
        raise RuntimeError(f"Unknown config key(s) in doe: {sorted(section.keys())}")

    return DOEConfig(cae=cae_cfg, cae_user=None, user=user, system=system)


def _build_modeler_config(
    *,
    payload: dict[str, Any],
    cae_cfg: CAEConfig,
    debug_level: str,
    enabled: bool,
    inputs: dict[str, Any],
) -> ModelerConfig | None:
    section, user_overrides, system_overrides = _split_user_system(
        _as_dict(payload.get("modeler"), name="modeler"),
        section_name="modeler",
    )
    if not enabled and not section and not user_overrides and not system_overrides:
        return None

    user = ModelerUserConfig(
        model_name=str(section.pop("model_name", user_overrides.pop("model_name", "xgb"))),
        use_hpo=bool(section.pop("use_hpo", user_overrides.pop("use_hpo", False))),
        target_col=str(section.pop("target_col", user_overrides.pop("target_col", "objective"))),
        use_secondary_selection=bool(
            section.pop(
                "use_secondary_selection",
                user_overrides.pop("use_secondary_selection", False),
            )
        ),
    )
    _apply_dataclass_overrides(user, user_overrides, section_name="modeler.user")

    system = ModelerSystemConfig(
        use_primary_selection=bool(
            section.pop(
                "use_primary_selection",
                system_overrides.pop("use_primary_selection", True),
            )
        ),
        debug_level=debug_level,
    )
    if "debug_level" in system_overrides:
        system_overrides.pop("debug_level")
    _apply_dataclass_overrides(system, system_overrides, section_name="modeler.system")

    doe_csv_path = section.pop("doe_csv_path", inputs.get("doe_csv_path"))
    if section:
        raise RuntimeError(f"Unknown config key(s) in modeler: {sorted(section.keys())}")

    return ModelerConfig(
        user=user,
        system=system,
        cae=cae_cfg,
        doe_csv_path=doe_csv_path,
        doe_metadata_path=None,
    )


def _build_explorer_config(
    *,
    payload: dict[str, Any],
    cae_cfg: CAEConfig,
    debug_level: str,
    enabled: bool,
    inputs: dict[str, Any],
    known_optimum: Any,
) -> ExplorerConfig | None:
    section, user_overrides, system_overrides = _split_user_system(
        _as_dict(payload.get("explorer"), name="explorer"),
        section_name="explorer",
    )
    if not enabled and not section and not user_overrides and not system_overrides:
        return None

    user = ExplorerUserConfig(
        known_optimum=section.pop(
            "known_optimum",
            user_overrides.pop("known_optimum", known_optimum),
        )
    )
    _apply_dataclass_overrides(user, user_overrides, section_name="explorer.user")

    strategy_id = str(section.pop("strategy_id", system_overrides.pop("strategy_id", "S4_obj")))
    custom_strategy_params = system_overrides.pop("strategy_params", None)
    system = ExplorerSystemConfig(
        strategy_id=strategy_id,
        debug_level=debug_level,
        save_plot=bool(section.pop("save_plot", system_overrides.pop("save_plot", True))),
    )
    if "debug_level" in system_overrides:
        system_overrides.pop("debug_level")
    apply_explorer_strategy_preset(system, system.strategy_id)
    if isinstance(custom_strategy_params, dict):
        merged_params = dict(system.strategy_params or {})
        merged_params.update(custom_strategy_params)
        system.strategy_params = merged_params
    elif custom_strategy_params is not None:
        raise RuntimeError("explorer.system.strategy_params must be an object.")
    _apply_dataclass_overrides(system, system_overrides, section_name="explorer.system")

    doe_csv_path = section.pop("doe_csv_path", inputs.get("doe_csv_path"))
    selected_features_csv_path = section.pop(
        "selected_features_csv_path",
        inputs.get("selected_features_csv_path"),
    )
    model_pkl_path = section.pop("model_pkl_path", inputs.get("model_pkl_path"))
    fi_scores_path = section.pop("fi_scores_path", inputs.get("fi_scores_path"))
    if section:
        raise RuntimeError(f"Unknown config key(s) in explorer: {sorted(section.keys())}")

    return ExplorerConfig(
        user=user,
        system=system,
        cae=cae_cfg,
        doe_csv_path=doe_csv_path,
        selected_features_csv_path=selected_features_csv_path,
        model_pkl_path=model_pkl_path,
        fi_scores_path=fi_scores_path,
    )


def _build_optimizer_config(
    *,
    payload: dict[str, Any],
    cae_cfg: CAEConfig,
    debug_level: str,
    enabled: bool,
    inputs: dict[str, Any],
) -> OptimizerConfig | None:
    section, user_overrides, system_overrides = _split_user_system(
        _as_dict(payload.get("optimizer"), name="optimizer"),
        section_name="optimizer",
    )
    if not enabled and not section and not user_overrides and not system_overrides:
        return None

    n_samples = int(section.pop("n_samples", user_overrides.pop("n_samples", 30)))
    doe_csv_path = section.pop("doe_csv_path", inputs.get("doe_csv_path"))
    explorer_bounds_path = section.pop(
        "explorer_bounds_path",
        inputs.get("explorer_bounds_path"),
    )
    user = OptimizerUserConfig(
        n_samples=n_samples,
        doe_csv_path=doe_csv_path,
        explorer_bounds_path=explorer_bounds_path,
    )
    _apply_dataclass_overrides(user, user_overrides, section_name="optimizer.user")

    system = OptimizerSystemConfig(debug_level=debug_level)
    if "debug_level" in system_overrides:
        system_overrides.pop("debug_level")
    _apply_dataclass_overrides(system, system_overrides, section_name="optimizer.system")

    if section:
        raise RuntimeError(f"Unknown config key(s) in optimizer: {sorted(section.keys())}")

    return OptimizerConfig(
        user=user,
        system=system,
        cae=cae_cfg,
        doe_csv_path=doe_csv_path,
        explorer_bounds_path=explorer_bounds_path,
    )
