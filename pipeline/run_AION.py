from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from CAE_tool_interface.config import CAEConfig, CAESystemConfig, CAEUserConfig
from DOE.config import DOEConfig, DOESystemConfig, DOEUserConfig
from Explorer.config import ExplorerConfig, ExplorerSystemConfig, ExplorerUserConfig
from Explorer.strategy_presets import apply_explorer_strategy_preset
from Modeler.config import ModelerConfig, ModelerSystemConfig, ModelerUserConfig
from Optimizer.config import OptimizerConfig, OptimizerSystemConfig, OptimizerUserConfig
from pipeline.config import PipelineConfig, PipelineReusePolicy, PipelineTasks
from pipeline.config_io import load_json_object
from pipeline.run_pipeline import run_pipeline


@dataclass
class AIONConfig:
    problem_name: str
    objective_sense: str = "min"
    seed: int = 42
    n_doe_samples: int = 100
    optimizer_n_samples: int = 30
    optimizer_goal: float | None = None
    variables: list[dict] | None = None
    run_root: str | None = None
    debug_level: str = "on"
    use_timestamp: bool = False
    reuse: PipelineReusePolicy = field(default_factory=PipelineReusePolicy)


def build_aion_pipeline_config(*, config: AIONConfig) -> PipelineConfig:
    cae_cfg = CAEConfig(
        user=CAEUserConfig(
            problem_name=str(config.problem_name),
            seed=int(config.seed),
            objective_sense=str(config.objective_sense),
            variables=config.variables,
        ),
        system=CAESystemConfig(
            use_timestamp=bool(config.use_timestamp),
            allow_latest_fallback=False,
        ),
    )

    doe_cfg = DOEConfig(
        cae=cae_cfg,
        cae_user=None,
        user=DOEUserConfig(
            algo_name="lhs",
            use_additional=True,
        ),
        system=DOESystemConfig(
            n_samples=int(config.n_doe_samples),
            debug_level=str(config.debug_level),
        ),
    )

    modeler_cfg = ModelerConfig(
        user=ModelerUserConfig(
            model_name="xgb",
            use_hpo=True,
            use_secondary_selection=False,
        ),
        system=ModelerSystemConfig(
            use_primary_selection=True,
            debug_level=str(config.debug_level),
        ),
        cae=cae_cfg,
        doe_csv_path=None,
        doe_metadata_path=None,
    )

    explorer_system = apply_explorer_strategy_preset(
        ExplorerSystemConfig(
            strategy_id="S4_dual",
            debug_level=str(config.debug_level),
            save_plot=True,
        ),
        "S4_dual",
    )

    explorer_cfg = ExplorerConfig(
        user=ExplorerUserConfig(),
        system=explorer_system,
        cae=cae_cfg,
        doe_csv_path=None,
        selected_features_csv_path=None,
        model_pkl_path=None,
    )

    optimizer_cfg = OptimizerConfig(
        user=OptimizerUserConfig(
            n_samples=int(config.optimizer_n_samples),
            goal=config.optimizer_goal,
        ),
        system=OptimizerSystemConfig(
            debug_level=str(config.debug_level),
        ),
        cae=cae_cfg,
        cae_metadata_path=None,
        doe_metadata_path=None,
        explorer_metadata_path=None,
        modeler_metadata_path=None,
    )

    return PipelineConfig(
        cae=cae_cfg,
        doe=doe_cfg,
        modeler=modeler_cfg,
        explorer=explorer_cfg,
        optimizer=optimizer_cfg,
        tasks=PipelineTasks(
            run_doe=True,
            run_modeler=True,
            run_explorer=True,
            run_optimizer=True,
        ),
        reuse=config.reuse,
        run_root=config.run_root,
    )


def run_aion(*, config: AIONConfig) -> dict:
    pipeline_config = build_aion_pipeline_config(config=config)
    return run_pipeline(config=pipeline_config)


def _as_dict(value: Any, *, name: str, required: bool = False) -> dict[str, Any]:
    if value is None:
        if required:
            raise RuntimeError(f"Missing required config section: {name}")
        return {}
    if not isinstance(value, dict):
        raise RuntimeError(f"Config section must be an object: {name}")
    return value


def _apply_reuse_overrides(reuse: PipelineReusePolicy, payload: dict[str, Any]) -> None:
    valid = {f.name for f in fields(reuse)}
    unknown = sorted(str(k) for k in payload.keys() if str(k) not in valid)
    if unknown:
        raise RuntimeError(
            f"Unknown config key(s) in reuse: {unknown}. Valid keys: {sorted(valid)}"
        )
    for key, value in payload.items():
        setattr(reuse, key, value)


def aion_config_from_dict(payload: dict[str, Any]) -> AIONConfig:
    if not isinstance(payload, dict):
        raise RuntimeError("AION config payload must be a dict.")
    valid_top = {"problem", "run", "reuse", "doe", "optimizer", "explorer"}
    unknown_top = sorted(str(k) for k in payload.keys() if str(k) not in valid_top)
    if unknown_top:
        raise RuntimeError(
            f"Unknown top-level AION config key(s): {unknown_top}. "
            f"Valid keys: {sorted(valid_top)}"
        )

    problem = _as_dict(payload.get("problem"), name="problem", required=True).copy()
    run = _as_dict(payload.get("run"), name="run").copy()
    reuse_payload = _as_dict(payload.get("reuse", run.get("reuse")), name="reuse").copy()
    doe = _as_dict(payload.get("doe"), name="doe").copy()
    optimizer = _as_dict(payload.get("optimizer"), name="optimizer").copy()
    explorer = _as_dict(payload.get("explorer"), name="explorer").copy()

    problem_name = problem.pop("name", None)
    if problem_name is None:
        problem_name = problem.pop("problem_name", None)
    if not str(problem_name or "").strip():
        raise RuntimeError("AION config problem.name is required.")

    reuse = PipelineReusePolicy()
    _apply_reuse_overrides(reuse, reuse_payload)

    n_doe_samples = doe.pop("n_samples", None)
    if n_doe_samples is None:
        n_doe_samples = doe.pop("n_doe_samples", 100)
    optimizer_n_samples = optimizer.pop("n_samples", None)
    if optimizer_n_samples is None:
        optimizer_n_samples = optimizer.pop("optimizer_n_samples", 30)
    optimizer_goal = optimizer.pop("goal", None)
    if optimizer_goal is None:
        optimizer_goal = optimizer.pop("goal_objective", None)

    cfg = AIONConfig(
        problem_name=str(problem_name),
        objective_sense=str(problem.pop("objective_sense", "min")),
        seed=int(problem.pop("seed", 42)),
        variables=problem.pop("variables", None),
        run_root=run.pop("run_root", None),
        debug_level=str(run.pop("debug_level", "on")),
        use_timestamp=bool(run.pop("use_timestamp", False)),
        n_doe_samples=int(n_doe_samples),
        optimizer_n_samples=int(optimizer_n_samples),
        optimizer_goal=float(optimizer_goal) if optimizer_goal is not None else None,
        reuse=reuse,
    )

    for section_name, section in (
        ("problem", problem),
        ("run", run),
        ("doe", doe),
        ("optimizer", optimizer),
        ("explorer", explorer),
    ):
        if section:
            raise RuntimeError(
                f"Unknown config key(s) in {section_name}: {sorted(section.keys())}"
            )

    return cfg


def aion_config_from_json(path: str) -> AIONConfig:
    return aion_config_from_dict(load_json_object(path))


def run_aion_from_dict(payload: dict[str, Any]) -> dict:
    return run_aion(config=aion_config_from_dict(payload))


def run_aion_from_json(config_path: str) -> dict:
    return run_aion(config=aion_config_from_json(config_path))


if __name__ == "__main__":
    raise SystemExit(
        "pipeline/run_AION.py exposes run_aion(config=...). "
        "CLI is disabled by policy; call run_aion_from_dict/json from backend/Python code."
    )
