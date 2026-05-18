from __future__ import annotations

from dataclasses import dataclass, field

from CAE_tool_interface.config import CAEConfig, CAESystemConfig, CAEUserConfig
from DOE.config import DOEConfig, DOESystemConfig, DOEUserConfig
from Explorer.config import ExplorerConfig, ExplorerSystemConfig, ExplorerUserConfig
from Modeler.config import ModelerConfig, ModelerSystemConfig, ModelerUserConfig
from Optimizer.config import OptimizerConfig, OptimizerSystemConfig, OptimizerUserConfig
from pipeline.config import PipelineConfig, PipelineReusePolicy, PipelineTasks
from pipeline.run_pipeline import run_pipeline


@dataclass
class AIONConfig:
    problem_name: str
    objective_sense: str = "min"
    seed: int = 42
    n_doe_samples: int = 100
    optimizer_n_samples: int = 30
    known_optimum: dict | list | None = None
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

    explorer_cfg = ExplorerConfig(
        user=ExplorerUserConfig(
            known_optimum=config.known_optimum,
        ),
        system=ExplorerSystemConfig(
            strategy_id="S4_dual",
            debug_level=str(config.debug_level),
            save_plot=True,
        ),
        cae=cae_cfg,
        doe_csv_path=None,
        doe_metadata_path=None,
        selected_features_csv_path=None,
        model_pkl_path=None,
        modeler_metadata_path=None,
    )

    optimizer_cfg = OptimizerConfig(
        user=OptimizerUserConfig(
            n_samples=int(config.optimizer_n_samples),
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


if __name__ == "__main__":
    raise SystemExit(
        "pipeline/run_AION.py exposes run_aion(config=...). "
        "CLI is not implemented yet; call it from backend/Python code."
    )
