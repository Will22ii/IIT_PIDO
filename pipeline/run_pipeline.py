import os

from CAE_tool_interface.run_CAE import run_cae
from CAE_tool_interface.task_metadata import save_cae_task
from DOE.run_DOE import run_doe
from Explorer.executor.explorer_orchestrator import ExplorerOrchestrator
from Modeler.run_Modeler import run_modeler
from pipeline.config import PipelineConfig, PipelineTasks
from pipeline.run_context import create_run_context
from pipeline.run_context import get_task_metadata_path


def run_pipeline(*, config: PipelineConfig) -> dict:
    cae_out = run_cae(config=config.cae)
    design_bounds = {
        v["name"]: [v["lb"], v["ub"]] for v in cae_out["variables"]
    }
    user_snapshot = {
        "problem": config.cae.user.problem_name,
        "seed": config.cae.user.seed,
        "objective_sense": config.cae.user.objective_sense,
        "design_bounds": design_bounds,
    }
    if config.doe is not None:
        user_snapshot["total_budget"] = (
            config.doe.system.n_samples
        )

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_context = create_run_context(
        project_root=project_root,
        user_config_snapshot=user_snapshot,
    )
    save_cae_task(
        run_context=run_context,
        cae_out=cae_out,
        use_timestamp=bool(config.cae.system.use_timestamp),
    )

    if config.tasks.run_doe and config.doe is not None:
        config.doe.cae_output = cae_out
        config.doe.cae_user = config.cae.user
        config.doe.cae = config.cae
        run_doe(config=config.doe, run_context=run_context)

    if config.tasks.run_modeler and config.modeler is not None:
        config.modeler.cae_user = config.cae.user
        config.modeler.cae = config.cae
        run_modeler(config=config.modeler, run_context=run_context)

    if config.tasks.run_explorer and config.explorer is not None:
        config.explorer.cae = config.cae
        ExplorerOrchestrator(config.explorer, run_context=run_context).run()

    return {
        "run_context": run_context,
        "cae_metadata": get_task_metadata_path(run_context, "CAE"),
        "doe_metadata": get_task_metadata_path(run_context, "DOE"),
        "modeler_metadata": get_task_metadata_path(run_context, "Modeler"),
        "explorer_metadata": get_task_metadata_path(run_context, "Explorer"),
    }


if __name__ == "__main__":
    # Minimal example; adjust configs per stage as needed.
    from CAE_tool_interface.config import CAEConfig, CAEUserConfig, CAESystemConfig
    from DOE.config import DOEConfig, DOESystemConfig, DOEUserConfig
    from Explorer.config import ExplorerConfig, ExplorerSystemConfig, ExplorerUserConfig
    from Modeler.config import ModelerConfig, ModelerSystemConfig, ModelerUserConfig

    cae_cfg = CAEConfig(
        user=CAEUserConfig(problem_name="goldstein_price", seed=42, objective_sense="min"),
        system=CAESystemConfig(use_timestamp=True),
    )

    cfg = PipelineConfig(
        cae=cae_cfg,
        doe=DOEConfig(
            cae=cae_cfg,
            cae_user=None,
            user=DOEUserConfig(algo_name="lhs", use_additional=True),
            system=DOESystemConfig(n_samples=166),
        ),
        modeler=ModelerConfig(
            user=ModelerUserConfig(
                model_name="xgb",
                use_hpo=True,
                use_secondary_selection=True,
            ),
            system=ModelerSystemConfig(),
            cae=cae_cfg,
            doe_csv_path=None,
            doe_metadata_path=None,
        ),
        explorer=ExplorerConfig(
            user=ExplorerUserConfig(
                known_optimum= {"x1": 0.0, "x2": -1.0, "d1": 0.0}
                # rosenbrock :{"x1": 1.0, "x2": 1.0, "x3": 1.0, "x4": 1.0, "x5": 1.0} 500
                # cantilever_beam : {"H": 7.0, "h1": 0.1, "b1" : 9.48482, "b2" : 0.1} 50
                # goldstein_price : {"x1": 0.0, "x2": -1.0} 166
                # six_hump_camel : [
                #{"x1": 0.0898, "x2": -0.7126},
                #{"x1": -0.0898, "x2": 0.7126},
                #] 16
            ),
            system=ExplorerSystemConfig(),
            cae=cae_cfg,
            doe_metadata_path=None,
            modeler_metadata_path=None,
        ),
        tasks=PipelineTasks(run_doe=True, run_modeler=True, run_explorer=True),
    )

    run_pipeline(config=cfg)
