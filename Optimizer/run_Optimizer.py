import os
import sys
import inspect

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Optimizer.algorithms.registry import get_optimizer_algorithm
from Optimizer.algorithms.result import OptimizerAlgorithmResult
from Optimizer.config import OptimizerConfig
from Optimizer.executor.algorithm_runtime import OptimizerRuntime
from Optimizer.executor.input_workflow import resolve_optimizer_inputs
from Optimizer.executor.output_workflow import save_optimizer_outputs
from pipeline.run_context import RunContext, create_run_context, get_task_metadata_path, update_run_index


def _normalize_debug_level(value: str | None) -> str:
    level = str(value or "on").strip().lower()
    if level == "full":
        level = "on"
    if level not in {"off", "on"}:
        raise ValueError("Optimizer debug_level must be one of: off, on")
    return level


def run_optimizer(
    *,
    config: OptimizerConfig,
    run_context: RunContext | None = None,
    doe_dataframe: "pd.DataFrame | None" = None,
    bounds_override: dict[str, tuple[float, float]] | None = None,
) -> dict:
    print("===================================")
    print(" OPTIMIZER 실행 시작")
    print("===================================")

    config.system.debug_level = _normalize_debug_level(config.system.debug_level)
    if int(config.user.n_samples) < 0:
        raise ValueError("Optimizer user.n_samples must be >= 0.")

    resolved = resolve_optimizer_inputs(
        config=config,
        run_context=run_context,
        doe_dataframe=doe_dataframe,
        bounds_override=bounds_override,
    )

    if run_context is None:
        design_bounds = {
            v["name"]: [v["lb"], v["ub"]]
            for v in resolved.variables
            if isinstance(v, dict) and {"name", "lb", "ub"}.issubset(v.keys())
        }
        user_snapshot = {
            "problem": resolved.problem_name,
            "seed": int(resolved.seed),
            "objective_sense": resolved.objective_sense,
            "design_bounds": design_bounds,
            "total_budget": int(config.user.n_samples),
            "goal": config.user.goal,
            "goal_objective": config.user.goal_objective,
            "task": "OPT",
        }
        run_context = create_run_context(
            project_root=PROJECT_ROOT,
            user_config_snapshot=user_snapshot,
        )

    if get_task_metadata_path(run_context, "CAE") is None:
        update_run_index(run_context, "CAE", os.path.abspath(resolved.cae_metadata_path))
    if resolved.doe_metadata_path and get_task_metadata_path(run_context, "DOE") is None:
        update_run_index(run_context, "DOE", os.path.abspath(resolved.doe_metadata_path))
    if resolved.explorer_metadata_path and get_task_metadata_path(run_context, "Explorer") is None:
        update_run_index(run_context, "Explorer", os.path.abspath(resolved.explorer_metadata_path))
    if resolved.modeler_metadata_path and get_task_metadata_path(run_context, "Modeler") is None:
        update_run_index(run_context, "Modeler", os.path.abspath(resolved.modeler_metadata_path))

    algorithm_id = str(getattr(config.system, "algorithm_id", "focus_bo") or "focus_bo")
    algorithm = get_optimizer_algorithm(algorithm_id)
    run_signature = inspect.signature(algorithm.run)
    if "runtime" in run_signature.parameters:
        runtime = OptimizerRuntime(
            config=config,
            resolved=resolved,
            algorithm_id=algorithm_id,
        )
        bo_result = algorithm.run(config=config, resolved=resolved, runtime=runtime)
    else:
        bo_result = algorithm.run(config=config, resolved=resolved)
    if not isinstance(bo_result, OptimizerAlgorithmResult):
        raise TypeError(
            f"Optimizer algorithm '{algorithm_id}' returned invalid result type: "
            f"{type(bo_result).__name__}. Expected OptimizerAlgorithmResult."
        )

    task_out = save_optimizer_outputs(
        config=config,
        run_context=run_context,
        resolved=resolved,
        bo_result=bo_result,
    )

    print(f"- Iterations completed: {bo_result.n_iterations}")
    print(f"- Algorithm: {algorithm_id}")
    print(f"- Best objective: {bo_result.best_objective:.6f}")
    print(f"- Best objective (raw): {bo_result.best_objective_raw:.6f}")
    print(f"- Best point: {bo_result.best_point}")
    if bo_result.post_penalty_active:
        print(
            f"- Post-constraint penalty active: mode={bo_result.post_score_mode}, "
            f"lambda={bo_result.post_penalty_lambda:.4f}, model={bo_result.feasibility_model_kind}"
        )
    print(f"- OPT metadata: {task_out['metadata']}")
    print("===================================")
    print(" OPTIMIZER 실행 완료")
    print("===================================")

    return {
        "csv": task_out["csv"],
        "metadata": task_out["metadata"],
        "best_objective": float(bo_result.best_objective),
        "best_objective_raw": float(bo_result.best_objective_raw),
        "best_point": dict(bo_result.best_point),
        "best_point_raw": dict(bo_result.best_point_raw),
        "n_iterations": int(bo_result.n_iterations),
    }


if __name__ == "__main__":
    raise SystemExit(
        "Optimizer/run_Optimizer.py is an internal task runner. "
        "Use pipeline/run_pipeline.py as the execution entrypoint."
    )
