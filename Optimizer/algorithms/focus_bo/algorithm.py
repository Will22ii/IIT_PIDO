from __future__ import annotations

from Optimizer.config import OptimizerConfig
from Optimizer.algorithms.focus_bo.bo_engine import run_bo_engine
from Optimizer.algorithms.result import OptimizerAlgorithmResult
from Optimizer.executor.input_workflow import ResolvedOptimizerInputs


class FocusBOAlgorithm:
    """Built-in default Optimizer algorithm.

    Focus0~3, focus budget orchestration, and the GP/BO execution logic are
    implementation details of this algorithm, not the global Optimizer task
    contract.
    """

    algorithm_id = "focus_bo"

    def run(
        self,
        *,
        config: OptimizerConfig,
        resolved: ResolvedOptimizerInputs,
    ) -> OptimizerAlgorithmResult:
        return run_bo_engine(
            problem_name=resolved.problem_name,
            variables=resolved.variables,
            doe_df=resolved.doe_df,
            selected_features=resolved.selected_features,
            selected_bounds=resolved.selected_bounds,
            evaluation_base_values=resolved.evaluation_base_values,
            bounds_source=resolved.bounds_source,
            objective_col=str(config.system.objective_col),
            objective_sense=str(resolved.objective_sense),
            n_samples=int(config.user.n_samples),
            goal=getattr(config.user, "goal", None) if getattr(config.user, "goal", None) is not None else getattr(config.user, "goal_objective", None),
            system=config.system,
            seed=int(resolved.seed),
            constraint_defs=resolved.constraint_defs,
            post_feasibility_payload=resolved.post_feasibility_payload,
        )
