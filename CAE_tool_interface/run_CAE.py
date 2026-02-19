# run_CAE.py

from CAE_tool_interface.config import CAEConfig, CAEUserConfig
from CAE_tool_interface.executor.configurator import (
    configure_objective_sense_from_config,
    configure_variables_from_spec,
    select_cae_by_name,
)


def run_cae(*, config: CAEConfig | dict) -> dict:
    if isinstance(config, CAEConfig):
        user_cfg = config.user
        problem_name = user_cfg.problem_name
        objective_sense = user_cfg.objective_sense
        overrides = user_cfg.variables
    else:
        if "problem_name" not in config:
            raise ValueError("config.problem_name is required.")
        problem_name = config["problem_name"]
        objective_sense = config.get("objective_sense")
        overrides = config.get("variables")

    problem_spec_func, evaluate_func = select_cae_by_name(problem_name)
    problem_spec = problem_spec_func()

    objective_sense = configure_objective_sense_from_config(objective_sense)

    variables = configure_variables_from_spec(
        problem_spec,
        overrides=overrides,
    )

    return {
        "problem_spec": problem_spec,
        "evaluate_func": evaluate_func,
        "variables": variables,
        "objective_sense": objective_sense,
    }


def main():
    cfg = CAEConfig(user=CAEUserConfig(problem_name="goldstein_price"))
    run_cae(config=cfg)


if __name__ == "__main__":
    main()
