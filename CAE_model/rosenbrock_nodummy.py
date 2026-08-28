import numpy as np


def get_problem_spec():
    real_variables = []
    for i in range(5):
        real_variables.append(
            {
                "name": f"x{i+1}",
                "default_lb": -2.048,
                "default_ub": 2.048,
            }
        )

    return {
        "name": "rosenbrock_nodummy",
        "variables": real_variables,
        "constraint_defs": [],
    }


def evaluate(x: np.ndarray, w_imp=1 * 1.0, w_dum=1 * 0.001) -> dict:
    _ = w_imp, w_dum
    if len(x) < 5:
        return {
            "objective": None,
            "success": False,
        }

    f = 0.0
    xb = x[:5]
    for i in range(len(xb) - 1):
        f += 100.0 * (xb[i + 1] - xb[i] ** 2) ** 2 + (1 - xb[i]) ** 2

    f_cae = f + 1
    return {
        "objective": float(f_cae),
        "success": True,
    }

