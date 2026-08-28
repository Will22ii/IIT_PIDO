import numpy as np


def get_problem_spec():
    real_vars = [
        {"name": "x1", "default_lb": -3.0, "default_ub": 3.0},
        {"name": "x2", "default_lb": -2.0, "default_ub": 2.0},
    ]

    return {
        "name": "six_hump_camel_nodummy",
        "variables": real_vars,
        "constraint_defs": [],
    }


def evaluate(x: np.ndarray, w_imp=1.0316 * 1.0, w_dum=1.0316 * 0.001) -> dict:
    _ = w_imp, w_dum
    x1, x2 = x

    f = (
        (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
        + x1 * x2
        + (-4 + 4 * x2**2) * x2**2
    )

    return {
        "objective": float(f),
        "success": True,
    }

