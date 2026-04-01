import numpy as np


def get_problem_spec():
    real_variables = [
        {
            "name": "x1",
            "default_lb": -2.0,
            "default_ub": 2.0,
            "default_baseline": 0.0,
        },
        {
            "name": "x2",
            "default_lb": -2.0,
            "default_ub": 2.0,
            "default_baseline": 0.0,
        },
    ]

    return {
        "name": "goldstein_price_nodummy",
        "variables": real_variables,
        "constraint_defs": [],
    }


def evaluate(x: np.ndarray, w_imp=3 * 1.0, w_dum=3 * 0.001) -> dict:
    _ = w_imp, w_dum
    x1, x2 = x

    term1 = (
        1
        + (x1 + x2 + 1) ** 2
        * (
            19
            - 14 * x1
            + 3 * x1**2
            - 14 * x2
            + 6 * x1 * x2
            + 3 * x2**2
        )
    )

    term2 = (
        30
        + (2 * x1 - 3 * x2) ** 2
        * (
            18
            - 32 * x1
            + 12 * x1**2
            + 48 * x2
            - 36 * x1 * x2
            + 27 * x2**2
        )
    )

    f_gp = term1 * term2
    return {
        "objective": float(f_gp),
        "success": True,
    }

