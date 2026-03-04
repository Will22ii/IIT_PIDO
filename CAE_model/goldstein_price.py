import numpy as np


def get_problem_spec():

    n_real = 2
    n_dummy = n_real + 1
    
    n_dim = n_dummy + n_real

    real_variables = []
    dummy_variables = []

    for i in range(n_real):
        real_variables.append({
            "name": f"x{i+1}",
            "default_lb": -2.0,
            "default_ub": 2.0,
            "default_baseline": 0.0,
        })

    for j in range(n_dummy):
        dummy_variables.append({
            "name": f"d{j+1}",
            "default_lb": 0.0,
            "default_ub": 1.0,
            "default_baseline": 0.5,
        })

    return {
        "name": "goldstein_price",
        "variables": real_variables + dummy_variables,
        # No constraints for this benchmark (kept for uniform problem_spec schema).
        "constraint_defs": [],
    }


def evaluate(x: np.ndarray, w_imp=3*1.0, w_dum=3*0.001) -> dict:
    """
    Goldstein–Price function
    Domain: -2 <= x1, x2 <= 2
    Global minimum: f = 3 at (0, -1)
    """

    # 방어 코드: 차원 불일치
    # if len(x) != 2:
    #     return {
    #         "objective": None,
    #         "constraints": {},
    #         "success": False,
    #     }

    x1, x2, x3, x4, x5 = x

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
    
    term_imp = w_imp * (x3**2)
    term_dum = w_dum * (x4**2 + x5**2)

    f_cae = f_gp + term_imp + term_dum

    return {
        "objective": float(f_cae),
        "success": True,
    }


