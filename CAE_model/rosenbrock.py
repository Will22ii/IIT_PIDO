import numpy as np


def get_problem_spec():
    """
    Rosenbrock 문제 정의
    
    """
    n_real = 5
    n_dummy = n_real + 1
    
    n_dim = n_dummy + n_real

    real_variables = []
    dummy_variables = []

    for i in range(n_real):
        real_variables.append({
            "name": f"x{i+1}",
            "default_lb": -2.048,
            "default_ub": 2.048,
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
        "name": "rosenbrock",
        "variables":  real_variables + dummy_variables,
        # No constraints for this benchmark (kept for uniform problem_spec schema).
        "constraint_defs": [],
    }


def evaluate(x: np.ndarray, w_imp=1*1.0, w_dum=1*0.001) -> dict:
    """
    Rosenbrock function (n-dimensional)
    """
    n_real = 5
    n_dummy = 6

    if len(x) < 5:
        return {
            "objective": None,
            "success": False,
        }

    f = 0.0
    xb = x[:n_real]
    for i in range(len(xb) - 1):
        f += 100.0 * (xb[i + 1] - xb[i] ** 2) ** 2 + (1 - xb[i]) ** 2

    x6, x7, x8, x9, x10, x11 = x[n_real:]

    term_imp = w_imp * (x6**2)
    term_dum = w_dum * (x7**2 + x8**2 + x9**2 + x10**2 + x11**2)

    f_cae = f + term_imp + term_dum + 1

    return {
        "objective": float(f_cae),
        "success": True,
    }
