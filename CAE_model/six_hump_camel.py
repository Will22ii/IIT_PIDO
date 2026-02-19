import numpy as np


def get_problem_spec():
    """
    Six-Hump Camel Back 문제 정의
    본질적으로 2차원 문제
    """

    n_real = 2
    n_dummy = n_real + 1
    n_dim = n_dummy + n_real

    real_vars = [
        {"name": "x1", "default_lb": -3.0, "default_ub": 3.0, "default_baseline": 0.0},
        {"name": "x2", "default_lb": -2.0, "default_ub": 2.0, "default_baseline": 0.0},
    ]

    dummy_vars = [
        {
            "name": f"d{i+1}",
            "default_lb": 0.0,
            "default_ub": 1.0,
            "default_baseline": 0.5,
        }
        for i in range(n_dummy)
    ]

    return {
        "name": "six_hump_camel",
        "variables": real_vars + dummy_vars,
        # No constraints for this benchmark (kept for uniform problem_spec schema).
        "constraint_defs": [],
    }



def evaluate(x: np.ndarray, w_imp=1*1.0, w_dum=1*0.001) -> dict:
    """
    Six-Hump Camel Back function
    Global minima:
        f = -1.0316 at (±0.0898, ∓0.7126)
    """

    # # 방어 코드
    # if len(x) != 2:
    #     return {
    #         "objective": None,
    #         "constraints": {},
    #         "success": False,
    #     }

    x1, x2, x3, x4, x5 = x

    f = (
        (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
        + x1 * x2
        + (-4 + 4 * x2**2) * x2**2
    )

    term_imp = w_imp * (x3**2)
    term_dum = w_dum * (x4**2 + x5**2)
    f_cae = f  + term_dum + term_imp

    return {
        "objective": float(f_cae),
        "success": True,
    }
