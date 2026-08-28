import numpy as np


def get_problem_spec():
    real_vars = [
        {"name": "H", "default_lb": 3.0, "default_ub": 7.0},
        {"name": "h1", "default_lb": 0.1, "default_ub": 1.0},
        {"name": "b1", "default_lb": 2.0, "default_ub": 12.0},
        {"name": "b2", "default_lb": 0.1, "default_ub": 2.0},
    ]

    return {
        "name": "cantilever_beam_nodummy",
        "variables": real_vars,
        "constraint_defs": [
            {
                "id": "sigma_max_ub",
                "name": "sigma_max",
                "scope": "pre",
                "type": "<=",
                "limit": 5000.0,
                "expr": "1000.0*36.0*H/(2*((1./12)*b2*(H - 2*h1)**3 + 2*((1./12)*b1*h1**3 + (1./4)*b1*h1*(H - h1)**2)))",
            },
            {
                "id": "delta_max_ub",
                "name": "delta_max",
                "scope": "pre",
                "type": "<=",
                "limit": 0.10,
                "expr": "1000.0*(36.0**3)/(3*(10.0e6)*((1./12)*b2*(H - 2*h1)**3 + 2*((1./12)*b1*h1**3 + (1./4)*b1*h1*(H - h1)**2)))",
            },
        ],
    }


def evaluate(x: np.ndarray, w_imp=92.77 * 1.0, w_dum=92.77 * 0.001) -> dict:
    _ = w_imp, w_dum
    H, h1, b1, b2 = x

    L = 36.0
    eps = 1e-9
    H = max(abs(H), eps)
    h1 = max(abs(h1), eps)
    b1 = max(abs(b1), eps)
    b2 = max(abs(b2), eps)

    V = (2 * h1 * b1 + (H - 2 * h1) * b2) * L
    return {
        "objective": float(V),
        "success": True,
    }

