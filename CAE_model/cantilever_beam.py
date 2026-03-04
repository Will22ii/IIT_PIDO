import numpy as np


def get_problem_spec():
    """
    Cantilever Beam 문제 정의
    4개의 설계변수를 가지는 구조 최적화 문제
    """
    n_real = 4
    n_dummy = n_real + 1
    
    n_dim = n_dummy + n_real

    real_vars = [
        {"name": "H", "default_lb": 3.0, "default_ub": 7.0, "default_baseline": 5},
        {"name": "h1", "default_lb": 0.1, "default_ub": 1.0, "default_baseline": 0.5},
        {"name": "b1", "default_lb": 2.0, "default_ub": 12.0, "default_baseline": 7.0},
        {"name": "b2", "default_lb": 0.1, "default_ub": 2.0, "default_baseline": 1.05},
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
        "name": "cantilever_beam",
        "variables": real_vars + dummy_vars,
        # Benchmark-only constraint definitions (treated as if user provided them).
        # Expressions are kept self-contained (no derived symbols) for now.
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

def evaluate(x: np.ndarray, w_imp=92.77*1.0, w_dum=92.77*0.001) -> dict:
    """
    Cantilever Beam surrogate evaluation
    목적: weight 최소화
    제약: stress <= 1, deflection <= 1
    """

    H, h1, b1, b2, dum1, dum2, dum3, dum4, dum5 = x

    # 상수 (정규화된 값)
    L = 36.0     # length

    eps = 1e-9
    H = max(abs(H), eps)
    h1   = max(abs(h1), eps)
    b1  = max(abs(b1), eps)
    b2   = max(abs(b2), eps)

    # 목적 함수: 무게
    V = (2*h1*b1 + (H - 2*h1) * b2 ) * L 

    # objective: weight + penalties on dummy variables
    term_imp = w_imp * (dum1**2)
    term_dum = w_dum * (dum2**2 + dum3**2+ dum4**2 + dum5**2)
    
    f_cae = float(V + term_imp + term_dum)

    return {
        "objective": float(f_cae),
        "success": True,
    }
