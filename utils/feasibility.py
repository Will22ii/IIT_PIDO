# utils/feasibility.py

def evaluate_feasibility(constraints: dict) -> bool:
    """
    Evaluate feasibility based on constraint dict

    constraints example:
    {
        "sigma_max": {"type": "<=", "value": 4800, "limit": 5000},
        "delta_max": {"type": "<=", "value": 0.09, "limit": 0.10},
    }
    """

    for c in constraints.values():
        if isinstance(c, dict) and "ok" in c:
            if not bool(c.get("ok")):
                return False
            continue

        ctype = c.get("type", "<=")
        value = c.get("value")
        limit = c.get("limit")

        if value is None or limit is None:
            continue  # 판단 불가 → 무시 (또는 정책적으로 False 처리 가능)

        if ctype == "<=":
            if value > limit:
                return False
        elif ctype == "<":
            if value >= limit:
                return False
        elif ctype == ">=":
            if value < limit:
                return False
        elif ctype == ">":
            if value <= limit:
                return False
        elif ctype == "==":
            eps = float(c.get("eps", 0.0)) if isinstance(c, dict) else 0.0
            if abs(float(value) - float(limit)) > eps:
                return False
        else:
            raise ValueError(f"Unsupported constraint type: {ctype}")

    return True
