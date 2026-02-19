from CAE_model.registry import get_cae


def select_cae_by_name(name: str):
    if not name:
        raise ValueError("CAE name is required.")
    return get_cae(name)


def configure_objective_sense_from_config(value: str | None) -> str:
    if value is None:
        return "min"
    value = value.strip().lower()
    if value in ("min", "max"):
        return value
    raise ValueError("objective_sense must be 'min' or 'max'.")


def configure_variables_from_spec(
    problem_spec: dict,
    overrides: list[dict] | None = None,
) -> list:
    variables = []
    override_map = {v.get("name"): v for v in (overrides or [])}

    for var in problem_spec["variables"]:
        name = var["name"]
        default_lb = var["default_lb"]
        default_ub = var["default_ub"]
        default_baseline = var["default_baseline"]

        override = override_map.get(name, {})
        lb = override.get("lb", default_lb)
        ub = override.get("ub", default_ub)
        baseline = override.get("baseline", default_baseline)

        if lb >= ub:
            print(f"[CAE] invalid bounds for {name}: lb={lb}, ub={ub}")
            raise ValueError("CAE variable bounds must satisfy lb < ub.")

        variables.append({
            "name": name,
            "lb": lb,
            "ub": ub,
            "baseline": baseline,
            "default_lb": default_lb,
            "default_ub": default_ub,
            "default_baseline": default_baseline,
        })

    return variables
