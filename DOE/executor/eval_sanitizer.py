from __future__ import annotations

from typing import Any

import numpy as np


def sanitize_evaluate_output(
    out: Any,
) -> tuple[bool, float, dict, str | None, str | None]:
    """
    Normalize CAE evaluate_func output.

    Policy:
    - invalid payload/objective -> success=False, objective=inf
    - explicit success=False -> objective=inf
    """
    if not isinstance(out, dict):
        return False, float("inf"), {}, "invalid_payload", None

    outputs = out.get("outputs", {})
    if not isinstance(outputs, dict):
        outputs = {}

    success = bool(out.get("success", True))
    if not success:
        return False, float("inf"), outputs, None, None

    raw_objective = out.get("objective", float("inf"))
    raw_repr = repr(raw_objective)
    if len(raw_repr) > 120:
        raw_repr = raw_repr[:117] + "..."

    try:
        objective = float(raw_objective)
    except (TypeError, ValueError):
        return False, float("inf"), outputs, "objective_non_numeric", raw_repr

    if not np.isfinite(objective):
        return False, float("inf"), outputs, "objective_non_finite", raw_repr

    return True, objective, outputs, None, raw_repr
