from __future__ import annotations

import math
from typing import Any


def build_evaluation_history_row(
    *,
    iteration: int,
    segment: str,
    opt_focus_level: int,
    source_mode: str,
    x_values: dict[str, float],
    objective_raw: float,
    objective_effective: float,
    previous_best_objective_raw: float,
    best_objective_raw: float,
    objective_sense: str,
    improved: bool,
    pre_feasible: bool,
    pre_margin: float,
    goal_state: dict[str, object],
    algorithm_id: str,
    feasible: bool | None = None,
    success: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Build the common per-evaluation Optimizer history row.

    FocusBO can append its focus-specific diagnostics on top of this base row;
    custom runtime algorithms can use it directly.
    """
    sense = str(objective_sense or "min").strip().lower()
    if sense == "max":
        raw_improvement = float(objective_raw) - float(previous_best_objective_raw)
    else:
        raw_improvement = float(previous_best_objective_raw) - float(objective_raw)
    margin = float(pre_margin)
    row: dict[str, object] = {
        "iter": int(iteration),
        "segment": str(segment),
        "opt_focus_level": int(opt_focus_level),
        "source_mode": str(source_mode),
        "success": bool(success),
        "feasible": bool(pre_feasible if feasible is None else feasible),
        "pre_feasible": bool(pre_feasible),
        "pre_margin": margin,
        "pre_violation": float(max(0.0, -margin)) if math.isfinite(margin) else 0.0,
        "pre_constraint_worst_id": "",
        "pre_constraint_worst_margin": margin,
        "objective": float(objective_raw),
        "objective_raw": float(objective_raw),
        "objective_effective": float(objective_effective),
        "raw_improved": bool(improved),
        "raw_improvement": float(raw_improvement),
        "prev_best_objective_raw": float(previous_best_objective_raw),
        "best_objective_raw": float(best_objective_raw),
        "algorithm_id": str(algorithm_id),
        **dict(goal_state),
        **{str(k): float(v) for k, v in x_values.items()},
    }
    if extra:
        row.update(dict(extra))
    return row


def update_goal_and_build_evaluation_row(
    *,
    goal_monitor: Any,
    iteration: int,
    best_objective: float,
    improved: bool,
    allow_early_stop: bool = True,
    segment: str,
    opt_focus_level: int,
    source_mode: str,
    x_values: dict[str, float],
    objective_raw: float,
    objective_effective: float,
    previous_best_objective_raw: float,
    objective_sense: str,
    algorithm_id: str,
    pre_feasible: bool,
    pre_margin: float,
    feasible: bool | None = None,
    success: bool = True,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Update the shared goal monitor and build the common history row."""
    goal_state = goal_monitor.update(
        iteration=int(iteration),
        best_objective=float(best_objective),
        improved=bool(improved),
        allow_early_stop=bool(allow_early_stop),
    )
    row = build_evaluation_history_row(
        iteration=int(iteration),
        segment=str(segment),
        opt_focus_level=int(opt_focus_level),
        source_mode=str(source_mode),
        x_values=x_values,
        objective_raw=float(objective_raw),
        objective_effective=float(objective_effective),
        previous_best_objective_raw=float(previous_best_objective_raw),
        best_objective_raw=float(best_objective),
        objective_sense=str(objective_sense),
        improved=bool(improved),
        pre_feasible=bool(pre_feasible),
        pre_margin=float(pre_margin),
        goal_state=goal_state,
        algorithm_id=str(algorithm_id),
        feasible=feasible,
        success=bool(success),
        extra=extra,
    )
    return row, goal_state
