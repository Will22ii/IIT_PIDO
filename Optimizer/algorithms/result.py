from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class OptimizerAlgorithmResult:
    """Common result contract returned by Optimizer algorithms."""

    history_df: pd.DataFrame
    archive_df: pd.DataFrame
    best_point: dict[str, float]
    best_objective: float
    best_point_raw: dict[str, float]
    best_objective_raw: float
    post_penalty_active: bool
    post_penalty_lambda: float
    post_score_mode: str
    feasibility_model_kind: str
    feasibility_status: str
    n_iterations: int
    gp_train_cap: int
    initial_archive_count: int
    final_train_count: int
    best_effort_point: dict[str, float] = field(default_factory=dict)
    best_effort_objective: float = float("nan")
    best_effort_pre_violation_score: float = float("nan")
    best_effort_score: float = float("nan")
    least_violation_point: dict[str, float] = field(default_factory=dict)
    least_violation_objective: float = float("nan")
    least_violation_pre_violation_score: float = float("nan")
    raw_best_infeasible_point: dict[str, float] = field(default_factory=dict)
    raw_best_infeasible_objective: float = float("nan")
    raw_best_infeasible_pre_violation_score: float = float("nan")
    best_effort_pareto_points: list[dict[str, object]] = field(default_factory=list)
    algorithm_summary: dict[str, object] = field(default_factory=dict)
    focus2_summary: dict[str, object] = field(default_factory=dict)
    focus_pipeline_summary: dict[str, object] = field(default_factory=dict)
