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
    algorithm_summary: dict[str, object] = field(default_factory=dict)
    focus2_summary: dict[str, object] = field(default_factory=dict)
    focus_pipeline_summary: dict[str, object] = field(default_factory=dict)
