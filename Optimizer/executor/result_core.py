from __future__ import annotations

from typing import Any

import pandas as pd

from Optimizer.algorithms.result import OptimizerAlgorithmResult


def build_optimizer_algorithm_result(
    *,
    history_df: pd.DataFrame,
    archive_df: pd.DataFrame,
    best_point: dict[str, float],
    best_objective: float,
    algorithm_id: str,
    engine: str,
    best_point_raw: dict[str, float] | None = None,
    best_objective_raw: float | None = None,
    post_penalty_active: bool = False,
    post_penalty_lambda: float = 0.0,
    post_score_mode: str = "none",
    feasibility_model_kind: str = "none",
    feasibility_status: str = "none",
    gp_train_cap: int = 0,
    initial_archive_count: int = 0,
    final_train_count: int | None = None,
    goal_summary: dict[str, object] | None = None,
    algorithm_summary_extra: dict[str, object] | None = None,
    focus2_summary: dict[str, object] | None = None,
    focus_pipeline_summary: dict[str, object] | None = None,
) -> OptimizerAlgorithmResult:
    """Build the shared Optimizer algorithm result contract.

    Algorithm-specific code should prepare its own history/archive/focus
    summaries, but the common result envelope stays centralized here.
    """
    summary: dict[str, object] = {
        "algorithm_id": str(algorithm_id),
        "engine": str(engine),
    }
    if goal_summary:
        summary.update(dict(goal_summary))
    if algorithm_summary_extra:
        summary.update(dict(algorithm_summary_extra))

    return OptimizerAlgorithmResult(
        history_df=history_df,
        archive_df=archive_df,
        best_point=dict(best_point),
        best_objective=float(best_objective),
        best_point_raw=dict(best_point if best_point_raw is None else best_point_raw),
        best_objective_raw=float(best_objective if best_objective_raw is None else best_objective_raw),
        post_penalty_active=bool(post_penalty_active),
        post_penalty_lambda=float(post_penalty_lambda),
        post_score_mode=str(post_score_mode),
        feasibility_model_kind=str(feasibility_model_kind),
        feasibility_status=str(feasibility_status),
        n_iterations=int(len(history_df)),
        gp_train_cap=int(gp_train_cap),
        initial_archive_count=int(initial_archive_count),
        final_train_count=int(len(archive_df) if final_train_count is None else final_train_count),
        algorithm_summary=summary,
        focus2_summary=dict(focus2_summary or {}),
        focus_pipeline_summary=dict(focus_pipeline_summary or {}),
    )
