from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from Optimizer.algorithms.result import OptimizerAlgorithmResult
from Optimizer.config import OptimizerConfig
from Optimizer.executor.archive_core import update_best_point
from Optimizer.executor.evaluation_core import (
    CaeObjectiveEvaluator,
    normalize_objective_sense,
    objective_best_index,
    objective_initial_best,
)
from utils.objective_sense import canonical_objective_for_result, objective_from_minimize
from Optimizer.executor.goal_monitor import GoalMonitor
from Optimizer.executor.history_core import update_goal_and_build_evaluation_row
from Optimizer.executor.input_workflow import ResolvedOptimizerInputs
from Optimizer.executor.final_best_policy import final_pre_candidate_payload, select_final_pre_feasible_best
from Optimizer.executor.pre_equality_penalty import PreEqualityPenaltyPolicy, objective_penalty_scale
from Optimizer.executor.pre_constraint_policy import PreInequalityPolicy
from Optimizer.executor.result_core import build_optimizer_algorithm_result


@dataclass(frozen=True)
class RuntimeEvaluation:
    """Single optimizer evaluation result returned to replaceable algorithms."""

    iteration: int
    x: np.ndarray
    point: dict[str, float]
    objective: float
    objective_raw: float
    feasible: bool
    pre_feasible: bool
    pre_margin: float
    improved: bool
    best_objective: float
    best_point: dict[str, float]
    goal_hit: bool
    should_stop: bool

class OptimizerRuntime:
    """Common runtime contract for replaceable Optimizer algorithms.

    Algorithms are responsible for proposing candidate points. The runtime owns
    objective evaluation, archive/history updates, best tracking, and optional
    goal-based early stop so task-level behavior stays consistent across
    built-in and user-provided algorithms.
    """

    def __init__(
        self,
        *,
        config: OptimizerConfig,
        resolved: ResolvedOptimizerInputs,
        algorithm_id: str,
    ) -> None:
        self.config = config
        self.resolved = resolved
        self.algorithm_id = str(algorithm_id or getattr(config.system, "algorithm_id", "custom"))
        self.problem_name = str(resolved.problem_name)
        self.raw_objective_sense = normalize_objective_sense(str(resolved.objective_sense))
        self.objective_sense = "min"

        self.selected_features = list(resolved.selected_features)
        if not self.selected_features:
            raise RuntimeError("OptimizerRuntime requires non-empty selected_features.")
        self.selected_bounds = dict(resolved.selected_bounds)
        self.lb = np.asarray([self.selected_bounds[f][0] for f in self.selected_features], dtype=float)
        self.ub = np.asarray([self.selected_bounds[f][1] for f in self.selected_features], dtype=float)
        if np.any(~np.isfinite(self.lb)) or np.any(~np.isfinite(self.ub)) or np.any(self.ub <= self.lb):
            raise RuntimeError("OptimizerRuntime received invalid selected bounds.")
        self.rng = np.random.default_rng(int(resolved.seed))

        self._cae_evaluator = CaeObjectiveEvaluator(
            problem_name=self.problem_name,
            variables=resolved.variables,
            selected_features=self.selected_features,
            base_values=resolved.evaluation_base_values,
            error_context="optimizer runtime evaluation",
        )
        self._mapper = self._cae_evaluator.mapper
        self._var_names = list(self._mapper.var_names)
        self._x_base = self._mapper.x_base.copy()
        self._selected_idx = self._mapper.selected_idx.copy()
        self.pre_inequality_policy = PreInequalityPolicy(
            constraint_defs=resolved.constraint_defs,
            var_names=self._var_names,
            to_full_x=self._mapper.to_full,
        )
        self.pre_equality_policy = PreEqualityPenaltyPolicy(
            constraint_defs=resolved.constraint_defs,
            var_names=self._var_names,
            to_full_x=self._mapper.to_full,
            enabled=bool(getattr(config.system, "pre_eq_penalty_enabled", True)),
            penalty_lambda=float(getattr(config.system, "pre_eq_penalty_lambda", 10.0)),
            violation_cap=float(getattr(config.system, "pre_eq_penalty_violation_cap", 1e6)),
        )
        self._pre_eq_scale_floor = float(max(float(getattr(config.system, "pre_eq_penalty_objective_scale_floor", 1.0)), 1e-12))
        goal = getattr(config.user, "goal", None)
        if goal is None:
            goal = getattr(config.user, "goal_objective", None)
        self.goal_monitor = GoalMonitor.from_goal(
            goal=goal,
            system_config=config.system,
            objective_sense=self.raw_objective_sense,
        )

        self._history_rows: list[dict[str, Any]] = []
        self._archive_rows: list[dict[str, Any]] = []
        self._initial_archive_count = 0
        self._best_x: np.ndarray | None = None
        self._best_y = objective_initial_best(self.objective_sense)
        self._best_y_raw_at_effective = objective_initial_best(self.objective_sense)
        self._best_x_raw: np.ndarray | None = None
        self._best_y_raw = objective_initial_best(self.objective_sense)
        self._load_initial_archive()
        self.goal_monitor.initialize(
            best_objective=float(objective_from_minimize(self._best_y_raw, self.raw_objective_sense))
        )

    def _to_array(self, x: np.ndarray | list[float] | tuple[float, ...] | dict[str, float]) -> np.ndarray:
        return self._mapper.to_selected_array(x)

    def _full_x(self, x_selected: np.ndarray) -> np.ndarray:
        return self._mapper.to_full(x_selected)

    def _evaluate_objective(self, x_selected: np.ndarray) -> float:
        return self._cae_evaluator.evaluate_selected(x_selected)

    def _evaluate_pre_constraints(self, x_selected: np.ndarray) -> tuple[dict, bool, float]:
        return self._cae_evaluator.evaluate_pre_constraints(
            x_selected,
            constraint_defs=self.resolved.constraint_defs,
        )

    @property
    def pre_inequality_active(self) -> bool:
        return bool(self.pre_inequality_policy.active)

    def check_pre_inequality(
        self,
        x: np.ndarray | list[float] | tuple[float, ...] | dict[str, float],
    ) -> tuple[dict, bool, float]:
        x_arr = self._to_array(x)
        return self.pre_inequality_policy.evaluate(x_arr)

    def is_pre_inequality_feasible(
        self,
        x: np.ndarray | list[float] | tuple[float, ...] | dict[str, float],
    ) -> bool:
        _payload, feasible, _margin = self.check_pre_inequality(x)
        return bool(feasible)

    def filter_pre_inequality_candidates(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        mask, _payloads, margins = self.pre_inequality_policy.evaluate_batch(X_arr)
        return X_arr[mask], mask, margins

    @property
    def pre_equality_penalty_active(self) -> bool:
        return bool(self.pre_equality_policy.active)

    def _pre_eq_objective_scale(self, extra_values: list[float] | None = None) -> float:
        values = [
            float(row.get("objective_raw", row.get("objective", float("nan"))))
            for row in self._archive_rows
            if isinstance(row, dict)
        ]
        if extra_values:
            values.extend(float(v) for v in extra_values)
        return objective_penalty_scale(values, floor=self._pre_eq_scale_floor)

    def _load_initial_archive(self) -> None:
        df = self.resolved.doe_df
        objective_col = str(self.config.system.objective_col)
        if not isinstance(df, pd.DataFrame) or df.empty or objective_col not in df.columns:
            return
        filter_result = self.pre_inequality_policy.filter_warm_start_df(
            df,
            selected_features=self.selected_features,
            warn_prefix="[OptimizerRuntime][WarmStart]",
        )
        df = filter_result.df
        if df.empty:
            return
        rows: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            try:
                x = np.asarray([float(row[f]) for f in self.selected_features], dtype=float)
                y = float(row[objective_col])
                y_raw = (
                    float(row["objective_raw"])
                    if "objective_raw" in row and np.isfinite(float(row["objective_raw"]))
                    else float(objective_from_minimize(y, self.raw_objective_sense))
                )
                y_internal_raw = float(canonical_objective_for_result(
                    raw_objective=float(y_raw),
                    objective_sense=self.raw_objective_sense,
                    success=True,
                ))
            except Exception:
                continue
            if x.shape[0] != len(self.selected_features) or not np.isfinite(y):
                continue
            item = {f: float(v) for f, v in zip(self.selected_features, x)}
            item.update(
                {
                    "iter": 0,
                    "segment": "initial_archive",
                    "source_mode": "input_csv",
                    "objective": float(y_raw),
                    "objective_raw": float(y_raw),
                    "objective_effective": float(objective_from_minimize(y, self.raw_objective_sense)),
                    "objective_internal": float(y_internal_raw),
                    "objective_effective_internal": float(y),
                    "success": True,
                    "feasible": True,
                    "pre_feasible": True,
                    "pre_margin": float("inf"),
                }
            )
            rows.append(item)
        if rows and self.pre_equality_policy.active:
            scale = self._pre_eq_objective_scale([float(r["objective_internal"]) for r in rows])
            for item in rows:
                x_item = np.asarray([float(item[f]) for f in self.selected_features], dtype=float)
                pre_eq_result = self.pre_equality_policy.penalty(x_item, objective_scale=scale)
                y_eff_internal = float(item["objective_effective_internal"] + pre_eq_result.penalty)
                item["objective_effective_internal"] = float(y_eff_internal)
                item["objective_effective"] = float(objective_from_minimize(y_eff_internal, self.raw_objective_sense))
                item["pre_eq_penalty"] = float(pre_eq_result.penalty)
                item["pre_eq_feasible"] = bool(pre_eq_result.feasible)
                item["pre_eq_violation_norm"] = float(pre_eq_result.violation_norm)
                item["pre_eq_worst_id"] = str(pre_eq_result.worst_id)
        self._archive_rows.extend(rows)
        self._initial_archive_count = len(rows)
        if rows:
            y_eff_arr = np.asarray([float(r["objective_effective_internal"]) for r in rows], dtype=float)
            best_idx = objective_best_index(y_eff_arr, self.objective_sense)
            self._best_y = float(y_eff_arr[best_idx])
            self._best_y_raw_at_effective = float(rows[best_idx]["objective_internal"])
            self._best_x = np.asarray(
                [float(rows[best_idx][f]) for f in self.selected_features],
                dtype=float,
            )
            y_raw_arr = np.asarray([float(r["objective_internal"]) for r in rows], dtype=float)
            raw_best_idx = objective_best_index(y_raw_arr, self.objective_sense)
            self._best_y_raw = float(y_raw_arr[raw_best_idx])
            self._best_x_raw = np.asarray(
                [float(rows[raw_best_idx][f]) for f in self.selected_features],
                dtype=float,
            )

    def sample_uniform(self, n: int = 1) -> np.ndarray:
        return self.rng.uniform(self.lb, self.ub, size=(int(max(n, 1)), len(self.selected_features)))

    def point_dict(self, x: np.ndarray | list[float] | tuple[float, ...] | dict[str, float]) -> dict[str, float]:
        return self._mapper.point_dict(x)

    @property
    def feature_names(self) -> list[str]:
        return list(self.selected_features)

    @property
    def n_dim(self) -> int:
        return int(len(self.selected_features))

    @property
    def bounds(self) -> dict[str, tuple[float, float]]:
        return {name: (float(self.selected_bounds[name][0]), float(self.selected_bounds[name][1])) for name in self.selected_features}

    @property
    def algorithm_params(self) -> dict[str, object]:
        params = getattr(self.config.system, "algorithm_params", {})
        return dict(params) if isinstance(params, dict) else {}

    @property
    def max_evaluations(self) -> int:
        return int(max(int(self.config.user.n_samples), 0))

    @property
    def remaining_budget(self) -> int:
        return int(max(self.max_evaluations - self.iteration, 0))

    @property
    def iteration(self) -> int:
        return int(len(self._history_rows))

    @property
    def should_stop(self) -> bool:
        return bool(self.goal_monitor.should_stop or self.remaining_budget <= 0)

    @property
    def goal_state(self) -> dict[str, object]:
        return dict(self.goal_monitor.state())

    @property
    def best_objective(self) -> float:
        return float(self._best_y)

    @property
    def best_point(self) -> dict[str, float]:
        if self._best_x is None:
            return {}
        return self.point_dict(self._best_x)

    @property
    def archive_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._archive_rows)

    @property
    def history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._history_rows)

    def evaluate(
        self,
        x: np.ndarray | list[float] | tuple[float, ...] | dict[str, float],
        *,
        source_mode: str = "custom",
        segment: str = "custom",
        extra: dict[str, Any] | None = None,
    ) -> RuntimeEvaluation:
        if self.remaining_budget <= 0:
            raise RuntimeError("OptimizerRuntime evaluation budget exhausted.")
        if self.goal_monitor.should_stop:
            raise RuntimeError("OptimizerRuntime.evaluate called after early stop.")
        x_arr = self._to_array(x)
        if np.any(x_arr < self.lb) or np.any(x_arr > self.ub):
            raise RuntimeError("Candidate is outside selected bounds.")

        constraints, pre_feasible, pre_margin = self._evaluate_pre_constraints(x_arr)
        y_raw = self._evaluate_objective(x_arr)
        y_effective = canonical_objective_for_result(
            raw_objective=float(y_raw),
            objective_sense=self.raw_objective_sense,
            success=True,
        )
        pre_eq_scale = self._pre_eq_objective_scale([float(y_effective)])
        pre_eq_result = self.pre_equality_policy.penalty(x_arr, objective_scale=pre_eq_scale)
        y_effective = float(y_effective + float(pre_eq_result.penalty))
        y_raw_internal = float(canonical_objective_for_result(
            raw_objective=float(y_raw),
            objective_sense=self.raw_objective_sense,
            success=True,
        ))
        previous_best = float(self._best_y)
        previous_best_raw = float(self._best_y_raw)
        best_update = update_best_point(
            best_x=self._best_x if self._best_x is not None else x_arr,
            best_y=previous_best,
            x_next=x_arr,
            y_next=float(y_effective),
            objective_sense=self.objective_sense,
        )
        improved = bool(best_update.improved or self._best_x is None)
        if improved:
            self._best_y = float(best_update.best_y)
            self._best_x = best_update.best_x.copy()
            self._best_y_raw_at_effective = float(y_raw_internal)
        raw_best_update = update_best_point(
            best_x=self._best_x_raw if self._best_x_raw is not None else x_arr,
            best_y=previous_best_raw,
            x_next=x_arr,
            y_next=float(y_raw_internal),
            objective_sense=self.objective_sense,
        )
        raw_improved = bool(raw_best_update.improved or self._best_x_raw is None)
        if raw_improved:
            self._best_y_raw = float(raw_best_update.best_y)
            self._best_x_raw = raw_best_update.best_x.copy()

        iteration = int(len(self._history_rows) + 1)
        row, goal_state = update_goal_and_build_evaluation_row(
            goal_monitor=self.goal_monitor,
            iteration=iteration,
            best_objective=float(objective_from_minimize(self._best_y_raw, self.raw_objective_sense)),
            improved=bool(raw_improved),
            segment=str(segment),
            opt_focus_level=-1,
            source_mode=str(source_mode),
            x_values={f: float(v) for f, v in zip(self.selected_features, x_arr)},
            objective_raw=float(y_raw),
            objective_effective=float(objective_from_minimize(y_effective, self.raw_objective_sense)),
            previous_best_objective_raw=float(objective_from_minimize(previous_best_raw, self.raw_objective_sense)),
            objective_sense=self.raw_objective_sense,
            pre_feasible=bool(pre_feasible),
            pre_margin=float(pre_margin),
            algorithm_id=str(self.algorithm_id),
        )
        if constraints:
            worst_id = min(
                constraints,
                key=lambda cid: float(constraints[cid].get("margin", float("inf"))),
            )
            row["pre_constraint_worst_id"] = str(worst_id)
            row["pre_constraint_worst_margin"] = float(constraints[worst_id].get("margin", pre_margin))
        row.update(
            self.pre_equality_policy.debug_fields(
                x_arr,
                objective_scale=pre_eq_scale,
            )
        )
        if extra:
            row.update(dict(extra))
        row["objective_internal"] = float(y_raw_internal)
        row["objective_effective_internal"] = float(y_effective)
        self._history_rows.append(row)
        self._archive_rows.append(dict(row))

        return RuntimeEvaluation(
            iteration=iteration,
            x=x_arr.copy(),
            point=self.point_dict(x_arr),
            objective=float(y_effective),
            objective_raw=float(y_raw),
            feasible=bool(pre_feasible),
            pre_feasible=bool(pre_feasible),
            pre_margin=float(pre_margin),
            improved=bool(improved),
            best_objective=float(objective_from_minimize(self._best_y, self.raw_objective_sense)),
            best_point=self.best_point,
            goal_hit=bool(goal_state.get("goal_hit", False)),
            should_stop=bool(self.should_stop),
        )

    def evaluate_batch(
        self,
        X: np.ndarray,
        *,
        source_mode: str = "custom_batch",
        segment: str = "custom",
    ) -> list[RuntimeEvaluation]:
        out: list[RuntimeEvaluation] = []
        for x in np.asarray(X, dtype=float):
            if self.should_stop:
                break
            out.append(self.evaluate(x, source_mode=source_mode, segment=segment))
        return out

    def build_result(self, *, algorithm_id: str | None = None) -> OptimizerAlgorithmResult:
        history_df = self.history_df
        archive_df = self.archive_df
        X_rows: list[list[float]] = []
        y_rows: list[float] = []
        for row in self._archive_rows:
            if not isinstance(row, dict):
                continue
            try:
                X_rows.append([float(row[f]) for f in self.selected_features])
                if "objective_internal" in row:
                    y_rows.append(float(row["objective_internal"]))
                else:
                    raw = float(row.get("objective_raw", row.get("objective", float("nan"))))
                    y_rows.append(float(canonical_objective_for_result(
                        raw_objective=raw,
                        objective_sense=self.raw_objective_sense,
                        success=True,
                    )))
            except Exception:
                if X_rows and len(X_rows) > len(y_rows):
                    X_rows.pop()
                continue
        X_archive = (
            np.asarray(X_rows, dtype=float)
            if X_rows
            else np.empty((0, len(self.selected_features)), dtype=float)
        )
        y_archive = np.asarray(y_rows, dtype=float) if y_rows else np.empty((0,), dtype=float)
        final_pre_best = select_final_pre_feasible_best(
            X=X_archive,
            y_raw_internal=y_archive,
            objective_sense=self.objective_sense,
            pre_inequality_policy=self.pre_inequality_policy,
            pre_equality_policy=self.pre_equality_policy,
            pre_eq_objective_scale=self._pre_eq_objective_scale(),
        )
        if final_pre_best.found:
            best_point = self.point_dict(final_pre_best.x)
            best_objective = float(objective_from_minimize(
                final_pre_best.y_raw_internal,
                self.raw_objective_sense,
            ))
        else:
            best_point = {}
            best_objective = float("nan")
        best_effort_payload = final_pre_candidate_payload(
            final_pre_best.best_effort,
            feature_names=self.selected_features,
            objective_from_internal=lambda value: float(objective_from_minimize(value, self.raw_objective_sense)),
        )
        least_violation_payload = final_pre_candidate_payload(
            final_pre_best.least_violation,
            feature_names=self.selected_features,
            objective_from_internal=lambda value: float(objective_from_minimize(value, self.raw_objective_sense)),
        )
        raw_best_infeasible_payload = final_pre_candidate_payload(
            final_pre_best.raw_best_infeasible,
            feature_names=self.selected_features,
            objective_from_internal=lambda value: float(objective_from_minimize(value, self.raw_objective_sense)),
        )
        pareto_payload = [
            final_pre_candidate_payload(
                candidate,
                feature_names=self.selected_features,
                objective_from_internal=lambda value: float(objective_from_minimize(value, self.raw_objective_sense)),
            )
            for candidate in final_pre_best.pareto_candidates
        ]
        best_point_raw = (
            self.point_dict(self._best_x_raw)
            if self._best_x_raw is not None
            else dict(best_point)
        )
        best_objective_raw = float(objective_from_minimize(self._best_y_raw, self.raw_objective_sense))
        best_objective_effective = float(objective_from_minimize(self._best_y, self.raw_objective_sense))
        return build_optimizer_algorithm_result(
            history_df=history_df,
            archive_df=archive_df,
            best_point=best_point,
            best_objective=best_objective,
            best_point_raw=best_point_raw,
            best_objective_raw=best_objective_raw,
            best_effort_point=dict(best_effort_payload.get("point", {})),
            best_effort_objective=float(best_effort_payload.get("objective", float("nan"))),
            best_effort_pre_violation_score=float(best_effort_payload.get("pre_violation_score", float("nan"))),
            best_effort_score=float(best_effort_payload.get("best_effort_score", float("nan"))),
            least_violation_point=dict(least_violation_payload.get("point", {})),
            least_violation_objective=float(least_violation_payload.get("objective", float("nan"))),
            least_violation_pre_violation_score=float(least_violation_payload.get("pre_violation_score", float("nan"))),
            raw_best_infeasible_point=dict(raw_best_infeasible_payload.get("point", {})),
            raw_best_infeasible_objective=float(raw_best_infeasible_payload.get("objective", float("nan"))),
            raw_best_infeasible_pre_violation_score=float(raw_best_infeasible_payload.get("pre_violation_score", float("nan"))),
            best_effort_pareto_points=[p for p in pareto_payload if p],
            algorithm_id=str(algorithm_id or self.algorithm_id),
            engine="optimizer_runtime",
            feasibility_status="runtime_pre_constraints_only",
            initial_archive_count=int(self._initial_archive_count),
            final_train_count=int(len(archive_df)),
            goal_summary=self.goal_monitor.summary(),
            algorithm_summary_extra={
                "best_objective_selected_raw": float(best_objective),
                "best_objective_effective": float(best_objective_effective),
                **final_pre_best.summary(),
            },
            focus_pipeline_summary={"mode": "custom_algorithm", "stages": []},
        )
