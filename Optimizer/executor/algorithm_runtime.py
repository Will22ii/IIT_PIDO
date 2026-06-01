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
from Optimizer.executor.goal_monitor import GoalMonitor
from Optimizer.executor.history_core import update_goal_and_build_evaluation_row
from Optimizer.executor.input_workflow import ResolvedOptimizerInputs
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
        self.objective_sense = normalize_objective_sense(str(resolved.objective_sense))

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
            error_context="optimizer runtime evaluation",
        )
        self._mapper = self._cae_evaluator.mapper
        self._var_names = list(self._mapper.var_names)
        self._x_base = self._mapper.x_base.copy()
        self._selected_idx = self._mapper.selected_idx.copy()
        goal = getattr(config.user, "goal", None)
        if goal is None:
            goal = getattr(config.user, "goal_objective", None)
        self.goal_monitor = GoalMonitor.from_goal(
            goal=goal,
            system_config=config.system,
            objective_sense=self.objective_sense,
        )

        self._history_rows: list[dict[str, Any]] = []
        self._archive_rows: list[dict[str, Any]] = []
        self._initial_archive_count = 0
        self._best_x: np.ndarray | None = None
        self._best_y = objective_initial_best(self.objective_sense)
        self._load_initial_archive()
        self.goal_monitor.initialize(best_objective=float(self._best_y))

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

    def _load_initial_archive(self) -> None:
        df = self.resolved.doe_df
        objective_col = str(self.config.system.objective_col)
        if not isinstance(df, pd.DataFrame) or df.empty or objective_col not in df.columns:
            return
        rows: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            try:
                x = np.asarray([float(row[f]) for f in self.selected_features], dtype=float)
                y = float(row[objective_col])
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
                    "objective": float(y),
                    "objective_raw": float(y),
                    "success": True,
                    "feasible": True,
                    "pre_feasible": True,
                    "pre_margin": float("inf"),
                }
            )
            rows.append(item)
        self._archive_rows.extend(rows)
        self._initial_archive_count = len(rows)
        if rows:
            y_arr = np.asarray([float(r["objective_raw"]) for r in rows], dtype=float)
            best_idx = objective_best_index(y_arr, self.objective_sense)
            self._best_y = float(y_arr[best_idx])
            self._best_x = np.asarray(
                [float(rows[best_idx][f]) for f in self.selected_features],
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
        y_effective = float(y_raw)
        previous_best = float(self._best_y)
        best_update = update_best_point(
            best_x=self._best_x if self._best_x is not None else x_arr,
            best_y=previous_best,
            x_next=x_arr,
            y_next=float(y_raw),
            objective_sense=self.objective_sense,
        )
        improved = bool(best_update.improved or self._best_x is None)
        if improved:
            self._best_y = float(best_update.best_y)
            self._best_x = best_update.best_x.copy()

        iteration = int(len(self._history_rows) + 1)
        row, goal_state = update_goal_and_build_evaluation_row(
            goal_monitor=self.goal_monitor,
            iteration=iteration,
            best_objective=float(self._best_y),
            improved=bool(improved),
            segment=str(segment),
            opt_focus_level=-1,
            source_mode=str(source_mode),
            x_values={f: float(v) for f, v in zip(self.selected_features, x_arr)},
            objective_raw=float(y_raw),
            objective_effective=float(y_effective),
            previous_best_objective_raw=float(previous_best),
            objective_sense=self.objective_sense,
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
        if extra:
            row.update(dict(extra))
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
            best_objective=float(self._best_y),
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
        best_point = self.best_point
        best_objective = float(self._best_y)
        return build_optimizer_algorithm_result(
            history_df=history_df,
            archive_df=archive_df,
            best_point=best_point,
            best_objective=best_objective,
            algorithm_id=str(algorithm_id or self.algorithm_id),
            engine="optimizer_runtime",
            feasibility_status="runtime_pre_constraints_only",
            initial_archive_count=int(self._initial_archive_count),
            final_train_count=int(len(archive_df)),
            goal_summary=self.goal_monitor.summary(),
            focus_pipeline_summary={"mode": "custom_algorithm", "stages": []},
        )
