from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from Optimizer.algorithms.focus_bo.bo_engine import _acquisition_scores_for_pool
from Optimizer.algorithms.focus_bo.bo_engine import run_bo_engine
from Optimizer.config import OptimizerSystemConfig
from Optimizer.executor.final_best_policy import select_final_pre_feasible_best
from Optimizer.executor.pre_equality_penalty import PreEqualityPenaltyPolicy
from Optimizer.executor.pre_constraint_policy import PreInequalityPolicy


def _variables() -> list[dict]:
    return [
        {"name": f"x{i}", "lb": -2.0, "ub": 2.0}
        for i in range(1, 6)
    ]


def _system() -> OptimizerSystemConfig:
    return OptimizerSystemConfig(
        debug_level="off",
        post_constraint_enabled=False,
    )


class OptimizerPreInequalityPolicyTests(unittest.TestCase):
    def test_pre_equality_penalty_normalizes_violation(self) -> None:
        policy = PreEqualityPenaltyPolicy(
            constraint_defs=[
                {
                    "id": "x1_eq",
                    "scope": "pre",
                    "type": "==",
                    "expr": "x1",
                    "limit": 0.6,
                    "eps": 0.0,
                },
                {
                    "id": "x1_ub",
                    "scope": "pre",
                    "type": "<=",
                    "expr": "x1",
                    "limit": 1.0,
                },
            ],
            var_names=["x1"],
            penalty_lambda=10.0,
        )

        result = policy.penalty(np.asarray([0.8], dtype=float), objective_scale=2.0)

        self.assertTrue(result.active)
        self.assertFalse(result.feasible)
        self.assertEqual(policy.constraint_count, 1)
        self.assertAlmostEqual(result.violation_norm, 0.2)
        self.assertAlmostEqual(result.penalty, 0.8)

    def test_pre_equality_penalty_changes_warm_start_effective_best_only(self) -> None:
        system = _system()
        system.pre_eq_penalty_lambda = 100.0
        result = run_bo_engine(
            problem_name="rosenbrock_nodummy",
            variables=_variables(),
            doe_df=pd.DataFrame(
                {
                    "x1": [0.9, 0.0],
                    "objective": [0.0, 10.0],
                    "objective_raw": [0.0, 10.0],
                    "success": [True, True],
                }
            ),
            selected_features=["x1"],
            selected_bounds={"x1": (-2.0, 2.0)},
            bounds_source="cae",
            evaluation_base_values={"x2": 0.0, "x3": 0.0, "x4": 0.0, "x5": 0.0},
            objective_col="objective",
            objective_sense="min",
            n_samples=0,
            goal=None,
            system=system,
            seed=123,
            constraint_defs=[
                {
                    "id": "x1_eq_zero",
                    "scope": "pre",
                    "type": "==",
                    "expr": "x1",
                    "limit": 0.0,
                    "eps": 0.0,
                }
            ],
        )

        self.assertAlmostEqual(result.best_point["x1"], 0.0)
        self.assertAlmostEqual(result.best_objective, 10.0)
        self.assertAlmostEqual(result.best_point_raw["x1"], 0.9)
        self.assertAlmostEqual(result.best_objective_raw, 0.0)
        self.assertAlmostEqual(float(result.archive_df.loc[0, "objective"]), 0.0)
        self.assertGreater(float(result.archive_df.loc[0, "objective_effective"]), 0.0)
        self.assertTrue(result.algorithm_summary["pre_eq_penalty_active"])
        self.assertTrue(result.algorithm_summary["final_pre_best_feasible"])
        self.assertEqual(result.algorithm_summary["final_pre_best_status"], "ok")

    def test_pre_equality_penalty_is_added_to_acquisition_scores(self) -> None:
        policy = PreEqualityPenaltyPolicy(
            constraint_defs=[
                {
                    "id": "x1_eq_zero",
                    "scope": "pre",
                    "type": "==",
                    "expr": "x1",
                    "limit": 0.0,
                    "eps": 0.0,
                }
            ],
            var_names=["x1"],
            penalty_lambda=10.0,
        )
        scores = _acquisition_scores_for_pool(
            pool=np.asarray([[0.0], [1.0]], dtype=float),
            model=None,
            y_best=0.0,
            objective_sense="min",
            acq_type="LCB",
            kappa=1.0,
            xi=0.01,
            post_feasible_prob_fn=None,
            post_penalty_lambda=0.0,
            soft_penalty_fn=lambda x: policy.penalty_value(x, objective_scale=1.0),
        )

        self.assertAlmostEqual(float(scores[0]), 0.0)
        self.assertGreater(float(scores[1]), float(scores[0]))

    def test_final_pre_best_exposes_best_effort_and_pareto_fallbacks(self) -> None:
        result = select_final_pre_feasible_best(
            X=np.asarray([[1.0], [0.01]], dtype=float),
            y_raw_internal=np.asarray([0.0, 0.5], dtype=float),
            objective_sense="min",
            pre_inequality_policy=PreInequalityPolicy(
                constraint_defs=[],
                var_names=["x1"],
            ),
            pre_equality_policy=PreEqualityPenaltyPolicy(
                constraint_defs=[
                    {
                        "id": "x1_eq_zero",
                        "scope": "pre",
                        "type": "==",
                        "expr": "x1",
                        "limit": 0.0,
                        "eps": 0.0,
                    }
                ],
                var_names=["x1"],
            ),
            pre_eq_objective_scale=1.0,
        )

        self.assertFalse(result.found)
        self.assertEqual(result.status, "no_pre_feasible_candidate")
        self.assertEqual(result.raw_best_infeasible.index, 0)
        self.assertEqual(result.least_violation.index, 1)
        self.assertEqual(result.best_effort.index, 1)
        self.assertEqual(len(result.pareto_candidates), 2)

    def test_policy_filters_selected_matrix_using_full_vector(self) -> None:
        policy = PreInequalityPolicy(
            constraint_defs=[
                {
                    "id": "x2_ub",
                    "scope": "pre",
                    "type": "<=",
                    "expr": "x2",
                    "limit": 0.5,
                }
            ],
            var_names=["x1", "x2"],
            to_full_x=lambda row: np.asarray([float(row[0]), float(row[0])], dtype=float),
        )

        X_keep, _payloads, margins, mask = policy.filter_matrix(np.asarray([[0.25], [0.75]], dtype=float))

        self.assertEqual(mask.tolist(), [True, False])
        self.assertEqual(X_keep.shape, (1, 1))
        self.assertAlmostEqual(float(X_keep[0, 0]), 0.25)
        self.assertGreater(float(margins[0]), 0.0)

    def test_policy_filters_warm_start_rows(self) -> None:
        policy = PreInequalityPolicy(
            constraint_defs=[
                {
                    "id": "x1_ub",
                    "scope": "pre",
                    "type": "<=",
                    "expr": "x1",
                    "limit": 0.5,
                }
            ],
            var_names=["x1"],
        )
        result = policy.filter_warm_start_df(
            pd.DataFrame(
                {
                    "x1": [0.1, 0.2, 0.8, 0.3],
                    "objective": [1.0, 2.0, 3.0, 4.0],
                    "success": [True, False, True, True],
                    "feasible_pre": [True, True, True, False],
                }
            ),
            selected_features=["x1"],
            warn_prefix="[TestWarmStart]",
        )

        self.assertEqual(result.df["x1"].tolist(), [0.1])
        self.assertEqual(result.diagnostics["raw_count"], 4)
        self.assertEqual(result.diagnostics["kept_count"], 1)
        self.assertTrue(result.diagnostics["pre_hard_filter_active"])

    def test_warm_start_pre_inequality_uses_frozen_full_vector(self) -> None:
        result = run_bo_engine(
            problem_name="rosenbrock_nodummy",
            variables=_variables(),
            doe_df=pd.DataFrame(
                {
                    "x1": [0.25],
                    "objective": [3.0],
                    "objective_raw": [3.0],
                    "success": [True],
                }
            ),
            selected_features=["x1"],
            selected_bounds={"x1": (-2.0, 2.0)},
            bounds_source="cae",
            evaluation_base_values={"x2": 0.0, "x3": 0.0, "x4": 0.0, "x5": 0.0},
            objective_col="objective",
            objective_sense="min",
            n_samples=0,
            goal=None,
            system=_system(),
            seed=123,
            constraint_defs=[
                {
                    "id": "omitted_x2_ub",
                    "scope": "pre",
                    "type": "<=",
                    "expr": "x2",
                    "limit": 0.1,
                }
            ],
        )

        self.assertEqual(result.initial_archive_count, 1)
        self.assertAlmostEqual(result.best_objective_raw, 3.0)
        self.assertTrue(result.algorithm_summary["pre_hard_filter_active"])
        self.assertTrue(result.algorithm_summary["pre_hard_filter_auto_enabled"])

    def test_warm_start_filters_pre_infeasible_rows_before_best_selection(self) -> None:
        result = run_bo_engine(
            problem_name="rosenbrock_nodummy",
            variables=_variables(),
            doe_df=pd.DataFrame(
                {
                    "x1": [0.9, 0.1],
                    "objective": [0.0, 10.0],
                    "objective_raw": [0.0, 10.0],
                    "success": [True, True],
                    "feasible_pre": [False, True],
                }
            ),
            selected_features=["x1"],
            selected_bounds={"x1": (-2.0, 2.0)},
            bounds_source="cae",
            evaluation_base_values={"x2": 0.0, "x3": 0.0, "x4": 0.0, "x5": 0.0},
            objective_col="objective",
            objective_sense="min",
            n_samples=0,
            goal=None,
            system=_system(),
            seed=123,
            constraint_defs=[
                {
                    "id": "x1_ub",
                    "scope": "pre",
                    "type": "<=",
                    "expr": "x1",
                    "limit": 0.2,
                }
            ],
        )

        self.assertEqual(result.initial_archive_count, 1)
        self.assertAlmostEqual(result.best_point_raw["x1"], 0.1)
        self.assertAlmostEqual(result.best_objective_raw, 10.0)

    def test_pre_equality_is_not_part_of_hard_filter(self) -> None:
        result = run_bo_engine(
            problem_name="rosenbrock_nodummy",
            variables=_variables(),
            doe_df=pd.DataFrame(
                {
                    "x1": [0.5],
                    "objective": [1.0],
                    "objective_raw": [1.0],
                    "success": [True],
                }
            ),
            selected_features=["x1"],
            selected_bounds={"x1": (-2.0, 2.0)},
            bounds_source="cae",
            evaluation_base_values={"x2": 0.0, "x3": 0.0, "x4": 0.0, "x5": 0.0},
            objective_col="objective",
            objective_sense="min",
            n_samples=0,
            goal=None,
            system=_system(),
            seed=123,
            constraint_defs=[
                {
                    "id": "x1_eq",
                    "scope": "pre",
                    "type": "==",
                    "expr": "x1",
                    "limit": 0.0,
                    "eps": 0.001,
                }
            ],
        )

        self.assertEqual(result.initial_archive_count, 1)
        self.assertFalse(result.algorithm_summary["pre_hard_filter_active"])
        self.assertFalse(result.algorithm_summary["final_pre_best_feasible"])
        self.assertEqual(result.algorithm_summary["final_pre_best_status"], "no_pre_feasible_candidate")


if __name__ == "__main__":
    unittest.main()
