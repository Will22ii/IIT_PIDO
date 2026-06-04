import unittest

import numpy as np
import pandas as pd

from Optimizer.executor.evaluation_core import (
    objective_from_minimize,
    objective_to_minimize,
)
from DOE.executor.eval_sanitizer import sanitize_evaluate_output
from utils.objective_sense import canonicalize_objective_columns
from utils.objective_sense import canonical_objective_for_result


class OptimizerObjectiveCanonicalizationTest(unittest.TestCase):
    def test_min_objective_is_identity(self) -> None:
        self.assertEqual(objective_to_minimize(3.5, "min"), 3.5)
        self.assertEqual(objective_from_minimize(3.5, "min"), 3.5)

    def test_max_objective_is_negated_internally(self) -> None:
        self.assertEqual(objective_to_minimize(3.5, "max"), -3.5)
        self.assertEqual(objective_from_minimize(-3.5, "max"), 3.5)

    def test_array_round_trip(self) -> None:
        raw = np.asarray([1.0, 2.0, -4.0], dtype=float)
        internal = objective_to_minimize(raw, "max")
        self.assertTrue(np.allclose(internal, [-1.0, -2.0, 4.0]))
        self.assertTrue(np.allclose(objective_from_minimize(internal, "max"), raw))

    def test_canonicalize_prefers_objective_raw(self) -> None:
        df = pd.DataFrame({"x": [0.0, 1.0], "objective_raw": [1.0, 5.0], "objective": [9.0, 9.0]})
        out = canonicalize_objective_columns(df, objective_sense="max")
        self.assertTrue(np.allclose(out["objective"].to_numpy(dtype=float), [-1.0, -5.0]))
        self.assertTrue(np.allclose(out["objective_raw"].to_numpy(dtype=float), [1.0, 5.0]))

    def test_canonicalize_treats_legacy_objective_as_raw(self) -> None:
        df = pd.DataFrame({"x": [0.0, 1.0], "objective": [1.0, 5.0]})
        out = canonicalize_objective_columns(df, objective_sense="max")
        self.assertTrue(np.allclose(out["objective"].to_numpy(dtype=float), [-1.0, -5.0]))
        self.assertTrue(np.allclose(out["objective_raw"].to_numpy(dtype=float), [1.0, 5.0]))

    def test_canonicalize_failed_rows_to_inf(self) -> None:
        df = pd.DataFrame({"objective_raw": [1.0, 5.0], "success": [False, True]})
        out = canonicalize_objective_columns(df, objective_sense="max")
        self.assertTrue(np.isinf(float(out.loc[0, "objective"])))
        self.assertEqual(float(out.loc[1, "objective"]), -5.0)

    def test_canonicalize_numeric_zero_success_to_inf(self) -> None:
        df = pd.DataFrame({"objective_raw": [1.0, 5.0], "success": [0.0, 1.0]})
        out = canonicalize_objective_columns(df, objective_sense="max")
        self.assertTrue(np.isinf(float(out.loc[0, "objective"])))
        self.assertEqual(float(out.loc[1, "objective"]), -5.0)

    def test_non_finite_raw_result_is_never_best_for_max(self) -> None:
        value = canonical_objective_for_result(
            raw_objective=float("inf"),
            objective_sense="max",
            success=True,
        )
        self.assertTrue(np.isposinf(value))

    def test_non_finite_array_rows_become_inf_before_negation(self) -> None:
        df = pd.DataFrame({"objective_raw": [float("inf"), float("nan"), 5.0]})
        out = canonicalize_objective_columns(df, objective_sense="max")
        self.assertTrue(np.isposinf(float(out.loc[0, "objective"])))
        self.assertTrue(np.isposinf(float(out.loc[1, "objective"])))
        self.assertEqual(float(out.loc[2, "objective"]), -5.0)

    def test_sanitizer_treats_string_false_success_as_failed(self) -> None:
        success, objective, _outputs, _reason, _raw = sanitize_evaluate_output(
            {"success": "false", "objective": 999.0}
        )
        self.assertFalse(success)
        self.assertTrue(np.isposinf(objective))


if __name__ == "__main__":
    unittest.main()
