import unittest

from DOE.executor.constraint_filter import evaluate_constraints_point


class ConstraintFilterEqualityToleranceTests(unittest.TestCase):
    def test_default_equality_tolerance_uses_minimum_unit_scale(self):
        cdefs = [
            {
                "id": "eq_zero",
                "type": "==",
                "expr": "x1",
                "limit": 0.0,
            }
        ]

        constraints, feasible, _margin = evaluate_constraints_point(
            x=[0.019],
            var_names=["x1"],
            constraint_defs=cdefs,
            scope="pre",
        )
        self.assertTrue(feasible)
        self.assertAlmostEqual(constraints["eq_zero"]["eps"], 0.02)

        _constraints, feasible, _margin = evaluate_constraints_point(
            x=[0.021],
            var_names=["x1"],
            constraint_defs=cdefs,
            scope="pre",
        )
        self.assertFalse(feasible)

    def test_default_equality_tolerance_scales_with_large_limit(self):
        cdefs = [
            {
                "id": "eq_large",
                "type": "==",
                "expr": "x1",
                "limit": 10.0,
            }
        ]

        constraints, feasible, _margin = evaluate_constraints_point(
            x=[10.19],
            var_names=["x1"],
            constraint_defs=cdefs,
            scope="pre",
        )
        self.assertTrue(feasible)
        self.assertAlmostEqual(constraints["eq_large"]["eps"], 0.2)

        _constraints, feasible, _margin = evaluate_constraints_point(
            x=[10.21],
            var_names=["x1"],
            constraint_defs=cdefs,
            scope="pre",
        )
        self.assertFalse(feasible)

    def test_explicit_equality_eps_overrides_default_ratio(self):
        cdefs = [
            {
                "id": "eq_explicit",
                "type": "==",
                "expr": "x1",
                "limit": 0.0,
                "eps": 0.005,
            }
        ]

        constraints, feasible, _margin = evaluate_constraints_point(
            x=[0.006],
            var_names=["x1"],
            constraint_defs=cdefs,
            scope="pre",
        )
        self.assertFalse(feasible)
        self.assertAlmostEqual(constraints["eq_explicit"]["eps"], 0.005)


if __name__ == "__main__":
    unittest.main()
