import unittest
import tempfile

import numpy as np

from DOE.executor.doe_orchestrator import run_doe_orchestrator
from pipeline.run_context import create_run_context


class PlainDOEConstraintPolicyTests(unittest.TestCase):
    def test_plain_doe_retries_and_keeps_pre_inequality_feasible_pool(self):
        calls = {"sampler": 0, "eval": 0}

        def sampler(*, n_samples, bounds, rng, n_divisions=None):
            calls["sampler"] += 1
            if calls["sampler"] == 1:
                return np.full((int(n_samples), len(bounds)), 0.9, dtype=float)
            return np.full((int(n_samples), len(bounds)), 0.05, dtype=float)

        def evaluate_func(x):
            calls["eval"] += 1
            return {"success": True, "objective": float(np.sum(x)), "outputs": {}}

        with tempfile.TemporaryDirectory() as tmpdir:
            context = create_run_context(
                project_root=tmpdir,
                user_config_snapshot={"problem": "plain_pre_ineq_retry"},
            )
            out = run_doe_orchestrator(
                problem_spec={
                    "name": "plain_pre_ineq_retry",
                    "constraint_defs": [
                        {
                            "id": "x1_max",
                            "scope": "pre",
                            "type": "<=",
                            "expr": "x1",
                            "limit": 0.1,
                        }
                    ],
                },
                evaluate_func=evaluate_func,
                variables=[
                    {"name": "x1", "lb": 0.0, "ub": 1.0, "baseline": 0.5},
                    {"name": "x2", "lb": 0.0, "ub": 1.0, "baseline": 0.5},
                ],
                algo_name="test_sampler",
                sampler=sampler,
                run_cfg={
                    "seed": 123,
                    "n_samples": 6,
                    "initial_corner_ratio": 0.0,
                    "initial_probe_multiplier": 1.0,
                    "plan_filter_safety": 1.0,
                    "plan_filter_r_floor": 0.5,
                    "success_rate_floor": 0.0,
                    "use_timestamp": False,
                    "debug_level": "off",
                },
                objective_sense="min",
                use_additional=False,
                run_context=context,
            )

        self.assertEqual(calls["sampler"], 2)
        self.assertEqual(calls["eval"], 6)
        self.assertEqual(len(out["results"]), 6)
        self.assertTrue(all(r["feasible_pre"] for r in out["results"]))
        self.assertTrue(all(r["x"][0] == 0.05 for r in out["results"]))

    def test_plain_doe_rejects_equality_constraints(self):
        def sampler(*, n_samples, bounds, rng, n_divisions=None):
            return np.zeros((int(n_samples), len(bounds)), dtype=float)

        with self.assertRaisesRegex(RuntimeError, "UNSUPPORTED_PLAIN_DOE_EQUALITY_CONSTRAINT"):
            with tempfile.TemporaryDirectory() as tmpdir:
                context = create_run_context(
                    project_root=tmpdir,
                    user_config_snapshot={"problem": "plain_eq_rejected"},
                )
                run_doe_orchestrator(
                    problem_spec={
                        "name": "plain_eq_rejected",
                        "constraint_defs": [
                            {
                                "id": "eq_x1",
                                "scope": "pre",
                                "type": "==",
                                "expr": "x1",
                                "limit": 0.0,
                            }
                        ],
                    },
                    evaluate_func=lambda x: {"success": True, "objective": 0.0, "outputs": {}},
                    variables=[
                        {"name": "x1", "lb": 0.0, "ub": 1.0, "baseline": 0.5},
                    ],
                    algo_name="test_sampler",
                    sampler=sampler,
                    run_cfg={
                        "seed": 123,
                        "n_samples": 6,
                        "initial_corner_ratio": 0.0,
                        "success_rate_floor": 0.0,
                        "use_timestamp": False,
                        "debug_level": "off",
                    },
                    objective_sense="min",
                    use_additional=False,
                    run_context=context,
                )


if __name__ == "__main__":
    unittest.main()
