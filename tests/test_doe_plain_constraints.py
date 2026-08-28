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
                    {"name": "x1", "lb": 0.0, "ub": 1.0},
                    {"name": "x2", "lb": 0.0, "ub": 1.0},
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

    def test_plain_doe_rejects_post_equality_constraints(self):
        """post 등식은 CAE 후에만 평가되어 샘플링을 유도할 방법이 없다."""

        def sampler(*, n_samples, bounds, rng, n_divisions=None):
            return np.zeros((int(n_samples), len(bounds)), dtype=float)

        with self.assertRaisesRegex(RuntimeError, "UNSUPPORTED_POST_EQUALITY_CONSTRAINT"):
            with tempfile.TemporaryDirectory() as tmpdir:
                context = create_run_context(
                    project_root=tmpdir,
                    user_config_snapshot={"problem": "plain_post_eq_rejected"},
                )
                run_doe_orchestrator(
                    problem_spec={
                        "name": "plain_post_eq_rejected",
                        "constraint_defs": [
                            {
                                "id": "eq_out",
                                "scope": "post",
                                "type": "==",
                                "expr": "out1",
                                "limit": 0.0,
                            }
                        ],
                    },
                    evaluate_func=lambda x: {
                        "success": True,
                        "objective": 0.0,
                        "outputs": {"out1": 0.0},
                    },
                    variables=[
                        {"name": "x1", "lb": 0.0, "ub": 1.0},
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

    def test_plain_doe_projects_pre_type2_equality(self):
        """pre Type 2 등식(eps 공란)은 투영으로 처리된다.

        rejection이었다면 x1+x2==1을 정확히 맞히는 무작위 점이 사실상 없어
        FAILED_FILTER_MIN이 났을 것이다.
        """

        def sampler(*, n_samples, bounds, rng, n_divisions=None):
            n = int(n_samples)
            return rng.random((n, len(bounds)))

        with tempfile.TemporaryDirectory() as tmpdir:
            context = create_run_context(
                project_root=tmpdir,
                user_config_snapshot={"problem": "plain_pre_type2_projection"},
            )
            out = run_doe_orchestrator(
                problem_spec={
                    "name": "plain_pre_type2_projection",
                    "constraint_defs": [
                        {
                            "id": "closure",
                            "scope": "pre",
                            "type": "==",
                            "expr": "x1 + x2",
                            "limit": 1.0,
                        }
                    ],
                },
                evaluate_func=lambda x: {
                    "success": True,
                    "objective": float(np.sum(x)),
                    "outputs": {},
                },
                variables=[
                    {"name": "x1", "lb": 0.0, "ub": 1.0},
                    {"name": "x2", "lb": 0.0, "ub": 1.0},
                ],
                algo_name="test_sampler",
                sampler=sampler,
                run_cfg={
                    "seed": 123,
                    "n_samples": 8,
                    "initial_corner_ratio": 0.1,
                    "success_rate_floor": 0.0,
                    "use_timestamp": False,
                    "debug_level": "off",
                },
                objective_sense="min",
                use_additional=False,
                run_context=context,
            )

        results = out["results"]
        self.assertEqual(len(results), 8)

        xs = np.asarray([r["x"] for r in results], dtype=float)
        # 모든 점이 등식을 만족해야 한다.
        residual = np.abs(xs[:, 0] + xs[:, 1] - 1.0)
        self.assertTrue(np.all(residual <= 1e-8), f"max residual={residual.max()}")
        self.assertTrue(all(r["feasible_pre"] for r in results))

        # bounds 안에 있어야 한다.
        self.assertTrue(np.all(xs >= -1e-12) and np.all(xs <= 1.0 + 1e-12))

        # maximin이 동작해 서로 다른 점이 뽑혀야 한다 (한 점으로 뭉치지 않음).
        self.assertGreater(float(np.ptp(xs[:, 0])), 0.3)

    def test_plain_doe_type1_equality_uses_band_rejection(self):
        """eps를 주면 Type 1. 밴드 안이면 통과하고 투영은 하지 않는다."""

        def sampler(*, n_samples, bounds, rng, n_divisions=None):
            # 절반은 밴드 안(0.5), 절반은 밖(0.9)
            n = int(n_samples)
            X = np.full((n, len(bounds)), 0.9, dtype=float)
            X[::2, :] = 0.5
            return X

        with tempfile.TemporaryDirectory() as tmpdir:
            context = create_run_context(
                project_root=tmpdir,
                user_config_snapshot={"problem": "plain_type1_band"},
            )
            out = run_doe_orchestrator(
                problem_spec={
                    "name": "plain_type1_band",
                    "constraint_defs": [
                        {
                            "id": "target",
                            "scope": "pre",
                            "type": "==",
                            "expr": "x1",
                            "limit": 0.5,
                            "eps": 0.1,
                        }
                    ],
                },
                evaluate_func=lambda x: {
                    "success": True,
                    "objective": float(np.sum(x)),
                    "outputs": {},
                },
                variables=[
                    {"name": "x1", "lb": 0.0, "ub": 1.0},
                ],
                algo_name="test_sampler",
                sampler=sampler,
                run_cfg={
                    "seed": 123,
                    "n_samples": 6,
                    "initial_corner_ratio": 0.0,
                    "initial_probe_multiplier": 4.0,
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

        results = out["results"]
        self.assertEqual(len(results), 6)
        # 밴드 밖(0.9)은 걸러지고 밴드 안(0.5)만 남는다. 투영이었다면 0.9가 0.5로
        # 이동했을 것이므로, 원래 값이 보존되었는지가 rejection의 증거다.
        self.assertTrue(all(abs(r["x"][0] - 0.5) < 1e-12 for r in results))
        self.assertTrue(all(r["feasible_pre"] for r in results))


if __name__ == "__main__":
    unittest.main()
