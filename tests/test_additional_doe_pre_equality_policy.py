import unittest

import numpy as np

from DOE.config import DOESystemConfig, build_additional_cfg_from_system
from DOE.executor.additional_orchestrator import AdditionalDOEOrchestrator


class AdditionalDOEPreEqualityPolicyTests(unittest.TestCase):
    def _make_orchestrator(self, constraint_defs):
        return AdditionalDOEOrchestrator(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            sampler=lambda **kwargs: np.zeros((int(kwargs["n_samples"]), 2), dtype=float),
            evaluate_func=lambda x: {"success": True, "objective": 0.0, "outputs": {}},
            feasibility_func=lambda payload: True,
            surrogate_factory=object(),
            gate1=object(),
            gate2=object(),
            gate_manager=object(),
            rng=np.random.default_rng(123),
            total_budget=20,
            phase1_global_ratio=0.8,
            phase2_global_ratio=0.3,
            constraint_defs=constraint_defs,
            var_names=["x1", "x2"],
            pre_equality_boost_base=2.0,
            pre_equality_boost_max=8.0,
            plan_filter_safety=1.0,
            plan_filter_r_floor=0.02,
        )

    def test_pre_equality_boost_scales_by_count_and_caps(self):
        orch = self._make_orchestrator(
            [
                {"id": "eq1", "scope": "pre", "type": "==", "expr": "x1", "limit": 0.0},
                {"id": "eq2", "scope": "pre", "type": "==", "expr": "x2", "limit": 0.0},
                {"id": "eq3", "scope": "pre", "type": "==", "expr": "x1 + x2", "limit": 0.0},
                {"id": "ineq", "scope": "pre", "type": "<=", "expr": "x1", "limit": 1.0},
            ]
        )

        self.assertEqual(orch.pre_equality_count, 3)
        self.assertEqual(orch.pre_equality_boost, 8.0)

    def test_plan_generation_count_uses_pre_equality_boost(self):
        orch = self._make_orchestrator(
            [
                {"id": "eq1", "scope": "pre", "type": "==", "expr": "x1", "limit": 0.0},
                {"id": "eq2", "scope": "pre", "type": "==", "expr": "x2", "limit": 0.0},
            ]
        )
        orch.constraint_rate_hat = 0.5

        self.assertEqual(orch.pre_equality_boost, 4.0)
        self.assertEqual(orch._plan_generation_count(target_count=10), 80)

    def test_system_config_exports_pre_equality_policy(self):
        cfg = build_additional_cfg_from_system(DOESystemConfig())

        self.assertEqual(cfg["pre_equality_boost_base"], 2.0)
        self.assertEqual(cfg["pre_equality_boost_max"], 8.0)
        self.assertEqual(cfg["pre_equality_warning_threshold"], 3)
        self.assertEqual(cfg["plan_filter_r_floor_equality"], 1e-5)
        self.assertEqual(cfg["plan_generation_max"], 2_000_000)


class AdditionalDOEGenerationCapTests(unittest.TestCase):
    """후보 생성 상한 완화. pre equality가 없으면 기존 동작이 그대로여야 한다."""

    def _make_orchestrator(self, constraint_defs, **overrides):
        kwargs = dict(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            sampler=lambda **kw: np.zeros((int(kw["n_samples"]), 2), dtype=float),
            evaluate_func=lambda x: {"success": True, "objective": 0.0, "outputs": {}},
            feasibility_func=lambda payload: True,
            surrogate_factory=object(),
            gate1=object(),
            gate2=object(),
            gate_manager=object(),
            rng=np.random.default_rng(123),
            total_budget=20,
            phase1_global_ratio=0.8,
            phase2_global_ratio=0.3,
            constraint_defs=constraint_defs,
            var_names=["x1", "x2"],
            plan_filter_safety=1.0,
            plan_filter_r_floor=0.02,
        )
        kwargs.update(overrides)
        return AdditionalDOEOrchestrator(**kwargs)

    def test_no_equality_keeps_existing_behavior(self):
        """등식이 없으면 floor도 인플레이션도 기존 값 그대로.

        벤치마크 4문제가 모두 이 경로다. 값이 바뀌면 벤치마크가 영향을 받는다.
        """
        orch = self._make_orchestrator(
            [{"id": "ineq", "scope": "pre", "type": "<=", "expr": "x1", "limit": 0.5}]
        )

        self.assertEqual(orch.pre_equality_count, 0)
        # floor는 기존 값 유지
        self.assertEqual(orch._effective_r_floor, 0.02)
        # 초기 probe 인플레이션은 항상 1.0 (관측 통과율과 무관)
        self.assertEqual(orch._initial_probe_equality_inflation(), 1.0)
        orch.constraint_rate_hat = 0.0001
        self.assertEqual(orch._initial_probe_equality_inflation(), 1.0)

        # 통과율이 아무리 낮게 관측돼도 floor에서 잘려 배수가 50에서 멈추는
        # 기존 동작이 유지된다.
        orch.constraint_rate_hat = 1.0
        orch._update_constraint_ratio(n_generated=10_000, n_feasible=1)
        self.assertEqual(orch.constraint_rate_hat, 0.02)
        self.assertEqual(orch._plan_generation_count(target_count=10), 500)

    def test_equality_lifts_floor_so_inflation_can_grow(self):
        """등식이 있으면 관측 통과율이 그대로 기록되고 배수가 커진다."""
        orch = self._make_orchestrator(
            [
                {"id": "eq1", "scope": "pre", "type": "==", "expr": "x1", "limit": 0.5},
                {"id": "eq2", "scope": "pre", "type": "==", "expr": "x2", "limit": 0.5},
            ]
        )

        self.assertEqual(orch.pre_equality_count, 2)
        self.assertEqual(orch._effective_r_floor, 1e-5)

        # 10,000개 중 4개 통과 = 0.04%. 기존에는 0.02로 잘려 50배에 그쳤다.
        orch._update_constraint_ratio(n_generated=10_000, n_feasible=4)
        self.assertAlmostEqual(orch.constraint_rate_hat, 0.0004)

        # target 10 x safety 1.0 x inv 2500 x boost 4.0 = 100,000
        self.assertEqual(orch._plan_generation_count(target_count=10), 100_000)

        # 초기 probe도 관측을 반영해 자기교정한다.
        self.assertEqual(orch._initial_probe_equality_inflation(), 2500.0)

    def test_generation_cap_truncates_and_warns_once(self):
        """절대 상한에서 잘리되 중단하지 않고, 경고는 한 번만 낸다."""
        orch = self._make_orchestrator(
            [{"id": "eq1", "scope": "pre", "type": "==", "expr": "x1", "limit": 0.5}],
            plan_generation_max=1000,
        )
        orch._update_constraint_ratio(n_generated=1_000_000, n_feasible=1)

        n_gen = orch._plan_generation_count(target_count=10)
        self.assertEqual(n_gen, 1000)
        self.assertTrue(orch._plan_generation_cap_warned)

        # target_count 밑으로는 자르지 않는다.
        self.assertEqual(
            orch._cap_generation_count(10**9, target_count=5000), 5000
        )


if __name__ == "__main__":
    unittest.main()
