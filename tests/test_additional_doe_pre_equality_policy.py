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


if __name__ == "__main__":
    unittest.main()
