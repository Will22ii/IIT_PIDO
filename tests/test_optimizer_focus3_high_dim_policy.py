from __future__ import annotations

import unittest

from Optimizer.algorithms.focus_bo.bo_engine import (
    _apply_focus3_goal_free_plateau_policy,
    _apply_focus3_recover_best_local_policy,
    _apply_focus3_recover_policy,
)
from Optimizer.config import OptimizerSystemConfig


class OptimizerFocus3HighDimPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.system = OptimizerSystemConfig(focus_planner_profile="aion")

    @staticmethod
    def _source_performance(*, local_improved: int = 0, nonlocal_improved: int = 0) -> dict[str, object]:
        info: dict[str, object] = {
            "best_local_refine_count": 20,
            "correlated_local_refine_count": 0,
            "local_probe_refine_count": 0,
            "topk_refine_count": 60,
            "random_refine_count": 20,
            "boundary_refine_count": 0,
        }
        for source in ("best_local", "correlated_local", "local_probe", "topk", "random", "boundary"):
            info[f"{source}_refine_improved_count"] = 0
            info[f"{source}_refine_improve_rate"] = 0.0
        info["best_local_refine_improved_count"] = int(local_improved)
        info["best_local_refine_improve_rate"] = float(local_improved / 20)
        info["topk_refine_improved_count"] = int(nonlocal_improved)
        info["topk_refine_improve_rate"] = float(nonlocal_improved / 60)
        return info

    def test_plateau_hold_does_not_override_strong_recovery(self) -> None:
        _, _, _, _, recover = _apply_focus3_recover_policy(
            system=self.system,
            p_topk=0.25,
            p_boundary=0.05,
            p_random=0.20,
            kappa=1.0,
            best_raw_history=[5.0] * 82,
            objective_sense="min",
            has_constraints=False,
            p_dim=5,
            focus3_available_budget=1050,
        )
        self.assertEqual(recover["level"], "strong")
        self.assertTrue(recover["force_exploration_acq"])

        topk, boundary, random_p, best_local, plateau = _apply_focus3_goal_free_plateau_policy(
            system=self.system,
            p_topk=0.25,
            p_boundary=0.05,
            p_random=0.20,
            p_best_local=0.50,
            best_raw_history=[5.0] * 101,
            objective_sense="min",
            focus3_eval_count=100,
            recover_info=recover,
            source_performance_info=self._source_performance(),
            has_constraints=False,
            p_dim=5,
            focus3_available_budget=1050,
        )

        self.assertEqual(plateau["state"], "plateau_hold")
        self.assertFalse(plateau["applied"])
        self.assertFalse(plateau["probability_adjusted"])
        self.assertEqual((topk, boundary, random_p, best_local), (0.25, 0.05, 0.20, 0.50))

    def test_plateau_exploit_still_applies_a_real_probability_change(self) -> None:
        _, _, _, best_local, plateau = _apply_focus3_goal_free_plateau_policy(
            system=self.system,
            p_topk=0.25,
            p_boundary=0.05,
            p_random=0.20,
            p_best_local=0.50,
            best_raw_history=[5.0] * 101,
            objective_sense="min",
            focus3_eval_count=100,
            recover_info={"no_improve_count": 100},
            source_performance_info=self._source_performance(local_improved=3),
            has_constraints=False,
            p_dim=5,
            focus3_available_budget=1050,
        )

        self.assertEqual(plateau["state"], "plateau_exploit")
        self.assertTrue(plateau["applied"])
        self.assertTrue(plateau["probability_adjusted"])
        self.assertGreater(best_local, 0.50)

    def test_plateau_hold_times_out_into_escape(self) -> None:
        # 개선 증거가 전혀 없는 상태로 hold 상한을 넘기면, 증거 요구를 면제하고 escape한다.
        # 증거 게이트는 window 내 개선 횟수를 요구하는데 plateau에서는 그 값이 0으로
        # 수렴하므로, 상한이 없으면 hold가 영구화된다.
        # escape는 기존 설계대로 escape_period 주기로만 발동한다. 상한 면제는
        # escape_supported를 켜줄 뿐 주기 계약을 바꾸지 않는다.
        hold_max = int(self.system.focus3_aion_high_dim_plateau_hold_max_no_improve)
        period = int(self.system.focus3_aion_high_dim_plateau_escape_period)
        no_improve = hold_max + (-hold_max % period)

        topk, boundary, random_p, best_local, plateau = _apply_focus3_goal_free_plateau_policy(
            system=self.system,
            p_topk=0.25,
            p_boundary=0.05,
            p_random=0.20,
            p_best_local=0.50,
            best_raw_history=[5.0] * (no_improve + 1),
            objective_sense="min",
            focus3_eval_count=no_improve,
            recover_info={"active": True, "level": "strong", "no_improve_count": no_improve},
            source_performance_info=self._source_performance(),
            has_constraints=False,
            p_dim=5,
            focus3_available_budget=1050,
        )

        self.assertEqual(plateau["state"], "plateau_escape")
        self.assertEqual(plateau["reason"], "plateau_escape_evidence_waived")
        self.assertTrue(plateau["escape_evidence_waived"])
        self.assertTrue(plateau["probability_adjusted"])
        self.assertLess(best_local, 0.50)
        self.assertGreater(random_p + topk, 0.25 + 0.20)
        self.assertGreaterEqual(boundary, 0.0)

    def test_plateau_hold_below_timeout_stays_hold(self) -> None:
        # 상한 미만에서는 기존 hold 동작이 그대로 유지된다(회귀 방지).
        hold_max = int(self.system.focus3_aion_high_dim_plateau_hold_max_no_improve)
        no_improve = hold_max - 1

        topk, boundary, random_p, best_local, plateau = _apply_focus3_goal_free_plateau_policy(
            system=self.system,
            p_topk=0.25,
            p_boundary=0.05,
            p_random=0.20,
            p_best_local=0.50,
            best_raw_history=[5.0] * (no_improve + 1),
            objective_sense="min",
            focus3_eval_count=no_improve,
            recover_info={"active": True, "level": "strong", "no_improve_count": no_improve},
            source_performance_info=self._source_performance(),
            has_constraints=False,
            p_dim=5,
            focus3_available_budget=1050,
        )

        self.assertEqual(plateau["state"], "plateau_hold")
        self.assertFalse(plateau["escape_evidence_waived"])
        self.assertFalse(plateau["probability_adjusted"])
        self.assertEqual((topk, boundary, random_p, best_local), (0.25, 0.05, 0.20, 0.50))

    def test_recover_best_local_preserves_explore_floor(self) -> None:
        # recover_best_local은 best_local quota를 random부터 회수한다.
        # 고차원 경로에서는 random 하한을 남기고, 부족분은 topk에서 충당해야 한다.
        floor = float(self.system.focus3_aion_recover_best_local_explore_floor)
        recover_info = {"active": True, "level": "strong", "no_improve_count": 500}

        topk, boundary, random_p, best_local, info = _apply_focus3_recover_best_local_policy(
            system=self.system,
            p_topk=0.30,
            p_boundary=0.00,
            p_random=0.20,
            p_best_local=0.50,
            recover_info=recover_info,
            has_constraints=False,
            p_dim=5,
            focus3_available_budget=1050,
        )

        self.assertTrue(info["applied"])
        self.assertTrue(info["explore_floor_applied"])
        self.assertGreaterEqual(random_p + 1e-12, floor)
        # best_local은 목표치까지 올라가되, 재원은 topk에서 나온다.
        self.assertGreater(best_local, 0.50)
        self.assertLess(topk, 0.30)
        self.assertAlmostEqual(topk + boundary + random_p + best_local, 1.0, places=9)

    def test_recover_best_local_explore_floor_skipped_when_not_high_dim(self) -> None:
        # 저차원/소예산은 기존 동작을 그대로 유지한다(random을 0까지 회수).
        recover_info = {"active": True, "level": "strong", "no_improve_count": 500}

        _, _, random_p, _, info = _apply_focus3_recover_best_local_policy(
            system=self.system,
            p_topk=0.30,
            p_boundary=0.00,
            p_random=0.20,
            p_best_local=0.50,
            recover_info=recover_info,
            has_constraints=False,
            p_dim=2,
            focus3_available_budget=350,
        )

        self.assertTrue(info["applied"])
        self.assertFalse(info["explore_floor_applied"])
        self.assertAlmostEqual(random_p, 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
