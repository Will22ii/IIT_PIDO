from __future__ import annotations

import unittest

import numpy as np

from Optimizer.algorithms.focus_bo.bo_engine import (
    _build_source_pool_starts,
    _focus3_polish_candidate,
    _resolve_focus3_best_local_sigma,
    _resolve_focus3_polish_budget,
    _resolve_focus3_restart_policy,
)
from Optimizer.config import OptimizerSystemConfig


class Focus3HighDimSigmaBudgetScaleTest(unittest.TestCase):
    """best_local sigma 전환 시점의 예산 비율 스케줄.

    절대 eval 수(mid=40, late=160)만 쓰면 예산이 큰 문제일수록 최소 보폭
    구간이 비대해진다(budget=1050이면 예산의 85%가 최소 보폭). 비율로도
    계산해 늦은 쪽을 쓰되, max()이므로 짧은 예산은 기존 동작을 유지한다.
    """

    def setUp(self) -> None:
        self.system = OptimizerSystemConfig(focus_planner_profile="aion")

    def _eval(self, *, p_dim: int, budget: int, system=None) -> dict:
        _sigma, info = _resolve_focus3_best_local_sigma(
            system=self.system if system is None else system,
            focus3_eval_count=0,
            recover_info=None,
            p_dim=int(p_dim),
            focus3_available_budget=int(budget),
        )
        return info

    def test_high_dim_large_budget_scales_by_ratio(self) -> None:
        info = self._eval(p_dim=5, budget=1050)
        self.assertTrue(bool(info["aion_high_dim_policy"]))
        self.assertTrue(bool(info["budget_scaled"]))
        self.assertEqual(int(info["mid_eval"]), 315)   # 0.30 * 1050
        self.assertEqual(int(info["late_eval"]), 630)  # 0.60 * 1050

    def test_low_dim_is_untouched(self) -> None:
        # 변수 2개(골드스타인/식스험프)는 예산과 무관하게 기존 절대값을 쓴다.
        for budget in (15, 350, 1050):
            info = self._eval(p_dim=2, budget=budget)
            self.assertFalse(bool(info["aion_high_dim_policy"]))
            self.assertFalse(bool(info["budget_scaled"]))
            self.assertEqual(int(info["mid_eval"]), 40)
            self.assertEqual(int(info["late_eval"]), 160)

    def test_small_budget_is_untouched(self) -> None:
        # 예산 200회 미만이면 high_dim 정책 자체가 꺼진다.
        info = self._eval(p_dim=5, budget=150)
        self.assertFalse(bool(info["aion_high_dim_policy"]))
        self.assertFalse(bool(info["budget_scaled"]))
        self.assertEqual(int(info["late_eval"]), 160)

    def test_absolute_value_wins_when_larger(self) -> None:
        # max()이므로 비율이 절대값보다 이르면 절대값이 유지된다.
        info = self._eval(p_dim=5, budget=200)  # 0.60 * 200 = 120 < 160
        self.assertEqual(int(info["late_eval"]), 160)

    def test_disabled_flag_restores_absolute_schedule(self) -> None:
        system = OptimizerSystemConfig(
            focus_planner_profile="aion",
            focus3_aion_high_dim_sigma_budget_scaled_enabled=False,
        )
        info = self._eval(p_dim=5, budget=1050, system=system)
        self.assertFalse(bool(info["budget_scaled"]))
        self.assertEqual(int(info["mid_eval"]), 40)
        self.assertEqual(int(info["late_eval"]), 160)


class Focus3HighDimRestartPolicyTest(unittest.TestCase):
    """탈출 장치 발동 게이트."""

    def setUp(self) -> None:
        self.system = OptimizerSystemConfig(focus_planner_profile="aion")

    def _resolve(self, *, p_dim=5, budget=1050, no_improve=400, archive=1500, system=None) -> dict:
        return _resolve_focus3_restart_policy(
            system=self.system if system is None else system,
            p_dim=int(p_dim),
            focus3_available_budget=int(budget),
            recover_info={"no_improve_count": int(no_improve)},
            archive_size=int(archive),
        )

    def test_active_on_deep_stall(self) -> None:
        info = self._resolve()
        self.assertTrue(bool(info["active"]))
        self.assertEqual(str(info["reason"]), "deep_stall_restart")
        self.assertAlmostEqual(float(info["radius"]), 0.25, places=6)

    def test_low_dim_blocked(self) -> None:
        # 캔틸레버(p=4)와 골드스타인/식스험프(p=2)는 구조적으로 차단된다.
        for p_dim in (2, 4):
            info = self._resolve(p_dim=p_dim)
            self.assertFalse(bool(info["active"]))
            self.assertEqual(str(info["reason"]), "not_high_dim")

    def test_small_budget_blocked(self) -> None:
        info = self._resolve(budget=150)
        self.assertFalse(bool(info["active"]))
        self.assertEqual(str(info["reason"]), "not_high_dim")

    def test_not_stalled_yet(self) -> None:
        info = self._resolve(no_improve=399)
        self.assertFalse(bool(info["active"]))
        self.assertEqual(str(info["reason"]), "not_stalled")
        self.assertEqual(int(info["threshold"]), 400)

    def test_archive_too_small(self) -> None:
        info = self._resolve(archive=31)
        self.assertFalse(bool(info["active"]))
        self.assertEqual(str(info["reason"]), "archive_too_small")

    def test_disabled_flag(self) -> None:
        system = OptimizerSystemConfig(
            focus_planner_profile="aion",
            focus3_aion_high_dim_restart_enabled=False,
        )
        info = self._resolve(system=system)
        self.assertFalse(bool(info["active"]))
        self.assertEqual(str(info["reason"]), "disabled")

    def test_standalone_profile_blocked(self) -> None:
        info = self._resolve(system=OptimizerSystemConfig(focus_planner_profile="standalone"))
        self.assertFalse(bool(info["active"]))
        self.assertEqual(str(info["reason"]), "not_aion")

    def test_missing_recover_info_is_safe(self) -> None:
        info = _resolve_focus3_restart_policy(
            system=self.system,
            p_dim=5,
            focus3_available_budget=1050,
            recover_info=None,
            archive_size=1500,
        )
        self.assertFalse(bool(info["active"]))
        self.assertEqual(str(info["reason"]), "not_stalled")


class Focus3RestartAnchorTest(unittest.TestCase):
    """탈출 장치의 앵커 치환.

    아카이브 구성:
      A (0.10, 0.10) y=1.0  전역 최고점
      B (0.12, 0.11) y=2.0  가깝고 더 나쁨
      D (0.85, 0.88) y=4.0  멀고, 먼 것들 중에서는 최선
      C (0.90, 0.90) y=5.0  멀고 최악
    """

    A = (0.10, 0.10)
    D = (0.85, 0.88)

    def setUp(self) -> None:
        self.system = OptimizerSystemConfig(focus_planner_profile="aion")
        self.X = np.array([self.A, (0.12, 0.11), self.D, (0.90, 0.90)], dtype=float)
        self.y = np.array([1.0, 2.0, 4.0, 5.0], dtype=float)
        self.lb = np.zeros(2, dtype=float)
        self.ub = np.ones(2, dtype=float)

    def _pool_center(self, radius: float) -> np.ndarray:
        pool = _build_source_pool_starts(
            rng=np.random.default_rng(0),
            source="best_local",
            X_train=self.X,
            y_train=self.y,
            objective_sense="min",
            lb=self.lb,
            ub=self.ub,
            pool_size=64,
            topk_fraction=0.2,
            topk_sigma=0.08,
            boundary_near_ratio=0.03,
            system=self.system,
            best_local_sigma=0.001,
            best_local_top_count=3,
            best_local_anisotropic_enabled=False,
            best_local_anchor_best_prob=1.0,
            restart_exclude_radius=float(radius),
        )
        return np.asarray(pool, dtype=float).mean(axis=0)

    def test_default_anchors_on_global_best(self) -> None:
        # 반경 0(평소)이면 기존 동작 그대로 전역 최고점에 붙는다.
        np.testing.assert_allclose(self._pool_center(0.0), np.array(self.A), atol=0.02)

    def test_restart_anchors_on_best_far_point(self) -> None:
        # 반경을 주면 "먼 것들 중 최선"(D)으로 앵커가 옮겨간다. 최악점 C가 아니다.
        np.testing.assert_allclose(self._pool_center(0.5), np.array(self.D), atol=0.02)

    def test_falls_back_when_no_far_candidate(self) -> None:
        # 반경 밖 후보가 없으면 조용히 기존 동작으로 돌아간다.
        np.testing.assert_allclose(self._pool_center(0.95), np.array(self.A), atol=0.02)

    def test_other_sources_are_unaffected(self) -> None:
        # topk 등 다른 source는 반경 인자의 영향을 받지 않는다.
        kwargs = dict(
            source="topk",
            X_train=self.X,
            y_train=self.y,
            objective_sense="min",
            lb=self.lb,
            ub=self.ub,
            pool_size=32,
            topk_fraction=0.5,
            topk_sigma=0.01,
            boundary_near_ratio=0.03,
            system=self.system,
            best_local_sigma=0.001,
            best_local_top_count=3,
        )
        a = _build_source_pool_starts(rng=np.random.default_rng(7), restart_exclude_radius=0.0, **kwargs)
        b = _build_source_pool_starts(rng=np.random.default_rng(7), restart_exclude_radius=0.5, **kwargs)
        np.testing.assert_allclose(a, b)


class Focus3FinalPolishBudgetTest(unittest.TestCase):
    """마무리 수렴 단계에 배정되는 예산."""

    def setUp(self) -> None:
        self.system = OptimizerSystemConfig(focus_planner_profile="aion")

    def _budget(self, *, budget: int, p_dim: int, system=None) -> int:
        return _resolve_focus3_polish_budget(
            system=self.system if system is None else system,
            focus3_budget=int(budget),
            p_dim=int(p_dim),
        )

    def test_large_budget_gets_ten_percent(self) -> None:
        self.assertEqual(self._budget(budget=1050, p_dim=5), 105)   # 로젠브록
        self.assertEqual(self._budget(budget=350, p_dim=2), 35)     # 골드스타인

    def test_small_budget_is_skipped(self) -> None:
        # compass는 1 sweep에 2*p회를 쓴다. 2 sweep도 못 돌리면 발동하지 않는다.
        self.assertEqual(self._budget(budget=60, p_dim=4), 0)       # 캔틸레버(10% = 6회)
        self.assertEqual(self._budget(budget=15, p_dim=2), 0)       # 식스험프(10% = 2회)

    def test_cap_is_respected(self) -> None:
        self.assertEqual(self._budget(budget=100000, p_dim=5), 200)

    def test_disabled_flag(self) -> None:
        system = OptimizerSystemConfig(
            focus_planner_profile="aion",
            focus3_final_polish_enabled=False,
        )
        self.assertEqual(self._budget(budget=1050, p_dim=5, system=system), 0)

    def test_zero_budget_is_safe(self) -> None:
        self.assertEqual(self._budget(budget=0, p_dim=5), 0)


class Focus3PolishCandidateTest(unittest.TestCase):
    """compass search 후보 생성."""

    def setUp(self) -> None:
        self.lb = np.zeros(3, dtype=float)
        self.ub = np.ones(3, dtype=float) * 2.0   # span = 2.0
        self.base = np.array([1.0, 1.0, 1.0], dtype=float)

    def _cand(self, dir_index: int, step: float = 0.05) -> np.ndarray:
        return _focus3_polish_candidate(
            base_x=self.base,
            lb=self.lb,
            ub=self.ub,
            step_ratio=step,
            dir_index=int(dir_index),
            p_dim=3,
        )

    def test_cycles_through_each_axis_both_signs(self) -> None:
        # step 0.05 * span 2.0 = 0.10 만큼 한 좌표씩 움직인다.
        expected = [
            (0, +0.10), (0, -0.10),
            (1, +0.10), (1, -0.10),
            (2, +0.10), (2, -0.10),
        ]
        for k, (dim, delta) in enumerate(expected):
            cand = self._cand(k)
            want = self.base.copy()
            want[dim] += delta
            np.testing.assert_allclose(cand, want, atol=1e-12)

    def test_wraps_around_after_full_sweep(self) -> None:
        np.testing.assert_allclose(self._cand(6), self._cand(0))

    def test_respects_bounds(self) -> None:
        base = np.array([2.0, 0.0, 1.0], dtype=float)  # 상/하한에 붙어 있음
        up = _focus3_polish_candidate(
            base_x=base, lb=self.lb, ub=self.ub, step_ratio=0.5, dir_index=0, p_dim=3
        )
        down = _focus3_polish_candidate(
            base_x=base, lb=self.lb, ub=self.ub, step_ratio=0.5, dir_index=3, p_dim=3
        )
        self.assertLessEqual(float(up[0]), 2.0)
        self.assertGreaterEqual(float(down[1]), 0.0)

    def test_step_scales_with_ratio(self) -> None:
        small = self._cand(0, step=0.01)
        large = self._cand(0, step=0.04)
        self.assertAlmostEqual(float(small[0] - self.base[0]), 0.02, places=12)
        self.assertAlmostEqual(float(large[0] - self.base[0]), 0.08, places=12)

    def test_base_is_not_mutated(self) -> None:
        before = self.base.copy()
        self._cand(0)
        np.testing.assert_allclose(self.base, before)


if __name__ == "__main__":
    unittest.main()
