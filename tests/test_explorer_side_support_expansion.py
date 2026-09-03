from __future__ import annotations

import unittest

import numpy as np

from Explorer.executor.explorer_utils import (
    _expand_bounds_by_side_support,
    apply_bounds_margin,
)


def _vol(sel, glb):
    r = 1.0
    for (lo, hi), (gl, gu) in zip(sel, glb):
        r *= max(hi - lo, 0.0) / max(gu - gl, 1e-12)
    return float(r)


class SideSupportExpansionTest(unittest.TestCase):
    """지지가중 최소부피 확장.

    각 변(차원x방향) 너머의 GP 상위 후보 수에 비례해 확장 폭을 배분한다.
    """

    def setUp(self) -> None:
        self.glb = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
        # 좁은 코어 (부피 0.4^4 = 0.0256)
        self.core = [(3.0, 7.0), (3.0, 7.0), (3.0, 7.0), (3.0, 7.0)]

    def test_reaches_target_volume_without_exceeding(self) -> None:
        out = _expand_bounds_by_side_support(
            selected_bounds=self.core,
            bounds=self.glb,
            min_volume_ratio=0.2499,
            support_lo=np.ones(4),
            support_hi=np.ones(4),
        )
        v = _vol(out, self.glb)
        self.assertLessEqual(v, 0.2499 + 1e-6)
        self.assertGreaterEqual(v, 0.2499 * 0.97)

    def test_supported_side_gets_more_width(self) -> None:
        # 0번 차원 위쪽에만 강한 지지 -> 그쪽 상한이 다른 차원보다 멀리 간다
        out = _expand_bounds_by_side_support(
            selected_bounds=self.core,
            bounds=self.glb,
            min_volume_ratio=0.2499,
            support_lo=np.ones(4),
            support_hi=np.array([30.0, 1.0, 1.0, 1.0]),
        )
        gain0 = out[0][1] - self.core[0][1]
        gain1 = out[1][1] - self.core[1][1]
        self.assertGreater(gain0, gain1 * 2.0)

    def test_respects_global_bounds(self) -> None:
        out = _expand_bounds_by_side_support(
            selected_bounds=[(9.0, 9.9), (0.1, 1.0), (0.1, 1.0), (0.1, 1.0)],
            bounds=self.glb,
            min_volume_ratio=0.2499,
            support_lo=np.ones(4),
            support_hi=np.full(4, 50.0),
        )
        for (lo, hi), (gl, gu) in zip(out, self.glb):
            self.assertGreaterEqual(lo, gl - 1e-12)
            self.assertLessEqual(hi, gu + 1e-12)

    def test_rollback_preserves_boundary_touch(self) -> None:
        # 확장이 전역 경계에 닿았다면, 초과분 되돌리기가 그 변을 도로 떼지 않는다.
        # (경계에 붙은 최적점을 0.0001 차이로 잘라내는 사고 방지)
        core = [(0.2, 4.0), (3.0, 7.0), (3.0, 7.0), (3.0, 7.0)]
        out = _expand_bounds_by_side_support(
            selected_bounds=core,
            bounds=self.glb,
            min_volume_ratio=0.2499,
            support_lo=np.array([50.0, 1.0, 1.0, 1.0]),  # 0번 아래쪽 강지지 -> 경계 도달
            support_hi=np.ones(4),
        )
        self.assertAlmostEqual(out[0][0], 0.0, places=9)
        self.assertLessEqual(_vol(out, self.glb), 0.25 + 1e-6)

    def test_already_above_floor_returns_unchanged(self) -> None:
        big = [(0.0, 9.0), (0.0, 9.0), (0.0, 9.0), (0.0, 9.0)]  # 0.9^4 = 0.656
        out = _expand_bounds_by_side_support(
            selected_bounds=big,
            bounds=self.glb,
            min_volume_ratio=0.2499,
            support_lo=np.ones(4),
            support_hi=np.ones(4),
        )
        for (a, b), (c, d) in zip(out, big):
            self.assertAlmostEqual(a, c, places=9)
            self.assertAlmostEqual(b, d, places=9)


class ApplyBoundsMarginHookTest(unittest.TestCase):
    def setUp(self) -> None:
        self.glb = [(0.0, 10.0)] * 4
        self.core = [(3.0, 7.0)] * 4

    def test_default_path_unchanged_without_side_weights(self) -> None:
        # side weights 미지정 -> 기존 균등+booster 경로. 목표 부피 도달 확인.
        out = apply_bounds_margin(
            selected_bounds=self.core,
            bounds=self.glb,
            margin_ratio=0.02,
            min_volume_ratio=0.2499,
        )
        self.assertGreaterEqual(_vol(out, self.glb), 0.2499 * 0.97)

    def test_side_weights_route_to_support_expansion(self) -> None:
        # 강한 편향 가중치 -> 기본 경로와 다른, 편향된 결과가 나와야 한다
        base = apply_bounds_margin(
            selected_bounds=self.core,
            bounds=self.glb,
            margin_ratio=0.02,
            min_volume_ratio=0.2499,
        )
        biased = apply_bounds_margin(
            selected_bounds=self.core,
            bounds=self.glb,
            margin_ratio=0.02,
            min_volume_ratio=0.2499,
            side_support_lo=np.ones(4),
            side_support_hi=np.array([40.0, 1.0, 1.0, 1.0]),
        )
        self.assertGreater(biased[0][1], base[0][1])
        self.assertLessEqual(_vol(biased, self.glb), 0.25 + 1e-6)


if __name__ == "__main__":
    unittest.main()
