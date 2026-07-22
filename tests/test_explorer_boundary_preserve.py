from __future__ import annotations

import unittest

from Explorer.config import ExplorerSystemConfig
from Explorer.executor.explorer_orchestrator import _apply_constrained_boundary_preserve


class ExplorerBoundaryPreserveTest(unittest.TestCase):
    """cap/shift 이후 설계 경계를 복원하는 보정 단계의 계약.

    이 단계는 width를 보존한 채 박스를 경계로 미끄러뜨린다. 부피가 늘지 않으므로
    volume cap 판정에 영향이 없고, 따라서 "불필요하게 복원했을 때의 비용"이 거의 없다.
    반대로 필요한데 복원하지 않으면 최적해가 bounds 밖에 남는다.
    """

    def setUp(self) -> None:
        self.system = ExplorerSystemConfig()

    @staticmethod
    def _points(rows: list[list[float]]):
        import numpy as np

        return np.asarray(rows, dtype=float)

    def test_small_gap_still_restores_to_edge(self) -> None:
        # cantilever_beam seed 2042 재현.
        # h1 축 설계범위 [0.1, 1.0](span 0.9)에서 앞 단계가 하한을 0.1110으로 남겼다.
        # 경계와의 간격 0.011은 span의 1.2%로, 구현상 취소 임계(2.5%)보다 작아
        # 복원이 취소됐고 최적해 h1=0.1이 bounds 밖에 남았다.
        global_bounds = [(0.1, 1.0)]
        selected = [(0.11104510989333827, 0.943295516850103)]
        # 선택된 앵커가 하한 근처까지 도달해 있다(touch 증거).
        points = self._points([[0.1005], [0.30], [0.55]])

        out, shifted = _apply_constrained_boundary_preserve(
            selected_bounds=selected,
            global_bounds=global_bounds,
            selected_points=points,
            protect_lb=None,
            protect_ub=None,
            system=self.system,
        )

        self.assertEqual(len(shifted), 1)
        self.assertEqual(shifted[0]["side"], "lb")
        self.assertTrue(shifted[0]["below_legacy_deadband"])
        self.assertAlmostEqual(out[0][0], 0.1, places=9)
        # 최적해가 복원 후 bounds 안에 들어온다.
        self.assertLessEqual(out[0][0], 0.1 + 1e-9)

    def test_width_is_preserved(self) -> None:
        # 부피가 늘지 않아야 volume cap 판정에 영향이 없다.
        global_bounds = [(0.1, 2.0)]
        selected = [(0.11304879998694384, 0.855734707592704)]
        points = self._points([[0.1002], [0.40], [0.80]])
        width_before = selected[0][1] - selected[0][0]

        out, shifted = _apply_constrained_boundary_preserve(
            selected_bounds=selected,
            global_bounds=global_bounds,
            selected_points=points,
            protect_lb=None,
            protect_ub=None,
            system=self.system,
        )

        self.assertEqual(len(shifted), 1)
        self.assertAlmostEqual(out[0][1] - out[0][0], width_before, places=9)
        self.assertAlmostEqual(out[0][0], 0.1, places=9)

    def test_no_evidence_no_restore(self) -> None:
        # 보호 판정 근거(policy pin / touch)가 없으면 복원하지 않는다.
        # 간격만으로 움직이지 않는다는 계약을 고정한다.
        global_bounds = [(0.0, 1.0)]
        selected = [(0.40, 0.70)]
        points = self._points([[0.45], [0.55], [0.65]])  # 어느 경계에도 닿지 않음

        out, shifted = _apply_constrained_boundary_preserve(
            selected_bounds=selected,
            global_bounds=global_bounds,
            selected_points=points,
            protect_lb=None,
            protect_ub=None,
            system=self.system,
        )

        self.assertEqual(shifted, [])
        self.assertEqual(out, selected)

    def test_both_sides_protected_is_skipped(self) -> None:
        # 양쪽 경계가 모두 보호 대상이면 미끄러뜨릴 방향이 정해지지 않으므로 건너뛴다.
        global_bounds = [(0.0, 1.0)]
        selected = [(0.30, 0.70)]
        points = self._points([[0.005], [0.50], [0.995]])  # 양쪽 다 touch

        out, shifted = _apply_constrained_boundary_preserve(
            selected_bounds=selected,
            global_bounds=global_bounds,
            selected_points=points,
            protect_lb=None,
            protect_ub=None,
            system=self.system,
        )

        self.assertEqual(shifted, [])
        self.assertEqual(out, selected)

    def test_already_at_edge_is_not_rewritten(self) -> None:
        # 이미 경계에 붙어 있으면 최소 간격 미만이므로 아무것도 하지 않는다.
        global_bounds = [(0.1, 1.0)]
        selected = [(0.1, 0.8)]
        points = self._points([[0.1001], [0.40], [0.75]])

        out, shifted = _apply_constrained_boundary_preserve(
            selected_bounds=selected,
            global_bounds=global_bounds,
            selected_points=points,
            protect_lb=None,
            protect_ub=None,
            system=self.system,
        )

        self.assertEqual(shifted, [])
        self.assertEqual(out, selected)


if __name__ == "__main__":
    unittest.main()
