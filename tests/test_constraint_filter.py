import unittest

from DOE.executor.constraint_filter import (
    EXACT_EQUALITY_EPS,
    evaluate_constraints_point,
    is_type2_equality,
    pre_type2_equality_defs,
    rejection_constraint_defs,
    validate_constraint_defs,
)


class ConstraintFilterEqualityToleranceTests(unittest.TestCase):
    """eps 유무가 Type 1(밴드) / Type 2(구조적 등식)를 가른다."""

    def test_missing_eps_means_exact_match_not_an_auto_band(self):
        """eps 공란은 정확 일치다. 과거처럼 limit의 2%를 자동으로 채우지 않는다.

        자동 채움은 limit 값만 보고 변수 범위를 보지 않아, limit=0이면 근거 없이
        1.0을 스케일로 써서 문제에 따라 무제약이 되거나 만족 불가능이 되었다.
        """
        cdefs = [
            {
                "id": "eq_zero",
                "type": "==",
                "expr": "x1",
                "limit": 0.0,
            }
        ]

        constraints, feasible, _margin = evaluate_constraints_point(
            x=[0.0],
            var_names=["x1"],
            constraint_defs=cdefs,
            scope="pre",
        )
        self.assertTrue(feasible)
        self.assertAlmostEqual(constraints["eq_zero"]["eps"], EXACT_EQUALITY_EPS)

        # 과거에는 eps가 0.02로 자동 설정되어 통과하던 값이다.
        _constraints, feasible, _margin = evaluate_constraints_point(
            x=[0.019],
            var_names=["x1"],
            constraint_defs=cdefs,
            scope="pre",
        )
        self.assertFalse(feasible)

    def test_exact_match_passes_for_discrete_outputs(self):
        """이산 출력(n_modes == 3)은 정확 일치로 정상 동작해야 한다."""
        cdefs = [
            {
                "id": "n_modes_eq",
                "scope": "post",
                "type": "==",
                "expr": "n_modes",
                "limit": 3.0,
            }
        ]

        _c, feasible, _m = evaluate_constraints_point(
            x=[0.0],
            var_names=["x1"],
            constraint_defs=cdefs,
            scope="post",
            env_extra={"n_modes": 3.0},
        )
        self.assertTrue(feasible)

        _c, feasible, _m = evaluate_constraints_point(
            x=[0.0],
            var_names=["x1"],
            constraint_defs=cdefs,
            scope="post",
            env_extra={"n_modes": 4.0},
        )
        self.assertFalse(feasible)

    def test_eps_ratio_and_eps_min_are_rejected(self):
        """비율 기반 공차는 기준 스케일이 없어 지원하지 않는다."""
        for legacy_key in ("eps_ratio", "eps_min"):
            with self.assertRaisesRegex(ValueError, "no longer supported"):
                validate_constraint_defs(
                    [
                        {
                            "id": "eq",
                            "type": "==",
                            "expr": "x1",
                            "limit": 0.0,
                            legacy_key: 0.02,
                        }
                    ]
                )

    def test_equality_kind_split_and_helpers(self):
        defs = validate_constraint_defs(
            [
                {"id": "band", "scope": "pre", "type": "==", "expr": "x1", "limit": 1.0, "eps": 0.5},
                {"id": "exact", "scope": "pre", "type": "==", "expr": "x2", "limit": 0.0},
                {"id": "ineq", "scope": "pre", "type": "<=", "expr": "x1", "limit": 9.0},
                {"id": "post_exact", "scope": "post", "type": "==", "expr": "y", "limit": 3.0},
            ]
        )
        by_id = {c["id"]: c for c in defs}

        self.assertFalse(is_type2_equality(by_id["band"]))
        self.assertTrue(is_type2_equality(by_id["exact"]))
        self.assertFalse(is_type2_equality(by_id["ineq"]))

        # 투영 대상은 pre + Type 2 뿐이다.
        self.assertEqual([c["id"] for c in pre_type2_equality_defs(defs)], ["exact"])

        # margin/통과율 계산에는 투영 대상만 빠진다.
        self.assertEqual(
            [c["id"] for c in rejection_constraint_defs(defs)],
            ["band", "ineq", "post_exact"],
        )

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
