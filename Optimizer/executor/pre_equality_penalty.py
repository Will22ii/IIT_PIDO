from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from DOE.executor.constraint_filter import evaluate_constraints_point
from Optimizer.executor.evaluation_core import safe_constraint_token


def pre_equality_constraint_defs(constraint_defs: list[dict] | None) -> list[dict]:
    return [
        c
        for c in (constraint_defs or [])
        if isinstance(c, dict)
        and str(c.get("scope", "pre")).strip().lower() == "pre"
        and str(c.get("type", "")).strip() == "=="
    ]


def objective_penalty_scale(values: np.ndarray | list[float] | tuple[float, ...], *, floor: float = 1.0) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    floor = float(max(float(floor), 1e-12))
    if finite.size >= 2:
        q10, q90 = np.percentile(finite, [10.0, 90.0])
        spread = max(float(q90 - q10), float(np.std(finite)))
        if np.isfinite(spread) and spread > floor:
            return float(spread)
    if finite.size >= 1:
        median_abs = float(abs(np.median(finite)))
        if np.isfinite(median_abs) and median_abs > floor:
            return float(median_abs)
    return float(floor)


@dataclass(frozen=True)
class PreEqualityPenaltyResult:
    active: bool
    feasible: bool
    penalty: float
    violation_norm: float
    violation_mean_square: float
    worst_id: str
    worst_violation: float
    payload: dict


@dataclass(frozen=True)
class PreEqualityPenaltyPolicy:
    constraint_defs: list[dict]
    var_names: list[str]
    to_full_x: Callable[[np.ndarray], np.ndarray] | None = None
    enabled: bool = True
    penalty_lambda: float = 10.0
    violation_cap: float = 1e6

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraint_defs", pre_equality_constraint_defs(list(self.constraint_defs or [])))
        object.__setattr__(self, "var_names", list(self.var_names or []))
        object.__setattr__(self, "penalty_lambda", float(max(float(self.penalty_lambda), 0.0)))
        object.__setattr__(self, "violation_cap", float(max(float(self.violation_cap), 1.0)))
        # id -> cdef 조회표. violation()이 점마다 제약마다 선형 탐색하던 것을 대체한다.
        object.__setattr__(
            self,
            "_defs_by_id",
            {str(c.get("id")): c for c in self.constraint_defs},
        )

    @property
    def active(self) -> bool:
        return bool(self.enabled and self.penalty_lambda > 0.0 and self.constraint_defs)

    @property
    def constraint_count(self) -> int:
        return int(len(self.constraint_defs))

    def full_x(self, x_selected: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_selected, dtype=float).reshape(-1)
        return self.to_full_x(x_arr) if self.to_full_x is not None else x_arr

    def _evaluate_payload(self, x_selected: np.ndarray) -> tuple[dict, bool]:
        if not self.constraint_defs:
            return {}, True
        payload, feasible, _margin = evaluate_constraints_point(
            x=self.full_x(x_selected),
            var_names=self.var_names,
            constraint_defs=self.constraint_defs,
            scope="pre",
            fail_fast_output_missing=False,
        )
        return payload, bool(feasible)

    def violation(self, x_selected: np.ndarray) -> tuple[float, float, bool, str, float, dict]:
        if not self.constraint_defs:
            return 0.0, 0.0, True, "", 0.0, {}
        try:
            payload, feasible = self._evaluate_payload(x_selected)
        except Exception:
            return (
                float(self.violation_cap),
                float(self.violation_cap * self.violation_cap),
                False,
                "__pre_eq_eval_error__",
                float(self.violation_cap),
                {},
            )

        weighted_squares: list[float] = []
        worst_id = ""
        worst_violation = float("-inf")
        for raw_id, cinfo in payload.items():
            if not isinstance(cinfo, dict):
                continue
            cid = str(cinfo.get("id", raw_id)).strip() or str(raw_id)
            try:
                value = float(cinfo.get("value", float("inf")))
                limit = float(cinfo.get("limit", float("inf")))
                eps = float(cinfo.get("eps", 0.0))
                cdef = getattr(self, "_defs_by_id", {}).get(cid, {})
                scale = float(cdef.get("penalty_scale", max(abs(value), abs(limit), 1.0)))
                weight = float(cdef.get("penalty_weight", 1.0))
                scale = max(abs(scale), 1e-12)
                weight = max(weight, 0.0)
                v = max(0.0, abs(value - limit) - max(eps, 0.0)) / scale
            except Exception:
                v = float(self.violation_cap)
                weight = 1.0
            if not np.isfinite(v):
                v = float(self.violation_cap)
            v = float(min(max(v, 0.0), self.violation_cap))
            if v > worst_violation:
                worst_violation = float(v)
                worst_id = cid
            weighted_squares.append(float(weight) * float(v * v))

        if not weighted_squares:
            return 0.0, 0.0, bool(feasible), "", 0.0, payload
        mean_square = float(np.mean(np.asarray(weighted_squares, dtype=float)))
        norm = float(np.sqrt(max(mean_square, 0.0)))
        return norm, mean_square, bool(feasible), worst_id, float(max(worst_violation, 0.0)), payload

    def penalty(self, x_selected: np.ndarray, *, objective_scale: float) -> PreEqualityPenaltyResult:
        if not self.active:
            return PreEqualityPenaltyResult(
                active=False,
                feasible=True,
                penalty=0.0,
                violation_norm=0.0,
                violation_mean_square=0.0,
                worst_id="",
                worst_violation=0.0,
                payload={},
            )
        norm, mean_square, feasible, worst_id, worst_violation, payload = self.violation(x_selected)
        scale = float(max(float(objective_scale), 1e-12))
        penalty = float(self.penalty_lambda) * scale * float(mean_square)
        if not np.isfinite(penalty):
            penalty = float(self.penalty_lambda) * scale * float(self.violation_cap * self.violation_cap)
        return PreEqualityPenaltyResult(
            active=True,
            feasible=bool(feasible),
            penalty=float(max(penalty, 0.0)),
            violation_norm=float(norm),
            violation_mean_square=float(mean_square),
            worst_id=str(worst_id),
            worst_violation=float(worst_violation),
            payload=payload,
        )

    def penalty_value(self, x_selected: np.ndarray, *, objective_scale: float) -> float:
        return float(self.penalty(x_selected, objective_scale=objective_scale).penalty)

    def penalty_values(self, X_selected: np.ndarray, *, objective_scale: float) -> np.ndarray:
        arr = np.asarray(X_selected, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.size == 0:
            return np.empty((0,), dtype=float)
        return np.asarray(
            [self.penalty_value(row, objective_scale=objective_scale) for row in arr],
            dtype=float,
        )

    def debug_fields(self, x_selected: np.ndarray, *, objective_scale: float) -> dict[str, object]:
        result = self.penalty(x_selected, objective_scale=objective_scale)
        fields: dict[str, object] = {
            "pre_eq_penalty_active": bool(result.active),
            "pre_eq_constraint_count": int(self.constraint_count),
            "pre_eq_feasible": bool(result.feasible),
            "pre_eq_penalty": float(result.penalty),
            "pre_eq_violation_norm": float(result.violation_norm),
            "pre_eq_violation_mean_square": float(result.violation_mean_square),
            "pre_eq_worst_id": str(result.worst_id),
            "pre_eq_worst_violation": float(result.worst_violation),
            "pre_eq_objective_scale": float(objective_scale),
            "pre_eq_penalty_lambda": float(self.penalty_lambda if result.active else 0.0),
        }
        for raw_id, cinfo in result.payload.items():
            if not isinstance(cinfo, dict):
                continue
            cid = str(cinfo.get("id", raw_id)).strip() or str(raw_id)
            token = safe_constraint_token(cid)
            fields[f"pre_eq_{token}_value"] = float(cinfo.get("value", float("nan")))
            fields[f"pre_eq_{token}_limit"] = float(cinfo.get("limit", float("nan")))
            fields[f"pre_eq_{token}_eps"] = float(cinfo.get("eps", float("nan")))
            fields[f"pre_eq_{token}_ok"] = bool(cinfo.get("ok", False))
            fields[f"pre_eq_{token}_margin"] = float(cinfo.get("margin", float("nan")))
            fields[f"pre_eq_{token}_g"] = float(cinfo.get("g", float("nan")))
        return fields
