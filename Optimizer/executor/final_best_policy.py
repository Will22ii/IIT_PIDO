from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from Optimizer.executor.evaluation_core import objective_best_index
from Optimizer.executor.pre_constraint_policy import PreInequalityPolicy
from Optimizer.executor.pre_equality_penalty import PreEqualityPenaltyPolicy


@dataclass(frozen=True)
class FinalPreFallbackCandidate:
    found: bool
    index: int
    x: np.ndarray
    y_raw_internal: float
    pre_violation_score: float
    objective_gap_norm: float
    best_effort_score: float

    @staticmethod
    def empty() -> "FinalPreFallbackCandidate":
        return FinalPreFallbackCandidate(
            found=False,
            index=-1,
            x=np.empty((0,), dtype=float),
            y_raw_internal=float("nan"),
            pre_violation_score=float("nan"),
            objective_gap_norm=float("nan"),
            best_effort_score=float("nan"),
        )


@dataclass(frozen=True)
class FinalPreBestResult:
    found: bool
    index: int
    x: np.ndarray
    y_raw_internal: float
    status: str
    candidate_count: int
    feasible_count: int
    pre_inequality_feasible_count: int
    pre_equality_feasible_count: int
    pre_inequality_active: bool
    pre_equality_active: bool
    mask: np.ndarray
    best_effort: FinalPreFallbackCandidate
    least_violation: FinalPreFallbackCandidate
    raw_best_infeasible: FinalPreFallbackCandidate
    pareto_candidates: tuple[FinalPreFallbackCandidate, ...]

    def summary(self) -> dict[str, object]:
        return {
            "final_pre_best_required": bool(self.pre_inequality_active or self.pre_equality_active),
            "final_pre_best_feasible": bool(self.found),
            "final_pre_best_status": str(self.status),
            "final_pre_best_index": int(self.index),
            "final_pre_best_candidate_count": int(self.candidate_count),
            "final_pre_best_feasible_count": int(self.feasible_count),
            "final_pre_inequality_active": bool(self.pre_inequality_active),
            "final_pre_inequality_feasible_count": int(self.pre_inequality_feasible_count),
            "final_pre_equality_active": bool(self.pre_equality_active),
            "final_pre_equality_feasible_count": int(self.pre_equality_feasible_count),
            "best_effort_available": bool(self.best_effort.found),
            "best_effort_index": int(self.best_effort.index),
            "best_effort_pre_violation_score": float(self.best_effort.pre_violation_score),
            "best_effort_objective_gap_norm": float(self.best_effort.objective_gap_norm),
            "best_effort_score": float(self.best_effort.best_effort_score),
            "least_violation_available": bool(self.least_violation.found),
            "least_violation_index": int(self.least_violation.index),
            "least_violation_pre_violation_score": float(self.least_violation.pre_violation_score),
            "raw_best_infeasible_available": bool(self.raw_best_infeasible.found),
            "raw_best_infeasible_index": int(self.raw_best_infeasible.index),
            "raw_best_infeasible_pre_violation_score": float(self.raw_best_infeasible.pre_violation_score),
            "best_effort_pareto_count": int(len(self.pareto_candidates)),
        }


def _empty_result(
    *,
    n_rows: int,
    status: str,
    pre_inequality_active: bool,
    pre_equality_active: bool,
) -> FinalPreBestResult:
    empty = FinalPreFallbackCandidate.empty()
    return FinalPreBestResult(
        found=False,
        index=-1,
        x=np.empty((0,), dtype=float),
        y_raw_internal=float("nan"),
        status=str(status),
        candidate_count=0,
        feasible_count=0,
        pre_inequality_feasible_count=0,
        pre_equality_feasible_count=0,
        pre_inequality_active=bool(pre_inequality_active),
        pre_equality_active=bool(pre_equality_active),
        mask=np.zeros((int(max(n_rows, 0)),), dtype=bool),
        best_effort=empty,
        least_violation=empty,
        raw_best_infeasible=empty,
        pareto_candidates=(),
    )


def _pre_inequality_violation(policy: PreInequalityPolicy, x: np.ndarray) -> tuple[float, bool]:
    if not policy.active:
        return 0.0, True
    try:
        payload, feasible, margin = policy.evaluate(x)
    except Exception:
        return 1e6, False
    violations: list[float] = []
    for cinfo in payload.values():
        if not isinstance(cinfo, dict):
            continue
        try:
            g = float(cinfo.get("g", float("inf")))
        except Exception:
            g = float("inf")
        if not np.isfinite(g):
            g = 1e6
        violations.append(float(max(0.0, g)))
    if violations:
        arr = np.asarray(violations, dtype=float)
        norm = float(np.sqrt(np.mean(arr * arr)))
    else:
        norm = float(max(0.0, -float(margin))) if np.isfinite(float(margin)) else (0.0 if feasible else 1e6)
    return float(norm), bool(feasible)


def _pre_equality_violation(policy: PreEqualityPenaltyPolicy, x: np.ndarray) -> tuple[float, bool]:
    if policy.constraint_count <= 0:
        return 0.0, True
    try:
        norm, _mean_square, feasible, _worst_id, _worst_violation, _payload = policy.violation(x)
    except Exception:
        return 1e6, False
    if not np.isfinite(norm):
        norm = 1e6
    return float(max(0.0, norm)), bool(feasible)


def _objective_gap_norms(y_arr: np.ndarray, indices: np.ndarray, objective_sense: str) -> np.ndarray:
    values = np.asarray(y_arr[indices], dtype=float).reshape(-1)
    if values.size == 0:
        return np.empty((0,), dtype=float)
    if str(objective_sense or "min").strip().lower() == "max":
        best = float(np.max(values))
        gaps = np.maximum(0.0, best - values)
    else:
        best = float(np.min(values))
        gaps = np.maximum(0.0, values - best)
    q10, q90 = np.percentile(values, [10.0, 90.0]) if values.size >= 2 else (0.0, abs(float(values[0])))
    scale = max(float(q90 - q10), float(np.std(values)), abs(float(np.median(values))) if values.size else 0.0, 1.0)
    return np.asarray(gaps, dtype=float) / float(scale)


def _candidate(
    *,
    idx: int,
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    violation_scores: np.ndarray,
    objective_gap_norms: np.ndarray,
    finite_indices: np.ndarray,
) -> FinalPreFallbackCandidate:
    idx = int(idx)
    pos_arr = np.where(finite_indices == idx)[0]
    if pos_arr.size == 0:
        return FinalPreFallbackCandidate.empty()
    pos = int(pos_arr[0])
    objective_gap = float(objective_gap_norms[pos])
    violation = float(violation_scores[pos])
    score = float(objective_gap + violation)
    return FinalPreFallbackCandidate(
        found=True,
        index=idx,
        x=X_arr[idx].copy(),
        y_raw_internal=float(y_arr[idx]),
        pre_violation_score=violation,
        objective_gap_norm=objective_gap,
        best_effort_score=score,
    )


def _pareto_indices(*, indices: np.ndarray, objective_gaps: np.ndarray, violations: np.ndarray, limit: int = 10) -> list[int]:
    keep: list[int] = []
    n = int(indices.size)
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            no_worse = objective_gaps[j] <= objective_gaps[i] and violations[j] <= violations[i]
            strictly_better = objective_gaps[j] < objective_gaps[i] or violations[j] < violations[i]
            if bool(no_worse and strictly_better):
                dominated = True
                break
        if not dominated:
            keep.append(int(indices[i]))
    keep.sort(key=lambda idx: float(objective_gaps[np.where(indices == idx)[0][0]] + violations[np.where(indices == idx)[0][0]]))
    return keep[: int(max(limit, 1))]


def final_pre_candidate_payload(
    candidate: FinalPreFallbackCandidate,
    *,
    feature_names: list[str],
    objective_from_internal: Callable[[float], float],
) -> dict[str, object]:
    if not candidate.found:
        return {}
    return {
        "archive_index": int(candidate.index),
        "point": {
            str(name): float(value)
            for name, value in zip(feature_names, np.asarray(candidate.x, dtype=float).reshape(-1))
        },
        "objective": float(objective_from_internal(float(candidate.y_raw_internal))),
        "pre_violation_score": float(candidate.pre_violation_score),
        "objective_gap_norm": float(candidate.objective_gap_norm),
        "best_effort_score": float(candidate.best_effort_score),
    }


def select_final_pre_feasible_best(
    *,
    X: np.ndarray,
    y_raw_internal: np.ndarray,
    objective_sense: str,
    pre_inequality_policy: PreInequalityPolicy,
    pre_equality_policy: PreEqualityPenaltyPolicy,
    pre_eq_objective_scale: float,
) -> FinalPreBestResult:
    """Select the user-facing final best from hard pre-feasible archive rows.

    Penalties may guide search, but this final selector uses only raw objective
    values after enforcing pre inequality and pre equality feasibility.
    """
    _ = float(pre_eq_objective_scale)
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y_raw_internal, dtype=float).reshape(-1)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)
    n_rows = int(X_arr.shape[0]) if X_arr.ndim == 2 else 0
    mask = np.zeros((n_rows,), dtype=bool)
    empty_x = np.empty((0,), dtype=float)
    pre_ineq_active = bool(pre_inequality_policy.active)
    pre_eq_active = bool(pre_equality_policy.constraint_count > 0)
    if X_arr.ndim != 2 or n_rows == 0 or y_arr.shape[0] != n_rows:
        return _empty_result(
            n_rows=n_rows,
            status="invalid_archive",
            pre_inequality_active=pre_ineq_active,
            pre_equality_active=pre_eq_active,
        )

    finite_mask = np.isfinite(y_arr) & np.all(np.isfinite(X_arr), axis=1)
    finite_idx = np.where(finite_mask)[0]
    if finite_idx.size == 0:
        return _empty_result(
            n_rows=n_rows,
            status="no_finite_candidate",
            pre_inequality_active=pre_ineq_active,
            pre_equality_active=pre_eq_active,
        )

    pre_ineq_mask = np.zeros((n_rows,), dtype=bool)
    pre_eq_mask = np.zeros((n_rows,), dtype=bool)
    violation_scores = np.full((finite_idx.size,), float("inf"), dtype=float)
    for idx in finite_idx.tolist():
        x = X_arr[int(idx)]
        ineq_violation, ineq_ok = _pre_inequality_violation(pre_inequality_policy, x)
        pre_ineq_mask[int(idx)] = bool(ineq_ok)
        eq_violation, eq_ok = _pre_equality_violation(pre_equality_policy, x)
        pre_eq_mask[int(idx)] = bool(eq_ok)
        pos = int(np.where(finite_idx == int(idx))[0][0])
        violation_scores[pos] = float(np.sqrt(float(ineq_violation * ineq_violation + eq_violation * eq_violation)))

    mask = finite_mask & pre_ineq_mask & pre_eq_mask
    keep_idx = np.where(mask)[0]
    objective_gaps = _objective_gap_norms(y_arr, finite_idx, objective_sense)
    if keep_idx.size == 0:
        fallback_local = objective_best_index(y_arr[finite_idx], objective_sense)
        fallback_idx = int(finite_idx[int(fallback_local)])
        best_effort_pos = int(np.argmin(objective_gaps + violation_scores))
        least_violation_pos = int(np.lexsort((objective_gaps, violation_scores))[0])
        pareto = tuple(
            _candidate(
                idx=idx,
                X_arr=X_arr,
                y_arr=y_arr,
                violation_scores=violation_scores,
                objective_gap_norms=objective_gaps,
                finite_indices=finite_idx,
            )
            for idx in _pareto_indices(indices=finite_idx, objective_gaps=objective_gaps, violations=violation_scores)
        )
        return FinalPreBestResult(
            found=False,
            index=fallback_idx,
            x=X_arr[fallback_idx].copy(),
            y_raw_internal=float(y_arr[fallback_idx]),
            status="no_pre_feasible_candidate",
            candidate_count=int(finite_idx.size),
            feasible_count=0,
            pre_inequality_feasible_count=int(np.sum(pre_ineq_mask & finite_mask)),
            pre_equality_feasible_count=int(np.sum(pre_eq_mask & finite_mask)),
            pre_inequality_active=pre_ineq_active,
            pre_equality_active=pre_eq_active,
            mask=mask,
            best_effort=_candidate(
                idx=int(finite_idx[best_effort_pos]),
                X_arr=X_arr,
                y_arr=y_arr,
                violation_scores=violation_scores,
                objective_gap_norms=objective_gaps,
                finite_indices=finite_idx,
            ),
            least_violation=_candidate(
                idx=int(finite_idx[least_violation_pos]),
                X_arr=X_arr,
                y_arr=y_arr,
                violation_scores=violation_scores,
                objective_gap_norms=objective_gaps,
                finite_indices=finite_idx,
            ),
            raw_best_infeasible=_candidate(
                idx=fallback_idx,
                X_arr=X_arr,
                y_arr=y_arr,
                violation_scores=violation_scores,
                objective_gap_norms=objective_gaps,
                finite_indices=finite_idx,
            ),
            pareto_candidates=pareto,
        )

    best_local = objective_best_index(y_arr[keep_idx], objective_sense)
    best_idx = int(keep_idx[int(best_local)])
    return FinalPreBestResult(
        found=True,
        index=best_idx,
        x=X_arr[best_idx].copy(),
        y_raw_internal=float(y_arr[best_idx]),
        status="ok",
        candidate_count=int(finite_idx.size),
        feasible_count=int(keep_idx.size),
        pre_inequality_feasible_count=int(np.sum(pre_ineq_mask & finite_mask)),
        pre_equality_feasible_count=int(np.sum(pre_eq_mask & finite_mask)),
        pre_inequality_active=pre_ineq_active,
        pre_equality_active=pre_eq_active,
        mask=mask,
        best_effort=FinalPreFallbackCandidate.empty(),
        least_violation=FinalPreFallbackCandidate.empty(),
        raw_best_infeasible=FinalPreFallbackCandidate.empty(),
        pareto_candidates=(),
    )
