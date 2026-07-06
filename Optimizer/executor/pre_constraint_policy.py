from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from DOE.executor.constraint_filter import evaluate_constraints_batch
from Optimizer.executor.evaluation_core import (
    build_pre_constraint_debug_fields,
    evaluate_pre_constraints_raw,
)
from utils.bool_mask import to_bool_mask


def pre_inequality_constraint_defs(constraint_defs: list[dict] | None) -> list[dict]:
    """Return only constraints that can be enforced before CAE evaluation."""
    return [
        c
        for c in (constraint_defs or [])
        if isinstance(c, dict)
        and str(c.get("scope", "pre")).strip().lower() == "pre"
        and str(c.get("type", "")).strip() != "=="
    ]


def constraint_eval_array(
    X: np.ndarray,
    *,
    to_full_x: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if to_full_x is None or arr.ndim != 2 or arr.shape[0] == 0:
        return arr
    return np.asarray([to_full_x(row) for row in arr], dtype=float)


@dataclass(frozen=True)
class WarmStartFilterResult:
    df: pd.DataFrame
    mask: np.ndarray
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class PreInequalityPolicy:
    constraint_defs: list[dict]
    var_names: list[str]
    to_full_x: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "constraint_defs",
            pre_inequality_constraint_defs(list(self.constraint_defs or [])),
        )
        object.__setattr__(self, "var_names", list(self.var_names or []))

    @property
    def active(self) -> bool:
        return bool(self.constraint_defs)

    @property
    def constraint_count(self) -> int:
        return int(len(self.constraint_defs))

    def full_x(self, x_selected: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_selected, dtype=float).reshape(-1)
        return self.to_full_x(x_arr) if self.to_full_x is not None else x_arr

    def evaluate(self, x_selected: np.ndarray) -> tuple[dict, bool, float]:
        if not self.active:
            return {}, True, float("inf")
        return evaluate_pre_constraints_raw(
            x=self.full_x(x_selected),
            var_names=self.var_names,
            constraint_defs=self.constraint_defs,
            fail_fast_output_missing=False,
        )

    def evaluate_batch(self, X_selected: np.ndarray) -> tuple[np.ndarray, list[dict], np.ndarray]:
        X_arr = np.asarray(X_selected, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.size == 0:
            return np.empty((0,), dtype=bool), [], np.empty((0,), dtype=float)
        if not self.active:
            return (
                np.ones((X_arr.shape[0],), dtype=bool),
                [{} for _ in range(X_arr.shape[0])],
                np.full((X_arr.shape[0],), float("inf"), dtype=float),
            )
        return evaluate_constraints_batch(
            X=constraint_eval_array(X_arr, to_full_x=self.to_full_x),
            var_names=self.var_names,
            constraint_defs=self.constraint_defs,
            scope="pre",
        )

    def filter_matrix(self, X_selected: np.ndarray) -> tuple[np.ndarray, list[dict], np.ndarray, np.ndarray]:
        X_arr = np.asarray(X_selected, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        mask, payloads, margins = self.evaluate_batch(X_arr)
        idx = np.where(mask)[0]
        if idx.size == 0:
            n_dim = X_arr.shape[1] if X_arr.ndim == 2 else 0
            return np.empty((0, n_dim), dtype=float), [], np.empty((0,), dtype=float), mask
        return X_arr[idx], [payloads[i] for i in idx], margins[idx], mask

    def filter_pool(
        self,
        pool: np.ndarray,
        labels: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str], np.ndarray, dict[str, object]]:
        pool_arr = np.asarray(pool, dtype=float)
        n_raw = int(pool_arr.shape[0]) if pool_arr.ndim == 2 else 0
        if pool_arr.ndim != 2 or n_raw == 0:
            return (
                np.empty((0, 0), dtype=float) if pool_arr.ndim != 2 else pool_arr,
                [],
                np.empty((0,), dtype=float),
                {
                    "active": bool(self.active),
                    "raw_count": 0,
                    "feasible_count": 0,
                    "feasible_ratio": float("nan"),
                    "margin_min": float("nan"),
                    "margin_median": float("nan"),
                    "margin_max": float("nan"),
                },
            )

        label_list = list(labels or ["unknown"] * n_raw)
        if len(label_list) < n_raw:
            label_list.extend(["unknown"] * (n_raw - len(label_list)))
        label_list = label_list[:n_raw]
        if not self.active:
            return (
                pool_arr,
                label_list,
                np.full((n_raw,), float("inf"), dtype=float),
                {
                    "active": False,
                    "raw_count": n_raw,
                    "feasible_count": n_raw,
                    "feasible_ratio": 1.0,
                    "margin_min": float("inf"),
                    "margin_median": float("inf"),
                    "margin_max": float("inf"),
                },
            )

        mask, _payloads, margins = self.evaluate_batch(pool_arr)
        keep_idx = np.where(mask)[0]
        finite_margins = margins[np.isfinite(margins)]
        stats = {
            "active": True,
            "raw_count": n_raw,
            "feasible_count": int(keep_idx.size),
            "feasible_ratio": float(keep_idx.size) / float(max(n_raw, 1)),
            "margin_min": float(np.min(finite_margins)) if finite_margins.size else float("nan"),
            "margin_median": float(np.median(finite_margins)) if finite_margins.size else float("nan"),
            "margin_max": float(np.max(finite_margins)) if finite_margins.size else float("nan"),
        }
        return (
            pool_arr[keep_idx],
            [label_list[int(i)] for i in keep_idx.tolist()],
            margins[keep_idx],
            stats,
        )

    def sample_feasible_candidate(
        self,
        *,
        x: np.ndarray,
        rng: np.random.Generator,
        lb: np.ndarray,
        ub: np.ndarray,
        max_attempts: int = 128,
    ) -> np.ndarray:
        if not self.active:
            return np.asarray(x, dtype=float).reshape(-1)

        x_cur = np.clip(np.asarray(x, dtype=float).reshape(-1), lb, ub)
        best_x = x_cur.copy()
        best_margin = float("-inf")
        for _ in range(max(int(max_attempts), 1)):
            _payload, feasible, margin = self.evaluate(x_cur)
            if bool(feasible):
                return x_cur
            if float(margin) > float(best_margin):
                best_margin = float(margin)
                best_x = x_cur.copy()
            x_cur = rng.uniform(lb, ub, size=(lb.shape[0],))
        return best_x

    def debug_fields(self, x_selected: np.ndarray) -> dict[str, object]:
        return build_pre_constraint_debug_fields(
            x=self.full_x(x_selected),
            var_names=self.var_names,
            constraint_defs=self.constraint_defs,
            enforce_pre_constraints=self.active,
        )

    def filter_warm_start_df(
        self,
        df: pd.DataFrame,
        *,
        selected_features: list[str],
        warn_prefix: str,
    ) -> WarmStartFilterResult:
        work = df.copy()
        n_rows = int(len(work))
        mask = np.ones((n_rows,), dtype=bool)
        feasibility_col = ""

        if "success" in work.columns:
            mask &= to_bool_mask(
                work["success"],
                column_name="success",
                warn_prefix=warn_prefix,
            )
        if "feasible_pre" in work.columns:
            feasibility_col = "feasible_pre"
            mask &= to_bool_mask(
                work["feasible_pre"],
                column_name="feasible_pre",
                warn_prefix=warn_prefix,
            )
        elif "feasible" in work.columns:
            feasibility_col = "feasible"
            mask &= to_bool_mask(
                work["feasible"],
                column_name="feasible",
                warn_prefix=warn_prefix,
            )

        missing_features = [f for f in selected_features if f not in work.columns]
        if missing_features:
            finite_mask = np.zeros((n_rows,), dtype=bool)
            X_all = np.empty((n_rows, 0), dtype=float)
        else:
            try:
                X_all = work[selected_features].to_numpy(dtype=float)
                finite_mask = np.isfinite(X_all).all(axis=1)
            except Exception:
                X_all = np.empty((n_rows, len(selected_features)), dtype=float)
                finite_mask = np.zeros((n_rows,), dtype=bool)
        mask &= finite_mask

        pre_feasible_count = int(np.sum(finite_mask))
        if self.active:
            pre_mask = np.zeros((n_rows,), dtype=bool)
            finite_idx = np.where(finite_mask)[0]
            if finite_idx.size > 0:
                try:
                    mask_pre, _payloads, _margins = self.evaluate_batch(X_all[finite_idx])
                    pre_mask[finite_idx] = mask_pre
                except Exception:
                    pre_mask[finite_idx] = False
            pre_feasible_count = int(np.sum(pre_mask))
            mask &= pre_mask

        diagnostics = {
            "raw_count": n_rows,
            "kept_count": int(np.sum(mask)),
            "success_col_used": bool("success" in work.columns),
            "feasibility_col_used": feasibility_col,
            "finite_count": int(np.sum(finite_mask)),
            "missing_features": missing_features,
            "pre_hard_filter_active": bool(self.active),
            "pre_hard_filter_constraint_count": int(self.constraint_count),
            "pre_hard_feasible_count": int(pre_feasible_count),
        }
        return WarmStartFilterResult(
            df=work.loc[mask].reset_index(drop=True),
            mask=mask,
            diagnostics=diagnostics,
        )
