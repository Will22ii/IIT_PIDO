from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Optimizer.executor.evaluation_core import objective_is_better


@dataclass(frozen=True)
class BestUpdateResult:
    best_x: np.ndarray
    best_y: float
    improved: bool


@dataclass(frozen=True)
class DualBestUpdateResult:
    best_x_raw: np.ndarray
    best_y_raw: float
    best_x_effective: np.ndarray
    best_y_effective: float
    raw_improved: bool
    effective_improved: bool


def append_archive_point(
    *,
    X_archive: np.ndarray,
    y_archive: np.ndarray,
    x_next: np.ndarray,
    y_next: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_next, dtype=float).reshape(1, -1)
    y = float(y_next)
    if X_archive.size == 0:
        X_new = x.copy()
    else:
        X_new = np.vstack([np.asarray(X_archive, dtype=float), x])
    y_new = np.append(np.asarray(y_archive, dtype=float).reshape(-1), y)
    return X_new, y_new


def update_best_point(
    *,
    best_x: np.ndarray,
    best_y: float,
    x_next: np.ndarray,
    y_next: float,
    objective_sense: str,
) -> BestUpdateResult:
    improved = objective_is_better(
        y_new=float(y_next),
        y_best=float(best_y),
        objective_sense=objective_sense,
    )
    if improved:
        return BestUpdateResult(
            best_x=np.asarray(x_next, dtype=float).reshape(-1).copy(),
            best_y=float(y_next),
            improved=True,
        )
    return BestUpdateResult(
        best_x=np.asarray(best_x, dtype=float).reshape(-1).copy(),
        best_y=float(best_y),
        improved=False,
    )


def update_raw_and_effective_best(
    *,
    best_x_raw: np.ndarray,
    best_y_raw: float,
    best_x_effective: np.ndarray,
    best_y_effective: float,
    x_next: np.ndarray,
    y_raw_next: float,
    y_effective_next: float,
    objective_sense: str,
) -> DualBestUpdateResult:
    raw = update_best_point(
        best_x=best_x_raw,
        best_y=float(best_y_raw),
        x_next=x_next,
        y_next=float(y_raw_next),
        objective_sense=objective_sense,
    )
    effective = update_best_point(
        best_x=best_x_effective,
        best_y=float(best_y_effective),
        x_next=x_next,
        y_next=float(y_effective_next),
        objective_sense=objective_sense,
    )
    return DualBestUpdateResult(
        best_x_raw=raw.best_x,
        best_y_raw=float(raw.best_y),
        best_x_effective=effective.best_x,
        best_y_effective=float(effective.best_y),
        raw_improved=bool(raw.improved),
        effective_improved=bool(effective.improved),
    )
