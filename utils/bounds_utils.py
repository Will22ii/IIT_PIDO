from typing import Sequence, Tuple

import numpy as np


def compute_spans_lbs(
    bounds: Sequence[Tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    spans = np.array([ub - lb for lb, ub in bounds], dtype=float)
    spans = np.where(spans == 0.0, 1.0, spans)
    lbs = np.array([lb for lb, _ in bounds], dtype=float)
    return spans, lbs


def spans_only(
    bounds: Sequence[Tuple[float, float]],
) -> np.ndarray:
    spans, _ = compute_spans_lbs(bounds)
    return spans


def bounds_to_array(
    bounds: Sequence[Tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    lbs = np.array([lb for lb, _ in bounds], dtype=float)
    ubs = np.array([ub for _, ub in bounds], dtype=float)
    return lbs, ubs


def normalize_with_bounds(
    X: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
) -> np.ndarray:
    spans, lbs = compute_spans_lbs(bounds)
    return (X - lbs) / spans


def clamp_to_bounds(
    X: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
) -> np.ndarray:
    X = np.asarray(X, dtype=float).copy()
    for j, (lb, ub) in enumerate(bounds):
        X[:, j] = np.clip(X[:, j], lb, ub)
    return X
