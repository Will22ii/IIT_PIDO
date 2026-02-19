from typing import Tuple

import numpy as np


def sample_boundary_corners(
    bounds: list[Tuple[float, float]],
    *,
    offset: np.ndarray,
) -> np.ndarray:
    dims = len(bounds)
    if dims <= 0:
        return np.empty((0, 0), dtype=float)
    lows = []
    highs = []
    for (lb, ub), off in zip(bounds, offset):
        low = lb + off
        high = ub - off
        if low > high:
            mid = (lb + ub) / 2.0
            low = mid
            high = mid
        lows.append(low)
        highs.append(high)
    choices = [np.array([lows[i], highs[i]], dtype=float) for i in range(dims)]
    grids = np.meshgrid(*choices, indexing="ij")
    pts = np.stack([g.reshape(-1) for g in grids], axis=1)
    return pts.astype(float)


def sample_boundary_partial(
    bounds: list[Tuple[float, float]],
    *,
    offset: np.ndarray,
    base_points: np.ndarray,
    n_samples: int,
    n_boundary_dims: int = 2,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_samples <= 0 or base_points.size == 0:
        return np.empty((0, len(bounds)), dtype=float)
    dims = len(bounds)
    n_boundary_dims = max(1, min(n_boundary_dims, dims))
    pts = []
    for _ in range(n_samples):
        base = base_points[rng.integers(0, base_points.shape[0])].copy()
        idxs = rng.choice(dims, size=n_boundary_dims, replace=False)
        for j in idxs:
            lb, ub = bounds[j]
            low = lb + offset[j]
            high = ub - offset[j]
            if low > high:
                low = high = (lb + ub) / 2.0
            base[j] = low if rng.random() < 0.5 else high
        pts.append(base)
    return np.vstack(pts).astype(float)
