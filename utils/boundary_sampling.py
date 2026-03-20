from typing import Tuple

import numpy as np


def _compute_boundary_low_high(
    bounds: list[Tuple[float, float]],
    *,
    offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dims = len(bounds)
    lows = np.empty((dims,), dtype=float)
    highs = np.empty((dims,), dtype=float)
    for i, ((lb, ub), off) in enumerate(zip(bounds, offset)):
        low = lb + off
        high = ub - off
        if low > high:
            mid = (lb + ub) / 2.0
            low = mid
            high = mid
        lows[i] = float(low)
        highs[i] = float(high)
    return lows, highs


def sample_boundary_corners(
    bounds: list[Tuple[float, float]],
    *,
    offset: np.ndarray,
) -> np.ndarray:
    dims = len(bounds)
    if dims <= 0:
        return np.empty((0, 0), dtype=float)
    lows, highs = _compute_boundary_low_high(bounds, offset=offset)
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
    n_boundary_dims_max = max(1, min(n_boundary_dims, dims))
    pts = []
    for _ in range(n_samples):
        base = base_points[rng.integers(0, base_points.shape[0])].copy()
        if n_boundary_dims_max == 1:
            n_fix = 1
        else:
            n_fix = int(rng.integers(1, n_boundary_dims_max + 1))
        idxs = rng.choice(dims, size=n_fix, replace=False)
        for j in idxs:
            lb, ub = bounds[j]
            low = lb + offset[j]
            high = ub - offset[j]
            if low > high:
                low = high = (lb + ub) / 2.0
            base[j] = low if rng.random() < 0.5 else high
        pts.append(base)
    return np.vstack(pts).astype(float)


def sample_boundary_corners_random(
    bounds: list[Tuple[float, float]],
    *,
    offset: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    max_attempt_factor: int = 64,
) -> np.ndarray:
    dims = len(bounds)
    if dims <= 0 or n_samples <= 0:
        return np.empty((0, max(dims, 0)), dtype=float)

    lows, highs = _compute_boundary_low_high(bounds, offset=offset)
    diff = np.abs(highs - lows)
    flex_idx = np.where(diff > 0.0)[0]
    n_flex = int(flex_idx.size)

    if n_flex == 0:
        return lows.reshape(1, -1).astype(float)

    max_unique = 1 << n_flex
    n_target = min(int(n_samples), int(max_unique))
    if n_target <= 0:
        return np.empty((0, dims), dtype=float)

    max_attempts = max(int(n_target * max_attempt_factor), int(n_target + 16))
    seen: set[bytes] = set()
    pts: list[np.ndarray] = []
    attempts = 0
    while len(pts) < n_target and attempts < max_attempts:
        attempts += 1
        bits = rng.integers(0, 2, size=n_flex, dtype=np.uint8)
        key = bits.tobytes()
        if key in seen:
            continue
        seen.add(key)
        p = lows.copy()
        p[flex_idx] = np.where(bits == 1, highs[flex_idx], lows[flex_idx])
        pts.append(p)

    if not pts:
        return np.empty((0, dims), dtype=float)
    return np.vstack(pts).astype(float)
