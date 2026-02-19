# doe/doe_algorithm/random.py

import numpy as np
from typing import Optional


def random_sampling(
    *,
    n_samples: int,
    bounds: list[tuple[float, float]],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Random Sampling within given bounds

    Parameters
    ----------
    n_samples : int
        Number of samples
    bounds : list of (lb, ub)
        Variable bounds
    rng : np.random.Generator or None
        Random number generator (for reproducibility)

    Returns
    -------
    X : ndarray, shape (n_samples, n_dim)
        Randomly sampled design points
    """
    if rng is None:
        rng = np.random.default_rng()

    n_dim = len(bounds)
    X = np.zeros((n_samples, n_dim), dtype=float)

    for j, (lb, ub) in enumerate(bounds):
        X[:, j] = rng.uniform(lb, ub, size=n_samples)

    return X
