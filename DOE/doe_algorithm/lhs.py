# DOE/doe_algorithm/lhs.py

import numpy as np
from typing import Optional, Sequence, Tuple


def latin_hypercube_sampling(
    *,
    n_samples: int,
    bounds: Sequence[Tuple[float, float]],
    n_divisions: int | None = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Latin Hypercube Sampling (LHC)

    Parameters
    ----------
    n_samples : int
        생성할 샘플 개수
    bounds : list of (lb, ub)
        각 설계변수의 하한/상한
    n_divisions : int or None
        각 축을 몇 구간으로 나눌지 (None이면 n_samples)
    rng : np.random.Generator or None
        재현성 제어용 RNG. None이면 default_rng() 사용

    Returns
    -------
    X : np.ndarray (n_samples, n_dim)
        LHC로 생성된 설계변수 샘플
    """

    if rng is None:
        rng = np.random.default_rng()

    n_dim = len(bounds)

    if n_divisions is None:
        n_divisions = n_samples

    X = np.zeros((n_samples, n_dim), dtype=float)

    for j in range(n_dim):
        divisions = np.linspace(0.0, 1.0, n_divisions + 1)

        perm = rng.permutation(n_divisions)[:n_samples]
        H = divisions[perm] + rng.random(n_samples) / n_divisions

        lb, ub = bounds[j]
        X[:, j] = lb + H * (ub - lb)

    return X
