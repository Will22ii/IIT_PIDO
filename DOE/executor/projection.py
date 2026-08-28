# DOE/executor/projection.py
"""구조적 등식(Type 2) 제약을 위한 매니폴드 투영과 maximin 부분선택.

등식 제약의 feasible 집합은 측도 0인 면이라, 무작위로 뽑아서 맞기를 기다리는
rejection sampling으로는 수율이 지수적으로 붕괴한다. 대신 뽑은 점을 면 위로
"계산해서 밀어 넣는다".

여기서 다루는 것은 scope="pre" 이면서 eps가 지정되지 않은 등식(Type 2)뿐이다.
- eps가 있는 등식(Type 1)은 유저가 선언한 밴드가 진짜 설계공간이므로 투영하면
  그 부피를 버리게 된다. rejection으로 처리한다.
- post 등식은 h 한 번 평가에 CAE 한 번이 필요해 투영이 원리적으로 불가능하다.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from DOE.executor.constraint_filter import (
    equality_residuals,
    equality_tolerances,
)


# 유한차분 스텝. 변수 크기에 비례시키되 0 근처에서는 절대값으로 떨어진다.
_FD_REL_STEP = 1e-6
# 유사역행렬 감쇠. J가 rank 부족일 때 폭발을 막는다.
_DAMPING = 1e-10


def _jacobian(
    *,
    x: np.ndarray,
    var_names: Sequence[str],
    equality_defs: list,
    r0: np.ndarray,
) -> np.ndarray:
    """전진차분 야코비안 (m x p). expr 평가 m*p회."""
    p = int(x.shape[0])
    m = int(r0.shape[0])
    J = np.zeros((m, p), dtype=float)
    for j in range(p):
        step = _FD_REL_STEP * max(abs(float(x[j])), 1.0)
        x_fwd = x.copy()
        x_fwd[j] += step
        r_fwd = equality_residuals(
            x=x_fwd, var_names=var_names, equality_defs=equality_defs
        )
        if not np.all(np.isfinite(r_fwd)):
            return np.zeros((m, p), dtype=float)
        J[:, j] = (r_fwd - r0) / step
    return J


def project_point_to_manifold(
    *,
    x0: np.ndarray,
    var_names: Sequence[str],
    equality_defs: list,
    lb: np.ndarray,
    ub: np.ndarray,
    tol: np.ndarray,
    max_iter: int = 8,
    max_line_search: int = 5,
) -> tuple[np.ndarray, bool]:
    """점 하나를 등식이 정의하는 면 위로 투영한다.

    min ||x - x0||^2  s.t.  h(x) = L,  lb <= x <= ub

    선형화하면 J dx = -r 이고, m < p 이므로 부정계다. 해가 무한히 많은 중에서
    ||dx||가 최소인 것(= 점을 가장 적게 움직이는 보정)을 고른다:

        dx = -J^T (J J^T + lam I)^-1 r

    이 한 식이 등식 m개를 동시에 처리한다. 커플링은 J J^T의 비대각 성분에
    그대로 담기므로, 여러 등식이 같은 변수를 공유해도 문제되지 않는다.

    박스 clip이 h를 다시 깨뜨리므로 잔차가 실제로 줄어드는 스텝만 받아들인다.

    Returns
    -------
    (x, ok) : 투영된 점과 수렴 여부. ok=False면 호출부가 그 점을 버린다.
    """
    x = np.clip(np.asarray(x0, dtype=float).reshape(-1), lb, ub)
    m = len(equality_defs)
    if m == 0:
        return x, True

    r = equality_residuals(x=x, var_names=var_names, equality_defs=equality_defs)
    if not np.all(np.isfinite(r)):
        return x, False

    for _ in range(int(max_iter)):
        if np.all(np.abs(r) <= tol):
            return x, True

        J = _jacobian(x=x, var_names=var_names, equality_defs=equality_defs, r0=r)
        gram = J @ J.T
        if not np.all(np.isfinite(gram)):
            return x, False
        gram = gram + _DAMPING * np.eye(m, dtype=float)
        try:
            dx = -(J.T @ np.linalg.solve(gram, r))
        except np.linalg.LinAlgError:
            return x, False
        if not np.all(np.isfinite(dx)):
            return x, False

        # 선탐색: clip 후에도 잔차가 줄어드는 스텝만 수용한다.
        norm_r = float(np.linalg.norm(r))
        accepted = False
        alpha = 1.0
        for _ls in range(int(max_line_search)):
            x_new = np.clip(x + alpha * dx, lb, ub)
            r_new = equality_residuals(
                x=x_new, var_names=var_names, equality_defs=equality_defs
            )
            if np.all(np.isfinite(r_new)) and float(np.linalg.norm(r_new)) < norm_r:
                x, r = x_new, r_new
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            # 더 이상 전진하지 못한다. 이미 허용치 안이면 성공으로 본다.
            return x, bool(np.all(np.abs(r) <= tol))

    return x, bool(np.all(np.abs(r) <= tol))


def project_points_to_manifold(
    *,
    X: np.ndarray,
    var_names: Sequence[str],
    equality_defs: list,
    bounds: Sequence[tuple[float, float]],
    max_iter: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """점 배열을 면 위로 투영한다.

    Returns
    -------
    (X_projected, ok_mask) : 실패한 점도 자리는 유지하고 mask로만 표시한다.
        호출부가 mask로 걸러내면 원래 인덱스 대응이 보존된다.
    """
    X = np.asarray(X, dtype=float)
    if X.size == 0 or not equality_defs:
        return X, np.ones((X.shape[0],), dtype=bool)

    lb = np.asarray([b[0] for b in bounds], dtype=float)
    ub = np.asarray([b[1] for b in bounds], dtype=float)
    tol = equality_tolerances(equality_defs)

    X_out = np.array(X, dtype=float, copy=True)
    ok = np.zeros((X.shape[0],), dtype=bool)
    for i in range(X.shape[0]):
        X_out[i], ok[i] = project_point_to_manifold(
            x0=X[i],
            var_names=var_names,
            equality_defs=equality_defs,
            lb=lb,
            ub=ub,
            tol=tol,
            max_iter=max_iter,
        )
    return X_out, ok


def maximin_subset(
    *,
    X: np.ndarray,
    k: int,
    bounds: Sequence[tuple[float, float]] | None = None,
    forced_indices: Iterable[int] | None = None,
) -> np.ndarray:
    """서로 가장 멀리 떨어지도록 k개를 탐욕 선택한다.

    투영은 면이 휘는 곳에 점을 뭉치게 만든다. 뭉친 곳에서는 하나만 골라지므로
    최종 표본이 다시 고르게 퍼진다. 투영이 LHS 층화를 훼손하는 것에 대한 대응이다.

    forced_indices는 무조건 포함한다.
    """
    X = np.asarray(X, dtype=float)
    n = int(X.shape[0])
    k = int(min(max(k, 0), n))
    if k <= 0:
        return np.empty((0,), dtype=int)
    if k >= n:
        return np.arange(n, dtype=int)

    if bounds is not None:
        spans = np.asarray([max(float(b[1]) - float(b[0]), 0.0) for b in bounds], dtype=float)
        spans[spans <= 0.0] = 1.0
    else:
        spans = np.ones((X.shape[1],), dtype=float)
    Z = X / spans

    selected: list[int] = []
    for idx in list(forced_indices or []):
        idx = int(idx)
        if 0 <= idx < n and idx not in selected:
            selected.append(idx)
        if len(selected) >= k:
            return np.asarray(selected[:k], dtype=int)

    if not selected:
        # 중심에 가장 가까운 점에서 시작한다. 인덱스 순서 의존을 없앤다.
        center = Z.mean(axis=0)
        selected.append(int(np.argmin(np.linalg.norm(Z - center, axis=1))))

    # 각 후보에서 이미 뽑힌 집합까지의 최소거리. 매 선택마다 증분 갱신한다.
    min_dist = np.full((n,), np.inf, dtype=float)
    for s in selected:
        min_dist = np.minimum(min_dist, np.linalg.norm(Z - Z[s], axis=1))
    min_dist[np.asarray(selected, dtype=int)] = -np.inf

    while len(selected) < k:
        nxt = int(np.argmax(min_dist))
        if not np.isfinite(min_dist[nxt]):
            break
        selected.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(Z - Z[nxt], axis=1))
        min_dist[nxt] = -np.inf

    return np.asarray(selected, dtype=int)
