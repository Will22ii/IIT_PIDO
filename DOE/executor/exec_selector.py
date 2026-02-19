# DOE/executor/exec_selector.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ExecSelectResult:
    X_exec: np.ndarray
    filled: bool  # True if len(X_exec) == n_exec


class ExecSelector:
    """
    Select X_exec from X_plan.

    Policy (per your requirements):
    - Remove already executed points (L2 distance > dedup_tol)
    - Pick a RANDOM subset (no "take first n")
    - If cannot fill n_exec after dedup -> filled=False
    """

    def _dedup_against_history(
        self,
        *,
        X_new: np.ndarray,
        X_old: np.ndarray,
        dedup_tol: float,
    ) -> np.ndarray:
        if X_new.size == 0:
            return X_new
        if X_old.size == 0:
            return X_new

        keep = []
        for x in X_new:
            dists = np.linalg.norm(X_old - x, axis=1)
            if np.all(dists > dedup_tol):
                keep.append(x)

        if not keep:
            return np.empty((0, X_new.shape[1]), dtype=float)
        return np.vstack(keep).astype(float)

    def select(
        self,
        *,
        X_plan: np.ndarray,
        X_executed: np.ndarray,
        n_exec: int,
        rng: np.random.Generator,
        dedup_tol: float,
    ) -> ExecSelectResult:
        X_plan = np.asarray(X_plan, dtype=float)
        X_executed = np.asarray(X_executed, dtype=float)

        if n_exec <= 0:
            return ExecSelectResult(X_exec=np.empty((0, X_plan.shape[1]), dtype=float), filled=True)

        X_unique = self._dedup_against_history(
            X_new=X_plan,
            X_old=X_executed,
            dedup_tol=float(dedup_tol),
        )

        if X_unique.shape[0] < n_exec:
            # not enough unique points to fill required execution batch
            # return what we have but mark filled=False
            return ExecSelectResult(X_exec=X_unique, filled=False)

        idx = rng.choice(X_unique.shape[0], size=n_exec, replace=False)
        X_exec = X_unique[idx]
        return ExecSelectResult(X_exec=X_exec, filled=True)
