# DOE/executor/dataset_store.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class StoredRow:
    x: np.ndarray
    objective: float
    constraints: Optional[Dict]
    feasible_pre: bool
    feasible_post: bool
    feasible_final: bool
    success: bool
    margin_pre: float
    margin_post: float
    source: Optional[str] = None          
    round_idx: Optional[int] = None       
    exec_scope: Optional[str] = None      


class DatasetStore:
    """
    Store for executed DOE results.

    Responsibilities:
    - Accumulate executed CAE results
    - Provide X / y access
    - Convert to DataFrame for surrogate training
    """

    def __init__(self, *, dim: int):
        self.dim = int(dim)
        self._rows: List[StoredRow] = []
    
    @property
    def successful_rows(self) -> List[StoredRow]:
        return [r for r in self._rows if r.success]
    
    @property
    def X_success(self) -> np.ndarray:
        rows = self.successful_rows
        if not rows:
            return np.empty((0, self.dim), dtype=float)
        return np.vstack([r.x.reshape(1, -1) for r in rows]).astype(float)

    @property
    def y_success(self) -> np.ndarray:
        rows = self.successful_rows
        if not rows:
            return np.empty((0,), dtype=float)
        return np.asarray([r.objective for r in rows], dtype=float)
    
    @property
    def rows(self) -> List[StoredRow]:
        return self._rows



    @property
    def size(self) -> int:
        return len(self._rows)

    @property
    def X(self) -> np.ndarray:
        if self.size == 0:
            return np.empty((0, self.dim), dtype=float)
        return np.vstack([r.x.reshape(1, -1) for r in self._rows]).astype(float)

    @property
    def y(self) -> np.ndarray:
        if self.size == 0:
            return np.empty((0,), dtype=float)
        return np.asarray([r.objective for r in self._rows], dtype=float)

    @property
    def constraints(self) -> List[Optional[Dict]]:
        return [r.constraints for r in self._rows]

    @property
    def feasible(self) -> np.ndarray:
        if self.size == 0:
            return np.empty((0,), dtype=bool)
        return np.asarray([r.feasible_final for r in self._rows], dtype=bool)

    @property
    def success(self) -> np.ndarray:
        if self.size == 0:
            return np.empty((0,), dtype=bool)
        return np.asarray([r.success for r in self._rows], dtype=bool)
    
    @property
    def constraint_margin(self) -> np.ndarray:
        if self.size == 0:
            return np.empty((0,), dtype=float)
        out = []
        for r in self._rows:
            vals = [r.margin_pre, r.margin_post]
            vals = [v for v in vals if np.isfinite(v)]
            out.append(min(vals) if vals else float("inf"))
        return np.asarray(out, dtype=float)


    def add(
        self,
        *,
        x: np.ndarray,
        objective: float,
        constraints: Optional[Dict],
        feasible_pre: bool,
        feasible_post: bool,
        feasible_final: bool,
        success: bool,
        margin_pre: float,
        margin_post: float,
        source: Optional[str] = None,       
        round_idx: Optional[int] = None,     
        exec_scope: Optional[str] = None,    

    ) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.dim:
            raise ValueError(f"Dimension mismatch: expected {self.dim}, got {x.shape[0]}")
        self._rows.append(
            StoredRow(
                x=x,
                objective=float(objective),
                constraints=constraints,
                feasible_pre=bool(feasible_pre),
                feasible_post=bool(feasible_post),
                feasible_final=bool(feasible_final),
                success=bool(success),
                margin_pre=float(margin_pre),
                margin_post=float(margin_post),
                source=source,                   
                round_idx=round_idx,
                exec_scope=exec_scope,
            )
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert successful executions to DataFrame (for surrogate training).
        """
        rows = self.successful_rows
        if not rows:
            return pd.DataFrame()

        data = {
            f"x_{j}": [r.x[j] for r in rows]
            for j in range(self.dim)
        }
        data["objective"] = [r.objective for r in rows]
        data["feasible_pre"] = [r.feasible_pre for r in rows]
        data["feasible_post"] = [r.feasible_post for r in rows]
        data["feasible_final"] = [r.feasible_final for r in rows]
        data["feasible"] = [r.feasible_final for r in rows]
        data["margin_pre"] = [r.margin_pre for r in rows]
        data["margin_post"] = [r.margin_post for r in rows]
        data["source"] = [r.source for r in rows]
        data["round"] = [r.round_idx for r in rows]
        data["exec_scope"] = [r.exec_scope for r in rows]

        return pd.DataFrame(data)

