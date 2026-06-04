from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_objective_sense(objective_sense: str | None) -> str:
    sense = str(objective_sense or "min").strip().lower()
    if sense not in {"min", "max"}:
        raise ValueError("objective_sense must be 'min' or 'max'.")
    return sense


def objective_to_minimize(value: float | np.ndarray, objective_sense: str | None):
    """Map a user-facing objective onto the canonical minimization score."""

    arr = np.asarray(value, dtype=float)
    if normalize_objective_sense(objective_sense) == "max":
        arr = -arr
    if np.isscalar(value):
        return float(arr.reshape(-1)[0])
    return arr


def objective_from_minimize(value: float | np.ndarray, objective_sense: str | None):
    """Map a canonical minimization score back to the user-facing objective."""

    arr = np.asarray(value, dtype=float)
    if normalize_objective_sense(objective_sense) == "max":
        arr = -arr
    if np.isscalar(value):
        return float(arr.reshape(-1)[0])
    return arr


def canonical_objective_for_result(
    *,
    raw_objective: float,
    objective_sense: str | None,
    success: bool,
) -> float:
    """Return the internal min score, keeping failed evaluations unattractive."""

    raw = float(raw_objective)
    if not success or not np.isfinite(raw):
        return float("inf")
    return float(objective_to_minimize(raw, objective_sense))


def _success_mask(values: pd.Series) -> np.ndarray:
    normalized = values.astype(str).str.strip().str.lower()
    false_tokens = {
        "false",
        "f",
        "0",
        "0.0",
        "no",
        "n",
        "fail",
        "failed",
        "nan",
        "none",
        "",
    }
    mask = ~normalized.isin(false_tokens).to_numpy(dtype=bool)
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_arr = numeric.to_numpy(dtype=float)
    numeric_known = np.isfinite(numeric_arr)
    mask[numeric_known] = numeric_arr[numeric_known] != 0.0
    return mask


def canonicalize_objective_columns(
    df: pd.DataFrame,
    *,
    objective_sense: str | None,
    objective_col: str = "objective",
    raw_col: str = "objective_raw",
    success_col: str = "success",
) -> pd.DataFrame:
    """Return a dataframe whose objective column is a minimization score.

    External CSV policy:
    - If objective_raw exists, it is the source of truth.
    - If only objective exists, treat it as legacy raw objective.
    - Failed/non-finite rows get internal objective=inf regardless of sense.
    """

    if not isinstance(df, pd.DataFrame):
        return df
    out = df.copy()
    if raw_col in out.columns:
        raw = pd.to_numeric(out[raw_col], errors="coerce")
    elif objective_col in out.columns:
        raw = pd.to_numeric(out[objective_col], errors="coerce")
        out[raw_col] = raw
    else:
        return out

    success = np.ones((len(out),), dtype=bool)
    if success_col in out.columns:
        success = _success_mask(out[success_col])

    raw_arr = raw.to_numpy(dtype=float)
    internal = np.full(raw_arr.shape, float("inf"), dtype=float)
    valid = success & np.isfinite(raw_arr)
    if np.any(valid):
        internal[valid] = np.asarray(
            objective_to_minimize(raw_arr[valid], objective_sense),
            dtype=float,
        ).reshape(-1)
    out[objective_col] = internal
    return out
