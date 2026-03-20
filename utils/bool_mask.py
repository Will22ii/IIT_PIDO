from __future__ import annotations

import numpy as np
import pandas as pd


_TRUE_TOKENS = {"true", "1", "y", "yes", "t"}
_FALSE_TOKENS = {"false", "0", "n", "no", "f", "", "none", "null", "na", "nan"}


def to_bool_mask(
    series: pd.Series,
    *,
    column_name: str | None = None,
    warn_prefix: str = "[BoolParse]",
) -> np.ndarray:
    """
    Convert a pandas Series to a bool mask with robust numeric/string handling.

    Rules:
    - bool dtype: keep as-is (NaN -> False)
    - numeric dtype: 0 -> False, non-zero -> True (NaN -> False)
    - string/object dtype:
      - known true/false tokens are parsed directly
      - numeric-like strings (e.g. "1.0", "-2", "0.0") are parsed as numbers
      - unknown tokens are treated as False and logged once
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)

    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        return values.fillna(0.0).ne(0.0).to_numpy(dtype=bool)

    normalized = series.astype("string").str.strip().str.lower()
    is_na = normalized.isna()

    out = pd.Series(False, index=series.index, dtype=bool)
    is_true = normalized.isin(_TRUE_TOKENS)
    is_false = normalized.isin(_FALSE_TOKENS)
    out.loc[is_true] = True

    unresolved = ~(is_na | is_true | is_false)
    if bool(unresolved.any()):
        unresolved_vals = normalized.loc[unresolved]
        parsed = pd.to_numeric(unresolved_vals, errors="coerce")
        parsed_ok = parsed.notna()
        if bool(parsed_ok.any()):
            idx_ok = parsed.index[parsed_ok]
            out.loc[idx_ok] = parsed.loc[idx_ok].ne(0.0).to_numpy(dtype=bool)

        unresolved_final = parsed.index[~parsed_ok]
        if len(unresolved_final) > 0:
            examples = (
                normalized.loc[unresolved_final]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .head(5)
                .tolist()
            )
            col = str(column_name or getattr(series, "name", None) or "<unknown>")
            print(
                f"{warn_prefix} column='{col}' "
                f"unrecognized_tokens={len(unresolved_final)} "
                f"examples={examples} -> treated as False"
            )

    return out.to_numpy(dtype=bool)

