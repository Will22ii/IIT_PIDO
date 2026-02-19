import pandas as pd
from dataclasses import dataclass
from typing import Dict


# =================================================
# Config
# =================================================

@dataclass
class FeatureSelectionConfig:
    perm_min_pass_rate: float = 0.9
    perm_epsilon: float = 0.1


# =================================================
# Selector
# =================================================

class FeatureSelector:
    """
    Feature selection based on permutation effect.
    """

    def __init__(self, config: FeatureSelectionConfig):
        self.config = config

    # -------------------------------------------------
    # Public API (file-based)
    # -------------------------------------------------

    def run(
        self,
        *,
        perm_effect_path: str,
        problem_name: str,
    ) -> Dict[str, pd.DataFrame]:
        perm_raw = pd.read_csv(perm_effect_path)
        processed, selected = self._select_perm_effect(perm_raw)

        return {
            "importance_processed": processed,
            "selected_features": selected,
        }

    # =================================================
    # Internal logic
    # =================================================

    def _select_perm_effect(
        self,
        perm_raw: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Version: permutation effect (mean (pred - pred_perm)^2), normalized by mean.
        """
        fold_pass = perm_raw.rename(columns={"delta": "perm_delta"}).copy()

        stats = (
            fold_pass.groupby("feature")["perm_delta"]
            .agg(delta_mean="mean", delta_min="min", delta_max="max")
            .reset_index()
        )

        global_max = float(stats["delta_mean"].max()) if len(stats) > 0 else 0.0
        if global_max == 0.0:
            global_max = 1.0

        stats["delta_mean_norm"] = stats["delta_mean"] / global_max

        fold_pass["perm_delta_norm"] = fold_pass["perm_delta"] / global_max
        fold_pass["pass"] = fold_pass["perm_delta_norm"] >= self.config.perm_epsilon

        selection_rate = (
            fold_pass.groupby("feature")["pass"]
            .mean()
            .reset_index(name="selection_rate")
        )

        summary = selection_rate.merge(stats, on="feature")
        summary["perm_epsilon"] = self.config.perm_epsilon
        summary["selected"] = (
            summary["selection_rate"] >= self.config.perm_min_pass_rate
        )
        summary["reason"] = summary["selected"].apply(
            lambda x: "perm_pass" if x else "perm_fail"
        )

        processed = fold_pass[
            ["feature", "fold", "perm_delta", "perm_delta_norm", "pass"]
        ].copy()

        selected = summary[
            [
                "feature",
                "selection_rate",
                "delta_mean",
                "delta_mean_norm",
                "delta_min",
                "delta_max",
                "perm_epsilon",
                "selected",
                "reason",
            ]
        ].sort_values("selected", ascending=False)

        return processed, selected
