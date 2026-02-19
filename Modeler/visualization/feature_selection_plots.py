import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_perm_effect(
    *,
    selected_df: pd.DataFrame,
    output_path: str,
    perm_epsilon: float | None = None,
    title: str | None = None,
):
    df = selected_df.sort_values("delta_mean_norm", ascending=False).copy()
    colors = np.where(df["selected"], "#1f77b4", "#d62728")

    plt.figure(figsize=(max(8, len(df) * 0.4), 4))
    x = np.arange(len(df))
    plt.bar(x, df["delta_mean_norm"], color=colors)

    global_max = float(df["delta_mean"].max()) if len(df) > 0 else 1.0
    yerr = np.vstack(
        [
            df["delta_mean_norm"] - (df["delta_min"] / global_max),
            (df["delta_max"] / global_max) - df["delta_mean_norm"],
        ]
    )
    plt.errorbar(
        x,
        df["delta_mean_norm"],
        yerr=yerr,
        fmt="none",
        ecolor="black",
        elinewidth=1,
        capsize=3,
        alpha=0.7,
    )
    if perm_epsilon is not None:
        plt.axhline(
            perm_epsilon,
            color="#444444",
            linestyle="--",
            linewidth=1.4,
            label=f"perm_epsilon={perm_epsilon:.3f}",
        )
        plt.legend(loc="upper right", frameon=False)

    plt.xticks(x, df["feature"], rotation=45, ha="right")
    plt.ylabel("mean((pred - pred_perm)^2) / max(all)")
    plt.ylim(0.0, 1.05)
    if title:
        plt.title(title)
    else:
        plt.title("Permutation Effect (delta mean sq, normalized to max)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
