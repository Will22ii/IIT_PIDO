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


def plot_drop_effect(
    *,
    selected_df: pd.DataFrame,
    output_path: str,
    drop_epsilon: float | None = None,
    metric_col: str = "drop_metric_mean",
    title: str | None = None,
):
    if selected_df is None or selected_df.empty or metric_col not in selected_df.columns:
        return
    df = selected_df.sort_values(metric_col, ascending=False).copy()
    colors = np.where(df["selected"], "#1f77b4", "#d62728")

    plt.figure(figsize=(max(8, len(df) * 0.4), 4))
    x = np.arange(len(df))
    vals = pd.to_numeric(df[metric_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    plt.bar(x, vals, color=colors)

    if {"drop_metric_min", "drop_metric_max"}.issubset(df.columns):
        vmin = pd.to_numeric(df["drop_metric_min"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        vmax = pd.to_numeric(df["drop_metric_max"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        yerr = np.vstack([vals - vmin, vmax - vals])
        yerr = np.maximum(yerr, 0.0)
        plt.errorbar(
            x,
            vals,
            yerr=yerr,
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=3,
            alpha=0.7,
        )

    if drop_epsilon is not None:
        plt.axhline(
            float(drop_epsilon),
            color="#444444",
            linestyle="--",
            linewidth=1.4,
            label=f"drop_epsilon={float(drop_epsilon):.6f}",
        )
        plt.legend(loc="upper right", frameon=False)

    plt.xticks(x, df["feature"], rotation=45, ha="right")
    plt.ylabel(metric_col)
    if title:
        plt.title(title)
    else:
        plt.title("Score Drop Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_perm_drop_compare(
    *,
    selected_df: pd.DataFrame,
    output_path: str,
    title: str | None = None,
):
    if selected_df is None or selected_df.empty:
        return
    req = {"delta_mean_norm", "drop_metric_mean"}
    if not req.issubset(selected_df.columns):
        return

    df = selected_df.sort_values("selected", ascending=False).copy()
    x = np.arange(len(df))
    w = 0.38

    perm_vals = pd.to_numeric(df["delta_mean_norm"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    drop_vals = pd.to_numeric(df["drop_metric_mean"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    drop_max = float(np.max(drop_vals)) if len(drop_vals) > 0 else 0.0
    if drop_max <= 0.0:
        drop_norm = np.zeros_like(drop_vals)
    else:
        drop_norm = drop_vals / drop_max

    plt.figure(figsize=(max(9, len(df) * 0.5), 4.5))
    plt.bar(x - w / 2, perm_vals, width=w, color="#1f77b4", alpha=0.9, label="perm_delta_norm")
    plt.bar(x + w / 2, drop_norm, width=w, color="#2ca02c", alpha=0.9, label="drop_metric_norm")

    selected_mask = df["selected"].astype(bool).to_numpy()
    if np.any(selected_mask):
        sel_x = x[selected_mask]
        plt.scatter(sel_x, np.full_like(sel_x, 1.03, dtype=float), marker="*", s=80, color="black", label="final_selected")

    plt.xticks(x, df["feature"], rotation=45, ha="right")
    plt.ylim(0.0, 1.08)
    plt.ylabel("normalized importance")
    if title:
        plt.title(title)
    else:
        plt.title("Permutation vs Score-Drop (normalized)")
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
