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
    if selected_df is None or selected_df.empty:
        return
    required = {"feature", "selected", "delta_mean_norm", "delta_mean", "delta_min", "delta_max"}
    if not required.issubset(selected_df.columns):
        return

    df = selected_df.copy()
    df["delta_mean_norm"] = pd.to_numeric(df["delta_mean_norm"], errors="coerce")
    df["delta_mean"] = pd.to_numeric(df["delta_mean"], errors="coerce")
    df["delta_min"] = pd.to_numeric(df["delta_min"], errors="coerce")
    df["delta_max"] = pd.to_numeric(df["delta_max"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["feature", "delta_mean_norm"]).copy()
    if df.empty:
        return
    df = df.sort_values("delta_mean_norm", ascending=False).copy()
    colors = np.where(df["selected"].astype(bool), "#1f77b4", "#d62728")

    plt.figure(figsize=(max(8, len(df) * 0.4), 4))
    x = np.arange(len(df))
    plt.bar(x, df["delta_mean_norm"], color=colors)

    global_max = float(pd.to_numeric(df["delta_mean"], errors="coerce").max()) if len(df) > 0 else 1.0
    if not np.isfinite(global_max) or global_max <= 0.0:
        global_max = 1.0
    mean_norm = pd.to_numeric(df["delta_mean_norm"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    dmin = pd.to_numeric(df["delta_min"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / global_max
    dmax = pd.to_numeric(df["delta_max"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / global_max
    lower = np.maximum(mean_norm - dmin, 0.0)
    upper = np.maximum(dmax - mean_norm, 0.0)
    yerr = np.vstack([lower, upper])
    plt.errorbar(
        x,
        mean_norm,
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


def plot_secondary_selection(
    *,
    diagnostics_df: pd.DataFrame,
    output_path: str,
    title: str | None = None,
):
    if diagnostics_df is None or diagnostics_df.empty:
        return

    df = diagnostics_df.copy()
    has_delta_r2_mode = {"mean_delta_r2", "freq", "passed"}.issubset(df.columns)
    if has_delta_r2_mode:
        df["mean_delta_r2"] = pd.to_numeric(df["mean_delta_r2"], errors="coerce")
        df["var_delta_r2"] = pd.to_numeric(df.get("var_delta_r2", np.nan), errors="coerce")
        df["freq"] = pd.to_numeric(df["freq"], errors="coerce")
        df["passed"] = df["passed"].astype(bool)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["feature", "mean_delta_r2", "freq"]).copy()
        if df.empty:
            return

        df = df.sort_values(
            by=["passed", "mean_delta_r2", "var_delta_r2", "feature"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
        x = np.arange(len(df))
        colors = np.where(df["passed"], "#1f77b4", "#d62728")

        fig, ax1 = plt.subplots(figsize=(max(9, len(df) * 0.55), 4.8))
        ax1.bar(x, df["mean_delta_r2"], color=colors, alpha=0.9, label="mean_delta_r2")
        ax1.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
        if "gate_min_delta_r2" in df.columns:
            gate_delta = pd.to_numeric(df["gate_min_delta_r2"], errors="coerce").dropna()
            if len(gate_delta) > 0:
                ax1.axhline(
                    float(gate_delta.iloc[0]),
                    color="#444444",
                    linestyle=":",
                    linewidth=1.2,
                    label=f"min_delta_r2={float(gate_delta.iloc[0]):.4f}",
                )

        ax2 = ax1.twinx()
        ax2.plot(
            x,
            df["freq"].to_numpy(dtype=float),
            color="#2ca02c",
            marker="o",
            linewidth=1.6,
            markersize=4,
            label="freq(delta>0)",
        )
        if "gate_min_freq" in df.columns:
            gate_freq = pd.to_numeric(df["gate_min_freq"], errors="coerce").dropna()
            if len(gate_freq) > 0:
                ax2.axhline(
                    float(gate_freq.iloc[0]),
                    color="#2ca02c",
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.8,
                    label=f"min_freq={float(gate_freq.iloc[0]):.2f}",
                )
        ax2.set_ylim(0.0, 1.05)

        ax1.set_xticks(x)
        ax1.set_xticklabels(df["feature"], rotation=45, ha="right")
        ax1.set_ylabel("delta_r2")
        ax2.set_ylabel("freq")
        if title:
            ax1.set_title(title)
        else:
            ax1.set_title("Secondary Selection (delta_r2 and freq)")

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best", frameon=False)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    has_eta_mode = {"eta2", "passed"}.issubset(df.columns)
    if has_eta_mode:
        df["eta2"] = pd.to_numeric(df["eta2"], errors="coerce")
        df["null_q95"] = pd.to_numeric(df.get("null_q95", np.nan), errors="coerce")
        df["q_value"] = pd.to_numeric(df.get("q_value", np.nan), errors="coerce")
        df["p_value"] = pd.to_numeric(df.get("p_value", np.nan), errors="coerce")
        df["passed"] = df["passed"].astype(bool)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["feature", "eta2"]).copy()
        if df.empty:
            return

        sig_col = "q_value" if np.isfinite(df["q_value"]).any() else "p_value"
        df["sig_plot"] = pd.to_numeric(df.get(sig_col, np.nan), errors="coerce").fillna(1.0)
        df = df.sort_values(
            by=["passed", "eta2", "sig_plot", "feature"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
        x = np.arange(len(df))
        colors = np.where(df["passed"], "#1f77b4", "#d62728")

        fig, ax1 = plt.subplots(figsize=(max(9, len(df) * 0.55), 4.8))
        ax1.bar(x, df["eta2"], color=colors, alpha=0.9, label="eta2")

        if np.isfinite(df["null_q95"]).any():
            ax1.plot(
                x,
                df["null_q95"].to_numpy(dtype=float),
                color="#7f7f7f",
                marker="s",
                linewidth=1.4,
                markersize=4,
                label="null_q95",
            )

        ax2 = ax1.twinx()
        ax2.plot(
            x,
            df["sig_plot"].to_numpy(dtype=float),
            color="#2ca02c",
            marker="o",
            linewidth=1.6,
            markersize=4,
            label=sig_col,
        )
        if "gate_alpha" in df.columns:
            gate_alpha = pd.to_numeric(df["gate_alpha"], errors="coerce").dropna()
            if len(gate_alpha) > 0:
                ax2.axhline(
                    float(gate_alpha.iloc[0]),
                    color="#2ca02c",
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.8,
                    label=f"alpha={float(gate_alpha.iloc[0]):.2f}",
                )
        ax2.set_ylim(0.0, 1.05)

        ax1.set_xticks(x)
        ax1.set_xticklabels(df["feature"], rotation=45, ha="right")
        ax1.set_ylabel("eta2")
        ax2.set_ylabel(sig_col)
        if title:
            ax1.set_title(title)
        else:
            ax1.set_title("Secondary Selection (eta2 vs null and significance)")

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best", frameon=False)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    has_delta_mode = {"mean_delta_mse_pct", "freq", "passed"}.issubset(df.columns)
    if not has_delta_mode:
        return

    df["mean_delta_mse_pct"] = pd.to_numeric(df["mean_delta_mse_pct"], errors="coerce")
    df["var_delta_mse_pct"] = pd.to_numeric(df.get("var_delta_mse_pct", np.nan), errors="coerce")
    df["freq"] = pd.to_numeric(df["freq"], errors="coerce")
    df["passed"] = df["passed"].astype(bool)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["feature", "mean_delta_mse_pct", "freq"]).copy()
    if df.empty:
        return

    df = df.sort_values(
        by=["passed", "mean_delta_mse_pct", "var_delta_mse_pct", "feature"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    x = np.arange(len(df))
    colors = np.where(df["passed"], "#1f77b4", "#d62728")

    fig, ax1 = plt.subplots(figsize=(max(9, len(df) * 0.55), 4.8))
    ax1.bar(x, df["mean_delta_mse_pct"], color=colors, alpha=0.9, label="mean_delta_mse_pct")
    ax1.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    if "gate_min_delta_mse_pct" in df.columns:
        gate_delta = pd.to_numeric(df["gate_min_delta_mse_pct"], errors="coerce").dropna()
        if len(gate_delta) > 0:
            ax1.axhline(
                float(gate_delta.iloc[0]),
                color="#444444",
                linestyle=":",
                linewidth=1.2,
                label=f"min_delta={float(gate_delta.iloc[0]):.2f}%",
            )

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        df["freq"].to_numpy(dtype=float),
        color="#2ca02c",
        marker="o",
        linewidth=1.6,
        markersize=4,
        label="freq(delta>0)",
    )
    if "gate_min_freq" in df.columns:
        gate_freq = pd.to_numeric(df["gate_min_freq"], errors="coerce").dropna()
        if len(gate_freq) > 0:
            ax2.axhline(
                float(gate_freq.iloc[0]),
                color="#2ca02c",
                linestyle=":",
                linewidth=1.2,
                alpha=0.8,
                label=f"min_freq={float(gate_freq.iloc[0]):.2f}",
            )
    ax2.set_ylim(0.0, 1.05)

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["feature"], rotation=45, ha="right")
    ax1.set_ylabel("delta_mse (%)")
    ax2.set_ylabel("freq")
    if title:
        ax1.set_title(title)
    else:
        ax1.set_title("Secondary Selection (delta_mse% and freq)")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
