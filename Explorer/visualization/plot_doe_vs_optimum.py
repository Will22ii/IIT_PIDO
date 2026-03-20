import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.bool_mask import to_bool_mask


def _known_optimum_map(known_optimum: object | None) -> dict[str, float]:
    if known_optimum is None:
        return {}
    if isinstance(known_optimum, dict):
        out = {}
        for k, v in known_optimum.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out
    return {}


def _constraint_mask(df: pd.DataFrame, *, respect_constraints: bool) -> np.ndarray:
    if df.empty:
        return np.zeros((0,), dtype=bool)
    mask = np.ones((len(df),), dtype=bool)
    if "success" in df.columns:
        mask &= to_bool_mask(
            df["success"],
            column_name="success",
            warn_prefix="[ExplorerPlot][BoolParse]",
        )
    if not respect_constraints:
        return mask

    if "feasible" in df.columns:
        mask &= to_bool_mask(
            df["feasible"],
            column_name="feasible",
            warn_prefix="[ExplorerPlot][BoolParse]",
        )
        return mask

    if "feasible_pre" in df.columns:
        mask &= to_bool_mask(
            df["feasible_pre"],
            column_name="feasible_pre",
            warn_prefix="[ExplorerPlot][BoolParse]",
        )
    if "feasible_post" in df.columns:
        mask &= to_bool_mask(
            df["feasible_post"],
            column_name="feasible_post",
            warn_prefix="[ExplorerPlot][BoolParse]",
        )
    return mask


def _pairwise_plot(
    *,
    df: pd.DataFrame,
    var_names: list[str],
    values: np.ndarray,
    known_optimum: dict[str, float],
    title: str,
    color_label: str,
    out_path: str,
) -> str | None:
    pairs = [(i, j) for i in range(len(var_names)) for j in range(i + 1, len(var_names))]
    if not pairs:
        return None
    if df.empty:
        return None

    cols = 3
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.4, rows * 4.1))
    axes = np.atleast_1d(axes).reshape(-1)

    values = np.asarray(values, dtype=float).reshape(-1)
    if values.shape[0] != len(df):
        values = np.zeros((len(df),), dtype=float)

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        vmax = vmin + 1e-12

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)

    for ax, (i, j) in zip(axes, pairs):
        xi = var_names[i]
        xj = var_names[j]
        ax.scatter(
            df[xi],
            df[xj],
            c=values,
            s=12,
            alpha=0.75,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        if xi in known_optimum and xj in known_optimum:
            ax.scatter(
                known_optimum[xi],
                known_optimum[xj],
                s=100,
                marker="*",
                color="red",
                edgecolor="white",
                linewidth=0.5,
                zorder=5,
            )
        ax.set_xlabel(xi)
        ax.set_ylabel(xj)
        ax.grid(True, alpha=0.25)

    for ax in axes[len(pairs):]:
        ax.axis("off")

    fig.suptitle(title)
    fig.colorbar(sm, ax=axes[: len(pairs)], label=color_label)
    fig.subplots_adjust(top=0.90, wspace=0.28, hspace=0.34)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _safe_token(name: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name))
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "var"


def _plot_source_split(
    *,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    known_optimum: dict[str, float],
    out_path: str,
) -> str | None:
    if "source" not in df.columns:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.4))
    styles = [
        ("basic", "#1f77b4", "DOE (basic)"),
        ("initial", "#2ca02c", "DOE (initial)"),
        ("additional", "#ff7f0e", "DOE (additional)"),
    ]
    has_any = False
    for source_name, color, label in styles:
        part = df[df["source"].astype(str).str.lower() == source_name]
        if part.empty:
            continue
        has_any = True
        ax.scatter(
            part[x_col],
            part[y_col],
            s=14,
            alpha=0.75,
            color=color,
            label=label,
        )
    if not has_any:
        plt.close(fig)
        return None

    if x_col in known_optimum and y_col in known_optimum:
        ax.scatter(
            known_optimum[x_col],
            known_optimum[y_col],
            s=120,
            marker="*",
            color="red",
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
            label="Known optimum",
        )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("DOE Sources vs Known Optimum")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _plot_exec_scope_split(
    *,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    known_optimum: dict[str, float],
    out_path: str,
) -> str | None:
    if "exec_scope" not in df.columns:
        return None

    scope = df["exec_scope"].astype(str).str.strip().str.lower()
    g_mask = scope.isin({"global", "initial", "basic"})
    l_mask = scope == "local"
    p_mask = scope == "probe"
    local_title_suffix = ""

    # Backward compatibility:
    # historical runs may only have "mixed" for additional stage.
    if not bool(np.any(l_mask)) and bool(np.any(scope == "mixed")):
        l_mask = scope == "mixed"
        local_title_suffix = " (mixed)"

    df_g = df[g_mask]
    df_l = df[l_mask]
    df_p = df[p_mask]
    if df_g.empty and df_l.empty and df_p.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), sharex=True, sharey=True)
    axg, axl = axes

    if not df_g.empty:
        axg.scatter(df_g[x_col], df_g[y_col], s=12, alpha=0.75, color="#1f77b4")
    if x_col in known_optimum and y_col in known_optimum:
        axg.scatter(
            known_optimum[x_col],
            known_optimum[y_col],
            s=120,
            marker="*",
            color="red",
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )
    axg.set_title("X_exec_global")
    axg.set_xlabel(x_col)
    axg.set_ylabel(y_col)
    axg.grid(True, alpha=0.25)

    if not df_l.empty:
        if "round" in df_l.columns and df_l["round"].notna().any():
            rounds = df_l["round"].astype(float).to_numpy()
            vmin = float(np.nanmin(rounds))
            vmax = float(np.nanmax(rounds))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmax = vmin + 1e-12
            sc = axl.scatter(
                df_l[x_col],
                df_l[y_col],
                c=rounds,
                cmap="Greys",
                s=14,
                alpha=0.8,
                edgecolor="black",
                linewidth=0.25,
                vmin=vmin,
                vmax=vmax,
            )
            fig.colorbar(sc, ax=axl, label="stage (round)")
        else:
            axl.scatter(
                df_l[x_col],
                df_l[y_col],
                s=14,
                alpha=0.8,
                color="#ff7f0e",
                edgecolor="black",
                linewidth=0.25,
            )
    if not df_p.empty:
        axl.scatter(
            df_p[x_col],
            df_p[y_col],
            s=34,
            alpha=0.95,
            color="#d62728",
            marker="x",
            linewidth=1.1,
            label="probe",
            zorder=6,
        )
    if x_col in known_optimum and y_col in known_optimum:
        axl.scatter(
            known_optimum[x_col],
            known_optimum[y_col],
            s=120,
            marker="*",
            color="red",
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )
    axl.set_title(f"X_exec_local{local_title_suffix}")
    axl.set_xlabel(x_col)
    axl.set_ylabel(y_col)
    axl.grid(True, alpha=0.25)
    if not df_p.empty:
        axl.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_doe_vs_optimum(
    *,
    doe_df: pd.DataFrame,
    explorer_df: pd.DataFrame | None,
    selected_features: list[str],
    known_optimum: object | None,
    objective_sense: str,
    out_dir: str,
    respect_constraints: bool = True,
    max_stage_plots: int = 20,
) -> dict:
    _ = explorer_df
    _ = objective_sense
    _ = max_stage_plots
    os.makedirs(out_dir, exist_ok=True)
    feature_cols = [str(c) for c in selected_features]
    if len(feature_cols) < 2:
        return {"saved": [], "reason": "need_at_least_two_features"}

    for col in feature_cols:
        if col not in doe_df.columns:
            return {"saved": [], "reason": f"missing_feature_in_doe:{col}"}

    known_map = _known_optimum_map(known_optimum)
    saved: list[str] = []
    saved_set: set[str] = set()

    def _append_saved(path: str | None) -> None:
        if not path:
            return
        if path in saved_set:
            return
        saved.append(path)
        saved_set.add(path)

    # DOE points: success + feasible(pre/post) aware
    doe_mask = _constraint_mask(doe_df, respect_constraints=respect_constraints)
    doe_used = doe_df.loc[doe_mask].copy()
    if doe_used.empty:
        return {
            "saved": saved,
            "n_saved": len(saved),
            "respect_constraints": bool(respect_constraints),
            "objective_sense": str(objective_sense),
        }

    # Requested output only:
    # for each variable pair, create one figure with
    # left=X_exec_global, right=X_exec_local(color=stage)+probe(marker).
    first_pair_saved = None
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            xi = feature_cols[i]
            xj = feature_cols[j]
            fname = (
                "doe_exec_global_local_"
                f"{_safe_token(xi)}_{_safe_token(xj)}.png"
            )
            p_exec_pair = _plot_exec_scope_split(
                df=doe_used,
                x_col=xi,
                y_col=xj,
                known_optimum=known_map,
                out_path=os.path.join(out_dir, fname),
            )
            if first_pair_saved is None and p_exec_pair:
                first_pair_saved = p_exec_pair
            _append_saved(p_exec_pair)

    # Backward-compatible alias (first pair)
    if first_pair_saved:
        x0 = feature_cols[0]
        x1 = feature_cols[1]
        p_alias = _plot_exec_scope_split(
            df=doe_used,
            x_col=x0,
            y_col=x1,
            known_optimum=known_map,
            out_path=os.path.join(out_dir, "doe_exec_global_local.png"),
        )
        _append_saved(p_alias)

    return {
        "saved": saved,
        "n_saved": len(saved),
        "respect_constraints": bool(respect_constraints),
        "objective_sense": str(objective_sense),
    }


def _find_latest_run_dir(result_root: str) -> str:
    root = Path(result_root)
    if not root.exists():
        raise FileNotFoundError(f"result root not found: {result_root}")
    runs = [p for p in root.glob("run_*") if p.is_dir()]
    if not runs:
        raise FileNotFoundError("no run_* dirs found")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(runs[0])


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_dir = _find_latest_run_dir(os.path.join(project_root, "result"))

    doe_csv = os.path.join(run_dir, "DOE", "artifacts", "public", "doe_results.csv")
    explorer_csv = os.path.join(run_dir, "Explorer", "artifacts", "public", "explorer_results.csv")
    snapshot_path = os.path.join(run_dir, "user_config_snapshot.json")
    if not os.path.exists(doe_csv):
        raise FileNotFoundError(f"DOE CSV not found: {doe_csv}")
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f"user_config_snapshot not found: {snapshot_path}")

    snap = _read_json(snapshot_path)
    design_bounds = snap.get("design_bounds", {})
    selected_features = list(design_bounds.keys())
    doe_df = pd.read_csv(doe_csv)
    explorer_df = pd.read_csv(explorer_csv) if os.path.exists(explorer_csv) else None

    out_dir = os.path.join(run_dir, "Explorer", "artifacts", "debug")
    result = plot_doe_vs_optimum(
        doe_df=doe_df,
        explorer_df=explorer_df,
        selected_features=selected_features,
        known_optimum=None,
        objective_sense=str(snap.get("objective_sense", "min")),
        out_dir=out_dir,
        respect_constraints=True,
    )
    print(f"[plot_doe_vs_optimum] saved={result.get('n_saved', 0)}")
    for p in result.get("saved", []):
        print(f" - {p}")


if __name__ == "__main__":
    main()
