import json
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DOE.doe_algorithm.lhs import latin_hypercube_sampling


def _pairwise_plot(
    *,
    df: pd.DataFrame,
    var_names: list[str],
    values: np.ndarray,
    known_optimum: dict,
    title: str,
    out_path: str,
) -> None:
    pairs = [(i, j) for i in range(len(var_names)) for j in range(i + 1, len(var_names))]
    if not pairs:
        return

    cols = 3
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.0))
    axes = np.atleast_1d(axes).reshape(-1)

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)

    for ax, (i, j) in zip(axes, pairs):
        xi = var_names[i]
        xj = var_names[j]
        ax.scatter(
            df[xi],
            df[xj],
            c=values,
            s=10,
            alpha=0.7,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        if xi in known_optimum and xj in known_optimum:
            ax.scatter(
                known_optimum[xi],
                known_optimum[xj],
                s=90,
                marker="*",
                color="red",
                zorder=5,
            )
        ax.set_xlabel(xi)
        ax.set_ylabel(xj)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(pairs):]:
        ax.axis("off")

    fig.suptitle(title)
    fig.colorbar(sm, ax=axes[: len(pairs)], label="objective")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")


def _load_latest_doe_csv(run_dir: str, stage_name: str = "DOE") -> str:
    meta_path = os.path.join(run_dir, stage_name, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"DOE metadata not found: {meta_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    results_csv = meta.get("artifacts", {}).get("results_csv")
    if not results_csv:
        raise RuntimeError("DOE metadata missing artifacts.results_csv")
    return os.path.join(os.path.dirname(meta_path), results_csv)


def _load_doe_metadata(run_dir: str, stage_name: str = "DOE") -> dict:
    meta_path = os.path.join(run_dir, stage_name, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"DOE metadata not found: {meta_path}")
    with open(meta_path, "r") as f:
        return json.load(f)


def _load_model_pkl(run_dir: str) -> str:
    meta_path = os.path.join(run_dir, "Modeler", "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Modeler metadata not found: {meta_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    model_path = meta.get("artifacts", {}).get("model_path")
    if not model_path:
        raise RuntimeError("Modeler metadata missing artifacts.model_path")
    return os.path.join(os.path.dirname(meta_path), model_path)

def _load_modeler_metadata(run_dir: str) -> dict:
    meta_path = os.path.join(run_dir, "Modeler", "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Modeler metadata not found: {meta_path}")
    with open(meta_path, "r") as f:
        return json.load(f)


def _load_selected_features(run_dir: str) -> list[str]:
    meta = _load_modeler_metadata(run_dir)
    rel_path = meta.get("artifacts", {}).get("selected_features")
    if not rel_path:
        return []
    sel_path = os.path.join(os.path.dirname(os.path.join(run_dir, "Modeler", "metadata.json")), rel_path)
    if not os.path.exists(sel_path):
        raise FileNotFoundError(f"Selected features not found: {sel_path}")
    df = pd.read_csv(sel_path)
    if "feature" in df.columns:
        return df["feature"].astype(str).tolist()
    if df.shape[1] >= 1:
        return df.iloc[:, 0].astype(str).tolist()
    return []


def _render_plots(
    *,
    stage_name: str,
    df: pd.DataFrame,
    doe_meta: dict,
    user_snapshot: dict,
    design_bounds: dict,
    var_names: list[str],
    name_to_idx: dict,
    has_x1x2: bool,
    known_optimum: dict,
    models: list,
    lhc_var_names: list[str],
    sel_indices: list[int],
    out_dir: str,
) -> None:
    y_pred = None
    X_lhc = None
    X_lhc_model = None
    if models:
        sample_multiplier = (
            doe_meta.get("inputs", {})
            .get("system_config_snapshot", {})
            .get("sample_multiplier")
        )
        if not sample_multiplier:
            raise RuntimeError("DOE metadata missing system_config_snapshot.sample_multiplier")
        n_vars = len(var_names)
        n_samples = int(sample_multiplier * n_vars)
        bounds = [(design_bounds[name][0], design_bounds[name][1]) for name in var_names]
        rng = np.random.default_rng(user_snapshot.get("seed", 42))
        X_lhc = latin_hypercube_sampling(
            n_samples=n_samples,
            bounds=bounds,
            rng=rng,
            n_divisions=n_samples,
        )

        if sel_indices:
            X_lhc_model = X_lhc[:, sel_indices]
        else:
            X_lhc_model = X_lhc

        preds = []
        for model in models:
            preds.append(np.asarray(model.predict(X_lhc_model), dtype=float).reshape(-1))
        y_pred = np.vstack(preds).mean(axis=0)

    suffix = stage_name.replace("DOE", "").lower()
    suffix = f"_{suffix}" if suffix else ""

    if has_x1x2:
        idx_x1 = name_to_idx["x1"]
        idx_x2 = name_to_idx["x2"]
        if y_pred is None:
            vmin = float(np.min(df["objective"].to_numpy()))
            vmax = float(np.max(df["objective"].to_numpy()))
        else:
            combined = np.concatenate([df["objective"].to_numpy(), y_pred])
            vmin = float(np.min(combined))
            vmax = float(np.max(combined))

        if y_pred is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
            ax = axes[0]
        sc = ax.scatter(
            df["x1"],
            df["x2"],
            c=df["objective"],
            s=12,
            alpha=0.7,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            label="DOE",
        )
        if "x1" in known_optimum and "x2" in known_optimum:
            ax.scatter(
                known_optimum["x1"],
                known_optimum["x2"],
                s=140,
                marker="*",
                color="red",
                label="Known optimum",
                zorder=5,
            )
        fig.colorbar(sc, ax=ax, label="objective")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(f"DOE (CAE) vs Known Optimum{suffix}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if y_pred is not None:
            ax2 = axes[1]
            sc2 = ax2.scatter(
                X_lhc_model[:, lhc_var_names.index("x1")] if "x1" in lhc_var_names else X_lhc[:, idx_x1],
                X_lhc_model[:, lhc_var_names.index("x2")] if "x2" in lhc_var_names else X_lhc[:, idx_x2],
                c=y_pred,
                s=12,
                alpha=0.7,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                label="LHC (model)",
            )
            if "x1" in known_optimum and "x2" in known_optimum:
                ax2.scatter(
                    known_optimum["x1"],
                    known_optimum["x2"],
                    s=140,
                    marker="*",
                    color="red",
                    label="Known optimum",
                    zorder=5,
                )
            fig.colorbar(sc2, ax=ax2, label="predicted y")
            ax2.set_xlabel("x1")
            ax2.set_ylabel("x2")
            ax2.set_title(f"LHC (model) vs Known Optimum{suffix}")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        out_path = os.path.join(out_dir, f"doe_vs_optimum{suffix}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        print(f"Saved: {out_path}")
    else:
        print("Skipping x1/x2 plot: required variables not found.")

    # -------------------------------------------------
    # Pairwise plots for all variables (n >= 3)
    # -------------------------------------------------
    if len(var_names) > 2:
        _pairwise_plot(
            df=df,
            var_names=var_names,
            values=df["objective"].to_numpy(),
            known_optimum=known_optimum,
            title=f"DOE (CAE) Pairwise vs Known Optimum{suffix}",
            out_path=os.path.join(out_dir, f"doe_pairwise{suffix}.png"),
        )
        if y_pred is not None:
            lhc_df = pd.DataFrame(X_lhc_model, columns=lhc_var_names)
            _pairwise_plot(
                df=lhc_df,
                var_names=lhc_var_names,
                values=y_pred,
                known_optimum=known_optimum,
                title=f"LHC (model) Pairwise vs Known Optimum{suffix}",
                out_path=os.path.join(out_dir, f"lhc_pairwise{suffix}.png"),
            )

    # -------------------------------------------------
    # Additional plot: DOE source split (no color by y)
    # -------------------------------------------------
    if "source" not in df.columns:
        print("Skipping source-split plot: DOE CSV has no 'source' column.")
        return

    df_basic = df[df["source"] == "basic"]
    df_init = df[df["source"] == "initial"]
    df_add = df[df["source"] == "additional"]

    fig2, ax3 = plt.subplots(1, 1, figsize=(6, 5.5))

    if not df_basic.empty:
        ax3.scatter(
            df_basic["x1"],
            df_basic["x2"],
            s=14,
            alpha=0.7,
            color="#1f77b4",
            label="DOE (basic)",
        )
    if not df_init.empty:
        ax3.scatter(
            df_init["x1"],
            df_init["x2"],
            s=14,
            alpha=0.7,
            color="#2ca02c",
            label="DOE (initial)",
        )
    if not df_add.empty:
        ax3.scatter(
            df_add["x1"],
            df_add["x2"],
            s=14,
            alpha=0.7,
            color="#ff7f0e",
            label="DOE (additional)",
        )

    if "x1" in known_optimum and "x2" in known_optimum:
        ax3.scatter(
            known_optimum["x1"],
            known_optimum["x2"],
            s=140,
            marker="*",
            color="red",
            label="Known optimum",
            zorder=5,
        )
    ax3.set_xlabel("x1")
    ax3.set_ylabel("x2")
    ax3.set_title(f"DOE Sources vs Known Optimum{suffix}")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    out_path2 = os.path.join(out_dir, f"doe_sources_vs_optimum{suffix}.png")
    fig2.tight_layout()
    fig2.savefig(out_path2, dpi=200)
    print(f"Saved: {out_path2}")

def _find_latest_run_dir(result_root: str) -> str:
    if not os.path.exists(result_root):
        raise FileNotFoundError(f"Result root not found: {result_root}")
    run_dirs = [
        d for d in os.listdir(result_root)
        if d.startswith("run_") and os.path.isdir(os.path.join(result_root, d))
    ]
    if not run_dirs:
        raise FileNotFoundError("No run_* directories found.")
    run_dirs.sort(reverse=True)
    return os.path.join(result_root, run_dirs[0])


def main():
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    result_root = os.path.join(project_root, "result")
    run_dir = _find_latest_run_dir(result_root)

    snapshot_path = os.path.join(run_dir, "user_config_snapshot.json")
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f"user_config_snapshot not found: {snapshot_path}")
    with open(snapshot_path, "r") as f:
        user_snapshot = json.load(f)

    design_bounds = user_snapshot.get("design_bounds", {})
    var_names = list(design_bounds.keys())
    if len(var_names) < 2:
        raise RuntimeError("design_bounds must contain at least 2 variables.")

    # Known optimum (optional; only plotted when keys exist)
    known_optimum = {"x1": 1, "x2": 1, "x3": +1, "x4": 1, "x5": 1} # "x3": +1, "x4": 1, "x5": 1

    has_x1x2 = {"x1", "x2"}.issubset(var_names)
    name_to_idx = {name: i for i, name in enumerate(var_names)}

    models = []
    sel_indices = []
    lhc_var_names = var_names
    try:
        model_path = _load_model_pkl(run_dir)
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        feature_cols = payload.get("feature_cols") or []
        selected_features = feature_cols or _load_selected_features(run_dir)
        if selected_features:
            missing_feats = [f for f in selected_features if f not in var_names]
            if missing_feats:
                raise RuntimeError(f"Selected features not in design_bounds: {missing_feats}")
            sel_indices = [name_to_idx[f] for f in selected_features]
            lhc_var_names = selected_features
        models = payload.get("models", [])
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Skipping model-based plots: {exc}")

    out_dir = os.path.join(run_dir, "Explorer", "artifacts")
    os.makedirs(out_dir, exist_ok=True)

    stage_names = []
    for name in ["DOE", "DOE_success", "DOE_topk"]:
        if os.path.isdir(os.path.join(run_dir, name)):
            stage_names.append(name)
    if not stage_names:
        raise FileNotFoundError(f"No DOE stage folders found under: {run_dir}")

    for stage_name in stage_names:
        csv_path = _load_latest_doe_csv(run_dir, stage_name)
        doe_meta = _load_doe_metadata(run_dir, stage_name)
        df = pd.read_csv(csv_path)
        missing_cols = [name for name in var_names if name not in df.columns]
        if missing_cols:
            raise RuntimeError(f"DOE CSV missing columns: {missing_cols}")
        _render_plots(
            stage_name=stage_name,
            df=df,
            doe_meta=doe_meta,
            user_snapshot=user_snapshot,
            design_bounds=design_bounds,
            var_names=var_names,
            name_to_idx=name_to_idx,
            has_x1x2=has_x1x2,
            known_optimum=known_optimum,
            models=models,
            lhc_var_names=lhc_var_names,
            sel_indices=sel_indices,
            out_dir=out_dir,
        )

    # -------------------------------------------------
    # Debug (A): local box width ratio
    # -------------------------------------------------
    if "exec_scope" in df.columns:
        df_local = df[df["exec_scope"] == "local"]
        if df_local.empty:
            print("Local width ratio: no local exec points found.")
        else:
            print("Local width ratio (local exec min/max vs orig bounds):")
            ratios = []
            for var in ["x1", "x2"]:
                lb, ub = design_bounds[var]
                span = float(ub - lb)
                local_min = float(df_local[var].min())
                local_max = float(df_local[var].max())
                local_span = local_max - local_min
                ratio = local_span / span if span > 0 else float("nan")
                ratios.append(ratio)
                print(f"- {var}: {ratio:.4f}")
            if ratios:
                avg_ratio = float(np.mean(ratios))
                print(f"- mean: {avg_ratio:.4f}")
    else:
        print("Local width ratio: exec_scope column missing.")

    # -------------------------------------------------
    # Debug (B): exec_global vs exec_local plot (separate)
    # -------------------------------------------------
    if "exec_scope" in df.columns:
        df_g = df[df["exec_scope"] == "global"]
        df_l = df[df["exec_scope"] == "local"]

        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5.5))
        axg, axl = axes3

        if not df_g.empty:
            axg.scatter(df_g["x1"], df_g["x2"], s=12, alpha=0.7, color="#1f77b4")
        axg.scatter(
            known_optimum["x1"],
            known_optimum["x2"],
            s=140,
            marker="*",
            color="red",
            zorder=5,
        )
        axg.set_xlim(design_bounds["x1"][0], design_bounds["x1"][1])
        axg.set_ylim(design_bounds["x2"][0], design_bounds["x2"][1])
        axg.set_aspect("equal", adjustable="box")
        axg.set_title("X_exec_global")
        axg.set_xlabel("x1")
        axg.set_ylabel("x2")
        axg.grid(True, alpha=0.3)

        if not df_l.empty:
            if "round" in df_l.columns and df_l["round"].notna().any():
                rounds = df_l["round"].astype(float).to_numpy()
                vmin = float(np.nanmin(rounds))
                vmax = float(np.nanmax(rounds))
                sc_l = axl.scatter(
                    df_l["x1"],
                    df_l["x2"],
                    c=rounds,
                    cmap="Greys",
                    vmin=vmin,
                    vmax=vmax,
                    s=14,
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=0.3,
                )
                fig3.colorbar(sc_l, ax=axl, label="stage (round)")
            else:
                axl.scatter(
                    df_l["x1"],
                    df_l["x2"],
                    s=14,
                    alpha=0.8,
                    color="#ff7f0e",
                    edgecolor="black",
                    linewidth=0.3,
                )
        axl.scatter(
            known_optimum["x1"],
            known_optimum["x2"],
            s=140,
            marker="*",
            color="red",
            zorder=5,
        )
        axl.set_xlim(design_bounds["x1"][0], design_bounds["x1"][1])
        axl.set_ylim(design_bounds["x2"][0], design_bounds["x2"][1])
        axl.set_aspect("equal", adjustable="box")
        axl.set_title("X_exec_local")
        axl.set_xlabel("x1")
        axl.set_ylabel("x2")
        axl.grid(True, alpha=0.3)

        out_path3 = os.path.join(out_dir, "doe_exec_global_local.png")
        fig3.tight_layout()
        fig3.savefig(out_path3, dpi=200)
        print(f"Saved: {out_path3}")
    else:
        print("Skipping global/local exec plot: exec_scope column missing.")

    # -------------------------------------------------
    # Debug (C): top-k successful points (proxy)
    # -------------------------------------------------
    success_df = df[df.get("success", True) == True]
    if not success_df.empty:
        k = 20
        obj_vals = success_df["objective"].to_numpy()
        if user_snapshot.get("objective_sense", "min") == "max":
            idx = np.argsort(-obj_vals)[:k]
        else:
            idx = np.argsort(obj_vals)[:k]
        topk = success_df.iloc[idx]

        fig4, ax4 = plt.subplots(1, 1, figsize=(6, 5.5))
        ax4.scatter(topk["x1"], topk["x2"], s=20, alpha=0.8, color="#9467bd")
        ax4.scatter(
            known_optimum["x1"],
            known_optimum["x2"],
            s=140,
            marker="*",
            color="red",
            zorder=5,
        )
        ax4.set_title("Top-k success points (proxy)")
        ax4.set_xlabel("x1")
        ax4.set_ylabel("x2")
        ax4.grid(True, alpha=0.3)

        out_path4 = os.path.join(out_dir, "doe_topk_proxy.png")
        fig4.tight_layout()
        fig4.savefig(out_path4, dpi=200)
        print(f"Saved: {out_path4}")


if __name__ == "__main__":
    main()
