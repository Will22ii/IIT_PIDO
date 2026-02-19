import os
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def plot_raw_overlay(
    *,
    X_all: np.ndarray,
    mask: np.ndarray,
    feature_names: list[str],
    problem_name: str,
    project_root: str,
    use_timestamp: bool,
    save_path: str | None = None,
) -> str:
    if X_all.ndim != 2 or X_all.shape[1] != 2:
        raise ValueError("X_all must be 2D with 2 features for raw overlay.")
    if mask.shape[0] != X_all.shape[0]:
        raise ValueError("mask length mismatch with X_all.")

    output_dir = os.path.join(project_root, "result", "explorer")
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") if use_timestamp else ""
    suffix = f"_{ts}" if ts else ""
    filename = f"explorer_raw_overlay_{problem_name}{suffix}.png"
    overlay_path = save_path or os.path.join(output_dir, filename)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(
        X_all[:, 0],
        X_all[:, 1],
        c="#a0a0a0",
        s=8,
        alpha=0.25,
        label="all",
    )
    ax.scatter(
        X_all[mask, 0],
        X_all[mask, 1],
        c="#ff7f0e",
        s=10,
        alpha=0.9,
        label="q90",
    )
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title("Raw overlay: all vs q90")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(overlay_path, dpi=150)
    plt.close(fig)
    return overlay_path


def plot_raw_dbscan(
    *,
    X_q90: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    problem_name: str,
    project_root: str,
    use_timestamp: bool,
    save_path: str | None = None,
) -> str:
    if X_q90.ndim != 2 or X_q90.shape[1] != 2:
        raise ValueError("X_q90 must be 2D with 2 features for raw DBSCAN.")
    if labels.shape[0] != X_q90.shape[0]:
        raise ValueError("labels length mismatch with X_q90.")

    output_dir = os.path.join(project_root, "result", "explorer")
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") if use_timestamp else ""
    suffix = f"_{ts}" if ts else ""
    filename = f"explorer_raw_dbscan_{problem_name}{suffix}.png"
    plot_path = save_path or os.path.join(output_dir, filename)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(
            X_q90[noise_mask, 0],
            X_q90[noise_mask, 1],
            c="#b0b0b0",
            s=8,
            alpha=0.4,
            label="noise",
        )
    non_noise = ~noise_mask
    if non_noise.any():
        scatter = ax.scatter(
            X_q90[non_noise, 0],
            X_q90[non_noise, 1],
            c=labels[non_noise],
            s=10,
            cmap="tab10",
            alpha=0.9,
            label="cluster",
        )
        fig.colorbar(scatter, ax=ax, label="cluster")
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title("Raw: q90 + DBSCAN clusters")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def plot_raw_known_optimum(
    *,
    X_q90: np.ndarray,
    labels: Optional[np.ndarray],
    x_opt: np.ndarray | list[np.ndarray],
    feature_names: list[str],
    problem_name: str,
    project_root: str,
    use_timestamp: bool,
    save_path: str | None = None,
) -> str:
    if X_q90.ndim != 2 or X_q90.shape[1] != 2:
        raise ValueError("X_q90 must be 2D with 2 features for raw optimum.")
    x_opts = x_opt if isinstance(x_opt, list) else [x_opt]
    for xo in x_opts:
        if xo.shape[0] != X_q90.shape[1]:
            raise ValueError("x_opt length mismatch with X_q90.")

    output_dir = os.path.join(project_root, "result", "explorer")
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") if use_timestamp else ""
    suffix = f"_{ts}" if ts else ""
    filename = f"explorer_raw_optimum_{problem_name}{suffix}.png"
    plot_path = save_path or os.path.join(output_dir, filename)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if labels is not None and labels.shape[0] == X_q90.shape[0]:
        noise_mask = labels == -1
        if noise_mask.any():
            ax.scatter(
                X_q90[noise_mask, 0],
                X_q90[noise_mask, 1],
                c="#b0b0b0",
                s=8,
                alpha=0.4,
                label="noise",
            )
        non_noise = ~noise_mask
        if non_noise.any():
            scatter = ax.scatter(
                X_q90[non_noise, 0],
                X_q90[non_noise, 1],
                c=labels[non_noise],
                s=10,
                cmap="tab10",
                alpha=0.9,
                label="cluster",
            )
            fig.colorbar(scatter, ax=ax, label="cluster")
    else:
        ax.scatter(
            X_q90[:, 0],
            X_q90[:, 1],
            c="#ff7f0e",
            s=10,
            alpha=0.6,
            label="q90",
        )

    for idx, xo in enumerate(x_opts):
        ax.scatter(
            xo[0],
            xo[1],
            marker="*",
            s=160,
            c="#111111",
            edgecolors="#f5f5f5",
            linewidths=0.8,
            label="known optimum" if idx == 0 else None,
            zorder=5,
        )

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title("Raw: known optimum overlay")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def plot_dual_cluster_pair(
    *,
    X_pred: np.ndarray,
    labels_pred: np.ndarray,
    X_obj: np.ndarray,
    labels_obj: np.ndarray,
    feature_names: list[str],
    pair: tuple[int, int],
    bounds: list[tuple[float, float]],
    x_opt: list[np.ndarray] | np.ndarray | None = None,
    problem_name: str,
    project_root: str,
    use_timestamp: bool,
    save_path: str | None = None,
) -> str:
    if X_pred.ndim != 2 or X_obj.ndim != 2:
        raise ValueError("X arrays must be 2D.")
    i, j = pair
    if i >= len(feature_names) or j >= len(feature_names):
        raise ValueError("pair indices out of range.")

    output_dir = os.path.join(project_root, "result", "explorer")
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") if use_timestamp else ""
    suffix = f"_{ts}" if ts else ""
    filename = f"explorer_pair_dual_{problem_name}_{feature_names[i]}_{feature_names[j]}{suffix}.png"
    plot_path = save_path or os.path.join(output_dir, filename)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=True, sharey=True)
    ax_left, ax_right = axes

    def _convex_hull(points: np.ndarray) -> np.ndarray | None:
        if points.shape[0] < 3:
            return None
        pts = np.unique(points, axis=0)
        if pts.shape[0] < 3:
            return None
        pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        hull = np.array(lower[:-1] + upper[:-1], dtype=float)
        return hull if hull.shape[0] >= 3 else None

    # Pred clusters
    if X_pred.size:
        for lab, color, name in [(0, "#1f77b4", "pred:best"), (1, "#17becf", "pred:second")]:
            mask = labels_pred == lab
            if np.any(mask):
                ax_left.scatter(X_pred[mask, i], X_pred[mask, j], s=14, alpha=0.85, color=color, label=name)
                hull = _convex_hull(X_pred[mask][:, [i, j]])
                if hull is not None:
                    ax_left.plot(
                        np.r_[hull[:, 0], hull[0, 0]],
                        np.r_[hull[:, 1], hull[0, 1]],
                        color=color,
                        linewidth=1.2,
                        alpha=0.9,
                    )
    ax_left.set_title("Model (pred) clusters")
    ax_left.set_xlabel(feature_names[i])
    ax_left.set_ylabel(feature_names[j])
    if i < len(bounds) and j < len(bounds):
        ax_left.set_xlim(bounds[i][0], bounds[i][1])
        ax_left.set_ylim(bounds[j][0], bounds[j][1])
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(loc="best", frameon=False)

    # Objective clusters
    if X_obj.size:
        for lab, color, name in [(0, "#ff7f0e", "obj:best"), (1, "#2ca02c", "obj:second")]:
            mask = labels_obj == lab
            if np.any(mask):
                ax_right.scatter(X_obj[mask, i], X_obj[mask, j], s=14, alpha=0.85, color=color, label=name)
                hull = _convex_hull(X_obj[mask][:, [i, j]])
                if hull is not None:
                    ax_right.plot(
                        np.r_[hull[:, 0], hull[0, 0]],
                        np.r_[hull[:, 1], hull[0, 1]],
                        color=color,
                        linewidth=1.2,
                        alpha=0.9,
                    )
    ax_right.set_title("CAE (objective) clusters")
    ax_right.set_xlabel(feature_names[i])
    ax_right.set_ylabel(feature_names[j])
    if i < len(bounds) and j < len(bounds):
        ax_right.set_xlim(bounds[i][0], bounds[i][1])
        ax_right.set_ylim(bounds[j][0], bounds[j][1])
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(loc="best", frameon=False)

    if x_opt is not None:
        x_opts = x_opt if isinstance(x_opt, list) else [x_opt]
        for xo in x_opts:
            if xo.shape[0] == len(feature_names):
                ax_left.scatter(xo[i], xo[j], marker="*", s=140, color="red", zorder=5)
                ax_right.scatter(xo[i], xo[j], marker="*", s=140, color="red", zorder=5)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def plot_dual_cluster_tsne(
    *,
    X_pred: np.ndarray,
    labels_pred: np.ndarray,
    X_obj: np.ndarray,
    labels_obj: np.ndarray,
    x_opt: list[np.ndarray] | np.ndarray | None = None,
    problem_name: str,
    project_root: str,
    use_timestamp: bool,
    max_points: int = 2000,
    save_path: str | None = None,
) -> str:
    output_dir = os.path.join(project_root, "result", "explorer")
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") if use_timestamp else ""
    suffix = f"_{ts}" if ts else ""
    filename = f"explorer_tsne_dual_{problem_name}{suffix}.png"
    plot_path = save_path or os.path.join(output_dir, filename)

    def _downsample(X: np.ndarray, y: np.ndarray):
        if X.shape[0] <= max_points:
            return X, y
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        return X[idx], y[idx]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    ax_left, ax_right = axes

    if X_pred.size:
        Xp, yp = _downsample(X_pred, labels_pred)
        x_opts = x_opt if isinstance(x_opt, list) else ([x_opt] if x_opt is not None else [])
        if x_opts and all(xo.shape[0] == Xp.shape[1] for xo in x_opts):
            Xp_aug = np.vstack([Xp] + [xo.reshape(1, -1) for xo in x_opts])
        else:
            Xp_aug = Xp
        n_pred = Xp_aug.shape[0]
        perplexity_pred = min(30.0, max(5.0, float(n_pred // 3)))
        if n_pred <= perplexity_pred:
            perplexity_pred = max(1.0, float(n_pred - 1))
        tsne_pred = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            random_state=42,
            perplexity=perplexity_pred,
        )
        Zp = tsne_pred.fit_transform(Xp_aug)
        Zp_main = Zp[:Xp.shape[0]]
        for lab, color, name in [(0, "#1f77b4", "pred:best"), (1, "#17becf", "pred:second")]:
            mask = yp == lab
            if np.any(mask):
                ax_left.scatter(Zp_main[mask, 0], Zp_main[mask, 1], s=14, alpha=0.85, color=color, label=name)
        if x_opts and all(xo.shape[0] == Xp.shape[1] for xo in x_opts):
            for k in range(len(x_opts)):
                z_opt = Zp[-(k + 1)]
                ax_left.scatter(z_opt[0], z_opt[1], marker="*", s=140, color="red", zorder=5)
        ax_left.set_title("t-SNE: model clusters")
        ax_left.grid(True, alpha=0.3)
        ax_left.legend(loc="best", frameon=False)
    else:
        ax_left.set_title("t-SNE: model clusters (empty)")
        ax_left.axis("off")

    if X_obj.size:
        Xo, yo = _downsample(X_obj, labels_obj)
        x_opts = x_opt if isinstance(x_opt, list) else ([x_opt] if x_opt is not None else [])
        if x_opts and all(xo.shape[0] == Xo.shape[1] for xo in x_opts):
            Xo_aug = np.vstack([Xo] + [xo.reshape(1, -1) for xo in x_opts])
        else:
            Xo_aug = Xo
        n_obj = Xo_aug.shape[0]
        perplexity_obj = min(30.0, max(5.0, float(n_obj // 3)))
        if n_obj <= perplexity_obj:
            perplexity_obj = max(1.0, float(n_obj - 1))
        tsne_obj = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            random_state=42,
            perplexity=perplexity_obj,
        )
        Zo = tsne_obj.fit_transform(Xo_aug)
        Zo_main = Zo[:Xo.shape[0]]
        for lab, color, name in [(0, "#ff7f0e", "obj:best"), (1, "#2ca02c", "obj:second")]:
            mask = yo == lab
            if np.any(mask):
                ax_right.scatter(Zo_main[mask, 0], Zo_main[mask, 1], s=14, alpha=0.85, color=color, label=name)
        if x_opts and all(xo.shape[0] == Xo.shape[1] for xo in x_opts):
            for k in range(len(x_opts)):
                z_opt = Zo[-(k + 1)]
                ax_right.scatter(z_opt[0], z_opt[1], marker="*", s=140, color="red", zorder=5)
        ax_right.set_title("t-SNE: objective clusters")
        ax_right.grid(True, alpha=0.3)
        ax_right.legend(loc="best", frameon=False)
    else:
        ax_right.set_title("t-SNE: objective clusters (empty)")
        ax_right.axis("off")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def plot_bounds_pair(
    *,
    X_points: np.ndarray | None = None,
    X_pred: np.ndarray | None = None,
    X_obj: np.ndarray | None = None,
    feature_names: list[str],
    pair: tuple[int, int],
    bounds: list[tuple[float, float]],
    selected_bounds: list[tuple[float, float]],
    x_opt: list[np.ndarray] | np.ndarray | None = None,
    problem_name: str,
    project_root: str,
    use_timestamp: bool,
    save_path: str | None = None,
) -> str:
    if X_pred is None and X_obj is None:
        if X_points is None:
            raise ValueError("Provide X_points or X_pred/X_obj.")
        if X_points.ndim != 2:
            raise ValueError("X_points must be 2D.")
    i, j = pair
    if i >= len(feature_names) or j >= len(feature_names):
        raise ValueError("pair indices out of range.")

    output_dir = os.path.join(project_root, "result", "explorer")
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") if use_timestamp else ""
    suffix = f"_{ts}" if ts else ""
    filename = f"explorer_bounds_{problem_name}_{feature_names[i]}_{feature_names[j]}{suffix}.png"
    plot_path = save_path or os.path.join(output_dir, filename)

    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    if X_pred is not None and X_pred.size:
        ax.scatter(
            X_pred[:, i],
            X_pred[:, j],
            s=12,
            alpha=0.8,
            color="#1f77b4",
            label="pred points",
        )
    if X_obj is not None and X_obj.size:
        ax.scatter(
            X_obj[:, i],
            X_obj[:, j],
            s=12,
            alpha=0.8,
            color="#ff7f0e",
            label="obj points",
        )
    if X_points is not None and X_points.size and X_pred is None and X_obj is None:
        ax.scatter(
            X_points[:, i],
            X_points[:, j],
            s=12,
            alpha=0.8,
            color="#1f77b4",
            label="selected points",
        )

    # draw selected bounds rectangle
    lb_i, ub_i = selected_bounds[i]
    lb_j, ub_j = selected_bounds[j]
    rect_x = [lb_i, ub_i, ub_i, lb_i, lb_i]
    rect_y = [lb_j, lb_j, ub_j, ub_j, lb_j]
    ax.plot(rect_x, rect_y, color="#d62728", linewidth=1.5, label="selected bounds")

    # set axis to original bounds
    if i < len(bounds) and j < len(bounds):
        ax.set_xlim(bounds[i][0], bounds[i][1])
        ax.set_ylim(bounds[j][0], bounds[j][1])

    if x_opt is not None:
        x_opts = x_opt if isinstance(x_opt, list) else [x_opt]
        for xo in x_opts:
            if xo.shape[0] == len(feature_names):
                ax.scatter(xo[i], xo[j], marker="*", s=140, color="red", zorder=5)

    ax.set_xlabel(feature_names[i])
    ax.set_ylabel(feature_names[j])
    ax.set_title("Selected bounds (merged clusters)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path
