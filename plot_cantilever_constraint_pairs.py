from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from CAE_model.cantilever_beam import get_problem_spec
from DOE.doe_algorithm.lhs import latin_hypercube_sampling


def _pair_indices(n_dim: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_dim) for j in range(i + 1, n_dim)]


def _cantilever_constraint_values(
    *,
    X: np.ndarray,
    index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    H = X[:, index["H"]]
    h1 = X[:, index["h1"]]
    b1 = X[:, index["b1"]]
    b2 = X[:, index["b2"]]

    i_term = (1.0 / 12.0) * b2 * (H - 2.0 * h1) ** 3 + 2.0 * (
        (1.0 / 12.0) * b1 * h1**3 + (1.0 / 4.0) * b1 * h1 * (H - h1) ** 2
    )

    sigma = np.full((X.shape[0],), np.inf, dtype=float)
    delta = np.full((X.shape[0],), np.inf, dtype=float)

    safe_i = np.isfinite(i_term) & (np.abs(i_term) > 1e-12)
    if np.any(safe_i):
        sigma[safe_i] = (1000.0 * 36.0 * H[safe_i]) / (2.0 * i_term[safe_i])
        delta[safe_i] = (1000.0 * (36.0**3)) / (3.0 * (10.0e6) * i_term[safe_i])

    return sigma, delta


def _plot_scatter_pairs(
    *,
    X: np.ndarray,
    names: list[str],
    bounds_map: dict[str, tuple[float, float]],
    mask: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    pairs = _pair_indices(len(names))
    cols = 3
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.8, rows * 4.1))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, (i, j) in zip(axes, pairs):
        xi = names[i]
        xj = names[j]
        x = X[:, i]
        y = X[:, j]
        infeasible = ~mask

        ax.scatter(x[infeasible], y[infeasible], s=2, alpha=0.14, c="#d9534f", marker="o", linewidths=0.0, rasterized=True)
        ax.scatter(x[mask], y[mask], s=2, alpha=0.14, c="#2b8a3e", marker="o", linewidths=0.0, rasterized=True)

        lbx, ubx = bounds_map[xi]
        lby, uby = bounds_map[xj]
        ax.set_xlim(lbx, ubx)
        ax.set_ylim(lby, uby)
        ax.set_xlabel(xi)
        ax.set_ylabel(xj)
        ax.grid(alpha=0.2)

    for ax in axes[len(pairs) :]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_feasible_boolean_pairs(
    *,
    X: np.ndarray,
    names: list[str],
    bounds_map: dict[str, tuple[float, float]],
    mask: np.ndarray,
    bins: int,
    title: str,
    out_path: Path,
) -> None:
    pairs = _pair_indices(len(names))
    cols = 3
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.8, rows * 4.1))
    axes = np.atleast_1d(axes).reshape(-1)

    im = None
    feasible_weights = mask.astype(float)
    infeasible_weights = (~mask).astype(float)
    cmap = ListedColormap(["#f7f7f7", "#d9534f", "#2b8a3e"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], ncolors=cmap.N)

    for ax, (i, j) in zip(axes, pairs):
        xi = names[i]
        xj = names[j]
        x = X[:, i]
        y = X[:, j]
        xb = bounds_map[xi]
        yb = bounds_map[xj]
        h_total, x_edges, y_edges = np.histogram2d(
            x,
            y,
            bins=bins,
            range=[[xb[0], xb[1]], [yb[0], yb[1]]],
        )
        h_ok, _, _ = np.histogram2d(
            x,
            y,
            bins=[x_edges, y_edges],
            weights=feasible_weights,
        )
        h_bad, _, _ = np.histogram2d(
            x,
            y,
            bins=[x_edges, y_edges],
            weights=infeasible_weights,
        )

        # -1: 비어 있음, 0: infeasible(False), 1: feasible(True)
        state = np.full(h_total.shape, -1.0, dtype=float)
        occupied = h_total > 0
        if np.any(occupied):
            state[occupied] = np.where(h_ok[occupied] >= h_bad[occupied], 1.0, 0.0)

        im = ax.imshow(
            state.T,
            origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="auto",
        )
        ax.set_xlabel(xi)
        ax.set_ylabel(xj)
        ax.grid(alpha=0.2)

    for ax in axes[len(pairs) :]:
        ax.axis("off")

    fig.suptitle(title)
    if im is not None:
        cb = fig.colorbar(im, ax=axes[: len(pairs)], shrink=0.95)
        cb.set_ticks([-1.0, 0.0, 1.0])
        cb.set_ticklabels(["empty", "False", "True"])
        cb.set_label("Boolean feasibility")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone cantilever constraint pair-plot checker."
    )
    parser.add_argument("--n-samples", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-plot-points", type=int, default=120000)
    parser.add_argument("--bins", type=int, default=110)
    parser.add_argument(
        "--include-dummy",
        action="store_true",
        help="Include dummy variables in pair plots (default: real vars only).",
    )
    args = parser.parse_args()

    spec = get_problem_spec()
    variables = list(spec.get("variables", []))
    if not variables:
        raise RuntimeError("No variables found in cantilever spec.")

    if args.include_dummy:
        selected_vars = variables
    else:
        selected_vars = [v for v in variables if not str(v.get("name", "")).startswith("d")]

    names = [str(v["name"]) for v in selected_vars]
    bounds = [(float(v["default_lb"]), float(v["default_ub"])) for v in selected_vars]
    bounds_map = {str(v["name"]): (float(v["default_lb"]), float(v["default_ub"])) for v in selected_vars}
    if any(req not in names for req in ("H", "h1", "b1", "b2")):
        raise RuntimeError("Required cantilever real variables are missing.")

    rng = np.random.default_rng(int(args.seed))
    X_all = latin_hypercube_sampling(
        n_samples=int(args.n_samples),
        bounds=bounds,
        rng=rng,
    )
    idx_map = {name: i for i, name in enumerate(names)}
    sigma, delta = _cantilever_constraint_values(X=X_all, index=idx_map)
    c1_ok = np.isfinite(sigma) & (sigma <= 5000.0)
    c2_ok = np.isfinite(delta) & (delta <= 0.10)
    both_ok = c1_ok & c2_ok

    n = X_all.shape[0]
    n_plot = int(min(max(int(args.max_plot_points), 1), n))
    if n_plot < n:
        plot_idx = rng.choice(n, size=n_plot, replace=False)
    else:
        plot_idx = np.arange(n)

    X = X_all[plot_idx, :]
    c1_plot = c1_ok[plot_idx]
    c2_plot = c2_ok[plot_idx]
    both_plot = both_ok[plot_idx]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("result") / f"cantilever_constraint_pairs_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("constraint1_sigma", c1_plot, "Constraint1 feasible: sigma_max <= 5000"),
        ("constraint2_delta", c2_plot, "Constraint2 feasible: delta_max <= 0.10"),
        ("constraint_both", both_plot, "Both constraints feasible"),
    ]

    for token, mask, title in configs:
        _plot_scatter_pairs(
            X=X,
            names=names,
            bounds_map=bounds_map,
            mask=mask,
            title=f"{title} (scatter, n_plot={n_plot})",
            out_path=out_dir / f"{token}_scatter.png",
        )
        _plot_feasible_boolean_pairs(
            X=X,
            names=names,
            bounds_map=bounds_map,
            mask=mask,
            bins=int(args.bins),
            title=f"{title} (boolean area map, n_plot={n_plot})",
            out_path=out_dir / f"{token}_area.png",
        )

    summary = {
        "problem": "cantilever_beam",
        "n_samples_total": int(n),
        "n_samples_plotted": int(n_plot),
        "variables_plotted": names,
        "constraint1": {"name": "sigma_max <= 5000", "pass_count": int(np.sum(c1_ok)), "pass_rate": float(np.mean(c1_ok))},
        "constraint2": {"name": "delta_max <= 0.10", "pass_count": int(np.sum(c2_ok)), "pass_rate": float(np.mean(c2_ok))},
        "both": {"name": "constraint1 & constraint2", "pass_count": int(np.sum(both_ok)), "pass_rate": float(np.mean(both_ok))},
        "output_dir": str(out_dir.resolve()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] output_dir={out_dir.resolve()}")
    print(
        "[done] pass_rates: "
        f"c1={summary['constraint1']['pass_rate']:.4f}, "
        f"c2={summary['constraint2']['pass_rate']:.4f}, "
        f"both={summary['both']['pass_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
