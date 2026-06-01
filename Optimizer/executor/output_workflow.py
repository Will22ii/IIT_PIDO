from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from itertools import combinations

import pandas as pd

from Optimizer.config import OptimizerConfig
from Optimizer.algorithms.result import OptimizerAlgorithmResult
from Optimizer.executor.input_workflow import ResolvedOptimizerInputs
from pipeline.run_context import RunContext, update_run_index
from utils.result_saver import ResultSaver


def _safe_plot_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))


def _known_optimum_records(known_optimum) -> list[dict[str, float]]:
    if known_optimum is None:
        return []
    items = known_optimum if isinstance(known_optimum, list) else [known_optimum]
    records: list[dict[str, float]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        record: dict[str, float] = {}
        for key, value in item.items():
            try:
                record[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        if record:
            records.append(record)
    return records


def _plot_known_optimum_markers(ax, *, known_optimum, fx: str, fy: str) -> None:
    plotted = False
    for point in _known_optimum_records(known_optimum):
        if str(fx) not in point or str(fy) not in point:
            continue
        ax.scatter(
            [float(point[str(fx)])],
            [float(point[str(fy)])],
            s=210,
            marker="*",
            facecolors="#ffd92f",
            edgecolors="#111111",
            linewidths=1.1,
            label="known optimum" if not plotted else None,
            zorder=12,
        )
        plotted = True


def _bounds_pair_rows(
    *,
    focus_region_output: dict,
    selected_features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    regions = focus_region_output.get("regions", []) if isinstance(focus_region_output, dict) else []
    if not isinstance(regions, list):
        regions = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        rid = str(region.get("region_id", ""))
        parent_bounds = region.get("bounds", {}) if isinstance(region.get("bounds", {}), dict) else {}
        for feature in selected_features:
            item = parent_bounds.get(str(feature))
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                rows.append({
                    "event_type": "parent",
                    "region_id": rid,
                    "feature": str(feature),
                    "lb": float(item[0]),
                    "ub": float(item[1]),
                    "batch_id": pd.NA,
                    "iter": pd.NA,
                    "volume_ratio": float(region.get("size_ratio", float("nan"))),
                    "status": str((region.get("turbo_lite_state", {}) or {}).get("status", "")) if isinstance(region.get("turbo_lite_state", {}), dict) else "",
                })
        state = region.get("turbo_lite_state", {}) if isinstance(region.get("turbo_lite_state", {}), dict) else {}
        active_bounds = state.get("active_bounds", {}) if isinstance(state.get("active_bounds", {}), dict) else {}
        for feature in selected_features:
            item = active_bounds.get(str(feature))
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                rows.append({
                    "event_type": "final_active",
                    "region_id": rid,
                    "feature": str(feature),
                    "lb": float(item[0]),
                    "ub": float(item[1]),
                    "batch_id": pd.NA,
                    "iter": pd.NA,
                    "volume_ratio": float(state.get("active_bounds_volume_ratio", float("nan"))),
                    "status": str(state.get("status", "")),
                })

    turbo = focus_region_output.get("turbo_lite", {}) if isinstance(focus_region_output, dict) else {}
    active_history = turbo.get("active_bounds_history", []) if isinstance(turbo, dict) else []
    if not isinstance(active_history, list):
        active_history = []
    for idx, event in enumerate(active_history, start=1):
        if not isinstance(event, dict):
            continue
        bounds = event.get("bounds", {}) if isinstance(event.get("bounds", {}), dict) else {}
        for feature in selected_features:
            item = bounds.get(str(feature))
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                rows.append({
                    "event_type": "active_update",
                    "event_index": int(idx),
                    "region_id": str(event.get("region_id", "")),
                    "feature": str(feature),
                    "lb": float(item[0]),
                    "ub": float(item[1]),
                    "batch_id": event.get("batch_id", pd.NA),
                    "iter": event.get("iter", pd.NA),
                    "volume_ratio": float(event.get("volume_ratio", float("nan"))),
                    "status": "",
                })
    return pd.DataFrame(rows)


def _plot_rect(ax, xlo: float, xhi: float, ylo: float, yhi: float, **kwargs) -> None:
    xs = [xlo, xhi, xhi, xlo, xlo]
    ys = [ylo, ylo, yhi, yhi, ylo]
    ax.plot(xs, ys, **kwargs)


def _coerce_bounds_item(item) -> tuple[float, float] | None:
    if isinstance(item, dict) and "lb" in item and "ub" in item:
        try:
            return float(item["lb"]), float(item["ub"])
        except (TypeError, ValueError):
            return None
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        try:
            return float(item[0]), float(item[1])
        except (TypeError, ValueError):
            return None
    return None


def _optimizer_result_status(
    *,
    history_df: pd.DataFrame,
    bounds_source: str,
    generated_bounds_available: bool = False,
) -> dict[str, object]:
    selected_bounds_available = str(bounds_source or "").strip().lower() in {"explorer", "path", "override"}
    optimization_bounds_available = bool(selected_bounds_available or generated_bounds_available)
    final_focus = ""
    if "segment" in history_df.columns and not history_df.empty:
        segments = [
            str(v).strip()
            for v in history_df["segment"].dropna().astype(str).tolist()
            if str(v).strip()
        ]
        if segments:
            final_focus = segments[-1]
    focus3_executed = bool("segment" in history_df.columns and (history_df["segment"].astype(str) == "focus3").any())
    if focus3_executed and optimization_bounds_available:
        result_status = "optimized"
        converged = True
    else:
        result_status = "exploratory_best"
        converged = False
    return {
        "result_status": result_status,
        "converged": bool(converged),
        "final_focus": str(final_focus),
        "selected_bounds_available": bool(selected_bounds_available),
        "generated_bounds_available": bool(generated_bounds_available),
        "optimization_bounds_available": bool(optimization_bounds_available),
        "focus3_executed": bool(focus3_executed),
    }


def _focus2_region_ids(focus_region_output: dict, history_df: pd.DataFrame) -> list[str]:
    ids: list[str] = []
    regions = focus_region_output.get("regions", []) if isinstance(focus_region_output, dict) else []
    if isinstance(regions, list):
        for region in regions:
            if isinstance(region, dict):
                rid = str(region.get("region_id", ""))
                if rid and rid not in ids:
                    ids.append(rid)
    if "focus2_selected_region_id" in history_df.columns:
        for rid in history_df["focus2_selected_region_id"].dropna().astype(str).tolist():
            if rid and rid != "nan" and rid not in ids:
                ids.append(rid)
    return ids


def _focus2_region_events(
    *,
    focus_region_output: dict,
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if "segment" in history_df.columns:
        focus2_rows = history_df[history_df["segment"].astype(str) == "focus2"].copy()
    else:
        focus2_rows = pd.DataFrame()
    for _, row in focus2_rows.iterrows():
        rid = str(row.get("focus2_selected_region_id", ""))
        if not rid or rid == "nan":
            continue
        rows.append({
            "event_type": "eval",
            "iter": int(row.get("iter", 0)),
            "batch_id": int(row.get("focus2_batch_id", 0)) if pd.notna(row.get("focus2_batch_id", pd.NA)) else 0,
            "region_id": rid,
            "source_mode": str(row.get("source_mode", "")),
            "objective": float(row.get("objective", float("nan"))),
            "objective_raw": float(row.get("objective_raw", float("nan"))),
            "scheduler_priority": float(row.get("focus2_scheduler_priority", float("nan"))),
            "batch_remaining": int(row.get("focus2_batch_remaining", 0)) if pd.notna(row.get("focus2_batch_remaining", pd.NA)) else 0,
            "duplicate_skip_count": int(row.get("focus2_duplicate_skip_count", 0)) if pd.notna(row.get("focus2_duplicate_skip_count", pd.NA)) else 0,
            "budget_skip_count": int(row.get("focus2_budget_skip_count", 0)) if pd.notna(row.get("focus2_budget_skip_count", pd.NA)) else 0,
        })

    batch_first_iter: dict[int, int] = {}
    region_last_iter: dict[str, int] = {}
    region_last_batch: dict[str, int] = {}
    if not focus2_rows.empty:
        for _, row in focus2_rows.iterrows():
            try:
                iter_value = int(row.get("iter", 0))
            except Exception:
                continue
            try:
                batch_value = int(row.get("focus2_batch_id", 0))
            except Exception:
                batch_value = 0
            rid = str(row.get("focus2_selected_region_id", ""))
            if batch_value > 0 and batch_value not in batch_first_iter:
                batch_first_iter[batch_value] = iter_value
            if rid and rid != "nan":
                region_last_iter[rid] = iter_value
                region_last_batch[rid] = batch_value

    turbo = focus_region_output.get("turbo_lite", {}) if isinstance(focus_region_output, dict) else {}
    if not isinstance(turbo, dict):
        turbo = {}

    scheduler_history = turbo.get("scheduler_history", [])
    if isinstance(scheduler_history, list):
        for batch in scheduler_history:
            if not isinstance(batch, dict):
                continue
            batch_id = int(batch.get("batch_id", 0) or 0)
            for item in list(batch.get("rows", []) or []):
                if not isinstance(item, dict):
                    continue
                rid = str(item.get("region_id", ""))
                if not rid:
                    continue
                rows.append({
                    "event_type": "scheduler_selected" if bool(item.get("selected", False)) else "scheduler_skipped",
                    "iter": batch_first_iter.get(batch_id, pd.NA),
                    "batch_id": batch_id,
                    "region_id": rid,
                    "scheduler_priority": float(item.get("priority", float("nan"))),
                    "scheduler_reason": str(item.get("reason", "")),
                    "eval_count": int(item.get("eval_count", 0)) if pd.notna(item.get("eval_count", pd.NA)) else 0,
                })

    active_bounds_history = turbo.get("active_bounds_history", [])
    if isinstance(active_bounds_history, list):
        for idx, event in enumerate(active_bounds_history, start=1):
            if not isinstance(event, dict):
                continue
            rows.append({
                "event_type": "active_bounds_update",
                "event_index": int(idx),
                "iter": event.get("iter", pd.NA),
                "batch_id": event.get("batch_id", pd.NA),
                "region_id": str(event.get("region_id", "")),
                "volume_ratio": float(event.get("volume_ratio", float("nan"))),
                "archive_count": int(event.get("archive_count", 0)) if pd.notna(event.get("archive_count", pd.NA)) else 0,
                "top_count": int(event.get("top_count", 0)) if pd.notna(event.get("top_count", pd.NA)) else 0,
            })

    merge_history = turbo.get("merge_history", [])
    if isinstance(merge_history, list):
        for idx, event in enumerate(merge_history, start=1):
            if not isinstance(event, dict):
                continue
            rows.append({
                "event_type": "merge_survivor",
                "event_index": int(idx),
                "iter": event.get("iter", pd.NA),
                "batch_id": event.get("batch_id", pd.NA),
                "region_id": str(event.get("survivor_region_id", "")),
                "other_region_id": str(event.get("absorbed_region_id", "")),
                "trigger": str(event.get("trigger", "")),
                "duplicate_count": int(event.get("duplicate_count", 0)) if pd.notna(event.get("duplicate_count", pd.NA)) else 0,
                "shared_count": int(event.get("shared_count", 0)) if pd.notna(event.get("shared_count", pd.NA)) else 0,
            })
            rows.append({
                "event_type": "merge_absorbed",
                "event_index": int(idx),
                "iter": event.get("iter", pd.NA),
                "batch_id": event.get("batch_id", pd.NA),
                "region_id": str(event.get("absorbed_region_id", "")),
                "other_region_id": str(event.get("survivor_region_id", "")),
                "trigger": str(event.get("trigger", "")),
            })

    regions = focus_region_output.get("regions", []) if isinstance(focus_region_output, dict) else []
    if isinstance(regions, list):
        for region in regions:
            if not isinstance(region, dict):
                continue
            state = region.get("turbo_lite_state", {}) if isinstance(region.get("turbo_lite_state", {}), dict) else {}
            status = str(state.get("status", ""))
            rid = str(region.get("region_id", ""))
            if rid and status in {"dropped", "merged"}:
                rows.append({
                    "event_type": f"final_status_{status}",
                    "iter": region_last_iter.get(rid, pd.NA),
                    "batch_id": region_last_batch.get(rid, pd.NA),
                    "region_id": rid,
                    "other_region_id": str(state.get("merged_into", "")),
                    "status": status,
                    "no_improve_count": int(state.get("no_improve_count", 0)),
                    "eval_count": int(state.get("eval_count", 0)),
                })
    return pd.DataFrame(rows)


def _plot_focus2_timeline(
    *,
    path: str,
    events_df: pd.DataFrame,
    region_ids: list[str],
) -> str | None:
    if events_df.empty or not region_ids:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    y_map = {rid: idx for idx, rid in enumerate(region_ids)}
    fig_h = max(3.8, 0.48 * len(region_ids) + 1.8)
    fig, ax = plt.subplots(figsize=(10.8, fig_h))

    for rid, y in y_map.items():
        ax.hlines(y, 0, max(1, int(pd.to_numeric(events_df.get("iter", pd.Series([0])), errors="coerce").max(skipna=True) or 1)), color="#e0e0e0", linewidth=0.8, zorder=0)

    def _x(row) -> float:
        val = row.get("iter", pd.NA)
        if pd.notna(val):
            try:
                return float(val)
            except Exception:
                pass
        val = row.get("batch_id", 0)
        if pd.isna(val):
            val = row.get("event_index", 0)
        try:
            return float(val)
        except Exception:
            return 0.0

    styles = {
        "eval": {"marker": "o", "color": "#1f77b4", "s": 48, "label": "eval"},
        "scheduler_selected": {"marker": "|", "color": "#2ca02c", "s": 70, "label": "scheduler selected"},
        "scheduler_skipped": {"marker": "v", "color": "#bdbdbd", "s": 34, "label": "scheduler skipped"},
        "active_bounds_update": {"marker": "s", "color": "#ff7f0e", "s": 36, "label": "bounds update"},
        "merge_survivor": {"marker": "*", "color": "#d62728", "s": 95, "label": "merge survivor"},
        "merge_absorbed": {"marker": "X", "color": "#7f7f7f", "s": 70, "label": "merge absorbed"},
        "final_status_dropped": {"marker": "x", "color": "#000000", "s": 70, "label": "dropped"},
        "final_status_merged": {"marker": "X", "color": "#7f7f7f", "s": 70, "label": "merged"},
    }
    used_labels: set[str] = set()
    for _, row in events_df.iterrows():
        rid = str(row.get("region_id", ""))
        if rid not in y_map:
            continue
        event_type = str(row.get("event_type", ""))
        style = styles.get(event_type)
        if style is None:
            continue
        label = str(style.get("label", event_type))
        ax.scatter(
            [_x(row)],
            [y_map[rid]],
            marker=str(style["marker"]),
            color=str(style["color"]),
            s=float(style["s"]),
            label=label if label not in used_labels else None,
            alpha=0.90,
            zorder=3,
        )
        used_labels.add(label)
        if event_type.startswith("merge_"):
            other = str(row.get("other_region_id", ""))
            if other:
                ax.annotate(
                    other,
                    xy=(_x(row), y_map[rid]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color="#555555",
                )

    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(region_ids)
    ax.set_xlabel("optimizer iter (fallback: batch id)")
    ax.set_ylabel("region")
    ax.set_title("Focus2 Region Timeline")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_focus2_debug_plots(
    *,
    debug_dir: str,
    focus_region_output: dict,
    history_df: pd.DataFrame,
    archive_df: pd.DataFrame,
    selected_features: list[str],
    objective_col: str,
    final_bounds: dict | None = None,
    known_optimum=None,
) -> dict[str, str | list[str]]:
    focus2_dir = os.path.join(debug_dir, "focus2")
    os.makedirs(focus2_dir, exist_ok=True)
    bounds_df = _bounds_pair_rows(
        focus_region_output=focus_region_output,
        selected_features=selected_features,
    )
    final_bounds_rows: list[dict[str, object]] = []
    if isinstance(final_bounds, dict) and final_bounds:
        for feature in selected_features:
            pair = _coerce_bounds_item(final_bounds.get(str(feature)))
            if pair is None:
                continue
            final_bounds_rows.append({
                "event_type": "focus3_final_bounds",
                "region_id": "focus3",
                "feature": str(feature),
                "lb": float(pair[0]),
                "ub": float(pair[1]),
                "batch_id": pd.NA,
                "iter": pd.NA,
                "volume_ratio": float("nan"),
                "status": "selected",
            })
    if final_bounds_rows:
        bounds_df = pd.concat([bounds_df, pd.DataFrame(final_bounds_rows)], ignore_index=True)
    bounds_csv_path = os.path.join(focus2_dir, "focus2_bounds_evolution.csv")
    bounds_df.to_csv(bounds_csv_path, index=False)
    events_df = _focus2_region_events(
        focus_region_output=focus_region_output,
        history_df=history_df,
    )
    events_csv_path = os.path.join(focus2_dir, "focus2_region_events.csv")
    events_df.to_csv(events_csv_path, index=False)
    region_ids = _focus2_region_ids(focus_region_output, history_df)
    timeline_path = _plot_focus2_timeline(
        path=os.path.join(focus2_dir, "focus2_region_timeline.png"),
        events_df=events_df,
        region_ids=region_ids,
    )

    plot_paths: list[str] = []
    if len(selected_features) < 2 or bounds_df.empty:
        return {
            "bounds_evolution_csv": bounds_csv_path,
            "region_events_csv": events_csv_path,
            "region_timeline_plot": timeline_path,
            "bounds_pair_plots": plot_paths,
        }
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {"bounds_evolution_csv": bounds_csv_path, "bounds_pair_plots": plot_paths}

    pairs = list(combinations(list(selected_features), 2))[:15]
    regions = focus_region_output.get("regions", []) if isinstance(focus_region_output, dict) else []
    if not isinstance(regions, list):
        regions = []
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    for fx, fy in pairs:
        fig, ax = plt.subplots(figsize=(6.6, 5.6))
        if fx in archive_df.columns and fy in archive_df.columns:
            c = None
            if objective_col in archive_df.columns:
                c = pd.to_numeric(archive_df[objective_col], errors="coerce")
            scatter_kwargs = {
                "s": 16,
                "alpha": 0.45,
                "edgecolors": "none",
            }
            if c is not None and c.notna().any():
                sc = ax.scatter(
                    pd.to_numeric(archive_df[fx], errors="coerce"),
                    pd.to_numeric(archive_df[fy], errors="coerce"),
                    c=c,
                    cmap="viridis",
                    **scatter_kwargs,
                )
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=objective_col)
            else:
                ax.scatter(
                    pd.to_numeric(archive_df[fx], errors="coerce"),
                    pd.to_numeric(archive_df[fy], errors="coerce"),
                    color="#bdbdbd",
                    label="archive",
                    **scatter_kwargs,
                )
        focus2_rows = history_df[history_df.get("segment", pd.Series(dtype=str)).astype(str) == "focus2"] if "segment" in history_df.columns else pd.DataFrame()
        if not focus2_rows.empty and fx in focus2_rows.columns and fy in focus2_rows.columns:
            ax.scatter(
                pd.to_numeric(focus2_rows[fx], errors="coerce"),
                pd.to_numeric(focus2_rows[fy], errors="coerce"),
                s=34,
                marker="x",
                color="#111111",
                linewidths=1.0,
                label="focus2 eval",
            )

        for idx, region in enumerate(regions):
            if not isinstance(region, dict):
                continue
            rid = str(region.get("region_id", ""))
            color = colors[idx % len(colors)]
            parent = region.get("bounds", {}) if isinstance(region.get("bounds", {}), dict) else {}
            px = parent.get(str(fx))
            py = parent.get(str(fy))
            if isinstance(px, (list, tuple)) and isinstance(py, (list, tuple)) and len(px) >= 2 and len(py) >= 2:
                _plot_rect(
                    ax,
                    float(px[0]),
                    float(px[1]),
                    float(py[0]),
                    float(py[1]),
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.55,
                    label=f"{rid} parent" if idx < 5 else None,
                )
            updates = bounds_df[
                (bounds_df["event_type"] == "active_update")
                & (bounds_df["region_id"].astype(str) == rid)
            ]
            update_indices = sorted(updates.get("event_index", pd.Series(dtype=int)).dropna().unique().tolist())
            for u_pos, event_index in enumerate(update_indices):
                rows = updates[updates["event_index"] == event_index]
                xrow = rows[rows["feature"].astype(str) == str(fx)]
                yrow = rows[rows["feature"].astype(str) == str(fy)]
                if xrow.empty or yrow.empty:
                    continue
                alpha = min(0.18 + 0.10 * u_pos, 0.70)
                _plot_rect(
                    ax,
                    float(xrow.iloc[0]["lb"]),
                    float(xrow.iloc[0]["ub"]),
                    float(yrow.iloc[0]["lb"]),
                    float(yrow.iloc[0]["ub"]),
                    color=color,
                    linestyle="-",
                    linewidth=0.9,
                    alpha=alpha,
                )
            state = region.get("turbo_lite_state", {}) if isinstance(region.get("turbo_lite_state", {}), dict) else {}
            active = state.get("active_bounds", {}) if isinstance(state.get("active_bounds", {}), dict) else {}
            axb = active.get(str(fx))
            ayb = active.get(str(fy))
            status = str(state.get("status", "active"))
            if isinstance(axb, (list, tuple)) and isinstance(ayb, (list, tuple)) and len(axb) >= 2 and len(ayb) >= 2:
                line_style = ":" if status in {"merged", "dropped"} else "-"
                line_width = 2.6 if status == "active" else 1.8
                _plot_rect(
                    ax,
                    float(axb[0]),
                    float(axb[1]),
                    float(ayb[0]),
                    float(ayb[1]),
                    color=color,
                    linestyle=line_style,
                    linewidth=line_width,
                    alpha=0.95,
                    label=f"{rid} final {status}" if idx < 5 else None,
                )

        final_x = _coerce_bounds_item((final_bounds or {}).get(str(fx))) if isinstance(final_bounds, dict) else None
        final_y = _coerce_bounds_item((final_bounds or {}).get(str(fy))) if isinstance(final_bounds, dict) else None
        if final_x is not None and final_y is not None:
            _plot_rect(
                ax,
                float(final_x[0]),
                float(final_x[1]),
                float(final_y[0]),
                float(final_y[1]),
                color="#0057b8",
                linestyle=(0, (4, 2)),
                linewidth=2.4,
                alpha=0.88,
                label="focus3 final bounds",
            )

        _plot_known_optimum_markers(ax, known_optimum=known_optimum, fx=str(fx), fy=str(fy))

        ax.set_xlabel(str(fx))
        ax.set_ylabel(str(fy))
        ax.set_title(f"Focus2 Bounds Evolution: {fx} vs {fy}")
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            dedup = {}
            for h, label in zip(handles, labels):
                if label and label not in dedup:
                    dedup[label] = h
            ax.legend(dedup.values(), dedup.keys(), fontsize=7, loc="best", frameon=False)
        fig.tight_layout()
        path = os.path.join(
            focus2_dir,
            f"focus2_bounds_evolution_{_safe_plot_token(fx)}_{_safe_plot_token(fy)}.png",
        )
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(path)
    return {
        "bounds_evolution_csv": bounds_csv_path,
        "region_events_csv": events_csv_path,
        "region_timeline_plot": timeline_path,
        "bounds_pair_plots": plot_paths,
    }


def _write_focus3_debug_plots(
    *,
    debug_dir: str,
    history_df: pd.DataFrame,
    archive_df: pd.DataFrame,
    selected_features: list[str],
    selected_bounds: dict[str, tuple[float, float]],
    objective_col: str,
    known_optimum=None,
) -> dict[str, str | list[str]]:
    focus3_dir = os.path.join(debug_dir, "focus3")
    os.makedirs(focus3_dir, exist_ok=True)
    if "segment" in history_df.columns:
        focus3_rows = history_df[history_df["segment"].astype(str) == "focus3"].copy()
    else:
        focus3_rows = pd.DataFrame()

    trajectory_cols = [
        "iter",
        "source_mode",
        "objective",
        "objective_raw",
        "raw_improvement",
        "raw_improved",
        "acq_type",
        "kappa",
        "focus3_plan_mode",
        "focus3_acq_effective",
        "focus3_acq_reason",
        "focus3_acq_data_ratio",
        "focus3_acq_mean_enabled",
        "focus3_acq_mean_min_data_ratio",
        "focus3_acq_mean_max_volume_ratio",
        "focus3_acq_best_local_active",
        "source_prob_topk",
        "source_prob_best_local",
        "source_prob_boundary",
        "source_prob_random",
        "focus3_recover_active",
        "focus3_recover_reason",
        "focus3_recover_level",
        "focus3_recover_no_improve_count",
        "focus3_recover_topk_bonus",
        "focus3_recover_boundary_bonus",
        "focus3_recover_random_bonus",
        "focus3_source_cap_applied",
        "focus3_source_cap_reason",
        "focus3_source_cap_boundary_max",
        "focus3_source_cap_topk_min",
        "focus3_source_cap_random_min",
        "focus3_boundary_gate_applied",
        "focus3_boundary_gate_allowed",
        "focus3_boundary_gate_reason",
        "focus3_boundary_gate_margin",
        "focus3_boundary_gate_boundary_score",
        "focus3_boundary_gate_non_boundary_best",
        "focus3_best_local_applied",
        "focus3_best_local_reason",
        "focus3_best_local_prob",
        "focus3_best_local_data_ratio",
        "focus3_best_local_min_focus3_evals",
        "focus3_best_local_min_data_ratio",
        "focus3_selected_best_local",
        "focus3_best_score_best_local",
        "focus3_dedup_applied",
        "focus3_dedup_fallback",
        "focus3_archive_distance_after_dedup",
    ] + list(selected_features)
    for col in trajectory_cols:
        if col not in focus3_rows.columns:
            focus3_rows[col] = pd.NA
    trajectory_path = os.path.join(focus3_dir, "focus3_trajectory.csv")
    focus3_rows[trajectory_cols].to_csv(trajectory_path, index=False)

    plot_paths: list[str] = []
    if focus3_rows.empty or len(selected_features) < 2:
        return {"trajectory_csv": trajectory_path, "trajectory_pair_plots": plot_paths}
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {"trajectory_csv": trajectory_path, "trajectory_pair_plots": plot_paths}

    pairs = list(combinations(list(selected_features), 2))[:15]
    for fx, fy in pairs:
        fig, ax = plt.subplots(figsize=(7.4, 5.8))
        if fx in archive_df.columns and fy in archive_df.columns:
            c = None
            if objective_col in archive_df.columns:
                c = pd.to_numeric(archive_df[objective_col], errors="coerce")
            if c is not None and c.notna().any():
                sc = ax.scatter(
                    pd.to_numeric(archive_df[fx], errors="coerce"),
                    pd.to_numeric(archive_df[fy], errors="coerce"),
                    c=c,
                    cmap="viridis",
                    s=13,
                    alpha=0.22,
                    edgecolors="none",
                    label="archive",
                    zorder=1,
                )
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=objective_col)
            else:
                ax.scatter(
                    pd.to_numeric(archive_df[fx], errors="coerce"),
                    pd.to_numeric(archive_df[fy], errors="coerce"),
                    color="#bdbdbd",
                    s=14,
                    alpha=0.22,
                    edgecolors="none",
                    label="archive",
                    zorder=1,
                )

        xs = pd.to_numeric(focus3_rows[fx], errors="coerce")
        ys = pd.to_numeric(focus3_rows[fy], errors="coerce")
        valid = xs.notna() & ys.notna()
        if valid.any():
            # BO does not follow a physical path. Keep the connection line very
            # light so the plot reads as evaluated points, not a trajectory.
            ax.plot(xs[valid], ys[valid], color="#111111", linewidth=0.55, alpha=0.18, label="eval order", zorder=2)
            plan_modes = focus3_rows.get("focus3_plan_mode", pd.Series("", index=focus3_rows.index)).astype(str)
            source_modes = focus3_rows.get("source_mode", pd.Series("", index=focus3_rows.index)).astype(str)
            marker_specs = [
                ("refine", "D", "#111111", "refine eval"),
                ("discrete", "o", "#111111", "discrete eval"),
            ]
            plotted = set()
            for mode, marker, color, label in marker_specs:
                mask = valid & plan_modes.str.lower().eq(mode)
                if mask.any():
                    ax.scatter(
                        xs[mask],
                        ys[mask],
                        s=46 if mode == "discrete" else 58,
                        marker=marker,
                        facecolors=color,
                        edgecolors="white",
                        linewidths=0.45,
                        alpha=0.82,
                        label=label,
                        zorder=4,
                    )
                    plotted.add(mode)
            fallback_mask = valid & ~plan_modes.str.lower().isin(plotted)
            if fallback_mask.any():
                # Older result files may not contain focus3_plan_mode. Fall back
                # to source_mode so historical debug plots remain readable.
                for source, marker, label in (
                    ("topk", "^", "topk eval"),
                    ("boundary", "s", "boundary eval"),
                    ("random", "o", "random eval"),
                    ("plan_refine", "D", "refine eval"),
                    ("plan_discrete", "o", "discrete eval"),
                ):
                    mask = fallback_mask & source_modes.str.lower().eq(source)
                    if mask.any():
                        ax.scatter(
                            xs[mask],
                            ys[mask],
                            s=48,
                            marker=marker,
                            facecolors="#111111",
                            edgecolors="white",
                            linewidths=0.45,
                            alpha=0.82,
                            label=label,
                            zorder=4,
                        )
                other_mask = fallback_mask & ~source_modes.str.lower().isin({"topk", "boundary", "random", "plan_refine", "plan_discrete"})
                if other_mask.any():
                    ax.scatter(
                        xs[other_mask],
                        ys[other_mask],
                        s=44,
                        marker="o",
                        facecolors="#111111",
                        edgecolors="white",
                        linewidths=0.45,
                        alpha=0.82,
                        label="focus3 eval",
                        zorder=4,
                    )
            first_i = xs[valid].index[0]
            last_i = xs[valid].index[-1]
            ax.scatter([xs.loc[first_i]], [ys.loc[first_i]], s=95, marker="P", color="#377eb8", edgecolors="#111111", linewidths=0.6, label="start", zorder=7)
            ax.scatter([xs.loc[last_i]], [ys.loc[last_i]], s=105, marker="X", color="#984ea3", edgecolors="#111111", linewidths=0.6, label="last", zorder=7)

        improved = focus3_rows.get("raw_improved", pd.Series(dtype=bool)).astype(str).str.lower().isin({"true", "1"})
        if improved.any():
            ax.scatter(
                xs[improved & valid],
                ys[improved & valid],
                s=120,
                marker="*",
                color="#2ca02c",
                edgecolors="#111111",
                linewidths=0.65,
                label="improved",
                zorder=8,
            )
        recovered = focus3_rows.get("focus3_recover_active", pd.Series(dtype=bool)).astype(str).str.lower().isin({"true", "1"})
        if recovered.any():
            ax.scatter(
                xs[recovered & valid],
                ys[recovered & valid],
                s=92,
                marker="s",
                facecolors="none",
                edgecolors="#ff7f0e",
                linewidths=1.6,
                label="recover",
                zorder=6,
            )
        deduped = focus3_rows.get("focus3_dedup_applied", pd.Series(dtype=bool)).astype(str).str.lower().isin({"true", "1"})
        if deduped.any():
            ax.scatter(
                xs[deduped & valid],
                ys[deduped & valid],
                s=74,
                marker="x",
                color="#d62728",
                linewidths=1.4,
                label="dedup fallback",
                zorder=9,
            )

        bx = selected_bounds.get(str(fx))
        by = selected_bounds.get(str(fy))
        if isinstance(bx, (list, tuple)) and isinstance(by, (list, tuple)) and len(bx) >= 2 and len(by) >= 2:
            _plot_rect(
                ax,
                float(bx[0]),
                float(bx[1]),
                float(by[0]),
                float(by[1]),
                color="#1f77b4",
                linestyle="--",
                linewidth=1.4,
                alpha=0.75,
                label="focus3 bounds",
            )

        _plot_known_optimum_markers(ax, known_optimum=known_optimum, fx=str(fx), fy=str(fy))

        ax.set_xlabel(str(fx))
        ax.set_ylabel(str(fy))
        ax.set_title(f"Focus3 Trajectory: {fx} vs {fy}")
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            dedup = {}
            for h, label in zip(handles, labels):
                if label and label not in dedup:
                    dedup[label] = h
            ax.legend(dedup.values(), dedup.keys(), fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
        fig.tight_layout()
        path = os.path.join(
            focus3_dir,
            f"focus3_trajectory_{_safe_plot_token(fx)}_{_safe_plot_token(fy)}.png",
        )
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(path)

    return {"trajectory_csv": trajectory_path, "trajectory_pair_plots": plot_paths}


def _write_optimizer_debug_tables(
    *,
    debug_dir: str,
    history_df: pd.DataFrame,
    bo_result: OptimizerAlgorithmResult,
    final_bounds: dict,
    resolved: ResolvedOptimizerInputs,
) -> dict[str, str]:
    out: dict[str, str] = {}
    if history_df.empty:
        return out

    summary_rows: list[dict[str, object]] = []
    if "segment" in history_df.columns:
        for segment, g in history_df.groupby(history_df["segment"].astype(str), dropna=False):
            y = pd.to_numeric(g.get("objective_raw", pd.Series(dtype=float)), errors="coerce")
            improved = g.get("raw_improved", pd.Series(False, index=g.index))
            summary_rows.append(
                {
                    "segment": str(segment),
                    "count": int(len(g)),
                    "objective_raw_min": float(y.min()) if y.notna().any() else float("nan"),
                    "objective_raw_median": float(y.median()) if y.notna().any() else float("nan"),
                    "objective_raw_max": float(y.max()) if y.notna().any() else float("nan"),
                    "improved_count": int(pd.Series(improved).fillna(False).astype(bool).sum()),
                    "first_iter": int(pd.to_numeric(g.get("iter", pd.Series(dtype=float)), errors="coerce").min()) if "iter" in g.columns else 0,
                    "last_iter": int(pd.to_numeric(g.get("iter", pd.Series(dtype=float)), errors="coerce").max()) if "iter" in g.columns else 0,
                }
            )
    focus_summary_path = os.path.join(debug_dir, "optimizer_focus_summary.csv")
    pd.DataFrame(summary_rows).to_csv(focus_summary_path, index=False)
    out["focus_summary_csv"] = focus_summary_path

    source_rows: list[dict[str, object]] = []
    if {"segment", "source_mode"}.issubset(history_df.columns):
        for (segment, source), g in history_df.groupby(
            [history_df["segment"].astype(str), history_df["source_mode"].astype(str)],
            dropna=False,
        ):
            y = pd.to_numeric(g.get("objective_raw", pd.Series(dtype=float)), errors="coerce")
            improved = g.get("raw_improved", pd.Series(False, index=g.index))
            source_rows.append(
                {
                    "segment": str(segment),
                    "source_mode": str(source),
                    "count": int(len(g)),
                    "objective_raw_min": float(y.min()) if y.notna().any() else float("nan"),
                    "objective_raw_median": float(y.median()) if y.notna().any() else float("nan"),
                    "improved_count": int(pd.Series(improved).fillna(False).astype(bool).sum()),
                }
            )
    source_summary_path = os.path.join(debug_dir, "optimizer_source_summary.csv")
    pd.DataFrame(source_rows).to_csv(source_summary_path, index=False)
    out["source_summary_csv"] = source_summary_path

    debug_plan_path = os.path.join(debug_dir, "optimizer_debug_plan.json")
    with open(debug_plan_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "problem_name": str(resolved.problem_name),
                "objective_sense": str(resolved.objective_sense),
                "selected_features": list(resolved.selected_features),
                "bounds_source": str(resolved.bounds_source),
                "final_bounds": final_bounds,
                "focus_pipeline_summary": bo_result.focus_pipeline_summary,
                "algorithm_summary": bo_result.algorithm_summary,
                "focus2_summary_keys": sorted(list(bo_result.focus2_summary.keys())) if isinstance(bo_result.focus2_summary, dict) else [],
                "history_columns": list(history_df.columns),
                "n_history_rows": int(len(history_df)),
                "n_archive_rows": int(len(bo_result.archive_df)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    out["debug_plan_json"] = debug_plan_path
    return out


def save_optimizer_outputs(
    *,
    config: OptimizerConfig,
    run_context: RunContext,
    resolved: ResolvedOptimizerInputs,
    bo_result: OptimizerAlgorithmResult,
) -> dict:
    output_t0 = time.perf_counter()
    task_name = "OPT"
    task_dir = os.path.join(run_context.run_root, "OPT")
    artifacts_dir = os.path.join(task_dir, "artifacts")
    public_dir = os.path.join(artifacts_dir, "public")
    meta_dir = os.path.join(artifacts_dir, "meta")
    debug_dir = os.path.join(artifacts_dir, "debug")
    os.makedirs(public_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    history_df = bo_result.history_df.copy()
    generated_bounds_available = bool(
        isinstance(bo_result.focus2_summary, dict)
        and bool(bo_result.focus2_summary.get("final_bounds", {}) or {})
    )
    result_status_info = _optimizer_result_status(
        history_df=history_df,
        bounds_source=str(resolved.bounds_source),
        generated_bounds_available=generated_bounds_available,
    )

    best_point_path = os.path.join(public_dir, "best_point.json")
    with open(best_point_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **result_status_info,
                "best_point": bo_result.best_point,
                "best_objective": float(bo_result.best_objective),
                "best_point_raw": bo_result.best_point_raw,
                "best_objective_raw": float(bo_result.best_objective_raw),
                "objective_sense": str(resolved.objective_sense),
                "goal": bo_result.algorithm_summary.get("goal") if isinstance(bo_result.algorithm_summary, dict) else None,
                "goal_objective": bo_result.algorithm_summary.get("goal_objective") if isinstance(bo_result.algorithm_summary, dict) else None,
                "goal_hit": bool(bo_result.algorithm_summary.get("goal_hit", False)) if isinstance(bo_result.algorithm_summary, dict) else False,
                "goal_first_hit_iter": int(bo_result.algorithm_summary.get("goal_first_hit_iter", 0)) if isinstance(bo_result.algorithm_summary, dict) else 0,
                "goal_early_stopped": bool(bo_result.algorithm_summary.get("goal_early_stopped", False)) if isinstance(bo_result.algorithm_summary, dict) else False,
                "goal_early_stop_reason": str(bo_result.algorithm_summary.get("goal_early_stop_reason", "")) if isinstance(bo_result.algorithm_summary, dict) else "",
                "post_penalty_active": bool(bo_result.post_penalty_active),
                "post_score_mode": str(bo_result.post_score_mode),
                "post_penalty_lambda": float(bo_result.post_penalty_lambda),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Main CSV는 "OPT 추가 평가점"만 저장한다 (DOE seed 제외).
    opt_point_cols = [
        "iter",
        "segment",
        "opt_focus_level",
        "source_mode",
        "success",
        "feasible",
        "objective",
        "objective_raw",
        "p_feasible",
        "pre_feasible",
        "pre_margin",
        "pre_violation",
        "pre_constraint_worst_id",
        "pre_constraint_worst_margin",
    ] + list(resolved.selected_features)
    if "success" not in history_df.columns:
        history_df["success"] = True
    if "feasible" not in history_df.columns:
        if "pre_feasible" in history_df.columns:
            history_df["feasible"] = history_df["pre_feasible"]
        else:
            history_df["feasible"] = True
    for col in opt_point_cols:
        if col not in history_df.columns:
            history_df[col] = pd.NA
    opt_points_df = history_df[opt_point_cols].copy()

    selected_features_path = os.path.join(meta_dir, "selected_features.csv")
    selected_features_df = pd.DataFrame(
        [
            {
                "order": int(idx),
                "feature": str(feature),
                "lb": float(resolved.selected_bounds[feature][0]),
                "ub": float(resolved.selected_bounds[feature][1]),
                "bounds_source": str(resolved.bounds_source),
            }
            for idx, feature in enumerate(resolved.selected_features)
        ]
    )
    selected_features_df.to_csv(selected_features_path, index=False)

    optimizer_inputs_path = os.path.join(meta_dir, "optimizer_inputs.json")
    with open(optimizer_inputs_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "problem_name": str(resolved.problem_name),
                "objective_col": str(config.system.objective_col),
                "objective_sense": str(resolved.objective_sense),
                "selected_features": list(resolved.selected_features),
                "selected_bounds": {
                    str(feature): {
                        "lb": float(resolved.selected_bounds[feature][0]),
                        "ub": float(resolved.selected_bounds[feature][1]),
                    }
                    for feature in resolved.selected_features
                },
                "bounds_source": str(resolved.bounds_source),
                "bounds_path": resolved.bounds_path,
                "cae_metadata_path": resolved.cae_metadata_path,
                "doe_metadata_path": resolved.doe_metadata_path,
                "explorer_metadata_path": resolved.explorer_metadata_path,
                "modeler_metadata_path": resolved.modeler_metadata_path,
                "post_feasibility_model_path": resolved.post_feasibility_model_path,
                "post_feasibility_model_kind": str(resolved.post_feasibility_model_kind),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    focus_regions_path = os.path.join(meta_dir, "focus_regions.json")
    focus_region_output = {}
    if isinstance(bo_result.focus2_summary, dict):
        focus_region_output = dict(bo_result.focus2_summary.get("region_output", {}) or {})
    if not focus_region_output:
        focus_region_output = {
            "enabled": False,
            "status": "skipped",
            "reason": "no_focus2_region_output",
            "regions": [],
        }
    with open(focus_regions_path, "w", encoding="utf-8") as f:
        json.dump(
            focus_region_output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    focus_bounds_path = os.path.join(meta_dir, "focus_bounds.json")
    final_bounds = {}
    final_info = {}
    if isinstance(bo_result.focus2_summary, dict):
        final_bounds = bo_result.focus2_summary.get("final_bounds", {}) or {}
        final_info = bo_result.focus2_summary.get("final", {}) or {}
    with open(focus_bounds_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_focus": "focus2" if final_bounds else "input",
                "source_region_id": (
                    str(final_info.get("selected_region_id", ""))
                    if isinstance(final_info, dict)
                    else ""
                ),
                "reason": (
                    str(final_info.get("method", "input_bounds"))
                    if isinstance(final_info, dict)
                    else "input_bounds"
                ),
                "selection": (
                    final_info.get("selection", {})
                    if isinstance(final_info, dict)
                    else {}
                ),
                "bounds": final_bounds if final_bounds else {
                    str(feature): {
                        "lb": float(resolved.selected_bounds[feature][0]),
                        "ub": float(resolved.selected_bounds[feature][1]),
                    }
                    for feature in resolved.selected_features
                },
                "bounds_source": str(resolved.bounds_source),
                "objective_sense": str(resolved.objective_sense),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    debug_enabled = str(config.system.debug_level).strip().lower() == "on"
    history_full_path = None
    focus2_debug_artifacts: dict[str, str | list[str]] = {}
    focus3_debug_artifacts: dict[str, str | list[str]] = {}
    optimizer_debug_tables: dict[str, str] = {}
    debug_write_sec = 0.0
    focus2_plot_sec = 0.0
    focus3_plot_sec = 0.0
    if debug_enabled:
        t0 = time.perf_counter()
        history_full_path = os.path.join(debug_dir, "optimizer_history_full.csv")
        history_df.to_csv(history_full_path, index=False)
        debug_write_sec += float(time.perf_counter() - t0)
        t0 = time.perf_counter()
        optimizer_debug_tables = _write_optimizer_debug_tables(
            debug_dir=debug_dir,
            history_df=history_df,
            bo_result=bo_result,
            final_bounds=final_bounds,
            resolved=resolved,
        )
        debug_write_sec += float(time.perf_counter() - t0)
        t0 = time.perf_counter()
        focus2_debug_artifacts = _write_focus2_debug_plots(
            debug_dir=debug_dir,
            focus_region_output=focus_region_output,
            history_df=history_df,
            archive_df=bo_result.archive_df.copy(),
            selected_features=list(resolved.selected_features),
            objective_col="objective",
            final_bounds=final_bounds,
            known_optimum=config.user.known_optimum,
        )
        focus2_plot_sec += float(time.perf_counter() - t0)
        focus3_plot_bounds: dict[str, tuple[float, float]] = {}
        if isinstance(final_bounds, dict) and final_bounds:
            for feature in resolved.selected_features:
                item = final_bounds.get(str(feature))
                if isinstance(item, dict) and "lb" in item and "ub" in item:
                    focus3_plot_bounds[str(feature)] = (float(item["lb"]), float(item["ub"]))
        if not focus3_plot_bounds:
            focus3_plot_bounds = dict(resolved.selected_bounds)
        t0 = time.perf_counter()
        focus3_debug_artifacts = _write_focus3_debug_plots(
            debug_dir=debug_dir,
            history_df=history_df,
            archive_df=bo_result.archive_df.copy(),
            selected_features=list(resolved.selected_features),
            selected_bounds=focus3_plot_bounds,
            objective_col="objective",
            known_optimum=config.user.known_optimum,
        )
        focus3_plot_sec += float(time.perf_counter() - t0)
        print(
            "[Optimizer][OutputTiming] "
            f"debug_csv={debug_write_sec:.3f}s "
            f"focus2_plots={focus2_plot_sec:.3f}s "
            f"focus3_plots={focus3_plot_sec:.3f}s",
            flush=True,
        )

    previous = {}
    if resolved.doe_metadata_path:
        try:
            previous["DOE"] = os.path.relpath(resolved.doe_metadata_path, task_dir)
        except ValueError:
            previous["DOE"] = resolved.doe_metadata_path
    if resolved.explorer_metadata_path:
        try:
            previous["Explorer"] = os.path.relpath(resolved.explorer_metadata_path, task_dir)
        except ValueError:
            previous["Explorer"] = resolved.explorer_metadata_path
    if resolved.modeler_metadata_path:
        try:
            previous["Modeler"] = os.path.relpath(resolved.modeler_metadata_path, task_dir)
        except ValueError:
            previous["Modeler"] = resolved.modeler_metadata_path

    inputs = {
        "user_config": os.path.relpath(run_context.user_config_snapshot_path, task_dir),
        "system_config_snapshot": asdict(config.system),
        "previous": previous,
        "selected_features": list(resolved.selected_features),
        "selected_bounds_path": resolved.bounds_path,
        "post_feasibility_model_path": resolved.post_feasibility_model_path,
        "post_feasibility_model_kind": str(resolved.post_feasibility_model_kind),
        "bounds_source": str(resolved.bounds_source),
    }

    resolved_params = {
        "seed": int(resolved.seed),
        "objective_sense": str(resolved.objective_sense),
        "n_features": int(len(resolved.selected_features)),
        "n_constraints": int(len(resolved.constraint_defs)),
        "n_samples_requested": int(config.user.n_samples),
        "optimizer_algorithm_id": str(config.system.algorithm_id),
        "optimizer_progress_scheme": "focus",
        "optimizer_output_schema": "optimizer_points_v2_focus",
        "optimizer_debug_schema": "optimizer_history_v2_focus",
    }

    focus3_budget_classes: list[str] = []
    if "focus3_budget_class" in history_df.columns:
        seen = set()
        for v in history_df["focus3_budget_class"].astype(str).tolist():
            if v not in seen:
                focus3_budget_classes.append(v)
                seen.add(v)

    results = {
        "n_iterations": int(bo_result.n_iterations),
        **result_status_info,
        "best_objective": float(bo_result.best_objective),
        "best_objective_raw": float(bo_result.best_objective_raw),
        "best_point": dict(bo_result.best_point),
        "best_point_raw": dict(bo_result.best_point_raw),
        "post_penalty_active": bool(bo_result.post_penalty_active),
        "post_penalty_lambda": float(bo_result.post_penalty_lambda),
        "post_score_mode": str(bo_result.post_score_mode),
        "feasibility_model_kind": str(bo_result.feasibility_model_kind),
        "feasibility_status": str(bo_result.feasibility_status),
        "n_history_rows": int(len(bo_result.history_df)),
        "n_archive_rows": int(len(bo_result.archive_df)),
        "initial_archive_count": int(bo_result.initial_archive_count),
        "final_train_count": int(bo_result.final_train_count),
        "gp_train_cap": int(bo_result.gp_train_cap),
        "algorithm_summary": dict(bo_result.algorithm_summary),
        "focus2_summary": bo_result.focus2_summary,
        "focus_pipeline_summary": bo_result.focus_pipeline_summary,
        "focus3_budget_classes": focus3_budget_classes,
    }

    saver = ResultSaver(use_timestamp=bool(config.cae.system.use_timestamp))
    task_out = saver.save_task_v3(
        run_root=run_context.run_root,
        task=task_name,
        problem_name=resolved.problem_name,
        df=opt_points_df,
        inputs=inputs,
        resolved_params=resolved_params,
        results=results,
        public_artifacts={
            "best_point": best_point_path,
            "optimizer_points": os.path.join(public_dir, "opt_results.csv"),
        },
        meta_artifacts={
            "selected_features": selected_features_path,
            "optimizer_inputs": optimizer_inputs_path,
            "focus_regions": focus_regions_path,
            "focus_bounds": focus_bounds_path,
        },
        debug_artifacts={
            **({"history_full": history_full_path} if history_full_path else {}),
            **optimizer_debug_tables,
            **(
                {
                    "focus2_bounds_evolution_csv": focus2_debug_artifacts.get("bounds_evolution_csv"),
                    "focus2_region_events_csv": focus2_debug_artifacts.get("region_events_csv"),
                    "focus2_region_timeline_plot": focus2_debug_artifacts.get("region_timeline_plot"),
                    "focus2_bounds_pair_plots": focus2_debug_artifacts.get("bounds_pair_plots", []),
                }
                if focus2_debug_artifacts
                else {}
            ),
            **(
                {
                    "focus3_trajectory_csv": focus3_debug_artifacts.get("trajectory_csv"),
                    "focus3_trajectory_pair_plots": focus3_debug_artifacts.get("trajectory_pair_plots", []),
                }
                if focus3_debug_artifacts
                else {}
            ),
        },
    )
    update_run_index(run_context, task_name, task_out["metadata"])
    print(
        "[Optimizer][OutputTiming] "
        f"total={float(time.perf_counter() - output_t0):.3f}s",
        flush=True,
    )
    return task_out
