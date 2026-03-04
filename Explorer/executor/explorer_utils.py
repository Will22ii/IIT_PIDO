from typing import Optional, Tuple

import os
import numpy as np
import pandas as pd


def resolve_selected_features(
    *,
    modeler_meta: dict | None,
    modeler_task_dir: str | None,
    modeler_df: pd.DataFrame | None,
    feature_cols: list[str] | None,
    doe_df: pd.DataFrame | None,
) -> list[str]:
    selected_features: list[str] = []
    if modeler_meta and modeler_meta.get("selected_features"):
        selected_features = list(modeler_meta["selected_features"])
    elif modeler_task_dir and modeler_meta and modeler_meta.get("artifacts", {}).get("selected_features"):
        selected_path = modeler_meta["artifacts"]["selected_features"]
        selected_df = pd.read_csv(
            os.path.join(modeler_task_dir, selected_path)
        )
        if {"feature", "selected"}.issubset(selected_df.columns):
            selected_features = (
                selected_df[selected_df["selected"]]["feature"].tolist()
            )
    elif feature_cols:
        selected_features = list(feature_cols)
    elif modeler_df is not None and {"feature", "selected"}.issubset(modeler_df.columns):
        selected_features = (
            modeler_df[modeler_df["selected"]]["feature"].tolist()
        )

    if not selected_features:
        if feature_cols:
            print("[Explorer] No selected features flagged; fallback to all feature_cols.")
            selected_features = list(feature_cols)
        elif doe_df is not None:
            ignore_cols = {"id", "objective", "constraints", "feasible", "success", "source", "round", "exec_scope"}
            selected_features = [c for c in doe_df.columns if c not in ignore_cols]
            if selected_features:
                print("[Explorer] No selected features flagged; fallback to DOE columns.")
            else:
                raise RuntimeError("Selected features not found.")
        else:
            raise RuntimeError("Selected features not found.")

    return selected_features


def resolve_bounds(
    *,
    selected_features: list[str],
    variables: Optional[list[dict]],
    df: pd.DataFrame | None,
) -> list[Tuple[float, float]]:
    bounds = []
    if not variables:
        print("[Explorer] Variables not found in DOE metadata.")
        raise RuntimeError("Cannot resolve bounds without DOE variables metadata.")
    for feature in selected_features:
        matched = next(
            (v for v in variables if v.get("name") == feature),
            None,
        )
        if matched:
            bounds.append((matched["lb"], matched["ub"]))
            continue
        print(f"[Explorer] Cannot resolve bounds for feature: {feature}")
        raise RuntimeError(f"Cannot resolve bounds for feature: {feature}")
    return bounds


def format_span_rows(
    *,
    kind: str,
    spans: list | None,
    vols: list | None,
    feature_names: list[str],
) -> None:
    if spans is None:
        return
    for idx, ratios in enumerate(spans):
        if ratios is None:
            continue
        parts = []
        for name, val in zip(feature_names, ratios):
            try:
                parts.append(f"{name}:{float(val):.2f}")
            except Exception:
                parts.append(f"{name}:nan")
        vol = None
        if vols is not None and idx < len(vols):
            vol = vols[idx]
        if vol is None:
            vol_str = "nan"
        else:
            try:
                vol_str = f"{float(vol):.4f}"
            except Exception:
                vol_str = "nan"
        print(f"[Explorer] {kind} cluster{idx+1} span: " + ", ".join(parts) + f" | volume={vol_str}")


def compute_selected_bounds(
    *,
    X_pred_sel: np.ndarray,
    X_obj_sel: np.ndarray,
) -> tuple[list[Tuple[float, float]] | None, list[Tuple[float, float]] | None, list[Tuple[float, float]] | None]:
    pred_bounds = None
    obj_bounds = None
    if X_pred_sel.size:
        pred_bounds = []
        for j in range(X_pred_sel.shape[1]):
            pred_bounds.append(
                (float(X_pred_sel[:, j].min()), float(X_pred_sel[:, j].max()))
            )
    if X_obj_sel.size:
        obj_bounds = []
        for j in range(X_obj_sel.shape[1]):
            obj_bounds.append(
                (float(X_obj_sel[:, j].min()), float(X_obj_sel[:, j].max()))
            )

    selected_bounds = None
    if pred_bounds is not None and obj_bounds is not None:
        selected_bounds = []
        for (p_lb, p_ub), (o_lb, o_ub) in zip(pred_bounds, obj_bounds):
            selected_bounds.append((min(p_lb, o_lb), max(p_ub, o_ub)))
    elif pred_bounds is not None:
        selected_bounds = pred_bounds
    elif obj_bounds is not None:
        selected_bounds = obj_bounds

    return selected_bounds, pred_bounds, obj_bounds


def apply_bounds_margin(
    *,
    selected_bounds: list[Tuple[float, float]],
    bounds: list[Tuple[float, float]],
    margin_ratio: float,
) -> list[Tuple[float, float]]:
    if margin_ratio <= 0:
        return selected_bounds
    spans = [float(ub - lb) for lb, ub in bounds]
    mean_span = float(np.mean(spans)) if spans else 0.0
    eps = mean_span * float(margin_ratio)
    if eps <= 0:
        return selected_bounds
    expanded = []
    for (s_lb, s_ub), (o_lb, o_ub) in zip(selected_bounds, bounds):
        lb_new = max(float(s_lb) - eps, float(o_lb))
        ub_new = min(float(s_ub) + eps, float(o_ub))
        expanded.append((lb_new, ub_new))
    return expanded
