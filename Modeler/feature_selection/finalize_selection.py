import ast
import os
from dataclasses import dataclass
import pandas as pd


@dataclass
class SelectionFinalizeResult:
    selected_df: pd.DataFrame
    selected_features: list[str]
    selected_path: str


def _extract_constraint_feature_vars(
    *,
    constraint_defs: list | None,
    candidate_features: list[str],
) -> list[str]:
    if not constraint_defs or not candidate_features:
        return []
    feature_set = set(str(f) for f in candidate_features)
    allowed_tokens = {
        "abs",
        "min",
        "max",
        "pow",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "exp",
        "log",
        "pi",
        "e",
        "objective",
    }
    used: set[str] = set()
    for c in constraint_defs:
        if not isinstance(c, dict):
            continue
        expr = str(c.get("expr", "")).strip()
        if not expr:
            continue
        try:
            tree = ast.parse(expr, mode="eval")
        except Exception:
            continue
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        for name in names:
            if name in allowed_tokens:
                continue
            if name in feature_set:
                used.add(name)
    return [f for f in candidate_features if f in used]


def finalize_selected_features(
    *,
    selected_df: pd.DataFrame,
    feature_cols: list[str],
    constraint_defs: list | None,
    public_dir: str,
    keep_debug: bool,
    use_score_drop: bool,
) -> SelectionFinalizeResult:
    selected_features = selected_df[selected_df["selected"]]["feature"].tolist()
    if not selected_features:
        selected_features = feature_cols
        print("- No selected features found; fallback to all features")

    forced_constraint_features = _extract_constraint_feature_vars(
        constraint_defs=constraint_defs,
        candidate_features=feature_cols,
    )
    appended_by_constraint = [
        f for f in forced_constraint_features
        if f not in set(selected_features)
    ]
    if appended_by_constraint:
        selected_features = list(selected_features) + appended_by_constraint
        print("- Constraint feature append: " + ", ".join(appended_by_constraint))

    if "feature" in selected_df.columns and "selected" in selected_df.columns:
        selected_df = selected_df.copy()
        if "forced_by_constraint" not in selected_df.columns:
            selected_df["forced_by_constraint"] = False
        for feat in appended_by_constraint:
            mask_feat = selected_df["feature"].astype(str) == str(feat)
            if bool(mask_feat.any()):
                selected_df.loc[mask_feat, "selected"] = True
                selected_df.loc[mask_feat, "forced_by_constraint"] = True
            else:
                selected_df = pd.concat(
                    [
                        selected_df,
                        pd.DataFrame(
                            [{"feature": feat, "selected": True, "forced_by_constraint": True}]
                        ),
                    ],
                    ignore_index=True,
                )

    if keep_debug and ("feature" in selected_df.columns):
        selected_count = int(
            pd.to_numeric(selected_df.get("selected", False), errors="coerce")
            .fillna(0)
            .astype(bool)
            .sum()
        )
        forced_count = int(
            pd.to_numeric(selected_df.get("forced_by_constraint", False), errors="coerce")
            .fillna(0)
            .astype(bool)
            .sum()
        )
        print(
            "[Modeler][FI-RESULT] "
            f"selected={selected_count}/{len(feature_cols)} "
            f"forced_by_constraint={forced_count} "
            f"use_score_drop={bool(use_score_drop)}"
        )
        top_cols = [
            "feature",
            "final_score",
            "global_score",
            "elite_score",
            "perm_selection_rate",
            "drop_selection_rate",
            "selected",
            "forced_by_constraint",
        ]
        top_cols = [c for c in top_cols if c in selected_df.columns]
        top_df = selected_df[top_cols].copy()
        if "final_score" in top_df.columns:
            top_df = top_df.sort_values(by="final_score", ascending=False)
        top_df = top_df.head(5).reset_index(drop=True)
        for idx, row in top_df.iterrows():
            feat = str(row.get("feature", ""))
            f_score = float(row.get("final_score", 0.0))
            g_score = float(row.get("global_score", 0.0))
            e_score = float(row.get("elite_score", 0.0))
            p_rate = float(row.get("perm_selection_rate", 0.0))
            d_rate = float(row.get("drop_selection_rate", 0.0))
            is_sel = bool(row.get("selected", False))
            is_forced = bool(row.get("forced_by_constraint", False))
            print(
                "[Modeler][FI-TOP] "
                f"rank={idx + 1} feature={feat} "
                f"final/global/elite={f_score:.3f}/{g_score:.3f}/{e_score:.3f} "
                f"perm/drop={p_rate:.3f}/{d_rate:.3f} "
                f"selected={is_sel} forced={is_forced}"
            )

    selected_path = os.path.join(public_dir, "selected_features.csv")
    selected_df.to_csv(selected_path, index=False)

    for _, row in selected_df.iterrows():
        feat = str(row.get("feature"))
        perm_rate = float(row.get("perm_selection_rate", 0.0))
        perm_pass = bool(row.get("perm_selected", False))
        drop_rate = float(row.get("drop_selection_rate", 0.0))
        drop_pass = bool(row.get("drop_selected", False))
        final_pass = bool(row.get("selected", False))
        print(
            "[FI] "
            f"feature={feat} "
            f"perm_rate={perm_rate:.3f} perm_pass={perm_pass} "
            f"drop_rate={drop_rate:.3f} drop_pass={drop_pass} "
            f"final={final_pass}"
        )

    print(
        f"- Selected features ({len(selected_features)}/{len(feature_cols)}): "
        f"{', '.join(selected_features)}"
    )
    return SelectionFinalizeResult(
        selected_df=selected_df,
        selected_features=selected_features,
        selected_path=selected_path,
    )
