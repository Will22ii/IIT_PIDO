from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from CAE_tool_interface.config import CAEConfig, CAEUserConfig, CAESystemConfig
from DOE.config import DOEConfig, DOESystemConfig, DOEUserConfig
from Explorer.config import ExplorerConfig, ExplorerSystemConfig, ExplorerUserConfig
from Explorer.executor.explorer_orchestrator import ExplorerOrchestrator
from Modeler.config import ModelerConfig, ModelerSystemConfig, ModelerUserConfig
from pipeline.config import PipelineConfig, PipelineTasks
from pipeline.run_pipeline import run_pipeline


@dataclass(frozen=True)
class ProblemCase:
    problem_name: str
    known_optimum: Any
    n_samples: int
    objective_sense: str = "min"
    repeats: int | None = None


@dataclass(frozen=True)
class ExplorerStrategy:
    strategy_id: str
    overrides: dict[str, Any]


PROBLEM_SUITE: list[ProblemCase] = [
    ProblemCase(
        problem_name="rosenbrock",
        known_optimum={"x1": 1.0, "x2": 1.0, "x3": 1.0, "x4": 1.0, "x5": 1.0},
        n_samples=450,
        repeats=10,
    ),
    ProblemCase(
        problem_name="cantilever_beam",
        known_optimum={"H": 7.0, "h1": 0.1, "b1": 9.48482, "b2": 0.1},
        n_samples=90,
        repeats=25,

    ),
    ProblemCase(
        problem_name="goldstein_price",
        known_optimum={"x1": 0.0, "x2": -1.0},
        n_samples=150,
        repeats=50,
    ),
    ProblemCase(
        problem_name="six_hump_camel",
        known_optimum=[
            {"x1": 0.0898, "x2": -0.7126},
            {"x1": -0.0898, "x2": 0.7126},
        ],
        n_samples=50,
        repeats=25,
    ),
]


EXPLORER_STRATEGIES: list[ExplorerStrategy] = [
    ExplorerStrategy(
        "S4_dual",
        {"strategy_params": {"mode": "dual_refine_ei"}},
    ),
    ExplorerStrategy(
        "S8_dual",
        {"strategy_params": {"mode": "dual_gradient_refine"}},
    ),
    ExplorerStrategy(
        "S4_pred",
        {"strategy_params": {"mode": "pred_refine_ei"}},
    ),
    ExplorerStrategy(
        "S8_pred",
        {"strategy_params": {"mode": "pred_refine_lcb"}},
    ),
    ExplorerStrategy(
        "S4_obj",
        {"strategy_params": {"mode": "obj_refine_ei"}},
    ),
    ExplorerStrategy(
        "S8_obj",
        {"strategy_params": {"mode": "obj_refine_lcb"}},
    ),
]


def _strategy_map() -> dict[str, ExplorerStrategy]:
    return {s.strategy_id: s for s in EXPLORER_STRATEGIES}


def _resolve_case_repeats(*, case: ProblemCase, default_repeats: int) -> int:
    if case.repeats is None:
        return int(max(int(default_repeats), 1))
    return int(max(int(case.repeats), 1))


def _case_real_variables(case: ProblemCase) -> list[str]:
    ko = case.known_optimum
    if isinstance(ko, dict):
        return sorted([str(k) for k in ko.keys()])
    if isinstance(ko, list):
        keys: set[str] = set()
        for item in ko:
            if isinstance(item, dict):
                keys.update(str(k) for k in item.keys())
        return sorted(keys)
    return []


def _load_modeler_selected_features(modeler_metadata_path: str | None) -> list[str]:
    if not modeler_metadata_path or not os.path.exists(modeler_metadata_path):
        return []
    try:
        with open(modeler_metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        resolved = meta.get("resolved_params", {}) if isinstance(meta, dict) else {}
        selected = resolved.get("selected_features", []) if isinstance(resolved, dict) else []
        if isinstance(selected, list):
            return [str(v) for v in selected]
    except Exception:
        return []
    return []


def _choose_strategies_by_dim(
    *,
    selected_feature_count: int,
    requested: list[ExplorerStrategy],
) -> tuple[str, list[ExplorerStrategy]]:
    requested_map = {s.strategy_id: s for s in requested}
    policy = "low_dim_all4" if selected_feature_count <= 3 else "high_dim_all4"
    ordered = [
        "S4_dual",
        "S8_dual",
        "S4_pred",
        "S8_pred",
        "S4_obj",
        "S8_obj",
    ]
    chosen = [requested_map[sid] for sid in ordered if sid in requested_map]
    if not chosen:
        chosen = requested[:]
    return policy, chosen


def _build_pipeline_config(
    *,
    case: ProblemCase,
    seed: int,
    run_doe: bool,
    run_modeler: bool,
    run_explorer: bool,
    use_additional: bool,
    use_hpo: bool,
    use_timestamp: bool,
) -> PipelineConfig:
    cae_cfg = CAEConfig(
        user=CAEUserConfig(
            problem_name=case.problem_name,
            seed=int(seed),
            objective_sense=str(case.objective_sense),
        ),
        system=CAESystemConfig(
            use_timestamp=bool(use_timestamp),
            allow_latest_fallback=False,
        ),
    )

    doe_cfg = DOEConfig(
        cae=cae_cfg,
        cae_user=None,
        user=DOEUserConfig(algo_name="lhs", use_additional=bool(use_additional)),
        system=DOESystemConfig(n_samples=int(case.n_samples)),
    )

    modeler_cfg = ModelerConfig(
        user=ModelerUserConfig(
            model_name="xgb",
            use_hpo=bool(use_hpo),
            use_secondary_selection=False,
        ),
        system=ModelerSystemConfig(),
        cae=cae_cfg,
        doe_csv_path=None,
        doe_metadata_path=None,
    )

    explorer_cfg = ExplorerConfig(
        user=ExplorerUserConfig(known_optimum=case.known_optimum),
        system=ExplorerSystemConfig(),
        cae=cae_cfg,
        doe_csv_path=None,
        doe_metadata_path=None,
        model_pkl_path=None,
        modeler_metadata_path=None,
    )

    return PipelineConfig(
        cae=cae_cfg,
        doe=doe_cfg if run_doe else None,
        modeler=modeler_cfg if run_modeler else None,
        explorer=explorer_cfg,
        tasks=PipelineTasks(
            run_doe=bool(run_doe),
            run_modeler=bool(run_modeler),
            run_explorer=bool(run_explorer),
        ),
    )


def _resolve_requested_strategies(raw: str) -> list[ExplorerStrategy]:
    catalog = _strategy_map()
    legacy_alias = {
        "S4_ei_focus": "S4_dual",
        "S8_dual_gradient_refine": "S8_dual",
        "S4_pred_focus": "S4_pred",
        "S8_pred_refine": "S8_pred",
    }
    wanted = [tok.strip() for tok in str(raw).split(",") if tok.strip()]
    if not wanted:
        wanted = [s.strategy_id for s in EXPLORER_STRATEGIES]
    out: list[ExplorerStrategy] = []
    for sid in wanted:
        sid = legacy_alias.get(sid, sid)
        if sid not in catalog:
            raise ValueError(f"Unknown explorer strategy: {sid}")
        out.append(catalog[sid])
    return out


def _build_strategy_explorer_config(
    *,
    base_explorer: ExplorerConfig,
    strategy: ExplorerStrategy,
) -> ExplorerConfig:
    system_cfg = copy.deepcopy(base_explorer.system)
    system_cfg.strategy_id = strategy.strategy_id

    merged_params = dict(system_cfg.strategy_params or {})
    for key, value in strategy.overrides.items():
        if key == "strategy_params" and isinstance(value, dict):
            merged_params.update(value)
            continue
        if hasattr(system_cfg, key):
            setattr(system_cfg, key, value)
        else:
            merged_params[key] = value
    system_cfg.strategy_params = merged_params

    return ExplorerConfig(
        user=copy.deepcopy(base_explorer.user),
        system=system_cfg,
        cae=base_explorer.cae,
        cae_metadata_path=base_explorer.cae_metadata_path,
        doe_csv_path=base_explorer.doe_csv_path,
        doe_metadata_path=base_explorer.doe_metadata_path,
        model_pkl_path=base_explorer.model_pkl_path,
        modeler_metadata_path=base_explorer.modeler_metadata_path,
    )


def _parse_selected_bounds(bounds_path: str | None) -> dict[str, tuple[float, float]]:
    if not bounds_path or (not os.path.exists(bounds_path)):
        return {}
    with open(bounds_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    selected = payload.get("selected_bounds")
    if isinstance(selected, dict):
        out = {}
        for name, item in selected.items():
            if not isinstance(item, dict):
                continue
            try:
                out[str(name)] = (float(item["lb"]), float(item["ub"]))
            except Exception:
                continue
        return out
    if isinstance(selected, list):
        order = payload.get("bounds_order", [])
        out = {}
        for idx, item in enumerate(selected):
            if idx >= len(order) or not isinstance(item, dict):
                continue
            try:
                out[str(order[idx])] = (float(item["lb"]), float(item["ub"]))
            except Exception:
                continue
        return out
    return {}


def _known_optimum_list(known_optimum: Any) -> list[dict[str, float]]:
    if isinstance(known_optimum, dict):
        return [{str(k): float(v) for k, v in known_optimum.items()}]
    if isinstance(known_optimum, list):
        out = []
        for item in known_optimum:
            if not isinstance(item, dict):
                continue
            out.append({str(k): float(v) for k, v in item.items()})
        return out
    return []


def _is_optimum_included(
    *,
    known_optimum: Any,
    selected_bounds: dict[str, tuple[float, float]],
) -> tuple[bool, int, int]:
    opts = _known_optimum_list(known_optimum)
    if not opts or not selected_bounds:
        return False, 0, len(opts)
    hit_count = 0
    for opt in opts:
        required_keys = list(opt.keys())
        if not required_keys:
            continue
        # strict mode: known optimum의 key가 selected_bounds에 하나라도 없으면 미포함 처리
        if any(k not in selected_bounds for k in required_keys):
            continue
        included = True
        for key in required_keys:
            lb, ub = selected_bounds[key]
            val = float(opt[key])
            if val < lb or val > ub:
                included = False
                break
        if included:
            hit_count += 1
    return hit_count > 0, hit_count, len(opts)


def _save_explorer_stats_csv(
    *,
    detail_rows: list[dict[str, Any]],
) -> tuple[str, str]:
    stats_root = os.path.join(PROJECT_ROOT, "result", "explorer_strategy_stats")
    os.makedirs(stats_root, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        detail_df = pd.DataFrame(
            columns=[
                "run",
                "repeat",
                "problem",
                "seed",
                "strategy",
                "executed_strategy",
                "fallback_from",
                "fallback_applied",
                "survivor_optimum_included",
                "optimum_hit_count",
                "optimum_total_count",
                "selected_feature_count",
                "strategy_policy",
                "modeler_selected_features",
                "modeler_selected_real_count",
                "modeler_selected_dummy_count",
                "modeler_real_coverage_pct",
                "modeler_all_real_only",
                "volume_ratio",
                "volume_ratio_pct",
                "volume_cap_pass",
                "joint_pass",
                "fail_type",
                "explorer_metadata",
                "selected_bounds_path",
                "run_root",
            ]
        )

    # --- derive DSE metrics per row ---
    if not detail_df.empty:
        _opt = detail_df["survivor_optimum_included"].astype(bool)
        _vr = pd.to_numeric(detail_df["volume_ratio"], errors="coerce").fillna(1.0)
        detail_df["volume_cap_pass"] = (_vr <= 0.25).astype(int)
        detail_df["joint_pass"] = (_opt & (_vr <= 0.25)).astype(int)
        detail_df["fail_type"] = "both_fail"
        detail_df.loc[_opt & (_vr <= 0.25), "fail_type"] = "pass"
        detail_df.loc[~_opt & (_vr <= 0.25), "fail_type"] = "over_shrink_fail"
        detail_df.loc[_opt & (_vr > 0.25), "fail_type"] = "over_wide_fail"

    detail_path = os.path.join(stats_root, f"explorer_strategy_try_stats_{ts}.csv")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    summary_df = detail_df.copy()
    if not summary_df.empty:
        summary_df["optimum_included_num"] = summary_df["survivor_optimum_included"].astype(bool).astype(int)
        summary_df["modeler_all_real_only_num"] = summary_df["modeler_all_real_only"].astype(bool).astype(int)
        summary_df["over_shrink_fail"] = (summary_df["fail_type"] == "over_shrink_fail").astype(int)
        summary_df["over_wide_fail"] = (summary_df["fail_type"] == "over_wide_fail").astype(int)
        summary_df["both_fail"] = (summary_df["fail_type"] == "both_fail").astype(int)
        grouped = (
            summary_df
            .groupby(["strategy", "problem"], as_index=False)
            .agg(
                tries=("strategy", "count"),
                survivor_optimum_included_pct=("optimum_included_num", "mean"),
                modeler_all_real_only_pct=("modeler_all_real_only_num", "mean"),
                modeler_real_coverage_pct_mean=("modeler_real_coverage_pct", "mean"),
                volume_ratio_pct_mean=("volume_ratio_pct", "mean"),
                volume_ratio_pct_std=("volume_ratio_pct", "std"),
                volume_ratio_pct_min=("volume_ratio_pct", "min"),
                volume_ratio_pct_max=("volume_ratio_pct", "max"),
                joint_pass_pct=("joint_pass", "mean"),
                joint_pass_std=("joint_pass", "std"),
                volume_cap_pass_pct=("volume_cap_pass", "mean"),
                over_shrink_fail_pct=("over_shrink_fail", "mean"),
                over_wide_fail_pct=("over_wide_fail", "mean"),
                both_fail_pct=("both_fail", "mean"),
            )
        )
        grouped["survivor_optimum_included_pct"] = grouped["survivor_optimum_included_pct"] * 100.0
        grouped["modeler_all_real_only_pct"] = grouped["modeler_all_real_only_pct"] * 100.0
        grouped["joint_pass_pct"] = grouped["joint_pass_pct"] * 100.0
        grouped["volume_cap_pass_pct"] = grouped["volume_cap_pass_pct"] * 100.0
        grouped["over_shrink_fail_pct"] = grouped["over_shrink_fail_pct"] * 100.0
        grouped["over_wide_fail_pct"] = grouped["over_wide_fail_pct"] * 100.0
        grouped["both_fail_pct"] = grouped["both_fail_pct"] * 100.0
    else:
        grouped = pd.DataFrame(
            columns=[
                "strategy",
                "problem",
                "tries",
                "survivor_optimum_included_pct",
                "modeler_all_real_only_pct",
                "modeler_real_coverage_pct_mean",
                "volume_ratio_pct_mean",
                "volume_ratio_pct_std",
                "volume_ratio_pct_min",
                "volume_ratio_pct_max",
                "joint_pass_pct",
                "joint_pass_std",
                "volume_cap_pass_pct",
                "over_shrink_fail_pct",
                "over_wide_fail_pct",
                "both_fail_pct",
            ]
        )

    summary_path = os.path.join(stats_root, f"explorer_strategy_problem_summary_{ts}.csv")
    grouped.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return detail_path, summary_path


def _save_fi_stats_csv(
    *,
    detail_rows: list[dict[str, Any]],
) -> tuple[str, str]:
    stats_root = os.path.join(PROJECT_ROOT, "result", "explorer_strategy_stats")
    os.makedirs(stats_root, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        detail_df = pd.DataFrame(
            columns=[
                "run",
                "repeat",
                "problem",
                "seed",
                "selected_feature_count",
                "modeler_selected_features",
                "modeler_selected_real_count",
                "modeler_selected_dummy_count",
                "modeler_real_coverage_pct",
                "fi_all_real_included",
                "fi_real_only_success",
                "modeler_metadata",
                "run_root",
            ]
        )

    detail_path = os.path.join(stats_root, f"fi_primary_try_stats_{ts}.csv")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    summary_df = detail_df.copy()
    if not summary_df.empty:
        summary_df["fi_all_real_included_num"] = summary_df["fi_all_real_included"].astype(bool).astype(int)
        summary_df["fi_real_only_success_num"] = summary_df["fi_real_only_success"].astype(bool).astype(int)
        grouped = (
            summary_df
            .groupby(["problem"], as_index=False)
            .agg(
                tries=("problem", "count"),
                fi_all_real_included_pct=("fi_all_real_included_num", "mean"),
                fi_real_only_success_pct=("fi_real_only_success_num", "mean"),
                modeler_real_coverage_pct_mean=("modeler_real_coverage_pct", "mean"),
                modeler_real_coverage_pct_std=("modeler_real_coverage_pct", "std"),
                selected_feature_count_mean=("selected_feature_count", "mean"),
                selected_feature_count_std=("selected_feature_count", "std"),
                modeler_selected_dummy_count_mean=("modeler_selected_dummy_count", "mean"),
            )
        )
        grouped["fi_all_real_included_pct"] = grouped["fi_all_real_included_pct"] * 100.0
        grouped["fi_real_only_success_pct"] = grouped["fi_real_only_success_pct"] * 100.0
    else:
        grouped = pd.DataFrame(
            columns=[
                "problem",
                "tries",
                "fi_all_real_included_pct",
                "fi_real_only_success_pct",
                "modeler_real_coverage_pct_mean",
                "modeler_real_coverage_pct_std",
                "selected_feature_count_mean",
                "selected_feature_count_std",
                "modeler_selected_dummy_count_mean",
            ]
        )

    summary_path = os.path.join(stats_root, f"fi_primary_problem_summary_{ts}.csv")
    grouped.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return detail_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch runner: execute multiple problems with repeated pipeline runs.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeats for full problem suite.")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed.")
    parser.add_argument(
        "--repeat-seed-step",
        type=int,
        default=1000,
        help="Seed offset added per repeat.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining runs even if one run fails.",
    )
    parser.add_argument("--no-additional", action="store_true", help="Disable DOE additional mode.")
    parser.add_argument("--no-hpo", action="store_true", help="Disable Modeler HPO.")
    parser.add_argument("--no-timestamp", action="store_true", help="Disable timestamp in CAE/system config.")
    parser.add_argument("--skip-doe", action="store_true", help="Skip DOE stage.")
    parser.add_argument("--skip-modeler", action="store_true", help="Skip Modeler stage.")
    parser.add_argument("--skip-explorer", action="store_true", help="Skip Explorer stage.")
    parser.add_argument(
        "--explorer-strategies",
        type=str,
        default="S4_dual,S8_dual,S4_pred,S8_pred,S4_obj,S8_obj",
        help="Comma-separated Explorer strategy IDs.",
    )
    args = parser.parse_args()

    repeats = max(int(args.repeats), 1)
    base_seed = int(args.base_seed)
    repeat_seed_step = int(args.repeat_seed_step)
    continue_on_error = bool(args.continue_on_error)

    run_doe = not bool(args.skip_doe)
    run_modeler = not bool(args.skip_modeler)
    run_explorer = not bool(args.skip_explorer)
    use_additional = not bool(args.no_additional)
    use_hpo = not bool(args.no_hpo)
    use_timestamp = not bool(args.no_timestamp)
    explorer_strategies = _resolve_requested_strategies(args.explorer_strategies)

    total_runs = sum(
        _resolve_case_repeats(case=case, default_repeats=repeats)
        for case in PROBLEM_SUITE
    )
    run_counter = 0
    failures: list[dict[str, Any]] = []
    explorer_failures: list[dict[str, Any]] = []
    explorer_detail_rows: list[dict[str, Any]] = []
    fi_detail_rows: list[dict[str, Any]] = []

    print("===================================")
    print(" Batch Pipeline 실행 시작")
    print("===================================")
    print(
        f"- repeats={repeats} total_runs={total_runs} "
        f"tasks(doe/modeler/explorer)={run_doe}/{run_modeler}/{run_explorer}"
    )
    print(
        "- problem_repeats="
        + ",".join(
            [
                f"{case.problem_name}:{_resolve_case_repeats(case=case, default_repeats=repeats)}"
                for case in PROBLEM_SUITE
            ]
        )
    )
    if run_explorer:
        print(f"- explorer_strategies={','.join([s.strategy_id for s in explorer_strategies])}")

    for idx, case in enumerate(PROBLEM_SUITE):
        case_repeats = _resolve_case_repeats(case=case, default_repeats=repeats)
        for rep in range(case_repeats):
            run_counter += 1
            seed = base_seed + rep * repeat_seed_step + idx
            print(
                "[Batch] "
                f"run={run_counter}/{total_runs} repeat={rep + 1}/{case_repeats} "
                f"problem={case.problem_name} seed={seed} n_samples={case.n_samples}"
            )

            cfg = _build_pipeline_config(
                case=case,
                seed=seed,
                run_doe=run_doe,
                run_modeler=run_modeler,
                run_explorer=False,
                use_additional=use_additional,
                use_hpo=use_hpo,
                use_timestamp=use_timestamp,
            )

            try:
                pipe_out = run_pipeline(config=cfg)
                run_context = pipe_out["run_context"]
                modeler_metadata_path = pipe_out.get("modeler_metadata")
                modeler_selected_features = _load_modeler_selected_features(modeler_metadata_path)
                selected_feature_count = int(len(modeler_selected_features))
                real_variables = set(_case_real_variables(case))
                selected_set = set(modeler_selected_features)
                modeler_selected_real_count = int(len(selected_set.intersection(real_variables)))
                modeler_selected_dummy_count = int(len(selected_set.difference(real_variables)))
                modeler_real_coverage_pct = (
                    float(modeler_selected_real_count) / float(len(real_variables)) * 100.0
                    if len(real_variables) > 0 else 0.0
                )
                # strict success: selected features exactly match real variables
                modeler_all_real_only = bool(selected_set == real_variables and len(real_variables) > 0)
                modeler_all_real_included = bool(
                    len(real_variables) > 0 and real_variables.issubset(selected_set)
                )

                fi_detail_rows.append(
                    {
                        "run": run_counter,
                        "repeat": rep + 1,
                        "problem": case.problem_name,
                        "seed": seed,
                        "selected_feature_count": selected_feature_count,
                        "modeler_selected_features": json.dumps(modeler_selected_features, ensure_ascii=False),
                        "modeler_selected_real_count": modeler_selected_real_count,
                        "modeler_selected_dummy_count": modeler_selected_dummy_count,
                        "modeler_real_coverage_pct": modeler_real_coverage_pct,
                        "fi_all_real_included": bool(modeler_all_real_included),
                        "fi_real_only_success": bool(modeler_all_real_only),
                        "modeler_metadata": modeler_metadata_path,
                        "run_root": run_context.run_root,
                    }
                )

                if run_explorer and cfg.explorer is not None:
                    if selected_feature_count <= 0:
                        strategy_policy, active_strategies = _choose_strategies_by_dim(
                            selected_feature_count=4,
                            requested=explorer_strategies,
                        )
                    else:
                        strategy_policy, active_strategies = _choose_strategies_by_dim(
                            selected_feature_count=selected_feature_count,
                            requested=explorer_strategies,
                        )

                    print(
                        f"[Explorer][Policy] problem={case.problem_name} "
                        f"selected_feature_count={selected_feature_count} "
                        f"strategy_policy={strategy_policy} "
                        f"selected_features={modeler_selected_features}"
                    )

                    for strategy in active_strategies:
                        requested_id = strategy.strategy_id

                        try:
                            explorer_cfg = _build_strategy_explorer_config(
                                base_explorer=cfg.explorer,
                                strategy=strategy,
                            )
                            exp_out = ExplorerOrchestrator(explorer_cfg, run_context=run_context).run()
                            selected_bounds = _parse_selected_bounds(exp_out.get("selected_bounds_path"))
                            vol_ratio = exp_out.get("selected_bounds_volume_ratio")

                            included, hit_count, total_count = _is_optimum_included(
                                known_optimum=case.known_optimum,
                                selected_bounds=selected_bounds,
                            )
                            vol_ratio_float = float(vol_ratio) if vol_ratio is not None else None
                            vol_pct = (vol_ratio_float * 100.0) if vol_ratio_float is not None else None
                            print(f"[Explorer][{requested_id}] executed={strategy.strategy_id}")
                            if vol_pct is not None:
                                print(
                                    f"[Explorer][{requested_id}] "
                                    f"selected_bounds volume_ratio={vol_ratio_float:.4f} ({vol_pct:.2f}%)"
                                )
                            else:
                                print(f"[Explorer][{requested_id}] selected_bounds volume_ratio=none")
                            print(
                                f"[Explorer][{requested_id}] optimum_included={included} "
                                f"(hits={hit_count}/{total_count})"
                            )

                            explorer_detail_rows.append(
                                {
                                    "run": run_counter,
                                    "repeat": rep + 1,
                                    "problem": case.problem_name,
                                    "seed": seed,
                                    "strategy": requested_id,
                                    "executed_strategy": strategy.strategy_id,
                                    "fallback_from": None,
                                    "fallback_applied": False,
                                    "survivor_optimum_included": bool(included),
                                    "optimum_hit_count": int(hit_count),
                                    "optimum_total_count": int(total_count),
                                    "selected_feature_count": selected_feature_count,
                                    "strategy_policy": strategy_policy,
                                    "modeler_selected_features": json.dumps(modeler_selected_features, ensure_ascii=False),
                                    "modeler_selected_real_count": modeler_selected_real_count,
                                    "modeler_selected_dummy_count": modeler_selected_dummy_count,
                                    "modeler_real_coverage_pct": modeler_real_coverage_pct,
                                    "modeler_all_real_only": bool(modeler_all_real_only),
                                    "volume_ratio": vol_ratio_float,
                                    "volume_ratio_pct": vol_pct,
                                    "explorer_metadata": exp_out.get("metadata"),
                                    "selected_bounds_path": exp_out.get("selected_bounds_path"),
                                    "run_root": run_context.run_root,
                                }
                            )
                        except Exception as exp_exc:
                            payload = {
                                "run": run_counter,
                                "repeat": rep + 1,
                                "problem": case.problem_name,
                                "seed": seed,
                                "strategy": requested_id,
                                "error": str(exp_exc),
                            }
                            explorer_failures.append(payload)
                            print(
                                "[Batch][Explorer] FAILED "
                                f"problem={case.problem_name} repeat={rep + 1} "
                                f"seed={seed} strategy={requested_id} error={exp_exc}"
                            )
                            if not continue_on_error:
                                raise

            except Exception as exc:
                payload = {
                    "run": run_counter,
                    "repeat": rep + 1,
                    "problem": case.problem_name,
                    "seed": seed,
                    "error": str(exc),
                }
                failures.append(payload)
                print(
                    "[Batch] FAILED "
                    f"problem={case.problem_name} repeat={rep + 1} seed={seed} error={exc}"
                )
                if not continue_on_error:
                    raise

    fi_detail_csv, fi_summary_csv = _save_fi_stats_csv(detail_rows=fi_detail_rows)
    detail_csv, summary_csv = _save_explorer_stats_csv(detail_rows=explorer_detail_rows)

    print("===================================")
    print(" Batch Pipeline 실행 완료")
    print("===================================")
    print(f"- total_runs={total_runs}")
    print(f"- success_runs={total_runs - len(failures)}")
    print(f"- failed_runs={len(failures)}")
    print(f"- explorer_strategy_failed_runs={len(explorer_failures)}")
    print(f"- fi_primary_try_stats_csv={fi_detail_csv}")
    print(f"- fi_primary_problem_summary_csv={fi_summary_csv}")
    print(f"- explorer_try_stats_csv={detail_csv}")
    print(f"- explorer_problem_summary_csv={summary_csv}")
    if failures:
        print("- failure_details:")
        for item in failures:
            problem = item.get("problem")
            repeat = item.get("repeat")
            seed = item.get("seed")
            run = item.get("run")
            strategy = item.get("strategy")
            err = item.get("error")
            if strategy:
                print(
                    f"  run={run} repeat={repeat} problem={problem} "
                    f"seed={seed} strategy={strategy} error={err}"
                )
            else:
                print(
                    f"  run={run} repeat={repeat} problem={problem} "
                    f"seed={seed} error={err}"
                )
    if explorer_failures:
        print("- explorer_failure_details:")
        for item in explorer_failures:
            print(
                f"  run={item.get('run')} repeat={item.get('repeat')} "
                f"problem={item.get('problem')} seed={item.get('seed')} "
                f"strategy={item.get('strategy')} error={item.get('error')}"
            )


if __name__ == "__main__":
    main()
