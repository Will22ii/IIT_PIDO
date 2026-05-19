from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StrategyHintDecision:
    strategy_alias_hint: str
    dual_obj_equivalent_forced: bool
    pred_uq_target_enabled: bool


@dataclass(frozen=True)
class StrategyExecutionDecision:
    strategy_alias_requested: str
    strategy_alias: str
    source_mode: str
    use_pred_model: bool
    use_obj_model: bool
    acq_type: str
    pred_family_alias: bool


def resolve_strategy_alias(strategy_id: str, mode: str) -> str:
    sid = str(strategy_id).strip().lower()
    if sid.startswith("s4_obj"):
        return "s4_obj"
    if sid.startswith("s4_dual"):
        return "s4"

    m = str(mode).strip().lower()
    mode_map = {
        "dual_refine_ei": "s4",
        "obj_refine_ei": "s4_obj",
    }
    return mode_map.get(m, "s4_obj")


def resolve_strategy_hint(
    *,
    strategy_id: str,
    strategy_params: dict[str, Any],
    has_pre_constraints: bool,
    p_dim: int,
) -> StrategyHintDecision:
    strategy_mode_hint = str(strategy_params.get("mode", "")).strip().lower()
    strategy_alias_hint = resolve_strategy_alias(
        strategy_id=strategy_id,
        mode=strategy_mode_hint,
    )
    dual_obj_equivalent_forced = bool(
        strategy_alias_hint == "s4"
        and (not has_pre_constraints)
        and int(p_dim) <= 3
    )
    if dual_obj_equivalent_forced:
        strategy_alias_hint = "s4_obj"
        print(
            "[Explorer] low-dim dual override: "
            f"p_dim={int(p_dim)}, has_pre_constraints={has_pre_constraints} "
            f"-> force obj-equivalent path ({strategy_alias_hint})"
        )

    return StrategyHintDecision(
        strategy_alias_hint=strategy_alias_hint,
        dual_obj_equivalent_forced=dual_obj_equivalent_forced,
        pred_uq_target_enabled=(strategy_alias_hint == "s4"),
    )


def resolve_strategy_execution(
    *,
    strategy_id: str,
    strategy_params: dict[str, Any],
    dual_obj_equivalent_forced: bool,
    has_model_layer: bool,
) -> StrategyExecutionDecision:
    strategy_mode = str(strategy_params.get("mode", "")).strip().lower()
    strategy_alias_requested = resolve_strategy_alias(
        strategy_id=strategy_id,
        mode=strategy_mode,
    )
    strategy_alias = strategy_alias_requested

    if dual_obj_equivalent_forced and strategy_alias == "s4":
        strategy_alias = "s4_obj"

    if not has_model_layer and strategy_alias == "s4":
        fallback_alias = "s4_obj"
        print(
            "[Explorer] model layer unavailable; "
            f"strategy {strategy_alias} degraded to {fallback_alias}."
        )
        strategy_alias = fallback_alias

    source_mode = "dual"
    if strategy_alias == "s4_obj":
        source_mode = "obj"

    return StrategyExecutionDecision(
        strategy_alias_requested=strategy_alias_requested,
        strategy_alias=strategy_alias,
        source_mode=source_mode,
        use_pred_model=(source_mode == "dual"),
        use_obj_model=(source_mode in {"dual", "obj"}),
        acq_type="EI",
        pred_family_alias=(strategy_alias == "s4"),
    )
