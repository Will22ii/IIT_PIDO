from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from Optimizer.config import OptimizerSystemConfig


_VALID_FOCUS = {"focus0", "focus1", "focus2", "focus3", "focus4"}
_BOUNDED_SOURCES = {"explorer", "path", "override"}


@dataclass(frozen=True)
class FocusPipelinePlan:
    stages: tuple[str, ...]
    mode: str
    bounds_source: str
    initial_objective_count: int
    p_dim: int
    n_samples: int
    budgets: dict[str, int]
    targets: dict[str, int]
    reasons: dict[str, str]
    input_state: dict[str, object]
    focus3_budget_class: str
    focus3_budget_ratio: float

    def has(self, name: str) -> bool:
        return str(name).strip().lower() in set(self.stages)

    def budget(self, name: str) -> int:
        return int(self.budgets.get(str(name).strip().lower(), 0))

    def target(self, name: str, default: int = 0) -> int:
        return int(self.targets.get(str(name), int(default)))

    def as_dict(self) -> dict[str, object]:
        return {
            "stages": list(self.stages),
            "mode": str(self.mode),
            "bounds_source": str(self.bounds_source),
            "initial_objective_count": int(self.initial_objective_count),
            "p_dim": int(self.p_dim),
            "n_samples": int(self.n_samples),
            "budgets": {str(k): int(v) for k, v in self.budgets.items()},
            "targets": {str(k): int(v) for k, v in self.targets.items()},
            "reasons": {str(k): str(v) for k, v in self.reasons.items()},
            "input_state": dict(self.input_state),
            "focus3_budget_class": str(self.focus3_budget_class),
            "focus3_budget_ratio": float(self.focus3_budget_ratio),
        }


def _normalize_stage(value: object) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "0": "focus0",
        "1": "focus1",
        "2": "focus2",
        "3": "focus3",
        "4": "focus4",
    }
    return aliases.get(raw, raw)


def _parse_pipeline(raw: object) -> tuple[str, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text or text.lower() == "auto":
            return None
        parts: Iterable[object] = [p.strip() for p in text.replace(">", ",").split(",")]
    elif isinstance(raw, (list, tuple)):
        if len(raw) == 1 and str(raw[0]).strip().lower() == "auto":
            return None
        parts = raw
    else:
        raise RuntimeError("Optimizer focus_pipeline must be 'auto', a comma string, or a list.")

    stages: list[str] = []
    for item in parts:
        stage = _normalize_stage(item)
        if not stage:
            continue
        if stage not in _VALID_FOCUS:
            raise RuntimeError(f"Unknown optimizer focus stage: {item}")
        if stage not in stages:
            stages.append(stage)
    if not stages:
        return None
    return tuple(stages)


def _target_count(*, ratio: float, p_dim: int, minimum: int = 0) -> int:
    return int(max(math.ceil(max(float(ratio), 0.0) * max(int(p_dim), 1)), int(minimum)))


def _focus3_budget_class(*, n_samples: int, p_dim: int, system: OptimizerSystemConfig) -> tuple[str, float]:
    ratio = float(max(int(n_samples), 0)) / float(max(int(p_dim), 1))
    ultra = float(getattr(system, "focus3_budget_ultra_low_np", 3.0))
    low = float(getattr(system, "focus3_budget_low_np", 8.0))
    normal = float(getattr(system, "focus3_budget_normal_np", 20.0))
    if ratio < ultra:
        return "ultra_low", ratio
    if ratio < low:
        return "low", ratio
    if ratio < normal:
        return "normal", ratio
    return "rich", ratio


def _focus3_min_budget(*, system: OptimizerSystemConfig, p_dim: int, available: int) -> int:
    if int(available) <= 0:
        return 0
    focus3_min = int(max(
        int(getattr(system, "focus3_min_budget", 5)),
        math.ceil(float(max(float(getattr(system, "focus3_min_budget_np_ratio", 2.0)), 0.0)) * max(int(p_dim), 1)),
    ))
    return int(min(max(focus3_min, 0), max(int(available), 0)))


def _allocate_budgets(
    *,
    stages: tuple[str, ...],
    system: OptimizerSystemConfig,
    n_samples: int,
    p_dim: int,
    initial_objective_count: int,
    targets: dict[str, int],
) -> tuple[dict[str, int], dict[str, str]]:
    budgets = {stage: 0 for stage in _VALID_FOCUS}
    reasons: dict[str, str] = {}
    remaining = int(max(int(n_samples), 0))
    expected_archive = int(max(int(initial_objective_count), 0))

    if "focus0" in stages and remaining > 0:
        need = int(max(int(targets.get("focus0_objective_count", 0)) - expected_archive, 0))
        budgets["focus0"] = int(min(need, remaining))
        expected_archive += budgets["focus0"]
        remaining -= budgets["focus0"]
        reasons["focus0_budget"] = f"fill objective archive to {targets.get('focus0_objective_count', 0)}"

    if "focus1" in stages and remaining > 0:
        need = int(max(int(targets.get("focus1_objective_count", 0)) - expected_archive, 0))
        max_fraction = float(max(float(getattr(system, "focus1_max_budget_fraction", 0.50)), 0.0))
        cap = int(math.floor(float(max(int(n_samples), 0)) * max_fraction))
        if "focus3" not in stages and "focus2" not in stages:
            cap = max(cap, remaining)
        cap = max(cap, 0)
        focus3_reserve = _focus3_min_budget(system=system, p_dim=p_dim, available=remaining) if "focus3" in stages else 0
        available = max(remaining - focus3_reserve, 0) if "focus3" in stages else remaining
        budgets["focus1"] = int(min(need, available, cap))
        expected_archive += budgets["focus1"]
        remaining -= budgets["focus1"]
        if "focus3" in stages:
            reasons["focus1_budget"] = (
                f"fill objective archive to {targets.get('focus1_objective_count', 0)} "
                "without consuming focus3 reserve"
            )
        else:
            reasons["focus1_budget"] = f"fill objective archive to {targets.get('focus1_objective_count', 0)}"

    focus3_reserve = 0
    if "focus3" in stages:
        focus3_reserve = _focus3_min_budget(system=system, p_dim=p_dim, available=remaining)

    if "focus2" in stages and remaining > 0:
        base = int(max(0, math.ceil(float(getattr(system, "focus2_budget_fraction", 0.30)) * int(n_samples))))
        available = max(remaining - focus3_reserve, 0) if "focus3" in stages else remaining
        budgets["focus2"] = int(min(base, available))
        remaining -= budgets["focus2"]
        reasons["focus2_budget"] = "region validation budget fraction with focus3 reserve"

    if "focus3" in stages and remaining > 0:
        budgets["focus3"] = int(remaining)
        remaining = 0
        reasons["focus3_budget"] = "remaining budget after focus0/focus1/focus2"

    return budgets, reasons


def build_focus_pipeline_plan(
    *,
    system: OptimizerSystemConfig,
    bounds_source: str,
    initial_objective_count: int,
    p_dim: int,
    n_samples: int,
) -> FocusPipelinePlan:
    source = str(bounds_source or "cae").strip().lower()
    n_total = int(max(int(n_samples), 0))
    p = int(max(int(p_dim), 1))
    focus0_target = _target_count(
        ratio=float(getattr(system, "focus0_min_np_ratio", 1.0)),
        p_dim=p,
        minimum=int(getattr(system, "focus0_min_points", 3)),
    )
    focus1_target = _target_count(
        ratio=float(getattr(system, "focus1_target_np_ratio", 10.0)),
        p_dim=p,
        minimum=0,
    )
    targets = {
        "focus0_objective_count": int(focus0_target),
        "focus1_objective_count": int(focus1_target),
    }
    archive_ratio = float(max(int(initial_objective_count), 0)) / float(p)
    has_selected_bounds = source in _BOUNDED_SOURCES
    explicit = _parse_pipeline(getattr(system, "focus_pipeline", "auto"))
    reasons: dict[str, str] = {}
    if explicit is not None:
        stages = explicit
        mode = "explicit"
        for stage in stages:
            reasons[stage] = "explicit focus_pipeline"
    else:
        if has_selected_bounds:
            stages_list: list[str] = []
            if bool(getattr(system, "focus0_enabled", True)) and int(initial_objective_count) < focus0_target:
                stages_list.append("focus0")
                reasons["focus0"] = "initial objective archive below focus0 target"
            if bool(getattr(system, "focus1_enabled", True)) and int(initial_objective_count) < focus1_target:
                stages_list.append("focus1")
                reasons["focus1"] = "initial objective archive below focus1 target"
            focus2_min_total = int(max(int(getattr(system, "focus2_min_total_budget", 10)), 0))
            if bool(getattr(system, "focus2_enabled", True)) and n_total >= focus2_min_total:
                stages_list.append("focus2")
                reasons["focus2"] = "selected bounds available; validate candidate regions before focus3"
            elif bool(getattr(system, "focus2_enabled", True)):
                reasons["focus2_skipped"] = f"n_samples below focus2_min_total_budget={focus2_min_total}"
            stages_list.append("focus3")
            reasons["focus3"] = "selected bounds available"
            stages = tuple(dict.fromkeys(stages_list))
        else:
            stages_list = []
            if bool(getattr(system, "focus0_enabled", True)):
                stages_list.append("focus0")
                reasons["focus0"] = "no selected bounds; generate objective archive"
            if bool(getattr(system, "focus1_enabled", True)):
                stages_list.append("focus1")
                reasons["focus1"] = "no selected bounds; classify/global evidence only"
            focus2_min_total = int(max(int(getattr(system, "focus2_min_total_budget", 10)), 0))
            if bool(getattr(system, "focus2_enabled", True)) and n_total >= focus2_min_total:
                stages_list.append("focus2")
                stages_list.append("focus3")
                reasons["focus2"] = "no selected bounds; generate internal region bounds"
                reasons["focus3"] = "use Focus2 generated bounds"
            elif bool(getattr(system, "focus2_enabled", True)):
                reasons["focus2_skipped"] = f"n_samples below focus2_min_total_budget={focus2_min_total}"
            stages = tuple(stages_list)
        mode = "auto"

    if "focus3" in stages and source not in _BOUNDED_SOURCES and "focus2" not in stages:
        raise RuntimeError(
            "Optimizer Focus3 requires selected bounds from Explorer/user/override "
            "or generated bounds from Focus2. "
            f"Current bounds_source={source!r}. Run Explorer/provide explorer_bounds_path, "
            "or include focus2 before focus3."
        )

    budgets, budget_reasons = _allocate_budgets(
        stages=tuple(stages),
        system=system,
        n_samples=n_total,
        p_dim=p,
        initial_objective_count=int(initial_objective_count),
        targets=targets,
    )
    reasons.update(budget_reasons)
    if mode == "auto":
        stages = tuple(stage for stage in stages if int(budgets.get(stage, 0)) > 0)
        if not has_selected_bounds and "focus3" in stages and "focus2" not in stages:
            stages = tuple(stage for stage in stages if stage != "focus3")
            budgets, retry_budget_reasons = _allocate_budgets(
                stages=tuple(stages),
                system=system,
                n_samples=n_total,
                p_dim=p,
                initial_objective_count=int(initial_objective_count),
                targets=targets,
            )
            for key in list(reasons.keys()):
                if key.endswith("_budget"):
                    reasons.pop(key, None)
            reasons.update(retry_budget_reasons)
            reasons.pop("focus3", None)
            reasons["focus3_skipped"] = "no selected bounds and no Focus2 budget to generate bounds"
        if (
            not stages
            and n_total > 0
            and not has_selected_bounds
            and bool(getattr(system, "focus0_enabled", True))
            and int(initial_objective_count) < focus0_target
        ):
            stages = ("focus0",)
            budgets["focus0"] = n_total
            reasons["focus0"] = "fallback: no auto stage received budget"
        active_stage_set = set(stages)
        for key in list(reasons.keys()):
            if key in _VALID_FOCUS and key not in active_stage_set:
                reasons.pop(key, None)
            if key.endswith("_budget") and key[:-7] in _VALID_FOCUS and int(budgets.get(key[:-7], 0)) <= 0:
                reasons.pop(key, None)
    focus3_budget_class, focus3_budget_ratio = _focus3_budget_class(
        n_samples=int(budgets.get("focus3", 0)),
        p_dim=p,
        system=system,
    )
    input_state = {
        "n_samples": int(n_total),
        "p_dim": int(p),
        "initial_objective_count": int(max(int(initial_objective_count), 0)),
        "archive_ratio": float(archive_ratio),
        "bounds_source": str(source),
        "has_selected_bounds": bool(has_selected_bounds),
    }

    return FocusPipelinePlan(
        stages=tuple(stages),
        mode=mode,
        bounds_source=source,
        initial_objective_count=int(max(int(initial_objective_count), 0)),
        p_dim=int(p),
        n_samples=int(n_total),
        budgets={k: int(v) for k, v in budgets.items() if int(v) > 0 or k in stages},
        targets=targets,
        reasons=reasons,
        input_state=input_state,
        focus3_budget_class=focus3_budget_class,
        focus3_budget_ratio=float(focus3_budget_ratio),
    )
