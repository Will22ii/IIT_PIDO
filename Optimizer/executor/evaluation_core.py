from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from CAE_tool_interface.executor.configurator import select_cae_by_name
from DOE.executor.constraint_filter import evaluate_constraints_point
from DOE.executor.eval_sanitizer import sanitize_evaluate_output


def normalize_objective_sense(objective_sense: str) -> str:
    sense = str(objective_sense or "min").strip().lower()
    if sense not in {"min", "max"}:
        raise ValueError("Optimizer objective_sense must be 'min' or 'max'.")
    return sense


def objective_is_better(*, y_new: float, y_best: float, objective_sense: str) -> bool:
    if normalize_objective_sense(objective_sense) == "max":
        return float(y_new) > float(y_best)
    return float(y_new) < float(y_best)


def objective_improvement(*, y_new: float, y_best: float, objective_sense: str) -> float:
    if normalize_objective_sense(objective_sense) == "max":
        return float(y_new) - float(y_best)
    return float(y_best) - float(y_new)


def objective_best_index(y: np.ndarray, objective_sense: str) -> int:
    arr = np.asarray(y, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot select best objective index from an empty array.")
    if normalize_objective_sense(objective_sense) == "max":
        return int(np.argmax(arr))
    return int(np.argmin(arr))


def objective_initial_best(objective_sense: str) -> float:
    return float("-inf") if normalize_objective_sense(objective_sense) == "max" else float("inf")


def safe_constraint_token(raw: str) -> str:
    token = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(raw).strip())
    token = "_".join(part for part in token.split("_") if part)
    return token or "constraint"


def evaluate_pre_constraints_raw(
    *,
    x: np.ndarray,
    var_names: list[str],
    constraint_defs: list[dict],
    fail_fast_output_missing: bool = False,
) -> tuple[dict, bool, float]:
    return evaluate_constraints_point(
        x=np.asarray(x, dtype=float).reshape(-1),
        var_names=var_names,
        constraint_defs=constraint_defs,
        scope="pre",
        fail_fast_output_missing=bool(fail_fast_output_missing),
    )


def build_pre_constraint_debug_fields(
    *,
    x: np.ndarray,
    var_names: list[str],
    constraint_defs: list[dict],
    enforce_pre_constraints: bool,
) -> dict[str, object]:
    if not enforce_pre_constraints or not constraint_defs:
        return {
            "pre_constraint_active": False,
            "pre_constraint_count": 0,
            "pre_constraint_failed_count": 0,
            "pre_constraint_worst_id": "",
            "pre_constraint_worst_margin": float("inf"),
        }
    payload, feasible, margin = evaluate_pre_constraints_raw(
        x=x,
        var_names=var_names,
        constraint_defs=constraint_defs,
        fail_fast_output_missing=False,
    )
    fields: dict[str, object] = {
        "pre_constraint_active": True,
        "pre_constraint_count": int(len(payload)),
        "pre_constraint_failed_count": int(
            sum(1 for c in payload.values() if isinstance(c, dict) and not bool(c.get("ok", False)))
        ),
        "pre_constraint_worst_id": "",
        "pre_constraint_worst_margin": float(margin),
        "pre_constraint_all_feasible": bool(feasible),
    }
    worst_id = ""
    worst_margin = float("inf")
    for raw_id, cinfo in payload.items():
        if not isinstance(cinfo, dict):
            continue
        cid = str(cinfo.get("id", raw_id)).strip() or str(raw_id)
        token = safe_constraint_token(cid)
        value = float(cinfo.get("value", float("nan")))
        c_margin = float(cinfo.get("margin", float("nan")))
        c_g = float(cinfo.get("g", float("nan")))
        fields[f"pre_constraint_{token}_value"] = value
        fields[f"pre_constraint_{token}_margin"] = c_margin
        fields[f"pre_constraint_{token}_g"] = c_g
        fields[f"pre_constraint_{token}_ok"] = bool(cinfo.get("ok", False))
        if np.isfinite(c_margin) and c_margin < worst_margin:
            worst_margin = c_margin
            worst_id = cid
    fields["pre_constraint_worst_id"] = worst_id
    fields["pre_constraint_worst_margin"] = float(worst_margin if np.isfinite(worst_margin) else margin)
    return fields


@dataclass(frozen=True)
class SelectedFeatureMapper:
    """Map Optimizer active-feature vectors to full CAE variable vectors."""

    var_names: list[str]
    x_base: np.ndarray
    selected_features: list[str]
    selected_idx: np.ndarray

    @classmethod
    def from_variables(
        cls,
        *,
        variables: list[dict],
        selected_features: list[str],
    ) -> "SelectedFeatureMapper":
        var_names: list[str] = []
        baselines: list[float] = []
        for var in variables:
            if not isinstance(var, dict):
                continue
            name = str(var.get("name", "")).strip()
            if not name:
                continue
            if "baseline" in var:
                base = float(var["baseline"])
            elif "lb" in var and "ub" in var:
                base = 0.5 * (float(var["lb"]) + float(var["ub"]))
            else:
                base = 0.0
            if not np.isfinite(base):
                base = 0.0
            var_names.append(name)
            baselines.append(float(base))

        if not var_names:
            raise RuntimeError("CAE variable list is empty for Optimizer evaluation.")
        selected = [str(name) for name in selected_features]
        name_to_idx = {name: idx for idx, name in enumerate(var_names)}
        missing = [name for name in selected if name not in name_to_idx]
        if missing:
            raise RuntimeError(
                "Selected features are not present in CAE variable list: "
                + ", ".join(missing)
            )
        return cls(
            var_names=var_names,
            x_base=np.asarray(baselines, dtype=float),
            selected_features=selected,
            selected_idx=np.asarray([name_to_idx[name] for name in selected], dtype=int),
        )

    def to_selected_array(self, x: np.ndarray | list[float] | tuple[float, ...] | dict[str, float]) -> np.ndarray:
        if isinstance(x, dict):
            missing = [name for name in self.selected_features if name not in x]
            if missing:
                raise RuntimeError("Candidate point missing selected features: " + ", ".join(missing))
            arr = np.asarray([float(x[name]) for name in self.selected_features], dtype=float)
        else:
            arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.shape[0] != len(self.selected_features):
            raise RuntimeError(
                f"Candidate dimension mismatch: got {arr.shape[0]}, "
                f"expected {len(self.selected_features)}."
            )
        if np.any(~np.isfinite(arr)):
            raise RuntimeError("Candidate contains non-finite value.")
        return arr

    def to_full(self, x_selected: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        x_sel = self.to_selected_array(x_selected)
        x_full = self.x_base.copy()
        x_full[self.selected_idx] = x_sel
        return x_full

    def point_dict(self, x_selected: np.ndarray | list[float] | tuple[float, ...] | dict[str, float]) -> dict[str, float]:
        x_sel = self.to_selected_array(x_selected)
        return {name: float(value) for name, value in zip(self.selected_features, x_sel)}


class CaeObjectiveEvaluator:
    """Common CAE objective evaluator used by FocusBO and replaceable runtimes."""

    def __init__(
        self,
        *,
        problem_name: str,
        variables: list[dict],
        selected_features: list[str],
        error_context: str = "optimizer evaluation",
    ) -> None:
        self.mapper = SelectedFeatureMapper.from_variables(
            variables=variables,
            selected_features=selected_features,
        )
        _, self._evaluate_func = select_cae_by_name(str(problem_name).strip())
        self.error_context = str(error_context or "optimizer evaluation")

    def evaluate_selected(self, x_selected: np.ndarray | list[float] | tuple[float, ...] | dict[str, float]) -> float:
        x_full = self.mapper.to_full(x_selected)
        out = self._evaluate_func(x_full)
        success, objective, _outputs, invalid_reason, raw_repr = sanitize_evaluate_output(out)
        if not success:
            reason = invalid_reason or "success_false"
            raise RuntimeError(
                f"CAE evaluation failed during {self.error_context}: "
                f"reason={reason}, raw={raw_repr}"
            )
        return float(objective)

    def evaluate_pre_constraints(
        self,
        x_selected: np.ndarray | list[float] | tuple[float, ...] | dict[str, float],
        *,
        constraint_defs: list[dict],
        fail_fast_output_missing: bool = True,
    ) -> tuple[dict, bool, float]:
        return evaluate_pre_constraints_raw(
            x=self.mapper.to_full(x_selected),
            var_names=self.mapper.var_names,
            constraint_defs=constraint_defs,
            fail_fast_output_missing=fail_fast_output_missing,
        )
