from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class GoalMonitor:
    """Optimizer-wide objective goal monitor.

    The monitor is optional. If no goal is provided, algorithms should run their
    full budget. If a goal is provided, the monitor reports goal-hit state and
    requests early stop after the best objective has not improved for the
    configured patience after reaching the goal.
    """

    goal: float | None
    objective_sense: str
    enabled: bool = True
    patience: int = 5
    hit: bool = False
    first_hit_iter: int | None = None
    no_improve_count: int = 0
    early_stopped: bool = False
    early_stop_reason: str = ""

    @classmethod
    def from_config(cls, *, user_config: object, system_config: object, objective_sense: str) -> "GoalMonitor":
        raw_goal = getattr(user_config, "goal", None)
        if raw_goal is None:
            raw_goal = getattr(user_config, "goal_objective", None)
        return cls.from_goal(
            goal=raw_goal,
            system_config=system_config,
            objective_sense=objective_sense,
        )

    @classmethod
    def from_goal(cls, *, goal: object, system_config: object, objective_sense: str) -> "GoalMonitor":
        goal = cls._coerce_goal(goal)
        objective_sense = cls._normalize_objective_sense(objective_sense)
        return cls(
            goal=goal,
            objective_sense=objective_sense,
            enabled=bool(getattr(system_config, "goal_early_stop_enabled", True)) and goal is not None,
            patience=int(max(int(getattr(system_config, "goal_early_stop_patience", 5)), 0)),
        )

    @staticmethod
    def _coerce_goal(value: object) -> float | None:
        if value is None:
            return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            raise ValueError("Optimizer goal must be a finite number or null.") from None
        if not math.isfinite(out):
            raise ValueError("Optimizer goal must be a finite number or null.")
        return out

    @staticmethod
    def _normalize_objective_sense(value: object) -> str:
        objective_sense = str(value or "min").strip().lower()
        if objective_sense not in {"min", "max"}:
            raise ValueError("Optimizer objective_sense must be 'min' or 'max'.")
        return objective_sense

    def _is_goal_hit(self, best_objective: float) -> bool:
        if self.goal is None or not math.isfinite(float(best_objective)):
            return False
        if str(self.objective_sense).strip().lower() == "max":
            return float(best_objective) >= float(self.goal)
        return float(best_objective) <= float(self.goal)

    def initialize(self, *, best_objective: float) -> None:
        self.hit = self._is_goal_hit(float(best_objective))
        self.first_hit_iter = 0 if self.hit else None
        self.no_improve_count = 0
        self.early_stopped = False
        self.early_stop_reason = ""

    def update(
        self,
        *,
        iteration: int,
        best_objective: float,
        improved: bool,
        allow_early_stop: bool = True,
    ) -> dict[str, object]:
        previous_hit = bool(self.hit)
        self.hit = self._is_goal_hit(float(best_objective))
        if self.hit and not previous_hit:
            self.first_hit_iter = int(iteration)
            self.no_improve_count = 0
        elif self.hit and bool(allow_early_stop):
            if bool(improved):
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1
        elif self.hit:
            self.no_improve_count = 0

        if (
            self.enabled
            and bool(allow_early_stop)
            and self.hit
            and int(self.no_improve_count) >= int(self.patience)
        ):
            self.early_stopped = True
            self.early_stop_reason = (
                "goal_reached_no_improvement:"
                f"goal={float(self.goal):.12g},"
                f"first_hit_iter={int(self.first_hit_iter or 0)},"
                f"patience={int(self.patience)}"
            )

        return self.state()

    def state(self) -> dict[str, object]:
        return {
            "goal": float(self.goal) if self.goal is not None else float("nan"),
            "goal_objective": float(self.goal) if self.goal is not None else float("nan"),
            "goal_hit": bool(self.hit),
            "goal_first_hit_iter": int(self.first_hit_iter or 0),
            "goal_no_improve_count": int(self.no_improve_count),
            "goal_early_stop_enabled": bool(self.enabled),
            "goal_early_stopped": bool(self.early_stopped),
            "goal_early_stop_reason": str(self.early_stop_reason),
            "goal_early_stop_patience": int(self.patience),
        }

    def summary(self) -> dict[str, object]:
        """Return a JSON-friendly goal state for result metadata.

        Per-iteration history keeps missing numeric values as NaN so CSV/debug
        analysis can treat them numerically. Result metadata should not expose
        NaN, so this summary converts missing goal values to null/None.
        """
        out: dict[str, object] = {}
        for key, value in self.state().items():
            if isinstance(value, float) and math.isnan(value):
                out[key] = None
            else:
                out[key] = value
        return out

    @property
    def should_stop(self) -> bool:
        return bool(self.early_stopped)
