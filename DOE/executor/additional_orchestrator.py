# DOE/executor/additional_orchestrator.py

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

import numpy as np

from utils.boundary_sampling import sample_boundary_corners, sample_boundary_partial
from utils.bounds_utils import compute_spans_lbs, clamp_to_bounds
from DOE.executor.constraint_filter import (
    clamp_ratio,
    evaluate_constraints_batch,
    evaluate_constraints_point,
    has_constraint_defs,
)
from DOE.executor.dataset_store import DatasetStore
from DOE.executor.plan_builder import PlanBuilder
from DOE.executor.execution_budget import ExecutionBudget
from DOE.executor.local_sampler import LocalSampler



class AdditionalDOEOrchestrator:
    """
    Additional DOE Orchestrator

    핵심 동작 요약:
    - 초기 DOE: 사용자 선택 sampler로 total_budget * init_ratio 만큼 실행
    - 추가 DOE는 stage 단위로 반복
    - X_plan: stage당 1회 생성, 중복 허용 (dedup 없음)
      * 현재 phase 비율로 global/local 분할
    - Gate 평가: X_plan으로 1회 평가
      * gate_stop이면 즉시 종료
      * phase 전환은 다음 stage부터 적용
    - X_exec: X_plan에서 선별 (중복 제거는 X_exec에서만)
      * n_exec = min(exec_ratio * total_budget, remaining)
      * global/local 비율은 현재 phase 기준
      * 부족하면 최대 3회 재시도 후 종료
    - tol: stage 기반으로 계산하며 X_exec dedup에만 적용
    - budget: 시도 횟수 기준으로 소모 (success 여부와 무관)
    - 실패(success=False)일 때 objective는 inf로 저장
    """
    def __init__(
        self,
        *,
        bounds: list[tuple[float, float]],
        sampler: Callable[..., np.ndarray],  # initial DOE sampler (lhs/rs)
        evaluate_func: Callable[[np.ndarray], Dict[str, Any]],
        feasibility_func: Callable[[Optional[Dict]], bool],
        surrogate_factory: Any,
        gate1: Any,
        gate2: Any,
        gate_manager: Any,
        rng: np.random.Generator,
        total_budget: int,
        init_ratio: float = 0.3,
        exec_ratio: float = 0.1,
        global_random_ratio: float = 0.3,
        global_boundary_ratio: float = 0.1,
        global_margin_ratio: float = 0.2,
        global_top_ratio: float = 0.2,
        global_boundary_corner_ratio: float = 0.5,
        plan_base_k: float = 200.0,
        plan_remaining_cap: float = 3.0,
        plan_decay: float = 0.85,
        phase1_global_ratio: Optional[float] = None,
        phase2_global_ratio: Optional[float] = None,
        min_additional_rounds: int = 2,
        stop_span_ratio_threshold: float = 0.3,
        stop_anchor_spread_streak: int = 2,
        initial_probe_multiplier: float = 2.0,
        plan_filter_safety: float = 1.2,
        plan_filter_r_floor: float = 0.02,
        max_additional_stages: int = 10,
        var_names: list[str] | None = None,
        constraint_defs: list[dict] | None = None,
        # local building knobs
        local_anchor_max_base: int = 8,
        local_anchor_max_decay: float = 0.9,
        local_anchor_best_k: int = 3,
        local_anchor_small_k: int = 2,
        local_anchor_best_ratio: float = 0.35,
        local_anchor_small_ratio: float = 0.2,
        local_radius_ratio_phase1: float = 0.5,
        local_radius_ratio_phase2: float = 0.3,
        local_top_p: float = 0.1,
        local_top_k_min: int = 10,
        local_dbscan_min_samples: int = 5,
        local_dbscan_q_eps: float = 0.85,
        local_dbscan_eps_max: float = 0.5,
        local_min_radius_ratio: float = 0.01,
        local_tol_ratio: float = 0.2,
        local_constraint_retry_count: int = 1,
        local_constraint_shrink_factor: float = 0.5,
        local_constraint_min_factor: float = 2.0,
        post_use_penalty: bool = True,
        post_lambda_init: float = 2.0,
        post_lambda_min: float = 0.25,
        post_lambda_max: float = 8.0,
        post_lambda_power: float = 1.0,
        post_feasible_rate_floor: float = 0.05,
        post_clf_min_samples: int = 30,
        post_clf_min_pos: int = 5,
        post_clf_min_neg: int = 5,
        hpo_runner: Optional[Any] = None,
        force_baseline: bool = False,
    ):
        self.bounds = bounds
        self.sampler = sampler

        self.evaluate_func = evaluate_func
        self.feasibility_func = feasibility_func

        self.surrogate_factory = surrogate_factory
        self.gate1 = gate1
        self.gate2 = gate2
        self.gate_manager = gate_manager

        self.rng = rng

        self.total_budget = int(total_budget)
        self.init_ratio = float(init_ratio)
        self.exec_ratio = float(exec_ratio)
        self.global_random_ratio = float(global_random_ratio)
        self.global_boundary_ratio = float(global_boundary_ratio)
        self.global_margin_ratio = float(global_margin_ratio)
        self.global_top_ratio = float(global_top_ratio)
        self.global_boundary_corner_ratio = float(global_boundary_corner_ratio)
        self.plan_base_k = float(plan_base_k)
        self.plan_remaining_cap = float(plan_remaining_cap)
        self.plan_decay = float(plan_decay)
        self.exec_min = 4
        if phase1_global_ratio is None or phase2_global_ratio is None:
            raise ValueError("phase1_global_ratio and phase2_global_ratio are required")
        self.phase1_global_ratio = float(phase1_global_ratio)
        self.phase2_global_ratio = float(phase2_global_ratio)
        self.min_additional_rounds = int(min_additional_rounds)
        self.stop_span_ratio_threshold = float(stop_span_ratio_threshold)
        self.stop_anchor_spread_streak = int(stop_anchor_spread_streak)
        self._anchor_spread_zero_streak = 0
        self.initial_probe_multiplier = float(initial_probe_multiplier)
        self.plan_filter_safety = float(plan_filter_safety)
        self.plan_filter_r_floor = float(plan_filter_r_floor)
        self.max_additional_stages = int(max_additional_stages)
        self.var_names = list(var_names or [f"x{i+1}" for i in range(len(bounds))])
        self.constraint_defs = list(constraint_defs or [])
        self.has_constraints = has_constraint_defs(self.constraint_defs)
        self.has_pre_constraints = any(
            str(c.get("scope", "x_only")).strip().lower() == "x_only"
            for c in self.constraint_defs
        )
        self.has_post_constraints = any(
            str(c.get("scope", "x_only")).strip().lower() == "cae_dependent"
            for c in self.constraint_defs
        )
        self.constraint_rate_hat = 1.0
        self._constraint_gen_total = 0
        self._constraint_feas_total = 0

        self.local_anchor_max_base = int(local_anchor_max_base)
        self.local_anchor_max_decay = float(local_anchor_max_decay)
        self.local_anchor_best_k = int(local_anchor_best_k)
        self.local_anchor_small_k = int(local_anchor_small_k)
        self.local_anchor_best_ratio = float(local_anchor_best_ratio)
        self.local_anchor_small_ratio = float(local_anchor_small_ratio)
        self.local_radius_ratio_phase1 = float(local_radius_ratio_phase1)
        self.local_radius_ratio_phase2 = float(local_radius_ratio_phase2)
        self.local_top_p = float(local_top_p)
        self.local_top_k_min = int(local_top_k_min)
        self.local_dbscan_min_samples = int(local_dbscan_min_samples)
        self.local_dbscan_q_eps = float(local_dbscan_q_eps)
        self.local_dbscan_eps_max = float(local_dbscan_eps_max)
        self.local_min_radius_ratio = float(local_min_radius_ratio)
        self.local_tol_ratio = float(local_tol_ratio)
        self.local_constraint_retry_count = int(local_constraint_retry_count)
        self.local_constraint_shrink_factor = float(local_constraint_shrink_factor)
        self.local_constraint_min_factor = float(local_constraint_min_factor)
        self.post_use_penalty = bool(post_use_penalty)
        self.post_lambda_init = float(post_lambda_init)
        self.post_lambda_min = float(post_lambda_min)
        self.post_lambda_max = float(post_lambda_max)
        self.post_lambda_power = float(post_lambda_power)
        self.post_feasible_rate_floor = float(post_feasible_rate_floor)
        self.post_clf_min_samples = int(post_clf_min_samples)
        self.post_clf_min_pos = int(post_clf_min_pos)
        self.post_clf_min_neg = int(post_clf_min_neg)
        self.post_lambda_current = float(post_lambda_init)
        self.post_feasible_rate_hat = 1.0
        self._post_feas_total = 0
        self._post_eval_total = 0
        self._post_model = None
        self.post_policy_log: list[dict] = []
        self.force_baseline = bool(force_baseline)

        self.store = DatasetStore(dim=len(bounds))
        self.budget = ExecutionBudget(total=self.total_budget)

        self.hpo_runner = hpo_runner
        self._hpo_done = False
        self.failure_reason: str | None = None
        self.local_metrics: list[dict] = []

        # Global plan builder is fixed LHS (per your scenario)
        self.plan_builder = PlanBuilder(bounds=bounds, rng=rng)

        self.local_sampler = LocalSampler(
            bounds=self.bounds,
            rng=self.rng,
            local_anchor_best_k=self.local_anchor_best_k,
            local_anchor_small_k=self.local_anchor_small_k,
            local_anchor_best_ratio=self.local_anchor_best_ratio,
            local_anchor_small_ratio=self.local_anchor_small_ratio,
            local_top_p=self.local_top_p,
            local_top_k_min=self.local_top_k_min,
            local_dbscan_min_samples=self.local_dbscan_min_samples,
            local_dbscan_q_eps=self.local_dbscan_q_eps,
            local_dbscan_eps_max=self.local_dbscan_eps_max,
            local_min_radius_ratio=self.local_min_radius_ratio,
            local_tol_ratio=self.local_tol_ratio,
        )

        # Exec selection is handled inline (rank + random mix)

    def _compute_n_plan(self, *, remaining: int, round_idx: int) -> int:
        dim = len(self.bounds)
        base = self.plan_base_k * float(dim)
        base = min(base, self.plan_remaining_cap * float(remaining))
        n_plan = int(base * (self.plan_decay ** float(round_idx)))
        return max(n_plan, 1)

    def _compute_anchor_counts(self, *, round_idx: int) -> tuple[int, int, int]:
        base = float(self.local_anchor_max_base) * (self.local_anchor_max_decay ** float(round_idx))
        max_k = int(np.ceil(base))
        max_k = max(max_k, 1)

        best_k = int(np.ceil(max_k * self.local_anchor_best_ratio))
        best_k = min(max(best_k, 0), self.local_anchor_best_k, max_k)

        small_k = int(np.ceil(max_k * self.local_anchor_small_ratio))
        small_k = min(max(small_k, 0), self.local_anchor_small_k, max(max_k - best_k, 0))

        # ensure at least one slot for big clusters when possible
        if max_k >= 2 and (max_k - best_k - small_k) < 1:
            if best_k > 1:
                best_k -= 1
            elif small_k > 1:
                small_k -= 1

        return max_k, best_k, small_k

    def _update_constraint_ratio(self, *, n_generated: int, n_feasible: int) -> None:
        if n_generated <= 0:
            return
        self._constraint_gen_total += int(n_generated)
        self._constraint_feas_total += int(n_feasible)
        raw_ratio = self._constraint_feas_total / max(self._constraint_gen_total, 1)
        self.constraint_rate_hat = clamp_ratio(raw_ratio, floor=self.plan_filter_r_floor)

    def _filter_by_constraints(
        self,
        X: np.ndarray,
        *,
        update_ratio: bool = False,
    ) -> tuple[np.ndarray, list[dict], np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        if X.size == 0:
            return (
                np.empty((0, len(self.bounds)), dtype=float),
                [],
                np.empty((0,), dtype=float),
                np.empty((0,), dtype=bool),
            )

        if not self.has_pre_constraints:
            mask = np.ones((X.shape[0],), dtype=bool)
            payloads = [{} for _ in range(X.shape[0])]
            margins = np.full((X.shape[0],), float("inf"), dtype=float)
        else:
            mask, payloads, margins = evaluate_constraints_batch(
                X=X,
                var_names=self.var_names,
                constraint_defs=self.constraint_defs,
                scope="x_only",
            )

        if update_ratio:
            self._update_constraint_ratio(
                n_generated=X.shape[0],
                n_feasible=int(mask.sum()),
            )

        idx = np.where(mask)[0]
        if idx.size == 0:
            return (
                np.empty((0, X.shape[1]), dtype=float),
                [],
                np.empty((0,), dtype=float),
                mask,
            )

        return (
            X[idx],
            [payloads[i] for i in idx],
            margins[idx],
            mask,
        )

    def _plan_generation_count(self, *, target_count: int) -> int:
        target_count = max(int(target_count), 1)
        if not self.has_pre_constraints:
            return target_count
        r_used = max(float(self.constraint_rate_hat), float(self.plan_filter_r_floor))
        inv = max(int(np.ceil(1.0 / r_used)), 1)
        n_gen = int(np.ceil(target_count * self.plan_filter_safety * inv))
        return max(n_gen, target_count)

    def _update_post_rate(self, *, feasible_final: bool) -> None:
        if not self.has_post_constraints:
            return
        self._post_eval_total += 1
        if bool(feasible_final):
            self._post_feas_total += 1
        raw = self._post_feas_total / max(self._post_eval_total, 1)
        self.post_feasible_rate_hat = clamp_ratio(raw, floor=self.post_feasible_rate_floor)

    def _update_post_lambda(self) -> None:
        if not (self.post_use_penalty and self.has_post_constraints):
            self.post_lambda_current = 0.0
            return
        r = clamp_ratio(self.post_feasible_rate_hat, floor=self.post_feasible_rate_floor)
        scale = ((1.0 - r) / r) ** self.post_lambda_power
        lam = self.post_lambda_init * scale
        self.post_lambda_current = float(
            np.clip(lam, self.post_lambda_min, self.post_lambda_max)
        )

    def _fit_post_feasibility_model(self) -> None:
        self._post_model = None
        if not (self.post_use_penalty and self.has_post_constraints):
            return
        if self.store.size < self.post_clf_min_samples:
            return

        X = self.store.X
        if X.size == 0:
            return
        y = np.asarray([1 if r.feasible_final else 0 for r in self.store.rows], dtype=int)
        if y.shape[0] != X.shape[0]:
            return

        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        if n_pos < self.post_clf_min_pos or n_neg < self.post_clf_min_neg:
            return

        try:
            from xgboost import XGBClassifier

            clf = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.0,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
            )
            clf.fit(X.astype(float), y.astype(int))
            self._post_model = clf
        except Exception as exc:
            print(f"[AdditionalDOE] post feasibility model training skipped: {exc}")
            self._post_model = None

    def _predict_post_feasible_prob(self, X_candidate: np.ndarray) -> np.ndarray:
        X_candidate = np.asarray(X_candidate, dtype=float)
        if X_candidate.size == 0:
            return np.empty((0,), dtype=float)
        if self._post_model is None:
            return np.ones((X_candidate.shape[0],), dtype=float)
        try:
            proba = self._post_model.predict_proba(X_candidate)[:, 1]
            proba = np.asarray(proba, dtype=float)
            return np.clip(proba, 0.0, 1.0)
        except Exception as exc:
            print(f"[AdditionalDOE] post feasibility model prediction skipped: {exc}")
            return np.ones((X_candidate.shape[0],), dtype=float)

    def _make_penalized_models(self, *, models: list, objective_sense: str) -> list:
        if not (self.post_use_penalty and self.has_post_constraints and self._post_model is not None):
            return models
        if self.post_lambda_current <= 0.0:
            return models

        parent = self
        sense = str(objective_sense).strip().lower()

        class _PenaltyWrappedModel:
            def __init__(self, base_model):
                self._base_model = base_model

            def predict(self, X):
                X_arr = np.asarray(X, dtype=float)
                y = np.asarray(self._base_model.predict(X_arr), dtype=float).reshape(-1)
                p_feas = parent._predict_post_feasible_prob(X_arr)
                penalty = parent.post_lambda_current * (1.0 - p_feas)
                if sense == "min":
                    return y + penalty
                return y - penalty

        return [_PenaltyWrappedModel(m) for m in models]

    def _dedup_mask_against_history_norm(
        self,
        *,
        X_new: np.ndarray,
        X_old: np.ndarray,
        spans: np.ndarray,
        dedup_tol: float,
    ) -> np.ndarray:
        if X_new.size == 0:
            return np.empty((0,), dtype=bool)
        if X_old.size == 0:
            return np.ones((X_new.shape[0],), dtype=bool)
        mask = np.ones((X_new.shape[0],), dtype=bool)
        for i, x in enumerate(X_new):
            dists = np.linalg.norm((X_old - x) / spans, axis=1)
            mask[i] = bool(np.all(dists > dedup_tol))
        return mask

    def _dedup_mask_against_history_raw(
        self,
        *,
        X_new: np.ndarray,
        X_old: np.ndarray,
        dedup_tol: float,
    ) -> np.ndarray:
        if X_new.size == 0:
            return np.empty((0,), dtype=bool)
        if X_old.size == 0:
            return np.ones((X_new.shape[0],), dtype=bool)
        mask = np.ones((X_new.shape[0],), dtype=bool)
        for i, x in enumerate(X_new):
            dists = np.linalg.norm(X_old - x, axis=1)
            mask[i] = bool(np.all(dists > dedup_tol))
        return mask

    def _dedup_mask_against_other_norm(
        self,
        *,
        X_new: np.ndarray,
        X_ref: np.ndarray,
        spans: np.ndarray,
        dedup_tol: float,
    ) -> np.ndarray:
        if X_new.size == 0:
            return np.empty((0,), dtype=bool)
        if X_ref.size == 0:
            return np.ones((X_new.shape[0],), dtype=bool)
        mask = np.ones((X_new.shape[0],), dtype=bool)
        for i, x in enumerate(X_new):
            dists = np.linalg.norm((X_ref - x) / spans, axis=1)
            mask[i] = bool(np.all(dists > dedup_tol))
        return mask

    # -------------------------------------------------
    # Main entry
    # -------------------------------------------------

    def run(
        self,
        *,
        baseline: np.ndarray,
        problem_name: str,  # reserved for logs / future HPO keys
        base_seed: int,     # reserved for deterministic pipelines
        objective_sense: str,
    ):
        # --------------------------------------------
        # Phase 0: Initial DOE (user-selected sampler)
        # --------------------------------------------
        n_init = max(int(self.total_budget * self.init_ratio), 1)

        if self.has_pre_constraints:
            probe_size = max(1, int(np.ceil(n_init * self.initial_probe_multiplier)))
            max_attempts = max(2, int(np.ceil(1.0 / max(self.plan_filter_r_floor, 1e-6))))
            pool_X: list[np.ndarray] = []
            pool_constraints: list[dict] = []
            pool_margins: list[float] = []

            for attempt in range(max_attempts):
                X_probe = self._sample_initial(n_samples=probe_size)
                if self.force_baseline and attempt == 0 and X_probe.shape[0] > 0:
                    X_probe[0, :] = np.asarray(baseline, dtype=float).reshape(-1)

                X_probe_f, constraints_f, margins_f, _ = self._filter_by_constraints(
                    X_probe,
                    update_ratio=True,
                )
                for i in range(X_probe_f.shape[0]):
                    pool_X.append(X_probe_f[i])
                    pool_constraints.append(constraints_f[i])
                    pool_margins.append(float(margins_f[i]))

                if len(pool_X) >= n_init:
                    break

            if len(pool_X) < n_init:
                self._set_failure("FAILED_FILTER_MIN", stage=0)
                return self._export_results()

            pick_idx = self.rng.choice(np.arange(len(pool_X)), size=n_init, replace=False)
            X_init = np.vstack([pool_X[i].reshape(1, -1) for i in pick_idx]).astype(float)
            init_constraints = [pool_constraints[i] for i in pick_idx]
            init_margins = np.asarray([pool_margins[i] for i in pick_idx], dtype=float)
        else:
            self.constraint_rate_hat = 1.0
            X_init = self._sample_initial(n_samples=n_init)
            if self.force_baseline and X_init.shape[0] > 0:
                X_init[0, :] = np.asarray(baseline, dtype=float).reshape(-1)
            init_constraints = [{} for _ in range(X_init.shape[0])]
            init_margins = np.full((X_init.shape[0],), float("inf"), dtype=float)

        executed = self._execute_points(
            X_init,
            source="initial",
            round_idx=None,
            exec_scope="initial",
            constraints_payloads=init_constraints,
            constraint_margins=init_margins,
        )
        if not executed:
            if self.failure_reason is None:
                self._set_failure("FAILED_BUDGET", stage=0)
            return self._export_results()

        self._fit_post_feasibility_model()
        self._update_post_lambda()



        # -------------------------------------------------
        # HPO: run ONCE after initial DOE
        # -------------------------------------------------
        if self.hpo_runner is not None and not self._hpo_done:
            # run HPO using initial DOE data
            result = self.hpo_runner.run_xgb(
                X=self.store.X,
                y=self.store.y,
                base_random_seed=base_seed,
                problem_name=problem_name,
            )

            best_params = result["best_params"]


            # inject best params into surrogate factory
            self.surrogate_factory.xgb_params = {
                **self.surrogate_factory.xgb_params,
                **best_params,
            }

            self._hpo_done = True

        # --------------------------------------------
        # Additional DOE loop
        # --------------------------------------------
        round_idx = 0
        phase = 1

        while not self.budget.exhausted() and round_idx < self.max_additional_stages:
            remaining = self.budget.remaining
            if remaining <= 0:
                break

            self._fit_post_feasibility_model()
            self._update_post_lambda()
            self.post_policy_log.append(
                {
                    "stage": int(round_idx),
                    "post_feasible_rate_hat": float(self.post_feasible_rate_hat),
                    "post_lambda": float(self.post_lambda_current),
                    "post_model_active": bool(self._post_model is not None),
                }
            )
            print(
                "[PostPolicy] "
                f"stage={round_idx} rate_hat={self.post_feasible_rate_hat:.4f} "
                f"lambda={self.post_lambda_current:.4f} model_active={self._post_model is not None}"
            )

            # -----------------------------
            # 2) Build X_plan (once per stage)
            # -----------------------------
            min_exec = max(int(self.total_budget * self.exec_ratio), self.exec_min)
            n_exec = min(min_exec, remaining)

            # -----------------------------
            # 1) Build surrogates (bundle)
            # -----------------------------
            df = self.store.to_dataframe()
            bundle = self.surrogate_factory.build(
                df=df,
                round_idx=round_idx,
                exec_size=n_exec,
            )

            gate1_models = bundle["gate1"]
            gate2_models = bundle["gate2"]
            gate1_eval_models = self._make_penalized_models(
                models=gate1_models,
                objective_sense=objective_sense,
            )
            gate2_eval_models = self._make_penalized_models(
                models=gate2_models,
                objective_sense=objective_sense,
            )

            n_plan_target = self._compute_n_plan(remaining=remaining, round_idx=round_idx)
            if n_plan_target < 2 * n_exec:
                n_plan_target = 2 * n_exec
            n_plan = self._plan_generation_count(target_count=n_plan_target)
            stage = round_idx + 1
            base_divisions = n_plan
            n_divisions = int(base_divisions * (2 ** (stage // 2)))
            # Tighten dedup radius to keep more local samples.
            tol = 0.5 * np.sqrt(len(self.bounds)) * (1.0 / n_divisions)

            ratio_global = self.phase1_global_ratio if phase == 1 else self.phase2_global_ratio
            n_global = int(round(n_plan * ratio_global))
            n_global = min(max(n_global, 0), n_plan)
            n_local = n_plan - n_global

            boundary_ratio = max(0.0, min(1.0, self.global_boundary_ratio))
            boundary_ratio *= float(0.9 ** round_idx)
            if self._anchor_spread_zero_streak >= self.stop_anchor_spread_streak:
                boundary_ratio = 0.0

            n_boundary = int(round(n_global * boundary_ratio))
            if n_boundary > 0 and n_global >= 2 and n_boundary < 2:
                # 코너/부분 경계를 모두 포함시키기 위해 최소 2개 확보
                n_boundary = 2
            spans, _ = compute_spans_lbs(self.bounds)
            offset = spans * tol

            boundary_flags = []
            if n_boundary > 0:
                corner_ratio = max(0.0, min(1.0, self.global_boundary_corner_ratio))
                corner_ratio *= float(0.9 ** round_idx)
                n_corner = int(round(n_boundary * corner_ratio))
                n_partial = n_boundary - n_corner
                if n_boundary >= 2:
                    # 경계 샘플이 2개 이상이면 코너/부분 경계 모두 최소 1개 보장
                    n_corner = min(max(n_corner, 1), n_boundary - 1)
                    n_partial = n_boundary - n_corner

                boundary_candidates = sample_boundary_corners(self.bounds, offset=offset)
                if n_corner > 0:
                    if boundary_candidates.shape[0] <= n_corner:
                        X_corner = boundary_candidates
                    else:
                        idx = self.rng.choice(boundary_candidates.shape[0], size=n_corner, replace=False)
                        X_corner = boundary_candidates[idx]
                else:
                    X_corner = np.empty((0, len(self.bounds)), dtype=float)

                if n_partial > 0:
                    # use LHC points as base for partial boundary
                    X_base = self.plan_builder.build(n_plan=max(n_partial, 1), n_divisions=n_divisions)
                    X_partial = sample_boundary_partial(
                        self.bounds,
                        offset=offset,
                        base_points=X_base,
                        n_samples=n_partial,
                        n_boundary_dims=min(2, len(self.bounds)),
                        rng=self.rng,
                    )
                else:
                    X_partial = np.empty((0, len(self.bounds)), dtype=float)

                Xg_boundary = np.vstack([X_corner, X_partial]) if X_corner.size or X_partial.size else np.empty((0, len(self.bounds)), dtype=float)
                boundary_flags = [True] * Xg_boundary.shape[0]
            else:
                Xg_boundary = np.empty((0, len(self.bounds)), dtype=float)

            n_lhc = max(n_global - Xg_boundary.shape[0], 0)
            Xg_lhc = self.plan_builder.build(n_plan=n_lhc, n_divisions=n_divisions) if n_lhc > 0 else np.empty((0, len(self.bounds)), dtype=float)

            Xg_boundary_f, constraints_boundary_f, margins_boundary_f, _ = self._filter_by_constraints(
                Xg_boundary,
                update_ratio=False,
            )
            Xg_lhc_f, constraints_lhc_f, margins_lhc_f, _ = self._filter_by_constraints(
                Xg_lhc,
                update_ratio=False,
            )
            Xg = np.vstack([Xg_boundary_f, Xg_lhc_f]) if Xg_boundary_f.size or Xg_lhc_f.size else np.empty((0, len(self.bounds)), dtype=float)
            constraints_g = constraints_boundary_f + constraints_lhc_f
            margins_g = np.concatenate([margins_boundary_f, margins_lhc_f]) if (margins_boundary_f.size or margins_lhc_f.size) else np.empty((0,), dtype=float)
            boundary_flags = np.concatenate(
                [
                    np.ones((Xg_boundary_f.shape[0],), dtype=bool),
                    np.zeros((Xg_lhc_f.shape[0],), dtype=bool),
                ]
            ) if (Xg_boundary_f.shape[0] + Xg_lhc_f.shape[0]) > 0 else np.zeros((0,), dtype=bool)

            if n_local > 0:
                # --- build local plan
                X_exec_success = self.store.X_success
                y_exec_success = self.store.y_success
                if phase == 1:
                    X_candidate = Xg
                    y_candidate = None
                else:
                    if X_exec_success.shape[0] > 0:
                        X_candidate = X_exec_success
                        y_candidate = y_exec_success
                    else:
                        X_candidate = Xg
                        y_candidate = None

                if X_candidate.shape[0] > 0:
                    radius_ratio = (
                        self.local_radius_ratio_phase1
                        if phase == 1
                        else self.local_radius_ratio_phase2
                    )
                    anchor_max, best_k, small_k = self._compute_anchor_counts(round_idx=round_idx)
                    debug_best_x = None
                    if phase == 2 and y_candidate is not None and y_candidate.size == X_candidate.shape[0]:
                        if objective_sense == "min":
                            best_idx = int(np.argmin(y_candidate))
                        else:
                            best_idx = int(np.argmax(y_candidate))
                        debug_best_x = X_candidate[best_idx]
                    Xl_raw, metrics, local_aux = self.local_sampler.build_local_plan(
                        models=gate1_models,
                        X_candidate=X_candidate,
                        y_candidate=y_candidate,
                        n_samples=n_local,
                        objective_sense=objective_sense,
                        local_radius_ratio=radius_ratio,
                        debug_best_x=debug_best_x,
                        anchor_max=anchor_max,
                        best_k=best_k,
                        small_k=small_k,
                    )
                    self._log_local_metrics(stage=round_idx, metrics=metrics)
                    self._update_convergence_streak(metrics)
                    Xl_anchors = np.asarray(local_aux.get("anchors", np.empty((0, Xg.shape[1]))), dtype=float)
                    Xl_anchor_ids = np.asarray(local_aux.get("anchor_ids", np.array([], dtype=int)))
                    tol_local = float(local_aux.get("tol_local", 0.0))
                    Xl, constraints_l, margins_l, mask_l = self._filter_by_constraints(
                        Xl_raw,
                        update_ratio=False,
                    )
                    Xl_anchor_ids = Xl_anchor_ids[mask_l] if Xl_anchor_ids.size == Xl_raw.shape[0] else np.array([], dtype=int)
                else:
                    Xl = np.empty((0, Xg.shape[1]), dtype=float)
                    constraints_l = []
                    margins_l = np.empty((0,), dtype=float)
                    Xl_anchors = np.empty((0, Xg.shape[1]), dtype=float)
                    Xl_anchor_ids = np.array([], dtype=int)
                    tol_local = 0.0

                X_plan = np.vstack([Xg, Xl]) if Xl.size else Xg
            else:
                Xl = np.empty((0, Xg.shape[1]), dtype=float)
                constraints_l = []
                margins_l = np.empty((0,), dtype=float)
                Xl_anchors = np.empty((0, Xg.shape[1]), dtype=float)
                Xl_anchor_ids = np.array([], dtype=int)
                tol_local = 0.0
                X_plan = Xg

            # -----------------------------
            # 3) Gate evaluation (on X_plan, once per stage)
            # -----------------------------
            if X_plan.shape[0] < 2 * n_exec:
                self._set_failure("FAILED_INSUFFICIENT_PLAN", stage=round_idx)
                break
            g2 = self.gate2.evaluate(models=gate2_eval_models, X_candidate=X_plan)
            g1 = self.gate1.evaluate(
                models=gate1_eval_models,
                X_candidate=X_plan,
                bounds=self.bounds,
                eps=0.5 * (1.0 / n_divisions),
            )
            print(
                f"[Gate1] stage={round_idx} score={g1.get('score')} passed={g1.get('passed')}"
            )
            print(
                f"[Gate2] stage={round_idx} score={g2.get('score')} passed={g2.get('passed')}"
            )

            decision = self.gate_manager.evaluate(
                gate1_result=g1,
                gate2_result=g2,
            )

            # If gates say stop, allow one final execution from current plan
            gate_stop = bool(decision.get("gate_stop", False))
            if round_idx + 1 >= self.min_additional_rounds:
                if self._should_stop_by_convergence():
                    gate_stop = True
            else:
                if gate_stop:
                    gate_stop = False
            if gate_stop:
                print(f"[AdditionalDOE] gate_stop=True, executing final batch (stage={round_idx})")

            g1_passed = bool(decision.get("gate1_passed", False))
            next_phase = 2 if g1_passed else 1

            # -----------------------------
            # 4) Filter X_plan then select X_exec
            # -----------------------------
            ratio_global = self.phase1_global_ratio if phase == 1 else self.phase2_global_ratio
            n_exec_global = int(round(n_exec * ratio_global))
            n_exec_global = min(max(n_exec_global, 0), n_exec)
            n_exec_local = n_exec - n_exec_global

            print(
                f"[AdditionalDOE] stage={round_idx} phase={phase} "
                f"n_plan={n_plan} n_exec={n_exec} "
                f"global/local={n_global}/{n_local} "
                f"exec_global/local={n_exec_global}/{n_exec_local}"
            )

            spans, lbs = compute_spans_lbs(self.bounds)
            mask_g = self._dedup_mask_against_history_norm(
                X_new=Xg,
                X_old=self.store.X,
                spans=spans,
                dedup_tol=tol,
            )
            Xg_f = Xg[mask_g]
            constraints_g_f = [constraints_g[i] for i in np.where(mask_g)[0]]
            margins_g_f = margins_g[mask_g] if margins_g.size == Xg.shape[0] else np.empty((Xg_f.shape[0],), dtype=float)
            boundary_flags_f = boundary_flags[mask_g] if boundary_flags.size == Xg.shape[0] else np.zeros((Xg_f.shape[0],), dtype=bool)

            if tol_local > 0.0:
                mask_l = self._dedup_mask_against_history_raw(
                    X_new=Xl,
                    X_old=self.store.X,
                    dedup_tol=tol_local,
                )
            else:
                mask_l = self._dedup_mask_against_history_norm(
                    X_new=Xl,
                    X_old=self.store.X,
                    spans=spans,
                    dedup_tol=tol,
                )
            Xl_f = Xl[mask_l]
            constraints_l_f = [constraints_l[i] for i in np.where(mask_l)[0]]
            margins_l_f = margins_l[mask_l] if margins_l.size == Xl.shape[0] else np.empty((Xl_f.shape[0],), dtype=float)
            Xl_anchor_ids_f = Xl_anchor_ids[mask_l] if Xl_anchor_ids.size == Xl.shape[0] else np.array([], dtype=int)

            if n_exec_local > 0 and n_exec_global > 0 and Xg_f.size > 0 and Xl_f.size > 0:
                mask_g_other = self._dedup_mask_against_other_norm(
                    X_new=Xg_f,
                    X_ref=Xl_f,
                    spans=spans,
                    dedup_tol=tol,
                )
                Xg_f = Xg_f[mask_g_other]
                constraints_g_f = [constraints_g_f[i] for i in np.where(mask_g_other)[0]]
                margins_g_f = margins_g_f[mask_g_other] if margins_g_f.size == mask_g_other.shape[0] else margins_g_f
                boundary_flags_f = boundary_flags_f[mask_g_other] if boundary_flags_f.size == mask_g_other.shape[0] else boundary_flags_f

            # local anchor는 local 실행에서 우선 포함될 수 있으므로,
            # global 실행 후보에서는 anchor와 좌표가 완전히 같은 점만 제거한다.
            if n_exec_local > 0 and Xg_f.size > 0 and Xl_anchors.size > 0:
                anchor_set = {tuple(row.tolist()) for row in np.asarray(Xl_anchors, dtype=float)}
                mask_g_anchor = np.asarray(
                    [tuple(row.tolist()) not in anchor_set for row in np.asarray(Xg_f, dtype=float)],
                    dtype=bool,
                )
                Xg_f = Xg_f[mask_g_anchor]
                constraints_g_f = [constraints_g_f[i] for i in np.where(mask_g_anchor)[0]]
                margins_g_f = margins_g_f[mask_g_anchor] if margins_g_f.size == mask_g_anchor.shape[0] else margins_g_f
                boundary_flags_f = boundary_flags_f[mask_g_anchor] if boundary_flags_f.size == mask_g_anchor.shape[0] else boundary_flags_f

            def _select_global_exec() -> tuple[np.ndarray, list[dict], np.ndarray]:
                if n_exec_global <= 0 or Xg_f.size == 0:
                    return np.empty((0, Xg_f.shape[1]), dtype=float), [], np.empty((0,), dtype=float)

                order_g = self._rank_by_pred(
                    models=gate1_models,
                    X_candidate=Xg_f,
                    objective_sense=objective_sense,
                )
                n_boundary_exec = int(round(n_exec_global * boundary_ratio))
                n_boundary_exec = min(max(n_boundary_exec, 0), n_exec_global)
                n_margin_exec = int(round(n_exec_global * self.global_margin_ratio))
                n_margin_exec = min(max(n_margin_exec, 0), n_exec_global)
                n_top_exec = int(round(n_exec_global * self.global_top_ratio))
                n_top_exec = min(max(n_top_exec, 0), n_exec_global)
                if n_exec_global > 0 and n_top_exec < 1:
                    n_top_exec = 1

                # top-k 최소 1개를 보장하기 위해 boundary/margin을 먼저 캡한다.
                max_for_pre_buckets = max(n_exec_global - n_top_exec, 0)
                if (n_boundary_exec + n_margin_exec) > max_for_pre_buckets:
                    overflow = (n_boundary_exec + n_margin_exec) - max_for_pre_buckets
                    reduce_margin = min(overflow, n_margin_exec)
                    n_margin_exec -= reduce_margin
                    overflow -= reduce_margin
                    if overflow > 0:
                        n_boundary_exec = max(n_boundary_exec - overflow, 0)

                selected: list[int] = []
                selected_set: set[int] = set()

                def _take(candidates: list[int], k: int) -> list[int]:
                    if k <= 0:
                        return []
                    out: list[int] = []
                    for cid in candidates:
                        if cid in selected_set:
                            continue
                        out.append(int(cid))
                        if len(out) >= k:
                            break
                    return out

                # 1) margin 작은 순(제약 경계 근처)
                if n_margin_exec > 0 and margins_g_f.size == Xg_f.shape[0]:
                    margin_order = np.argsort(margins_g_f).astype(int).tolist()
                    picked = _take(margin_order, n_margin_exec)
                    selected.extend(picked)
                    selected_set.update(picked)

                # 2) top-k
                if n_top_exec > 0:
                    picked = _take(order_g.astype(int).tolist(), n_top_exec)
                    selected.extend(picked)
                    selected_set.update(picked)

                # 3) boundary quota (가능한 만큼 확보)
                if n_boundary_exec > 0 and boundary_flags_f.size == Xg_f.shape[0]:
                    boundary_order = [idx for idx in order_g.tolist() if boundary_flags_f[idx]]
                    picked = _take(boundary_order, n_boundary_exec)
                    selected.extend(picked)
                    selected_set.update(picked)

                # 4) random(남은 슬롯)
                remaining_slots = max(n_exec_global - len(selected), 0)
                remain_order = [idx for idx in order_g.astype(int).tolist() if idx not in selected_set]
                if remaining_slots > 0 and remain_order:
                    rand_pool = np.asarray(remain_order, dtype=int)
                    n_take = min(remaining_slots, rand_pool.shape[0])
                    if n_take > 0:
                        rand_idx = self.rng.choice(rand_pool, size=n_take, replace=False).astype(int).tolist()
                        selected.extend(rand_idx)
                        selected_set.update(rand_idx)

                # 여전히 부족하면 예측 순으로 채움
                if len(selected) < n_exec_global:
                    remain = [idx for idx in order_g.astype(int).tolist() if idx not in selected_set]
                    fill = remain[: (n_exec_global - len(selected))]
                    selected.extend(fill)
                    selected_set.update(fill)

                selected = selected[:n_exec_global]
                if not selected:
                    return np.empty((0, Xg_f.shape[1]), dtype=float), [], np.empty((0,), dtype=float)
                idx = np.asarray(selected, dtype=int)
                return Xg_f[idx], [constraints_g_f[i] for i in idx], margins_g_f[idx]

            def _select_local_exec_equal() -> tuple[np.ndarray, list[dict], np.ndarray]:
                if n_exec_local <= 0 or Xl_f.size == 0:
                    return np.empty((0, Xl_f.shape[1]), dtype=float), [], np.empty((0,), dtype=float)

                anchors = Xl_anchors
                if anchors is None or anchors.size == 0:
                    return np.empty((0, Xl_f.shape[1]), dtype=float), [], np.empty((0,), dtype=float)

                if tol_local > 0.0:
                    anchor_keep = self._dedup_mask_against_history_raw(
                        X_new=anchors,
                        X_old=self.store.X,
                        dedup_tol=tol_local,
                    )
                else:
                    anchor_keep = self._dedup_mask_against_history_norm(
                        X_new=anchors,
                        X_old=self.store.X,
                        spans=spans,
                        dedup_tol=tol,
                    )

                anchor_count = anchors.shape[0]
                Xl_work = np.asarray(Xl_f, dtype=float).copy()
                constraints_work = list(constraints_l_f)
                margins_work = np.asarray(margins_l_f, dtype=float).copy()
                groups = {i: [] for i in range(anchor_count)}
                if Xl_anchor_ids_f.size == Xl_work.shape[0]:
                    for idx, aid in enumerate(Xl_anchor_ids_f.tolist()):
                        if 0 <= int(aid) < anchor_count:
                            groups[int(aid)].append(idx)

                active = []
                for i in range(anchor_count):
                    available = len(groups[i]) + (1 if anchor_keep[i] else 0)
                    if available > 0:
                        active.append(i)
                if not active:
                    return np.empty((0, Xl_f.shape[1]), dtype=float), [], np.empty((0,), dtype=float)

                per = n_exec_local // len(active)
                remainder = n_exec_local - per * len(active)
                quotas = {i: per + (1 if j < remainder else 0) for j, i in enumerate(active)}
                base_quotas = dict(quotas)
                deficits = 0

                # 제약 부족 anchor: 반경 축소 1회 재시도 후 실패 시 anchor 1개만 유지
                for i in active:
                    q_i = quotas[i]
                    if q_i <= 0:
                        continue
                    required_feasible = int(np.ceil(self.local_constraint_min_factor * (q_i + 1)))
                    f_i = len(groups[i])
                    if self.has_pre_constraints and f_i < required_feasible:
                        for retry_idx in range(self.local_constraint_retry_count):
                            shrink = self.local_constraint_shrink_factor ** float(retry_idx + 1)
                            X_retry = clamp_to_bounds(
                                anchors[i].reshape(1, -1)
                                + self.rng.normal(
                                    loc=0.0,
                                    scale=np.maximum(spans * radius_ratio * shrink, spans * self.local_min_radius_ratio),
                                    size=(max(4, required_feasible), Xl_work.shape[1]),
                                ),
                                self.bounds,
                            )
                            X_retry_f, c_retry_f, m_retry_f, _ = self._filter_by_constraints(X_retry, update_ratio=False)
                            if tol_local > 0.0:
                                mask_retry_hist = self._dedup_mask_against_history_raw(
                                    X_new=X_retry_f,
                                    X_old=self.store.X,
                                    dedup_tol=tol_local,
                                )
                            else:
                                mask_retry_hist = self._dedup_mask_against_history_norm(
                                    X_new=X_retry_f,
                                    X_old=self.store.X,
                                    spans=spans,
                                    dedup_tol=tol,
                                )
                            idx_retry = np.where(mask_retry_hist)[0]
                            for k in idx_retry.tolist():
                                if Xl_work.size > 0:
                                    if tol_local > 0.0:
                                        d_dup = np.linalg.norm(Xl_work - X_retry_f[k].reshape(1, -1), axis=1)
                                        if np.any(d_dup <= tol_local):
                                            continue
                                    else:
                                        d_dup = np.linalg.norm((Xl_work - X_retry_f[k].reshape(1, -1)) / spans, axis=1)
                                        if np.any(d_dup <= tol):
                                            continue
                                groups[i].append(len(Xl_work))
                                Xl_work = np.vstack([Xl_work, X_retry_f[k].reshape(1, -1)])
                                constraints_work.append(c_retry_f[k])
                                margins_work = np.concatenate([margins_work, np.asarray([m_retry_f[k]], dtype=float)])
                            f_i = len(groups[i])
                            if f_i >= required_feasible:
                                break
                    if self.has_pre_constraints and f_i < required_feasible:
                        deficits += max(q_i - 1, 0)
                        quotas[i] = 1

                if deficits > 0:
                    donors = [i for i in active if quotas[i] > 0]
                    for j in donors:
                        if deficits <= 0:
                            break
                        available_j = len(groups[j]) + (1 if anchor_keep[j] else 0)
                        start_q = max(base_quotas[j] - 1, 0)
                        spare = available_j - max(quotas[j], start_q)
                        if spare <= 0:
                            continue
                        take = min(spare, deficits)
                        quotas[j] += take
                        deficits -= take
                if deficits > 0:
                    return np.empty((0, Xl_work.shape[1]), dtype=float), [], np.empty((0,), dtype=float)

                picks = []
                picks_constraints = []
                picks_margins = []
                for i in active:
                    required = quotas[i]
                    if required <= 0:
                        continue
                    already = 0
                    if anchor_keep[i]:
                        picks.append(anchors[i].reshape(1, -1))
                        c_anchor, f_anchor, m_anchor = evaluate_constraints_point(
                            x=anchors[i],
                            var_names=self.var_names,
                            constraint_defs=self.constraint_defs,
                            scope="x_only",
                        )
                        if not self.has_pre_constraints or f_anchor:
                            picks_constraints.append(c_anchor if self.has_pre_constraints else {})
                            picks_margins.append(float(m_anchor if self.has_pre_constraints else float("inf")))
                            already = 1
                    need = required - already
                    if need <= 0:
                        continue
                    cand_idx = groups[i]
                    if len(cand_idx) < need:
                        return np.empty((0, Xl_work.shape[1]), dtype=float), [], np.empty((0,), dtype=float)
                    pts = Xl_work[cand_idx]
                    dists = np.linalg.norm(pts - anchors[i].reshape(1, -1), axis=1)
                    order = np.argsort(dists)[:need]
                    chosen_idx = [cand_idx[k] for k in order.tolist()]
                    picks.append(Xl_work[chosen_idx])
                    picks_constraints.extend([constraints_work[k] for k in chosen_idx])
                    picks_margins.extend([float(margins_work[k]) for k in chosen_idx])

                if not picks:
                    return np.empty((0, Xl_work.shape[1]), dtype=float), [], np.empty((0,), dtype=float)
                X_pick = np.vstack(picks).astype(float)
                if X_pick.shape[0] < n_exec_local:
                    return np.empty((0, Xl_work.shape[1]), dtype=float), [], np.empty((0,), dtype=float)
                return X_pick[:n_exec_local], picks_constraints[:n_exec_local], np.asarray(picks_margins[:n_exec_local], dtype=float)

            if n_exec_local > 0:
                X_exec_g, constraints_exec_g, margins_exec_g = _select_global_exec()
                X_exec_l, constraints_exec_l, margins_exec_l = _select_local_exec_equal()
                if X_exec_g.shape[0] < n_exec_global:
                    self._set_failure("FAILED_DEDUP", stage=round_idx)
                    break
                if X_exec_l.shape[0] < n_exec_local:
                    self._set_failure("FAILED_LOCAL_FEASIBILITY", stage=round_idx)
                    break
                X_exec = np.vstack([X_exec_g, X_exec_l]) if X_exec_l.size else X_exec_g
                constraints_exec = constraints_exec_g + constraints_exec_l
                margins_exec = np.concatenate([margins_exec_g, margins_exec_l]) if (margins_exec_g.size or margins_exec_l.size) else np.empty((0,), dtype=float)
            else:
                X_exec_g, constraints_exec_g, margins_exec_g = _select_global_exec()
                if X_exec_g.shape[0] < n_exec_global:
                    self._set_failure("FAILED_DEDUP", stage=round_idx)
                    break
                X_exec = X_exec_g
                constraints_exec = constraints_exec_g
                margins_exec = margins_exec_g

            if X_exec.shape[0] == 0:
                self._set_failure("FAILED_DEDUP", stage=round_idx)
                break

            # -----------------------------
            # 5) Execute CAE
            # -----------------------------
            executed = self._execute_points(
                X_exec,
                source="additional",
                round_idx=round_idx,
                exec_scope="mixed" if n_exec_local > 0 else "global",
                constraints_payloads=constraints_exec,
                constraint_margins=margins_exec,
            )
            if not executed:
                if self.failure_reason is None:
                    self._set_failure("FAILED_BUDGET", stage=round_idx)
                break

            if gate_stop:
                break

            # increase round only when actual execution happened
            round_idx += 1
            phase = next_phase

        if round_idx >= self.max_additional_stages and not self.budget.exhausted() and self.failure_reason is None:
            self._set_failure("STOP_MAX_ADDITIONAL_STAGES", stage=round_idx)

        if self.failure_reason is not None:
            print(f"[AdditionalDOE] terminated: {self.failure_reason}")
        else:
            print("[AdditionalDOE] terminated: COMPLETED")
        return self._export_results()

    def get_diagnostics(self) -> dict:
        out = {
            "failure_reason": self.failure_reason,
            "local_metrics": self.local_metrics,
            "constraint_rate_hat": float(self.constraint_rate_hat),
            "post_feasible_rate_hat": float(self.post_feasible_rate_hat),
            "post_lambda": float(self.post_lambda_current),
            "post_model_active": bool(self._post_model is not None),
            "post_policy_log": self.post_policy_log,
        }
        if self.has_pre_constraints:
            out["constraint_stats"] = {
                "generated": int(self._constraint_gen_total),
                "feasible": int(self._constraint_feas_total),
            }
        return out

    # -------------------------------------------------
    # Internal helpers
    # -------------------------------------------------
    def _update_convergence_streak(self, metrics: dict) -> None:
        anchor_spread = metrics.get("anchor_spread_mean")
        if anchor_spread == 0:
            self._anchor_spread_zero_streak += 1
        else:
            self._anchor_spread_zero_streak = 0

    def _should_stop_by_convergence(self) -> bool:
        if not self.local_metrics:
            return False
        latest = self.local_metrics[-1]
        span_ratio = latest.get("local_span_ratio_mean")
        if span_ratio is not None and span_ratio < self.stop_span_ratio_threshold:
            return True
        if self._anchor_spread_zero_streak >= self.stop_anchor_spread_streak:
            return True
        return False

    def _sample_initial(self, *, n_samples: int) -> np.ndarray:
        """
        Call user-selected sampler with compatible kwargs.
        """
        sig = inspect.signature(self.sampler)
        kwargs = {"n_samples": n_samples, "bounds": self.bounds, "rng": self.rng}
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        X = self.sampler(**accepted)
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.bounds):
            raise ValueError(f"Sampler returned invalid shape: {X.shape}")
        return X

    def _rank_by_pred(
        self,
        *,
        models,
        X_candidate: np.ndarray,
        objective_sense: str,
    ) -> np.ndarray:
        X_candidate = np.asarray(X_candidate, dtype=float)
        if X_candidate.ndim != 2 or X_candidate.shape[0] == 0:
            return np.array([], dtype=int)
        Yhat = np.vstack([m.predict(X_candidate) for m in models])
        y_mean = np.mean(Yhat, axis=0)
        if self.post_use_penalty and self.has_post_constraints and self.post_lambda_current > 0.0:
            p_feas = self._predict_post_feasible_prob(X_candidate)
            penalty = self.post_lambda_current * (1.0 - p_feas)
            if objective_sense == "min":
                score = y_mean + penalty
                order = np.argsort(score)
            else:
                score = y_mean - penalty
                order = np.argsort(-score)
        else:
            if objective_sense == "min":
                order = np.argsort(y_mean)
            else:
                order = np.argsort(-y_mean)
        return order.astype(int)

    def _execute_points(
        self,
        X: np.ndarray,
        *,
        source: str,
        round_idx: Optional[int],
        exec_scope: Optional[str] = None,
        constraints_payloads: list[dict] | None = None,
        constraint_margins: np.ndarray | None = None,
    ) -> bool:
        """
        Execute CAE on points.
        Returns False if budget exhausted, else True.
        """

        X = np.asarray(X, dtype=float)

        for i, x in enumerate(X):
            # 🔴 budget 먼저 확인
            if self.budget.remaining <= 0:
                return False

            out = self.evaluate_func(x)

            success = bool(out.get("success", True))
            if success:
                objective = float(out.get("objective"))
            else:
                objective = float("inf")
            outputs = out.get("outputs", {})
            if not isinstance(outputs, dict):
                outputs = {}

            if constraints_payloads is not None and i < len(constraints_payloads):
                constraints_pre = constraints_payloads[i] if constraints_payloads[i] is not None else {}
                feasible_pre = bool(self.feasibility_func(constraints_pre)) if constraints_pre is not None else True
                if constraint_margins is not None and i < len(constraint_margins):
                    margin_pre = float(constraint_margins[i])
                else:
                    margin_pre = float("inf")
            elif self.has_pre_constraints:
                constraints_pre, feasible_pre, margin_pre = evaluate_constraints_point(
                    x=x,
                    var_names=self.var_names,
                    constraint_defs=self.constraint_defs,
                    scope="x_only",
                )
            else:
                constraints_pre = {}
                feasible_pre = True
                margin_pre = float("inf")

            constraints_post: dict = {}
            feasible_post = True
            margin_post = float("inf")
            if self.has_post_constraints:
                try:
                    constraints_post, feasible_post, margin_post = evaluate_constraints_point(
                        x=x,
                        var_names=self.var_names,
                        constraint_defs=self.constraint_defs,
                        scope="cae_dependent",
                        env_extra={**outputs, "objective": objective},
                        fail_fast_output_missing=True,
                    )
                except Exception as exc:
                    self._set_failure("FAILED_POST_CONSTRAINT_OUTPUT", stage=int(round_idx or 0))
                    print(f"[AdditionalDOE] post-constraint evaluation failed: {exc}")
                    return False

            constraints = {**constraints_pre, **constraints_post}
            feasible_final = bool(success and feasible_pre and feasible_post)
            self._update_post_rate(feasible_final=bool(feasible_final))

            # store에는 기록 (실패도 기록 가능)
            self.store.add(
                x=x,
                objective=objective,
                constraints=constraints,
                feasible_pre=bool(feasible_pre),
                feasible_post=bool(feasible_post),
                feasible_final=bool(feasible_final),
                success=success,
                margin_pre=float(margin_pre),
                margin_post=float(margin_post),
                source=source,
                round_idx=round_idx,
                exec_scope=exec_scope,
            )

            consumed = self.budget.consume(1)
            if consumed < 1:
                return False

        return True

    def _set_failure(self, reason: str, *, stage: int) -> None:
        if self.failure_reason is None:
            self.failure_reason = reason
        print(f"[AdditionalDOE] stop_reason={reason} stage={stage}")

    def _log_local_metrics(self, *, stage: int, metrics: dict) -> None:
        payload = {"stage": stage, **metrics}
        self.local_metrics.append(payload)
        span_ratio = metrics.get("local_span_ratio_mean")
        pairwise_mean = metrics.get("exec_pairwise_dist_mean")
        anchor_spread = metrics.get("anchor_spread_mean")
        if span_ratio is None:
            print(f"[LocalMetrics] stage={stage} local empty")
            return
        print(
            "[LocalMetrics] "
            f"stage={stage} span_ratio_mean={span_ratio:.4f} "
            f"pairwise_mean={pairwise_mean} anchor_spread_mean={anchor_spread}"
        )
        best_in_topk = metrics.get("best_in_topk")
        best_anchor_dist = metrics.get("best_anchor_dist")
        best_local_min_dist = metrics.get("best_local_min_dist")
        if best_in_topk is not None or best_anchor_dist is not None or best_local_min_dist is not None:
            print(
                "[LocalDebug] "
                f"stage={stage} best_in_topk={best_in_topk} "
                f"best_anchor_dist={best_anchor_dist} best_local_min_dist={best_local_min_dist}"
            )




    def _export_results(self):
        results = []
        rows = self.store.rows  

        for i, r in enumerate(rows):
            results.append(
                {
                    "id": i,
                    "x": r.x.tolist(),
                    "objective": float(r.objective),
                    "constraints": r.constraints,
                    "margin_pre": float(r.margin_pre),
                    "margin_post": float(r.margin_post),
                    "constraint_margin": float(
                        np.min(
                            np.asarray(
                                [v for v in [r.margin_pre, r.margin_post] if np.isfinite(v)]
                                or [float("inf")],
                                dtype=float,
                            )
                        )
                    ),
                    "feasible_pre": bool(r.feasible_pre),
                    "feasible_post": bool(r.feasible_post),
                    "feasible_final": bool(r.feasible_final),
                    "feasible": bool(r.feasible_final),
                    "success": bool(r.success),
                    "source": r.source,
                    "round": r.round_idx,
                    "exec_scope": r.exec_scope,
                }
            )
        return results
