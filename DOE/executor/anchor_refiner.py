from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    Matern,
    WhiteKernel,
)


ObjectiveSense = Literal["min", "max"]


class BoundsScaledGPModel:
    """Expose a raw-coordinate predict API while the wrapped GP is trained in scaled coordinates."""

    def __init__(self, model: GaussianProcessRegressor, *, lb: np.ndarray, span: np.ndarray):
        self._model = model
        self._lb = np.asarray(lb, dtype=float).reshape(-1)
        self._span = np.maximum(np.asarray(span, dtype=float).reshape(-1), 1e-12)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        return (X_arr - self._lb.reshape(1, -1)) / self._span.reshape(1, -1)

    def predict(self, X: np.ndarray, *args: Any, **kwargs: Any):
        return self._model.predict(self._scale(X), *args, **kwargs)


def gp_x_scaling_policy(
    *,
    bounds: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    enabled: bool = True,
    mode: str = "auto",
    auto_span_ratio_threshold: float = 3.0,
    span_floor: float = 1e-12,
) -> dict[str, Any]:
    requested_enabled = bool(enabled)
    mode_norm = str(mode or "auto").strip().lower()
    if mode_norm not in {"off", "auto", "bounds"}:
        mode_norm = "auto"
    threshold = float(max(float(auto_span_ratio_threshold), 1.0))
    floor = float(max(float(span_floor), 1e-15))
    arr = np.asarray(bounds, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
        return {
            "gp_x_scaling_enabled": requested_enabled,
            "gp_x_scaling_applied": False,
            "gp_x_scaling_mode": mode_norm,
            "gp_x_scaling_reason": "invalid_bounds",
            "gp_x_scaling_auto_span_ratio_threshold": threshold,
            "gp_x_scaling_span_min": None,
            "gp_x_scaling_span_max": None,
            "gp_x_scaling_span_ratio": None,
        }
    spans = np.maximum(arr[:, 1] - arr[:, 0], floor)
    span_min = float(np.min(spans))
    span_max = float(np.max(spans))
    span_ratio = float(span_max / max(span_min, floor))
    if not requested_enabled or mode_norm == "off":
        applied = False
        reason = "disabled" if not requested_enabled else "mode_off"
    elif mode_norm == "bounds":
        applied = True
        reason = "mode_bounds"
    else:
        applied = bool(span_ratio >= threshold)
        reason = "auto_span_ratio" if applied else "auto_span_ratio_below_threshold"
    return {
        "gp_x_scaling_enabled": requested_enabled,
        "gp_x_scaling_applied": bool(applied),
        "gp_x_scaling_mode": mode_norm,
        "gp_x_scaling_reason": reason,
        "gp_x_scaling_auto_span_ratio_threshold": threshold,
        "gp_x_scaling_span_min": span_min,
        "gp_x_scaling_span_max": span_max,
        "gp_x_scaling_span_ratio": span_ratio,
    }


def _scale_args(
    d: int,
    *,
    noise: float = 1e-6,
    l_kernel: float = 1.0,
    l_scale_const: float = 0.0001,
) -> tuple[float, tuple[float, float], list[tuple[float, float]]]:
    _ = noise
    l_scale = l_scale_const * l_kernel
    const_bounds = (max(l_scale, 1e-4), min(1.0 / l_scale, 1e4))
    ard_bounds = [const_bounds] * int(d)
    return l_scale, const_bounds, ard_bounds


def kernel_common_best(
    d: int,
    *,
    noise: float = 1e-6,
    l_kernel: float = 1.0,
    l_scale_const: float = 0.0001,
    include_white: bool = False,
):
    _l_scale, const_bounds, ard_bounds = _scale_args(
        d,
        noise=noise,
        l_kernel=l_kernel,
        l_scale_const=l_scale_const,
    )
    safe_noise = max(float(noise), 1e-8)
    base = C(constant_value=1.0 * l_kernel, constant_value_bounds=const_bounds) * Matern(
        length_scale=[1.0 * l_kernel] * int(d),
        length_scale_bounds=ard_bounds,
        nu=2.5,
    )
    if not include_white:
        return base
    return base + WhiteKernel(noise_level=safe_noise, noise_level_bounds=(1e-8, 1e-2))


def kernel_stable_conservative(
    d: int,
    *,
    noise: float = 1e-6,
    include_white: bool = False,
):
    safe_noise = max(float(noise), 1e-6)
    base = C(1.0, (0.1, 10.0)) * Matern(
        length_scale=[1.0] * int(d),
        length_scale_bounds=[(5e-3, 2e2)] * int(d),
        nu=2.5,
    )
    if not include_white:
        return base
    return base + WhiteKernel(noise_level=safe_noise, noise_level_bounds=(1e-6, 1e-1))


def fit_gp_with_fallback(
    *,
    X: np.ndarray,
    y: np.ndarray,
    include_white: bool = False,
    random_state: int = 0,
) -> tuple[GaussianProcessRegressor | None, bool]:
    """GP 학습: kernel_common_best 시도 → 실패 시 kernel_stable_conservative fallback."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[0] != y.shape[0]:
        if X.ndim == 2 and X.shape[0] < 2:
            print(f"[GP] skipped: n_samples={X.shape[0]} < 2 (dim={X.shape[1] if X.ndim == 2 else '?'})")
        return None, False
    dim = X.shape[1]
    try:
        gp = GaussianProcessRegressor(
            kernel=kernel_common_best(dim, include_white=include_white),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=random_state,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            gp.fit(X, y)
        return gp, False
    except Exception as e:
        print(f"[GP] fallback to conservative kernel: {e}")
        try:
            gp = GaussianProcessRegressor(
                kernel=kernel_stable_conservative(dim, include_white=include_white),
                alpha=1e-5,
                normalize_y=False,
                n_restarts_optimizer=1,
                random_state=random_state,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                gp.fit(X, y)
            return gp, True
        except Exception as e2:
            print(f"[GP] all kernels failed: {e2}")
            return None, True


def fit_gp_with_optional_x_scaling(
    *,
    X: np.ndarray,
    y: np.ndarray,
    bounds: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    include_white: bool = False,
    random_state: int = 0,
    gp_x_scaling_enabled: bool = True,
    gp_x_scaling_mode: str = "auto",
    gp_x_scaling_auto_span_ratio_threshold: float = 3.0,
    gp_x_scaling_span_floor: float = 1e-12,
) -> tuple[Any | None, bool, dict[str, Any]]:
    policy = gp_x_scaling_policy(
        bounds=bounds,
        enabled=gp_x_scaling_enabled,
        mode=gp_x_scaling_mode,
        auto_span_ratio_threshold=gp_x_scaling_auto_span_ratio_threshold,
        span_floor=gp_x_scaling_span_floor,
    )
    X_arr = np.asarray(X, dtype=float)
    arr = np.asarray(bounds, dtype=float)
    if X_arr.ndim != 2 or arr.ndim != 2 or arr.shape[0] != X_arr.shape[1]:
        policy = {**policy, "gp_x_scaling_applied": False, "gp_x_scaling_reason": "bounds_dim_mismatch"}
    if not bool(policy.get("gp_x_scaling_applied", False)):
        model, fallback = fit_gp_with_fallback(
            X=X_arr,
            y=y,
            include_white=include_white,
            random_state=random_state,
        )
        return model, fallback, policy

    lb = arr[:, 0]
    span = np.maximum(arr[:, 1] - arr[:, 0], max(float(gp_x_scaling_span_floor), 1e-15))
    X_scaled = (X_arr - lb.reshape(1, -1)) / span.reshape(1, -1)
    model, fallback = fit_gp_with_fallback(
        X=X_scaled,
        y=y,
        include_white=include_white,
        random_state=random_state,
    )
    if model is None:
        return None, fallback, policy
    return BoundsScaledGPModel(model, lb=lb, span=span), fallback, policy


class AcquisitionOptimizer:
    def __init__(self):
        pass

    @staticmethod
    def acquisition_lcb(
        x: np.ndarray,
        model: GaussianProcessRegressor,
        *,
        kappa: float,
        objective_sense: ObjectiveSense,
    ) -> float:
        mu, sigma = model.predict(np.asarray(x, dtype=float).reshape(1, -1), return_std=True)
        m = float(mu.reshape(-1)[0])
        s = float(sigma.reshape(-1)[0])
        if objective_sense == "min":
            return m - float(kappa) * s
        return -(m + float(kappa) * s)

    @staticmethod
    def acquisition_ei(
        x: np.ndarray,
        model: GaussianProcessRegressor,
        *,
        y_best: float,
        objective_sense: ObjectiveSense,
        xi: float = 0.01,
    ) -> float:
        mu, sigma = model.predict(np.asarray(x, dtype=float).reshape(1, -1), return_std=True)
        m = float(mu.reshape(-1)[0])
        s = max(float(sigma.reshape(-1)[0]), 1e-9)
        if objective_sense == "min":
            imp = float(y_best) - m - float(xi)
        else:
            imp = m - float(y_best) - float(xi)
        z = imp / s
        ei = imp * norm.cdf(z) + s * norm.pdf(z)
        return -float(ei)

    @staticmethod
    def acquisition_mean(
        x: np.ndarray,
        model: GaussianProcessRegressor,
        *,
        objective_sense: ObjectiveSense,
    ) -> float:
        mu = model.predict(np.asarray(x, dtype=float).reshape(1, -1))
        m = float(np.asarray(mu, dtype=float).reshape(-1)[0])
        if objective_sense == "max":
            return -m
        return m

    def optimize(
        self,
        *,
        model: GaussianProcessRegressor,
        y_best: float,
        lb: np.ndarray,
        ub: np.ndarray,
        starts: np.ndarray,
        objective_sense: ObjectiveSense,
        acq_type: str = "LCB",
        kappa: float = 2.0,
        xi: float = 0.01,
        pre_feasible_fn: Callable[[np.ndarray], bool] | None = None,
        soft_penalty_fn: Callable[[np.ndarray], float] | None = None,
        post_feasible_prob_fn: Callable[[np.ndarray], float] | None = None,
        post_penalty_lambda: float = 0.0,
        pre_hard_penalty: float = 1e9,
        maxiter: int | None = None,
        maxfun: int | None = None,
    ) -> np.ndarray | None:
        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        starts = np.asarray(starts, dtype=float)
        if starts.ndim != 2 or starts.shape[1] != lb.shape[0]:
            return None

        bounds = list(zip(lb.tolist(), ub.tolist()))
        best_x = None
        best_val = float("inf")
        acq = str(acq_type).strip().upper()

        options: dict[str, int] = {}
        if maxiter is not None and int(maxiter) > 0:
            options["maxiter"] = int(maxiter)
        if maxfun is not None and int(maxfun) > 0:
            options["maxfun"] = int(maxfun)

        for x0 in starts:
            x0 = np.clip(np.asarray(x0, dtype=float).reshape(-1), lb, ub)

            def fn(x: np.ndarray) -> float:
                x_arr = np.asarray(x, dtype=float).reshape(-1)
                if pre_feasible_fn is not None and not bool(pre_feasible_fn(x_arr)):
                    return float(pre_hard_penalty)
                if acq == "EI":
                    base = self.acquisition_ei(
                        x_arr,
                        model,
                        y_best=y_best,
                        objective_sense=objective_sense,
                        xi=xi,
                    )
                elif acq == "MEAN":
                    base = self.acquisition_mean(
                        x_arr,
                        model,
                        objective_sense=objective_sense,
                    )
                else:
                    base = self.acquisition_lcb(
                        x_arr,
                        model,
                        kappa=kappa,
                        objective_sense=objective_sense,
                    )
                if post_feasible_prob_fn is not None and post_penalty_lambda > 0.0:
                    try:
                        p_post = float(post_feasible_prob_fn(x_arr))
                    except Exception:
                        p_post = 1.0
                    p_post = float(np.clip(p_post, 0.0, 1.0))
                    base = float(base) + float(post_penalty_lambda) * (1.0 - p_post)
                if soft_penalty_fn is not None:
                    try:
                        soft_penalty = float(soft_penalty_fn(x_arr))
                    except Exception:
                        soft_penalty = 0.0
                    if np.isfinite(soft_penalty) and soft_penalty > 0.0:
                        base = float(base) + float(soft_penalty)
                return float(base)

            try:
                result = minimize(
                    fn,
                    x0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options=options if options else None,
                )
            except Exception:
                continue
            if not result.success and not np.isfinite(float(getattr(result, "fun", float("nan")))):
                continue
            x_opt = np.asarray(result.x, dtype=float).reshape(-1)
            if pre_feasible_fn is not None and not bool(pre_feasible_fn(x_opt)):
                continue
            val = float(result.fun)
            if val < best_val:
                best_val = val
                best_x = x_opt
        return best_x


@dataclass
class RefineResult:
    refined_anchor: np.ndarray | None
    refined_score: float | None
    box_lb: np.ndarray
    box_ub: np.ndarray
    used_gp: bool
    used_fallback_kernel: bool
    status: str


class GPAnchorRefiner:
    def __init__(
        self,
        *,
        bounds: list[tuple[float, float]],
        rng: np.random.Generator,
        random_seed: int,
        refine_min_points: int = 15,
        delta_ratio: float = 0.02,
        singleton_box_ratio: float = 0.03,
        phase1_kappa: float = 0.75,
        phase2_kappa: float = 0.5,
        q_expand_step: int = 10,
        use_white_kernel: bool = False,
    ):
        self.bounds = list(bounds)
        self.rng = rng
        self.random_seed = int(random_seed)
        self.refine_min_points = int(refine_min_points)
        self.delta_ratio = float(delta_ratio)
        self.singleton_box_ratio = float(singleton_box_ratio)
        self.phase1_kappa = float(phase1_kappa)
        self.phase2_kappa = float(phase2_kappa)
        self.q_expand_step = int(q_expand_step)
        self.use_white_kernel = bool(use_white_kernel)
        self._acq = AcquisitionOptimizer()

        self._lb = np.asarray([b[0] for b in self.bounds], dtype=float)
        self._ub = np.asarray([b[1] for b in self.bounds], dtype=float)
        self._span = np.maximum(self._ub - self._lb, 1e-12)

    def build_cluster_box(
        self,
        *,
        cluster_points: np.ndarray,
        base_anchor: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(cluster_points, dtype=float)
        base = np.asarray(base_anchor, dtype=float).reshape(-1)
        dim = self._lb.shape[0]

        if pts.ndim != 2 or pts.shape[1] != dim or pts.shape[0] == 0:
            half = 0.5 * self.singleton_box_ratio * self._span
            lb = base - half
            ub = base + half
        else:
            delta = self.delta_ratio * self._span
            lb = np.min(pts, axis=0) - delta
            ub = np.max(pts, axis=0) + delta
            raw_span = np.maximum(ub - lb, 0.0)
            min_span = self.singleton_box_ratio * self._span
            center = 0.5 * (lb + ub)
            need = raw_span < min_span
            lb[need] = center[need] - 0.5 * min_span[need]
            ub[need] = center[need] + 0.5 * min_span[need]

        lb = np.maximum(lb, self._lb)
        ub = np.minimum(ub, self._ub)
        too_small = (ub - lb) < 1e-12
        if np.any(too_small):
            c = np.clip(base, self._lb, self._ub)
            eps = 0.5 * self.singleton_box_ratio * self._span
            lb[too_small] = np.maximum(c[too_small] - eps[too_small], self._lb[too_small])
            ub[too_small] = np.minimum(c[too_small] + eps[too_small], self._ub[too_small])
        return lb, ub

    def _in_box_mask(
        self,
        *,
        X: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.all((X >= lb.reshape(1, -1)) & (X <= ub.reshape(1, -1)), axis=1)

    def _good_mask_by_q(
        self,
        *,
        y: np.ndarray,
        q: float,
        objective_sense: ObjectiveSense,
    ) -> np.ndarray:
        y = np.asarray(y, dtype=float).reshape(-1)
        q_clip = float(np.clip(q, 0.0, 100.0))
        if objective_sense == "min":
            thr = float(np.percentile(y, q_clip))
            return y <= thr
        thr = float(np.percentile(y, 100.0 - q_clip))
        return y >= thr

    def _fit_gp(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[GaussianProcessRegressor | None, bool]:
        return fit_gp_with_fallback(
            X=X,
            y=y,
            include_white=self.use_white_kernel,
            random_state=self.random_seed,
        )

    def _build_multistarts(
        self,
        *,
        base_anchor: np.ndarray,
        box_lb: np.ndarray,
        box_ub: np.ndarray,
        perturb_ratio: float,
    ) -> np.ndarray:
        base = np.asarray(base_anchor, dtype=float).reshape(-1)
        lb = np.asarray(box_lb, dtype=float).reshape(-1)
        ub = np.asarray(box_ub, dtype=float).reshape(-1)
        span = np.maximum(ub - lb, 1e-12)
        starts = [np.clip(base, lb, ub)]
        # 2 random in box
        starts.append(self.rng.uniform(lb, ub))
        starts.append(self.rng.uniform(lb, ub))
        # 2 perturb around base
        scale = float(perturb_ratio) * span
        starts.append(np.clip(base + self.rng.normal(0.0, scale, size=base.shape[0]), lb, ub))
        starts.append(np.clip(base + self.rng.normal(0.0, scale, size=base.shape[0]), lb, ub))
        return np.asarray(starts, dtype=float)

    def refine_phase1(
        self,
        *,
        base_anchor: np.ndarray,
        cluster_points: np.ndarray,
        X_source: np.ndarray,
        y_source: np.ndarray,
        objective_sense: ObjectiveSense,
        acq_type: str = "LCB",
        perturb_ratio: float = 0.02,
        pre_feasible_fn: Callable[[np.ndarray], bool] | None = None,
        post_feasible_prob_fn: Callable[[np.ndarray], float] | None = None,
        post_penalty_lambda: float = 0.0,
    ) -> RefineResult:
        base = np.asarray(base_anchor, dtype=float).reshape(-1)
        X_source = np.asarray(X_source, dtype=float)
        y_source = np.asarray(y_source, dtype=float).reshape(-1)
        box_lb, box_ub = self.build_cluster_box(cluster_points=cluster_points, base_anchor=base)
        in_box = self._in_box_mask(X=X_source, lb=box_lb, ub=box_ub)
        idx = np.where(in_box)[0]
        if idx.size < 2:
            return RefineResult(
                refined_anchor=base.copy(),
                refined_score=None,
                box_lb=box_lb,
                box_ub=box_ub,
                used_gp=False,
                used_fallback_kernel=False,
                status="phase1_insufficient_box_points",
            )

        X_train = X_source[idx]
        y_train = y_source[idx]
        gp_model, fallback_used = self._fit_gp(
            X=X_train,
            y=y_train,
        )
        if gp_model is None:
            return RefineResult(
                refined_anchor=base.copy(),
                refined_score=None,
                box_lb=box_lb,
                box_ub=box_ub,
                used_gp=False,
                used_fallback_kernel=fallback_used,
                status="phase1_gp_fit_failed",
            )

        if objective_sense == "min":
            y_best = float(np.min(y_train))
        else:
            y_best = float(np.max(y_train))
        starts = self._build_multistarts(
            base_anchor=base,
            box_lb=box_lb,
            box_ub=box_ub,
            perturb_ratio=perturb_ratio,
        )
        refined = self._acq.optimize(
            model=gp_model,
            y_best=y_best,
            lb=box_lb,
            ub=box_ub,
            starts=starts,
            objective_sense=objective_sense,
            acq_type=acq_type,
            kappa=self.phase1_kappa,
            pre_feasible_fn=pre_feasible_fn,
            post_feasible_prob_fn=post_feasible_prob_fn,
            post_penalty_lambda=post_penalty_lambda,
        )
        if refined is None:
            return RefineResult(
                refined_anchor=base.copy(),
                refined_score=None,
                box_lb=box_lb,
                box_ub=box_ub,
                used_gp=True,
                used_fallback_kernel=fallback_used,
                status="phase1_opt_failed",
            )

        mu_ref, _std_ref = gp_model.predict(refined.reshape(1, -1), return_std=True)
        return RefineResult(
            refined_anchor=np.asarray(refined, dtype=float).reshape(-1),
            refined_score=float(mu_ref.reshape(-1)[0]),
            box_lb=box_lb,
            box_ub=box_ub,
            used_gp=True,
            used_fallback_kernel=fallback_used,
            status="phase1_refined",
        )

    def refine_phase2(
        self,
        *,
        base_anchor: np.ndarray,
        cluster_points: np.ndarray,
        X_source: np.ndarray,
        y_source: np.ndarray,
        objective_sense: ObjectiveSense,
        base_q: float,
        acq_type: str = "LCB",
        perturb_ratio: float = 0.02,
        pre_feasible_fn: Callable[[np.ndarray], bool] | None = None,
        post_feasible_prob_fn: Callable[[np.ndarray], float] | None = None,
        post_penalty_lambda: float = 0.0,
    ) -> RefineResult:
        base = np.asarray(base_anchor, dtype=float).reshape(-1)
        X_source = np.asarray(X_source, dtype=float)
        y_source = np.asarray(y_source, dtype=float).reshape(-1)
        box_lb, box_ub = self.build_cluster_box(cluster_points=cluster_points, base_anchor=base)

        in_box = self._in_box_mask(X=X_source, lb=box_lb, ub=box_ub)
        if not np.any(in_box):
            return RefineResult(
                refined_anchor=None,
                refined_score=None,
                box_lb=box_lb,
                box_ub=box_ub,
                used_gp=False,
                used_fallback_kernel=False,
                status="phase2_no_history_in_box",
            )

        q = float(np.clip(base_q, 0.0, 100.0))
        upper_bound = 100.0
        selected = np.array([], dtype=int)
        while True:
            good = self._good_mask_by_q(y=y_source, q=q, objective_sense=objective_sense)
            idx = np.where(in_box & good)[0]
            selected = idx
            if selected.size >= self.refine_min_points:
                break
            if q >= upper_bound:
                break
            q = min(q + self.q_expand_step, upper_bound)
            if q == upper_bound:
                good = self._good_mask_by_q(y=y_source, q=q, objective_sense=objective_sense)
                selected = np.where(in_box & good)[0]
                if selected.size >= self.refine_min_points:
                    break
                break

        if selected.size < self.refine_min_points:
            return RefineResult(
                refined_anchor=None,
                refined_score=None,
                box_lb=box_lb,
                box_ub=box_ub,
                used_gp=False,
                used_fallback_kernel=False,
                status="phase2_insufficient_history",
            )

        X_train = X_source[selected]
        y_train = y_source[selected]
        gp_model, fallback_used = self._fit_gp(
            X=X_train,
            y=y_train,
        )
        if gp_model is None:
            return RefineResult(
                refined_anchor=None,
                refined_score=None,
                box_lb=box_lb,
                box_ub=box_ub,
                used_gp=False,
                used_fallback_kernel=fallback_used,
                status="phase2_gp_fit_failed",
            )

        if objective_sense == "min":
            y_best = float(np.min(y_train))
        else:
            y_best = float(np.max(y_train))
        starts = self._build_multistarts(
            base_anchor=base,
            box_lb=box_lb,
            box_ub=box_ub,
            perturb_ratio=perturb_ratio,
        )
        refined = self._acq.optimize(
            model=gp_model,
            y_best=y_best,
            lb=box_lb,
            ub=box_ub,
            starts=starts,
            objective_sense=objective_sense,
            acq_type=acq_type,
            kappa=self.phase2_kappa,
            pre_feasible_fn=pre_feasible_fn,
            post_feasible_prob_fn=post_feasible_prob_fn,
            post_penalty_lambda=post_penalty_lambda,
        )
        if refined is None:
            return RefineResult(
                refined_anchor=None,
                refined_score=None,
                box_lb=box_lb,
                box_ub=box_ub,
                used_gp=True,
                used_fallback_kernel=fallback_used,
                status="phase2_opt_failed",
            )

        mu_ref, _std_ref = gp_model.predict(refined.reshape(1, -1), return_std=True)
        return RefineResult(
            refined_anchor=np.asarray(refined, dtype=float).reshape(-1),
            refined_score=float(mu_ref.reshape(-1)[0]),
            box_lb=box_lb,
            box_ub=box_ub,
            used_gp=True,
            used_fallback_kernel=fallback_used,
            status="phase2_refined",
        )
