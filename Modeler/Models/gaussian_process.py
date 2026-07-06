from typing import Optional
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel

from .base import BaseModel


class GaussianProcessModel(BaseModel):
    """
    Gaussian Process regression model wrapper.
    """

    def __init__(
        self,
        *,
        random_state: Optional[int] = None,
        alpha: float = 1e-6,
        normalize_y: bool = True,
        n_restarts_optimizer: int = 1,
        kernel_id: str = "matern_2.5",
        length_scale_bounds_id: str = "additional_doe",
        length_scale_bounds: tuple[float, float] = (1e-4, 1e4),
        constant_value_bounds: tuple[float, float] = (1e-4, 1e4),
        use_white_kernel: bool = False,
        noise_level: float = 1e-6,
        noise_level_bounds_id: str = "default",
        noise_level_bounds: tuple[float, float] = (1e-8, 1e-1),
        matern_nu: float = 2.5,
        optimizer: str | None = "fmin_l_bfgs_b",
    ):
        super().__init__(model_name="GaussianProcess")
        self.params = dict(
            random_state=random_state,
            alpha=float(alpha),
            normalize_y=bool(normalize_y),
            n_restarts_optimizer=int(max(int(n_restarts_optimizer), 0)),
            kernel_id=str(kernel_id),
            length_scale_bounds_id=str(length_scale_bounds_id),
            length_scale_bounds=tuple(length_scale_bounds),
            constant_value_bounds=tuple(constant_value_bounds),
            use_white_kernel=bool(use_white_kernel),
            noise_level=float(noise_level),
            noise_level_bounds_id=str(noise_level_bounds_id),
            noise_level_bounds=tuple(noise_level_bounds),
            matern_nu=float(matern_nu),
            optimizer=optimizer,
        )

    @staticmethod
    def _resolve_length_scale_bounds(bounds_id: object, fallback: tuple[float, float]) -> tuple[float, float]:
        presets = {
            "additional_doe": (1e-4, 1e4),
            "tight": (1e-1, 1e1),
            "default": (1e-2, 1e2),
            "wide": (1e-3, 1e3),
            "conservative": (5e-3, 2e2),
        }
        return presets.get(str(bounds_id).strip().lower(), tuple(fallback))

    @staticmethod
    def _resolve_noise_bounds(bounds_id: object, fallback: tuple[float, float]) -> tuple[float, float]:
        presets = {
            "low_noise": (1e-10, 1e-3),
            "default": (1e-8, 1e-1),
        }
        return presets.get(str(bounds_id).strip().lower(), tuple(fallback))

    @staticmethod
    def _resolve_kernel(*, kernel_id: object, length_scale: np.ndarray, length_scale_bounds: tuple[float, float], matern_nu: float):
        key = str(kernel_id or "matern_2.5").strip().lower()
        if key == "rbf":
            return RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
        if key == "matern_1.5":
            return Matern(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=1.5)
        if key == "matern_0.5":
            return Matern(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=0.5)
        if key == "matern_2.5":
            return Matern(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=2.5)
        return Matern(
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds,
            nu=float(matern_nu),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        n_features = int(X_arr.shape[1]) if X_arr.ndim == 2 else 1
        length_scale = np.ones(n_features, dtype=float)
        length_scale_bounds = self._resolve_length_scale_bounds(
            self.params["length_scale_bounds_id"],
            self.params["length_scale_bounds"],
        )
        noise_level_bounds = self._resolve_noise_bounds(
            self.params["noise_level_bounds_id"],
            self.params["noise_level_bounds"],
        )
        base_kernel = self._resolve_kernel(
            kernel_id=self.params["kernel_id"],
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds,
            matern_nu=float(self.params["matern_nu"]),
        )
        kernel = ConstantKernel(1.0, self.params["constant_value_bounds"]) * base_kernel
        if bool(self.params["use_white_kernel"]):
            kernel = kernel + WhiteKernel(
                noise_level=float(self.params["noise_level"]),
                noise_level_bounds=noise_level_bounds,
            )
        self._estimator = GaussianProcessRegressor(
            kernel=kernel,
            alpha=float(self.params["alpha"]),
            normalize_y=bool(self.params["normalize_y"]),
            n_restarts_optimizer=int(self.params["n_restarts_optimizer"]),
            random_state=self.params["random_state"],
            optimizer=self.params["optimizer"],
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self._estimator.fit(X_arr, y_arr)
        self.is_fitted = True
