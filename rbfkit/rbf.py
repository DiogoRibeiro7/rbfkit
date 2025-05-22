from __future__ import annotations
from typing import Optional, Union
import numpy as np
from enum import Enum


class RBFFunction(Enum):
    """
    Supported radial basis functions.
    """

    GAUSSIAN = "gaussian"
    MULTIQUADRIC = "multiquadric"
    INVERSE_MULTIQ = "inverse_multiquadric"
    THIN_PLATE = "thin_plate"


def _pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute squared Euclidean distances between two sets of points.

    Args:
        x: Array of shape (m, d).
        y: Array of shape (n, d).
    Returns:
        (m, n) array of squared distances.
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Input arrays must be 2D")
    if x.shape[1] != y.shape[1]:
        raise ValueError("Points must have same dimension")

    x_sq = np.sum(x**2, axis=1)[:, None]
    y_sq = np.sum(y**2, axis=1)[None, :]
    cross = x @ y.T
    return x_sq + y_sq - 2 * cross


class RBFInterpolator:
    """
    Interpolator using various radial basis functions, with support for
    anisotropic length-scales and Tikhonov regularization.

    Attributes:
        centers: training points (n, d).
        weights: solution coefficients (n,).
        epsilon: per-dimension or scalar length-scale(s).
        function: type of RBF.
        reg: regularization parameter.
    """

    def __init__(
        self,
        centers: np.ndarray,
        values: np.ndarray,
        epsilon: Union[float, np.ndarray],
        function: Optional[str] = None,
        reg: float = 0.0,
    ) -> None:
        """
        Fit the RBF interpolator to data.

        Args:
            centers: Array (n, d) of input points.
            values: Array (n,) of target values.
            epsilon: scalar or (d,) array of length-scales.
            function: RBF name or None for Gaussian.
            reg: non-negative Tikhonov regularization parameter.
        """
        # validate shapes
        if centers.ndim != 2 or values.ndim != 1:
            raise ValueError("centers must be 2D and values 1D arrays")
        if centers.shape[0] != values.shape[0]:
            raise ValueError("centers and values length mismatch")
        if reg < 0:
            raise ValueError("reg must be non-negative")

        self.centers = centers
        self.values = values
        self.function = function or RBFFunction.GAUSSIAN.value
        self.reg = reg

        # process epsilon into array of shape (d,)
        eps_arr = np.atleast_1d(epsilon).astype(float)
        d = centers.shape[1]
        if eps_arr.ndim == 1 and eps_arr.size == d:
            self.epsilon = eps_arr
        elif eps_arr.ndim == 0:
            self.epsilon = np.full((d,), eps_arr.item())
        else:
            raise ValueError("epsilon must be a float or an array of length d")

        # build kernel matrix, add regularization, solve weights
        K = self._compute_kernel(centers, centers)
        K_reg = K + self.reg * np.eye(K.shape[0])
        self.weights = np.linalg.solve(K_reg, values)

    def _compute_kernel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the RBF kernel matrix between x and y, applying
        anisotropic scaling and dispatching the chosen RBF function.

        Args:
            x: Array (m, d).
            y: Array (n, d).
        Returns:
            Kernel matrix of shape (m, n).
        """
        d = x.shape[1]
        # pairwise difference scaled by epsilon
        diff = x[:, None, :] - y[None, :, :]  # (m, n, d)
        scaled = diff * self.epsilon[None, None, :]  # apply per-dim scale
        d2 = np.sum(scaled**2, axis=2)  # (m, n)

        # dispatch on RBF type
        if self.function == RBFFunction.GAUSSIAN.value:
            return np.exp(-d2)
        if self.function == RBFFunction.MULTIQUADRIC.value:
            return np.sqrt(d2 + 1.0 / (np.prod(self.epsilon) ** 2))
        if self.function == RBFFunction.INVERSE_MULTIQ.value:
            return 1.0 / np.sqrt(d2 + 1.0 / (np.prod(self.epsilon) ** 2))
        if self.function == RBFFunction.THIN_PLATE.value:
            r = np.sqrt(d2)
            with np.errstate(divide="ignore", invalid="ignore"):
                phi = r**2 * np.log(r)
            phi[np.isnan(phi)] = 0.0
            return phi
        raise ValueError(f"Unsupported RBF function '{self.function}'")

    def __call__(self, x_new: np.ndarray) -> np.ndarray:
        """
        Predict at new input locations.

        Args:
            x_new: Array (m, d) points to predict.
        Returns:
            Array (m,) interpolated values.
        """
        K_new = self._compute_kernel(x_new, self.centers)
        return K_new.dot(self.weights)

    @staticmethod
    def select_epsilon(
        centers: np.ndarray,
        values: np.ndarray,
        eps_candidates: np.ndarray,
        function: Optional[str] = None,
    ) -> float:
        """
        Choose optimal epsilon via leave-one-out cross-validation.

        Args:
            centers: (n, d) input points.
            values: (n,) targets.
            eps_candidates: 1D array of epsilons.
            function: RBF name or None.
        Returns:
            epsilon with lowest CV error.
        """
        n = centers.shape[0]
        best_eps = eps_candidates[0]
        best_err = np.inf
        for eps in eps_candidates:
            errors: list[float] = []
            for i in range(n):
                idx = np.arange(n) != i
                interp = RBFInterpolator(centers[idx], values[idx], eps, function)
                pred = interp(centers[i : i + 1])[0]
                errors.append((pred - values[i]) ** 2)
            mse = np.mean(errors)
            if mse < best_err:
                best_err, best_eps = float(mse), float(eps)
        return best_eps
