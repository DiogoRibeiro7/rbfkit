from __future__ import annotations
from typing import Literal, Optional
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
    # runtime checks
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Input arrays must be 2D")
    if x.shape[1] != y.shape[1]:
        raise ValueError("Points must have same dimension")

    x_sq = np.sum(x**2, axis=1)[:, None]
    y_sq = np.sum(y**2, axis=1)[None, :]
    cross = x @ y.T
    return x_sq + y_sq - 2 * cross


def rbf_kernel(
    x: np.ndarray,
    centers: np.ndarray,
    epsilon: float,
    function: Literal[
        "gaussian", "multiquadric", "inverse_multiquadric", "thin_plate"
    ] = RBFFunction.GAUSSIAN.value,
) -> np.ndarray:
    """
    Evaluate an RBF kernel between points and centers.

    Args:
        x: Array (m, d) points to evaluate.
        centers: Array (n, d) RBF centers.
        epsilon: Positive shape parameter.
        function: One of the supported RBF names.
    Returns:
        (m, n) kernel matrix.
    """
    # type and value checks
    if not isinstance(x, np.ndarray) or not isinstance(centers, np.ndarray):
        raise TypeError("x and centers must be numpy arrays")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    d2 = _pairwise_sq_dists(x, centers)

    if function == RBFFunction.GAUSSIAN.value:
        return np.exp(-(epsilon**2) * d2)
    elif function == RBFFunction.MULTIQUADRIC.value:
        return np.sqrt(d2 + (1.0 / epsilon**2))
    elif function == RBFFunction.INVERSE_MULTIQ.value:
        return 1.0 / np.sqrt(d2 + (1.0 / epsilon**2))
    elif function == RBFFunction.THIN_PLATE.value:
        # phi(r) = r^2 * log(r), handle r=0
        r = np.sqrt(d2)
        # avoid log(0) by replacing zeros with 1 before log, then zero out
        with np.errstate(divide="ignore", invalid="ignore"):
            result = r**2 * np.log(r)
        result[np.isnan(result)] = 0.0
        return result
    else:
        raise ValueError(f"Unsupported RBF function '{function}'")


class RBFInterpolator:
    """
    Interpolator using various radial basis functions.

    Attributes:
        centers: training points (n, d).
        weights: solution coefficients (n,).
        epsilon: shape parameter.
        function: type of RBF.
    """

    def __init__(
        self,
        centers: np.ndarray,
        values: np.ndarray,
        epsilon: float,
        function: Optional[str] = None,
    ) -> None:
        """
        Fit the RBF interpolator to data.

        Args:
            centers: Array (n, d) input points.
            values: Array (n,) target values.
            epsilon: Positive shape parameter.
            function: RBF name or None for Gaussian.
        """
        # validate inputs
        if centers.ndim != 2 or values.ndim != 1:
            raise ValueError("centers must be 2D and values 1D arrays")
        if centers.shape[0] != values.shape[0]:
            raise ValueError("centers and values length mismatch")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.centers = centers
        self.epsilon = epsilon
        self.function: Literal[
            "gaussian", "multiquadric", "inverse_multiquadric", "thin_plate"
        ] = function or RBFFunction.GAUSSIAN.value  # type: ignore

        # build kernel matrix and solve for weights
        phi = rbf_kernel(centers, centers, epsilon, self.function)
        self.weights = np.linalg.solve(phi, values)

    def __call__(self, x_new: np.ndarray) -> np.ndarray:
        """
        Predict at new input locations.

        Args:
            x_new: Array (m, d) points to predict.
        Returns:
            Array (m,) interpolated values.
        """
        kernel = rbf_kernel(x_new, self.centers, self.epsilon, self.function)
        return kernel.dot(self.weights)

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
            errors = []
            for i in range(n):
                # leave out i-th sample
                idx = np.arange(n) != i
                interp = RBFInterpolator(centers[idx], values[idx], eps, function)
                pred = interp(centers[i : i + 1])[0]
                errors.append((pred - values[i]) ** 2)

            mse = np.mean(errors)
            if mse < best_err:
                best_err = mse
                best_eps = eps

        return best_eps
