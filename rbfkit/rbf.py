from __future__ import annotations
from typing import Literal, Optional, Union, Sequence
import numpy as np
from enum import Enum
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt


class RBFFunction(Enum):
    """
    Supported radial basis functions.
    """
    GAUSSIAN = "gaussian"
    MULTIQUADRIC = "multiquadric"
    INVERSE_MULTIQ = "inverse_multiquadric"
    THIN_PLATE = "thin_plate"


# def _pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
#     """
#     Compute squared Euclidean distances between two sets of points.

#     Args:
#         x: Array of shape (m, d).
#         y: Array of shape (n, d).
#     Returns:
#         (m, n) array of squared distances.
#     """
#     if x.ndim != 2 or y.ndim != 2:
#         raise ValueError("Input arrays must be 2D")
#     if x.shape[1] != y.shape[1]:
#         raise ValueError("Points must have same dimension")

#     x_sq = np.sum(x**2, axis=1)[:, None]
#     y_sq = np.sum(y**2, axis=1)[None, :]
#     cross = x @ y.T
#     return x_sq + y_sq - 2 * cross


class RBFInterpolator(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible interpolator using various radial basis functions,
    with support for anisotropic length-scales, Tikhonov regularization,
    automatic epsilon selection, and derivative evaluation.

    Follows the estimator API so it can be used in pipelines and grid search.
    """
    def __init__(
        self,
        epsilon: Union[float, np.ndarray] = 1.0,
        function: Optional[str] = None,
        reg: float = 0.0
    ) -> None:
        """
        Initialize the RBF interpolator.

        Args:
            epsilon: scalar or (d,) array of length-scales.
            function: RBF name or None for Gaussian.
            reg: non-negative Tikhonov regularization parameter.
        """
        self.epsilon = epsilon
        self.function_ = function or RBFFunction.GAUSSIAN.value
        self.reg = reg
        self.cv_errors_: Optional[np.ndarray] = None
        self.eps_candidates_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> RBFInterpolator:
        """
        Fit the interpolator to data.

        Args:
            X: Array (n, d) of input points.
            y: Array (n,) of target values.
        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y 1D arrays")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y length mismatch")
        if self.reg < 0:
            raise ValueError("reg must be non-negative")

        self.centers_ = X
        self.values_ = y

        # process epsilon
        eps_arr = np.atleast_1d(self.epsilon).astype(float)
        d = X.shape[1]
        if eps_arr.ndim == 1 and eps_arr.size == d:
            self.epsilon_ = eps_arr
        elif eps_arr.ndim == 0:
            self.epsilon_ = np.full((d,), eps_arr.item())
        else:
            raise ValueError("epsilon must be a float or an array of length d")

        # build and solve
        K = self._compute_kernel(self.centers_, self.centers_)
        K_reg = K + self.reg * np.eye(K.shape[0])
        self.weights_ = np.linalg.solve(K_reg, self.values_)
        return self

    def _compute_kernel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the RBF kernel matrix between x and y with anisotropic scaling.
        """
        diff = x[:, None, :] - y[None, :, :]
        scaled = diff * self.epsilon_[None, None, :]
        d2 = np.sum(scaled**2, axis=2)

        if self.function_ == RBFFunction.GAUSSIAN.value:
            return np.exp(-d2)
        if self.function_ == RBFFunction.MULTIQUADRIC.value:
            return np.sqrt(d2 + 1.0/(np.prod(self.epsilon_)**2))
        if self.function_ == RBFFunction.INVERSE_MULTIQ.value:
            return 1.0/np.sqrt(d2 + 1.0/(np.prod(self.epsilon_)**2))
        if self.function_ == RBFFunction.THIN_PLATE.value:
            r = np.sqrt(d2)
            with np.errstate(divide='ignore', invalid='ignore'):
                phi = r**2 * np.log(r)
            phi[np.isnan(phi)] = 0.0
            return phi
        raise ValueError(f"Unsupported RBF function '{self.function_}'")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values at new points.
        """
        K_new = self._compute_kernel(np.asarray(X), self.centers_)
        return K_new.dot(self.weights_)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.
        """
        y_pred = self.predict(X)
        u = ((y - y_pred)**2).sum()
        v = ((y - np.mean(y))**2).sum()
        return 1 - u/v

    def gradient(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the gradient at new points (Gaussian only).
        """
        if self.function_ != RBFFunction.GAUSSIAN.value:
            raise NotImplementedError
        diff = X[:, None, :] - self.centers_[None, :, :]
        scaled = diff * self.epsilon_[None, None, :]
        phi = np.exp(-np.sum(scaled**2, axis=2))
        grad_basis = -2*(scaled*self.epsilon_[None, None, :])*phi[:,:,None]
        return np.tensordot(grad_basis, self.weights_, axes=([1],[0]))

    def hessian(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian at new points (Gaussian only).
        """
        if self.function_ != RBFFunction.GAUSSIAN.value:
            raise NotImplementedError
        m, d = X.shape
        diff = X[:, None, :] - self.centers_[None, :, :]
        scaled = diff * self.epsilon_[None, None, :]
        phi = np.exp(-np.sum(scaled**2, axis=2))
        H = np.zeros((m,d,d))
        for i in range(d):
            for j in range(d):
                term = 4*self.epsilon_[i]*self.epsilon_[j]*scaled[:,:,i]*scaled[:,:,j]
                if i==j:
                    term -= 2*(self.epsilon_[i]**2)
                H[:,i,j] = np.tensordot(term*phi, self.weights_, axes=([1],[0]))
        return H

    def select_epsilon(
        self,
        candidates: Sequence[float],
        method: Literal['loo', 'grid'] = 'loo'
    ) -> float:
        """
        Choose epsilon by cross-validation or grid evaluation.

        Args:
            candidates: list or array of epsilon values to try.
            method: 'loo' for leave-one-out, 'grid' for direct error scan.
        Returns:
            best epsilon value.
        """
        errors = []
        for eps in candidates:
            if method == 'loo':
                # leave-one-out CV
                errs = []
                n = self.centers_.shape[0]
                for i in range(n):
                    idx = np.arange(n) != i
                    interp = RBFInterpolator(
                        epsilon=eps,
                        function=self.function_,
                        reg=self.reg
                    ).fit(self.centers_[idx], self.values_[idx])
                    pred = interp.predict(self.centers_[i:i+1])[0]
                    errs.append((pred - self.values_[i])**2)
                errors.append(np.mean(errs))
            elif method == 'grid':
                # fit once and compute training error
                interp = RBFInterpolator(
                    epsilon=eps,
                    function=self.function_,
                    reg=self.reg
                ).fit(self.centers_, self.values_)
                y_pred = interp.predict(self.centers_)
                errors.append(np.mean((self.values_ - y_pred)**2))
            else:
                raise ValueError("Unknown method")
        self.eps_candidates_ = np.array(candidates)
        self.cv_errors_ = np.array(errors)
        return candidates[int(np.argmin(errors))]

    def plot_epsilon_curve(self) -> None:
        """
        Plot CV or grid errors vs epsilon candidates.

        Requires matplotlib.
        """
        if self.eps_candidates_ is None or self.cv_errors_ is None:
            raise RuntimeError("No CV data. Run select_epsilon first.")
        plt.figure()
        plt.plot(self.eps_candidates_, self.cv_errors_, marker='o')
        plt.xlabel('epsilon')
        plt.ylabel('CV error')
        plt.title('Epsilon selection curve')
        plt.xscale('log')
        plt.show()
