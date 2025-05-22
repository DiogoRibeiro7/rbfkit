from __future__ import annotations
from typing import Literal, Optional, Union, Sequence, Tuple, Callable
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin


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


class RBFInterpolator(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible interpolator using various radial basis functions,
    with support for anisotropic length-scales, Tikhonov regularization,
    automatic epsilon selection, derivative evaluation,
    and fast approximate methods.

    Follows the estimator API so it can be used in pipelines and grid search.
    """
    def __init__(
        self,
        epsilon: Union[float, np.ndarray] = 1.0,
        function: Optional[str] = None,
        reg: float = 0.0,
        method: Literal['exact', 'nystrom', 'rff'] = 'exact',
        n_features: int = 100
    ) -> None:
        """
        Initialize the RBF interpolator.

        Args:
            epsilon: scalar or (d,) array of length-scales.
            function: RBF name or None for Gaussian.
            reg: non-negative Tikhonov regularization parameter.
            method: 'exact' for full kernel, 'nystrom' for Nyström approx.,
                    'rff' for random Fourier features.
            n_features: number of landmarks or random features.
        """
        self.epsilon = epsilon
        self.function_ = function or RBFFunction.GAUSSIAN.value
        self.reg = reg
        self.method = method
        self.n_features = n_features
        self.random_weights_: Optional[np.ndarray] = None
        self.random_offset_: Optional[np.ndarray] = None
        self.landmarks_: Optional[np.ndarray] = None
        self.weights_: Optional[np.ndarray] = None
        self.centers_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> RBFInterpolator:
        X = np.asarray(X)
        y = np.asarray(y)
        self.centers_ = X
        self.values_ = y

        eps_arr = np.atleast_1d(self.epsilon).astype(float)
        d = X.shape[1]
        if eps_arr.ndim == 1 and eps_arr.size == d:
            self.epsilon_ = eps_arr
        elif eps_arr.ndim == 0:
            self.epsilon_ = np.full((d,), eps_arr.item())
        else:
            raise ValueError("epsilon must be a float or an array of length d")

        if self.method == 'exact':
            K = self._compute_kernel(self.centers_, self.centers_)
            K += self.reg * np.eye(K.shape[0])
            self.weights_ = np.linalg.solve(K, self.values_)
        elif self.method == 'nystrom':
            self._init_nystrom()
            Z = self._compute_features(self.centers_)
            G = Z @ Z.T + self.reg * np.eye(Z.shape[0])
            alpha = np.linalg.solve(G, self.values_)
            self.weights_ = Z.T @ alpha
        elif self.method == 'rff':
            self._init_rff()
            Z = self._compute_features(self.centers_)
            A = Z.T @ Z + self.reg * np.eye(Z.shape[1])
            B = Z.T @ self.values_
            self.weights_ = np.linalg.solve(A, B)
        else:
            raise ValueError(f"Unknown method '{self.method}'")
        return self

    def _init_nystrom(self) -> None:
        idx = np.random.choice(self.centers_.shape[0], self.n_features, replace=False)
        self.landmarks_ = self.centers_[idx]

    def _init_rff(self) -> None:
        d = self.centers_.shape[1]
        variance = 2.0 * np.mean(self.epsilon_**2)
        self.random_weights_ = np.random.normal(0, np.sqrt(variance), (d, self.n_features))
        self.random_offset_ = np.random.uniform(0, 2*np.pi, self.n_features)

    def _compute_features(self, X: np.ndarray) -> np.ndarray:
        if self.method == 'nystrom':
            K_nm = self._compute_kernel(X, self.landmarks_)
            W = self._compute_kernel(self.landmarks_, self.landmarks_)
            U, S, _ = np.linalg.svd(W)
            S_inv_sqrt = np.diag(1.0/np.sqrt(S))
            return K_nm @ U @ S_inv_sqrt
        if self.method == 'rff':
            projection = X @ self.random_weights_ + self.random_offset_
            return np.sqrt(2.0/self.n_features) * np.cos(projection)
        raise ValueError(f"Features not available for method '{self.method}'")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if self.method == 'exact':
            K_new = self._compute_kernel(X, self.centers_)
            return K_new.dot(self.weights_)
        Z = self._compute_features(X)
        return Z @ self.weights_

    def _compute_kernel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
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

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        u = ((y - y_pred)**2).sum()
        v = ((y - np.mean(y))**2).sum()
        return 1 - u/v

    def gradient(self, X: np.ndarray) -> np.ndarray:
        if self.function_ != RBFFunction.GAUSSIAN.value:
            raise NotImplementedError
        diff = X[:, None, :] - self.centers_[None, :, :]
        scaled = diff * self.epsilon_[None, None, :]
        phi = np.exp(-np.sum(scaled**2, axis=2))
        grad_basis = -2*(scaled*self.epsilon_[None, None, :])*phi[:,:,None]
        return np.tensordot(grad_basis, self.weights_, axes=([1],[0]))

    def hessian(self, X: np.ndarray) -> np.ndarray:
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
        errors = []
        for eps in candidates:
            if method == 'loo':
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
        if not hasattr(self, 'eps_candidates_') or not hasattr(self, 'cv_errors_'):
            raise RuntimeError("No CV data. Run select_epsilon first.")
        plt.figure()
        plt.plot(self.eps_candidates_, self.cv_errors_, marker='o')
        plt.xlabel('epsilon')
        plt.ylabel('CV error')
        plt.title('Epsilon selection curve')
        plt.xscale('log')
        plt.show()

    def partition_of_unity(self,
        X: np.ndarray,
        values: np.ndarray,
        patches: Sequence[np.ndarray],
        overlap: float = 0.1
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Create a partition-of-unity interpolator by fitting local RBFs on overlapping patches.
        """
        local_models: list[Tuple[np.ndarray, RBFInterpolator]] = []
        for idx in patches:
            interp = RBFInterpolator(
                epsilon=self.epsilon,
                function=self.function_,
                reg=self.reg
            ).fit(X[idx], values[idx])
            local_models.append((idx, interp))

        def interpolator(x_new: np.ndarray) -> np.ndarray:
            m, d = x_new.shape
            blend_weights = np.zeros((m, len(local_models)))
            preds = np.zeros((m, len(local_models)))
            for i, (idx, model) in enumerate(local_models):
                centers = X[idx]
                centroid = np.mean(centers, axis=0)
                dist = np.linalg.norm(x_new - centroid, axis=1)
                scale = np.mean(np.std(centers, axis=0))
                w = np.exp(- (dist/(overlap * scale + 1e-8))**2)
                blend_weights[:, i] = w
                preds[:, i] = model.predict(x_new)
            W = blend_weights / np.sum(blend_weights, axis=1, keepdims=True)
            return np.sum(preds * W, axis=1)

        return interpolator

