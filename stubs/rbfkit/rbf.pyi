import numpy as np
from _typeshed import Incomplete
from enum import Enum
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Literal, Sequence

class RBFFunction(Enum):
    GAUSSIAN = 'gaussian'
    MULTIQUADRIC = 'multiquadric'
    INVERSE_MULTIQ = 'inverse_multiquadric'
    THIN_PLATE = 'thin_plate'

class RBFInterpolator(BaseEstimator, RegressorMixin):
    epsilon: Incomplete
    function_: Incomplete
    reg: Incomplete
    cv_errors_: np.ndarray | None
    eps_candidates_: np.ndarray | None
    def __init__(self, epsilon: float | np.ndarray = 1.0, function: str | None = None, reg: float = 0.0) -> None: ...
    centers_: Incomplete
    values_: Incomplete
    epsilon_: Incomplete
    weights_: Incomplete
    def fit(self, X: np.ndarray, y: np.ndarray) -> RBFInterpolator: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def score(self, X: np.ndarray, y: np.ndarray) -> float: ...
    def gradient(self, X: np.ndarray) -> np.ndarray: ...
    def hessian(self, X: np.ndarray) -> np.ndarray: ...
    def select_epsilon(self, candidates: Sequence[float], method: Literal['loo', 'grid'] = 'loo') -> float: ...
    def plot_epsilon_curve(self) -> None: ...
