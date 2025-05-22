import numpy as np
from rbf_package.rbf import gaussian_rbf, RBFInterpolator

def test_gaussian_rbf_shape():
    x = np.zeros((5, 2))
    centers = np.ones((3, 2))
    out = gaussian_rbf(x, centers, epsilon=1.0)
    assert out.shape == (5, 3)

def test_interpolator_exact():
    # simple 1D check: f(x)=x, three points
    xs = np.array([[0.0], [0.5], [1.0]])
    ys = np.array([0.0, 0.5, 1.0])
    interp = RBFInterpolator(xs, ys, epsilon=2.0)
    x_test = np.array([[0.25], [0.75]])
    y_pred = interp(x_test)
    # since data is linear, interpolation should be close to true
    assert np.allclose(y_pred, [0.25, 0.75], atol=1e-2)
