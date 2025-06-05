import numpy as np
from scipy.stats import linregress

import qfeval_functions.functions as QF


def test_slope_with_scipy() -> None:
    a = QF.randn(100, 200)
    b = QF.randn(100, 200)
    actual = QF.slope(a, b, dim=1)
    expected = np.zeros((100,))
    for i in range(a.shape[0]):
        expected[i] = linregress(a[i].numpy(), b[i].numpy()).slope
    np.testing.assert_allclose(actual, expected, 1e-6, 1e-6)


def test_slope_when_aggregating_multiple_dims() -> None:
    a = QF.randn(10, 20, 30)
    b = QF.randn(10, 20, 30)
    actual = QF.slope(a, b, dim=(0, 1))
    expected = QF.slope(a.reshape(-1, 30), b.reshape(-1, 30), dim=0)
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), 1e-6, 1e-6)


def test_slope_with_broadcasting() -> None:
    a = QF.randn(2, 3, 1, 5, 1, 7)
    b = QF.randn(2, 1, 4, 5, 6, 1)
    actual = QF.slope(a, b, dim=(0, 1, 2))
    expected = QF.slope(
        a.expand(2, 3, 4, 5, 6, 7),
        b.expand(2, 3, 4, 5, 6, 7),
        dim=(0, 1, 2),
    )
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), 1e-6, 1e-6)
