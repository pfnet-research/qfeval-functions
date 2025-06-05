import numpy as np
import pytest

import qfeval_functions.functions as QF


@pytest.mark.parametrize(
    "ddof",
    [0, 1],
)
def test_covar(ddof: int) -> None:
    a = QF.randn(10, 1, 200)
    b = QF.randn(1, 20, 200)
    actual = QF.covar(a, b, ddof=ddof)
    expected = np.zeros((10, 20))
    for i in range(10):
        for j in range(20):
            xa, xb = a[i, 0].numpy(), b[0, j].numpy()
            expected[i, j] = np.cov(xa, xb, ddof=ddof)[0, 1]
    np.testing.assert_allclose(actual, expected, 1e-4, 1e-4)
