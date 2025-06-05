import numpy as np
from scipy.stats import linregress

import qfeval_functions.functions as QF


def test_correl() -> None:
    a = QF.randn(100, 200)
    b = QF.randn(100, 200)
    actual = QF.correl(a, b, dim=1)
    expected = np.zeros((100,))
    for i in range(a.shape[0]):
        expected[i] = linregress(a[i].numpy(), b[i].numpy()).rvalue
    np.testing.assert_allclose(actual, expected, 1e-6, 1e-6)
