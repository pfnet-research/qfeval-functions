import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_nancovar() -> None:
    a = QF.randn(100, 200)
    a = torch.where(a < -1, torch.as_tensor(math.nan), a)
    b = QF.randn(100, 200)
    b = torch.where(b < -1, torch.as_tensor(math.nan), b)
    actual = QF.nancovar(a, b)
    expected = np.zeros((100,))
    for i in range(a.shape[0]):
        xa, xb = a[i].numpy(), b[i].numpy()
        mask = ~np.isnan(xa + xb)
        expected[i] = np.cov(xa[mask], xb[mask])[0, 1]
    np.testing.assert_allclose(actual, expected, 1e-4, 1e-4)
