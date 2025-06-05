import math

import numpy as np
import torch
from scipy.stats import linregress

import qfeval_functions.functions as QF


def test_nanslope() -> None:
    a = QF.randn(100, 200)
    a = torch.where(a < 0, torch.as_tensor(math.nan), a)
    b = QF.randn(100, 200)
    b = torch.where(b < 0, torch.as_tensor(math.nan), b)
    actual = QF.nanslope(a, b, dim=1)
    expected = np.zeros((100,))
    for i in range(a.shape[0]):
        xa, xb = a[i].numpy(), b[i].numpy()
        mask = ~np.isnan(xa + xb)
        expected[i] = linregress(xa[mask], xb[mask]).slope
    np.testing.assert_allclose(actual, expected, 1e-6, 1e-6)
