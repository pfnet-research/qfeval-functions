from math import nan

import numpy as np
import scipy
import torch

import qfeval_functions.functions as QF


def test_nankurtosis() -> None:
    x = torch.tensor(
        [
            [1.0, nan, 2.0, nan, 3.0, 4.0],
            [nan, 3.0, 7.0, nan, 8.0, nan],
            [9.0, 10.0, 11.0, 12.0, 13.0, 20.0],
            [nan, nan, nan, nan, nan, nan],
        ],
        dtype=torch.float64,
    )
    np.testing.assert_allclose(
        QF.nankurtosis(x, dim=1, unbiased=False).numpy(),
        scipy.stats.kurtosis(x, axis=1, bias=True, nan_policy="omit"),
    )


def test_nankurtosis_with_randn() -> None:
    x = QF.randn(73, 83, dtype=torch.float64)
    np.testing.assert_allclose(
        QF.nankurtosis(x, dim=1, unbiased=False).numpy(),
        scipy.stats.kurtosis(x, axis=1, bias=True),
    )
    np.testing.assert_allclose(
        QF.nankurtosis(x, dim=1, unbiased=True).numpy(),
        scipy.stats.kurtosis(x, axis=1, bias=False),
    )
