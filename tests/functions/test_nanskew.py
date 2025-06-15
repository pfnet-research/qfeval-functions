import warnings
from math import nan

import numpy as np
import scipy.stats
import torch
from scipy.stats._axis_nan_policy import SmallSampleWarning

import qfeval_functions.functions as QF


def test_nanskew() -> None:
    x = torch.tensor(
        [
            [1.0, nan, 2.0, nan, 3.0, 4.0],
            [nan, 3.0, 7.0, nan, 8.0, nan],
            [9.0, 10.0, 11.0, 12.0, 13.0, 20.0],
            [nan, nan, nan, nan, nan, nan],
        ],
        dtype=torch.float64,
    )
    # Suppress SmallSampleWarning from scipy.stats.skew when testing with sparse data.
    # This warning occurs because some rows have very few non-NaN values after omission,
    # which is insufficient for reliable statistical computation. However, this is the
    # intended test behavior to verify our function handles edge cases consistently
    # with scipy's reference implementation.
    # TODO(claude): Consider restructuring test data to have sufficient sample sizes
    # in all rows, or create separate tests for small sample edge cases vs normal cases.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=SmallSampleWarning)
        expected = scipy.stats.skew(x, axis=1, bias=True, nan_policy="omit")

    np.testing.assert_allclose(
        QF.nanskew(x, dim=1, unbiased=False).numpy(),
        expected,
    )


def test_randn() -> None:
    x = torch.randn((73, 83), dtype=torch.float64)
    np.testing.assert_allclose(
        QF.nanskew(x, dim=1, unbiased=False).numpy(),
        scipy.stats.skew(x, axis=1, bias=True),
    )
    np.testing.assert_allclose(
        QF.nanskew(x, dim=1, unbiased=True).numpy(),
        scipy.stats.skew(x, axis=1, bias=False),
    )
