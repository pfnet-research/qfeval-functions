import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_signif() -> None:
    x = torch.tensor(
        [0.123456789, 123456789, 1.23456789e-20], dtype=torch.float64
    )
    np.testing.assert_array_almost_equal(
        QF.signif(x).numpy(),
        [0.123457, 123457000, 1.23457e-20],
        decimal=10,
    )
    np.testing.assert_array_almost_equal(
        QF.signif(x.float(), 3).numpy(),
        [0.123, 123000000, 1.23e-20],
        decimal=8,
    )
    x = torch.tensor(
        [
            0.1234,
            -12.34,
            9.876,
            -9.999,
            1.234e-100,
            1.234e100,
            0.0,
            math.nan,
            math.inf,
            -math.inf,
        ],
        dtype=torch.float64,
    )
    np.testing.assert_array_almost_equal(
        QF.signif(x, 3).numpy(),
        [
            0.123,
            -12.3,
            9.88,
            -10.00,
            1.23e-100,
            1.23e100,
            0.0,
            math.nan,
            math.inf,
            -math.inf,
        ],
        decimal=5,
    )
