import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_shift() -> None:
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    np.testing.assert_allclose(
        QF.shift(x, 0, 0).numpy(),
        x.numpy(),
    )
    np.testing.assert_allclose(
        QF.shift(x, 1, 0).numpy(),
        np.array(
            [
                [math.nan, math.nan, math.nan],
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ),
    )
    np.testing.assert_allclose(
        QF.shift(x, (-1, 1), (0, 1)).numpy(),
        np.array(
            [
                [math.nan, 4.0, 5.0],
                [math.nan, 7.0, 8.0],
                [math.nan, math.nan, math.nan],
            ]
        ),
    )
    np.testing.assert_allclose(
        QF.shift(x, -3, 1).numpy(),
        x.numpy() * math.nan,
    )
    np.testing.assert_allclose(
        QF.shift(x, 3, 1).numpy(),
        x.numpy() * math.nan,
    )
    np.testing.assert_allclose(
        QF.shift(x, -100, 1).numpy(),
        x.numpy() * math.nan,
    )
    np.testing.assert_allclose(
        QF.shift(x, 100, 1).numpy(),
        x.numpy() * math.nan,
    )
