from math import nan

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_nanshift() -> None:
    x = torch.tensor(
        [
            [1.0, nan, 2.0, nan, 3.0, 4.0],
            [nan, 6.0, 7.0, nan, 8.0, nan],
            [9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            [nan, nan, nan, nan, nan, nan],
        ]
    )
    np.testing.assert_allclose(
        QF.nanshift(x, 0, 1).numpy(),
        x.numpy(),
    )
    np.testing.assert_allclose(
        QF.nanshift(x, 1, 1).numpy(),
        np.array(
            [
                [nan, nan, 1.0, nan, 2.0, 3.0],
                [nan, nan, 6.0, nan, 7.0, nan],
                [nan, 9.0, 10.0, 11.0, 12.0, 13.0],
                [nan, nan, nan, nan, nan, nan],
            ]
        ),
    )
    np.testing.assert_allclose(
        QF.nanshift(x, -1, 1).numpy(),
        np.array(
            [
                [2.0, nan, 3.0, nan, 4.0, nan],
                [nan, 7.0, 8.0, nan, nan, nan],
                [10.0, 11.0, 12.0, 13.0, 14.0, nan],
                [nan, nan, nan, nan, nan, nan],
            ]
        ),
    )
    np.testing.assert_allclose(
        QF.nanshift(x, -3, 1).numpy(),
        np.array(
            [
                [4.0, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan],
                [12.0, 13.0, 14.0, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan],
            ]
        ),
    )
    np.testing.assert_allclose(
        QF.nanshift(x, 4, 1).numpy(),
        np.array(
            [
                [nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, 9.0, 10.0],
                [nan, nan, nan, nan, nan, nan],
            ]
        ),
    )
    np.testing.assert_allclose(
        QF.nanshift(x, 10, 1).numpy(),
        np.array(
            [
                [nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan],
            ]
        ),
    )


def test_nanshift_randn() -> None:
    x = QF.randn(3, 5, 7)
    np.testing.assert_allclose(
        QF.shift(x, 3, 1).numpy(),
        QF.nanshift(x, 3, 1).numpy(),
    )
    np.testing.assert_allclose(
        QF.shift(x, -3, 2).numpy(),
        QF.nanshift(x, -3, 2).numpy(),
    )
    np.testing.assert_allclose(
        QF.shift(x, 30, 1).numpy(),
        QF.nanshift(x, 30, 1).numpy(),
    )
    np.testing.assert_allclose(
        QF.shift(x, -30, 2).numpy(),
        QF.nanshift(x, -30, 2).numpy(),
    )
