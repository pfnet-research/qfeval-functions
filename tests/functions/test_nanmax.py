import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_nanmax() -> None:
    x = torch.tensor(
        [
            [0.0, -1.0, 1.0, math.nan],
            [math.nan, math.nan, math.nan, math.nan],
            [0.0, -1.0, 2.0, -2.0],
        ]
    )
    np.testing.assert_allclose(
        QF.nanmax(x, dim=1).values.numpy(),
        np.array([1.0, math.nan, 2.0]),
    )
    np.testing.assert_allclose(
        QF.nanmax(x, dim=1).indices.numpy(),
        np.array([2, 0, 2]),
    )


def test_nanmax_with_nan_and_neginf() -> None:
    x = torch.tensor(
        [
            [-math.inf, -1.0, 1.0, math.nan],
            [math.inf, -1.0, 1.0, math.nan],
            [math.nan, math.nan, math.nan, math.nan],
            [-math.inf, -math.inf, -math.inf, -math.inf],
            [math.nan, -math.inf, math.nan, math.nan],
            [math.nan, math.nan, math.inf, math.nan],
            [0.0, -1.0, 2.0, -2.0],
        ]
    )
    np.testing.assert_allclose(
        QF.nanmax(x, dim=1).values.numpy(),
        np.array(
            [1.0, math.inf, math.nan, -math.inf, -math.inf, math.inf, 2.0]
        ),
    )
    np.testing.assert_allclose(
        QF.nanmax(x, dim=1).indices.numpy(),
        np.array([2, 0, 0, 0, 1, 2, 2]),
    )
