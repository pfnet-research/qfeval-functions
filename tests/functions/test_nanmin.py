import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_nanmin() -> None:
    x = torch.tensor(
        [
            [0.0, -1.0, 1.0, math.nan],
            [math.nan, math.nan, math.nan, math.nan],
            [0.0, -1.0, 2.0, -2.0],
        ]
    )
    np.testing.assert_allclose(
        QF.nanmin(x, dim=1).values.numpy(),
        np.array([-1.0, math.nan, -2.0]),
    )
    np.testing.assert_allclose(
        QF.nanmin(x, dim=1).indices.numpy(),
        np.array([1, 0, 3]),
    )


def test_nanmin_with_nan_and_neginf() -> None:
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
        QF.nanmin(x, dim=1).values.numpy(),
        np.array([-math.inf, -1, math.nan, -math.inf, -math.inf, math.inf, -2]),
    )
    np.testing.assert_allclose(
        QF.nanmin(x, dim=1).indices.numpy(),
        np.array([0, 1, 0, 0, 1, 2, 3]),
    )
