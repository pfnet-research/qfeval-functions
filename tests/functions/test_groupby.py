import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_groupby() -> None:
    idx = torch.tensor([0, 1, 3, 2, 1, 0, 3, 1, 2])
    a = torch.arange(idx.shape[0], dtype=float) * 10  # type: ignore
    np.testing.assert_array_almost_equal(
        QF.groupby(a, idx).numpy(),
        np.array(
            [
                [0, 50, math.nan],
                [10, 40, 70],
                [30, 80, math.nan],
                [20, 60, math.nan],
            ]
        ),
    )


def test_multidimension() -> None:
    idx = torch.tensor([0, 1, 3, 2, 1, 0, 3, 1, 2])
    a = torch.arange(idx.shape[0] * 2, dtype=float) * 10  # type: ignore
    np.testing.assert_array_almost_equal(
        QF.groupby(a.reshape(2, -1), idx).numpy(),
        np.array(
            [
                [
                    [0, 50, math.nan],
                    [10, 40, 70],
                    [30, 80, math.nan],
                    [20, 60, math.nan],
                ],
                [
                    [90, 140, math.nan],
                    [100, 130, 160],
                    [120, 170, math.nan],
                    [110, 150, math.nan],
                ],
            ]
        ),
    )
