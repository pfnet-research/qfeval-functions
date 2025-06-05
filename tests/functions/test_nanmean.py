import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_nanmean() -> None:
    x = torch.tensor(
        [
            [0.0, -1.0, 1.5, math.nan],
            [math.nan, math.nan, math.nan, math.nan],
            [0.0, -1.0, 2.0, -2.0],
        ]
    )
    np.testing.assert_allclose(
        QF.nanmean(x, dim=1).numpy(),
        np.array([0.5 / 3, math.nan, -0.25]),
    )
