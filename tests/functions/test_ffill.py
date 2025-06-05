import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_ffill() -> None:
    x = torch.tensor(
        [math.nan, 1.0, math.nan, 2.0, 3.0, math.nan, 4.0, math.nan]
    )
    np.testing.assert_allclose(
        QF.ffill(x, dim=0).numpy(),
        np.array([math.nan, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 4.0]),
    )
