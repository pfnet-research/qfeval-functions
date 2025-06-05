import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_nancumsum() -> None:
    x = torch.tensor([math.nan, 1.0, math.nan, 2.0, 3.0, math.nan, 4.0])
    np.testing.assert_allclose(
        QF.nancumsum(x, dim=0).numpy(),
        np.array([math.nan, 1.0, math.nan, 3.0, 6.0, math.nan, 10.0]),
    )
