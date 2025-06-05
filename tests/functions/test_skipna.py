import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_skipna() -> None:
    # Test if skipna enables to implement nancumsum.
    x = torch.tensor([math.nan, 1.0, math.nan, 2.0, 3.0, math.nan, 4.0])
    np.testing.assert_allclose(
        QF.skipna(lambda x: x.cumsum(dim=0), x, dim=0).numpy(),  # type: ignore
        np.array([math.nan, 1.0, math.nan, 3.0, 6.0, math.nan, 10.0]),
    )
