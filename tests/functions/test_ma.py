from math import nan

import numpy as np
import pandas as pd
import torch

import qfeval_functions.functions as QF


def test_ma() -> None:
    x = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0.0])
    np.testing.assert_allclose(
        QF.ma(x, 4).numpy(),
        np.array([nan, nan, nan, 0.25, 0.25, 0.25, 0.25, 0]),
    )


def test_ma_with_random_values() -> None:
    a = QF.randn(100, 10)
    df = pd.DataFrame(a.numpy())
    np.testing.assert_allclose(
        QF.ma(a, 10, dim=0).numpy(),
        df.rolling(10).mean().to_numpy(),
        1e-6,
        1e-6,
    )
