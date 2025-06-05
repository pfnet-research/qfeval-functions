import numpy as np
import pandas as pd
import torch

import qfeval_functions.functions as QF


def test_mmax() -> None:
    # Simple case.
    x = torch.tensor([2.0, 3.0, 1.0, 5.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(
        QF.mmax(x, 3).numpy(),
        np.array([2.0, 3.0, 3.0, 5.0, 5.0, 5.0, 4.0]),
    )
    # If all values are the same, their max should also be the same.
    np.testing.assert_allclose(
        QF.mmax(torch.full((100,), 42.0), 10).numpy(),
        torch.full((100,), 42.0),
    )


def test_mmax_with_random_values() -> None:
    a = QF.randn(100, 10)
    df = pd.DataFrame(a.numpy())
    np.testing.assert_allclose(
        QF.mmax(a, 10, dim=0).numpy()[10:],
        df.rolling(10).max().to_numpy()[10:],
        1e-6,
        1e-6,
    )
