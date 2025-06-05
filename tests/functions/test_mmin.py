import numpy as np
import pandas as pd
import torch

import qfeval_functions.functions as QF


def test_mmin() -> None:
    # Simple case.
    x = torch.tensor([2.0, 3.0, 1.0, 5.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(
        QF.mmin(x, 3).numpy(),
        np.array([2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0]),
    )
    # If all values are the same, their max should also be the same.
    np.testing.assert_allclose(
        QF.mmin(torch.full((100,), 42.0), 10).numpy(),
        torch.full((100,), 42.0),
    )


def test_mmin_with_random_values() -> None:
    a = QF.randn(100, 10)
    df = pd.DataFrame(a.numpy())
    np.testing.assert_allclose(
        QF.mmin(a, 10, dim=0).numpy()[10:],
        df.rolling(10).min().to_numpy()[10:],
        1e-6,
        1e-6,
    )
