import numpy as np
import pandas as pd
import torch

import qfeval_functions.functions as QF


def test_msum_1d() -> None:
    x = QF.randn(100)
    span = 5
    df = pd.Series(x.numpy())
    expected = df.rolling(span).sum().to_numpy()
    actual = QF.msum(x, span=span)
    np.testing.assert_allclose(actual.numpy(), expected, equal_nan=True, atol=1e-6, rtol=1e-6)


def test_msum_dim1() -> None:
    x = QF.randn(10, 20)
    span = 4
    df = pd.DataFrame(x.numpy())
    expected = df.rolling(span, axis=1).sum().to_numpy()
    actual = QF.msum(x, span=span, dim=1)
    np.testing.assert_allclose(actual.numpy(), expected, equal_nan=True, atol=1e-6, rtol=1e-6)
