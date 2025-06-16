import numpy as np
import pandas as pd
import torch

import qfeval_functions.functions as QF


def test_mstd() -> None:
    x = QF.randn(100)
    span = 5
    df = pd.Series(x.numpy())
    expected = df.rolling(span).std().to_numpy()
    actual = QF.mstd(x, span=span)
    np.testing.assert_allclose(actual.numpy(), expected, equal_nan=True, atol=1e-6)


def test_mstd_dim1() -> None:
    x = QF.randn(20, 30)
    span = 7
    df = pd.DataFrame(x.numpy())
    expected = df.rolling(span, axis=1).std().to_numpy()
    actual = QF.mstd(x, span=span, dim=1)
    np.testing.assert_allclose(actual.numpy(), expected, equal_nan=True, atol=1e-6)
