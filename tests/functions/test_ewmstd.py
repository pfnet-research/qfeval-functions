import math

import numpy as np
import pandas as pd
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties

# Tolerances per alpha for comparisons against pandas (see
# test_ewmvar.py for why the precision depends on alpha).
STD_TOLS = {0.05: (1e-10, 1e-13), 0.3: (1e-6, 1e-9), 0.7: (1e-4, 1e-7)}


def test_ewmstd_pandas_comparison_1d() -> None:
    """Compare 1D results with pandas ewm().std() for both bias modes."""
    torch.manual_seed(0)
    x = torch.randn(200, dtype=torch.float64)
    s = pd.Series(x.numpy())
    for alpha, (rtol, atol) in STD_TOLS.items():
        for bias in [False, True]:
            result = QF.ewmstd(x, alpha, dim=0, bias=bias)
            expected = s.ewm(alpha=alpha, adjust=True).std(bias=bias)
            np.testing.assert_allclose(
                result.numpy(),
                expected.to_numpy(),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )


def test_ewmstd_pandas_comparison_2d() -> None:
    """Compare 2D results with pandas DataFrame.ewm().std()."""
    torch.manual_seed(1)
    x = torch.randn(50, 4, dtype=torch.float64)
    df = pd.DataFrame(x.numpy())
    for alpha, (rtol, atol) in STD_TOLS.items():
        result = QF.ewmstd(x, alpha, dim=0)
        expected = df.ewm(alpha=alpha, adjust=True).std()
        np.testing.assert_allclose(
            result.numpy(),
            expected.to_numpy(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )


def test_ewmstd_squared_equals_ewmvar() -> None:
    """The squared standard deviation equals the variance."""
    torch.manual_seed(2)
    x = torch.randn(100, dtype=torch.float64)
    for bias in [False, True]:
        std = QF.ewmstd(x, 0.3, dim=0, bias=bias)
        var = QF.ewmvar(x, 0.3, dim=0, bias=bias)
        torch.testing.assert_close(
            std**2, var, rtol=1e-12, atol=1e-15, equal_nan=True
        )


def test_ewmstd_non_negative() -> None:
    """All defined values are non-negative."""
    torch.manual_seed(3)
    x = torch.randn(200)
    result = QF.ewmstd(x, 0.7, dim=0)
    valid = result[~result.isnan()]
    assert (valid >= 0).all()


def test_ewmstd_no_spurious_nan() -> None:
    """Tiny negative variances are clamped, so sqrt never yields NaN."""
    torch.manual_seed(4)
    # A nearly constant float32 series with a large offset stresses the
    # cancellation in the variance formula.
    x = 1e6 + torch.rand(100, dtype=torch.float32) * 0.01
    result = QF.ewmstd(x, 0.3, dim=0)
    assert not result[1:].isnan().any()


def test_ewmstd_constant_input() -> None:
    """A constant series has zero standard deviation."""
    x = torch.full((10,), 3.0, dtype=torch.float64)
    result = QF.ewmstd(x, 0.3, dim=0)
    assert math.isnan(result[0].item())
    assert (result[1:] == 0).all()
    assert (QF.ewmstd(x, 0.3, dim=0, bias=True) == 0).all()


def test_ewmstd_nan_contamination() -> None:
    """A NaN contaminates all subsequent outputs, unlike pandas."""
    torch.manual_seed(5)
    x = torch.randn(60, dtype=torch.float64)
    x[15] = math.nan
    result = QF.ewmstd(x, 0.3, dim=0).numpy()
    expected = pd.Series(x.numpy()).ewm(alpha=0.3).std().to_numpy()
    assert np.isnan(result[15:]).all()
    np.testing.assert_allclose(
        result[1:15], expected[1:15], rtol=1e-6, atol=1e-9
    )
    assert not np.isnan(expected[15:]).all()


def test_ewmstd_multi_dimensional() -> None:
    """Results along any dimension match the equivalent 1D computation."""
    torch.manual_seed(6)
    x = torch.randn(4, 5, 6, dtype=torch.float64)
    for dim in range(3):
        result = QF.ewmstd(x, 0.5, dim=dim)
        assert result.shape == x.shape
        index: list = [0] * 3
        index[dim] = slice(None)
        slice_result = QF.ewmstd(x[tuple(index)], 0.5, dim=0)
        np.testing.assert_allclose(
            result[tuple(index)].numpy(),
            slice_result.numpy(),
            equal_nan=True,
        )
    np.testing.assert_allclose(
        QF.ewmstd(x, 0.5, dim=-1).numpy(),
        QF.ewmstd(x, 0.5, dim=2).numpy(),
        equal_nan=True,
    )


def test_ewmstd_dtype_preservation() -> None:
    """The result preserves shape, dtype and device of the input."""
    torch.manual_seed(7)
    for dtype in [torch.float32, torch.float64]:
        x = torch.randn(3, 20, dtype=dtype)
        result = QF.ewmstd(x, 0.5, dim=1)
        assert_basic_properties(result, x)
