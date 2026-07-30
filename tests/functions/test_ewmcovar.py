import math

import numpy as np
import pandas as pd
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties

# Tolerances per alpha for comparisons against pandas (see
# test_ewmvar.py for why the precision depends on alpha).
COV_TOLS = {0.05: (1e-10, 1e-12), 0.3: (1e-6, 1e-8), 0.7: (1e-4, 1e-6)}


def test_ewmcovar_pandas_comparison_1d() -> None:
    """Compare 1D results with pandas ewm().cov() for both bias modes."""
    torch.manual_seed(0)
    x = torch.randn(200, dtype=torch.float64)
    y = torch.randn(200, dtype=torch.float64)
    sx = pd.Series(x.numpy())
    sy = pd.Series(y.numpy())
    for alpha, (rtol, atol) in COV_TOLS.items():
        for bias in [False, True]:
            result = QF.ewmcovar(x, y, alpha, dim=0, bias=bias)
            expected = sx.ewm(alpha=alpha, adjust=True).cov(sy, bias=bias)
            np.testing.assert_allclose(
                result.numpy(),
                expected.to_numpy(),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )


def test_ewmcovar_pandas_comparison_2d() -> None:
    """Compare 2D results column-wise with pandas ewm().cov()."""
    torch.manual_seed(1)
    x = torch.randn(50, 4, dtype=torch.float64)
    y = torch.randn(50, 4, dtype=torch.float64)
    for alpha, (rtol, atol) in COV_TOLS.items():
        result = QF.ewmcovar(x, y, alpha, dim=0)
        for col in range(x.shape[1]):
            sx = pd.Series(x[:, col].numpy())
            sy = pd.Series(y[:, col].numpy())
            expected = sx.ewm(alpha=alpha, adjust=True).cov(sy)
            np.testing.assert_allclose(
                result[:, col].numpy(),
                expected.to_numpy(),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )


def test_ewmcovar_with_itself_equals_ewmvar() -> None:
    """ewmcovar(x, x) coincides with ewmvar(x) in both bias modes."""
    torch.manual_seed(2)
    x = torch.randn(150, dtype=torch.float64)
    for bias in [False, True]:
        cov = QF.ewmcovar(x, x, 0.3, dim=0, bias=bias)
        var = QF.ewmvar(x, 0.3, dim=0, bias=bias)
        # ewmvar clamps tiny negative rounding results while ewmcovar
        # does not; the atol absorbs this difference.
        torch.testing.assert_close(
            cov, var, rtol=1e-12, atol=1e-12, equal_nan=True
        )


def test_ewmcovar_sign_flip() -> None:
    """Negating one input negates the covariance."""
    torch.manual_seed(3)
    x = torch.randn(100, dtype=torch.float64)
    cov = QF.ewmcovar(x, -x, 0.3, dim=0)
    var = QF.ewmvar(x, 0.3, dim=0)
    torch.testing.assert_close(
        cov, -var, rtol=1e-12, atol=1e-12, equal_nan=True
    )


def test_ewmcovar_broadcasting() -> None:
    """Inputs of shapes (N,) and (B, N) broadcast to (B, N)."""
    torch.manual_seed(4)
    x = torch.randn(200, dtype=torch.float64)
    y = torch.randn(3, 200, dtype=torch.float64)
    result = QF.ewmcovar(x, y, 0.3, dim=-1)
    assert result.shape == (3, 200)
    for row in range(3):
        expected = QF.ewmcovar(x, y[row], 0.3, dim=0)
        torch.testing.assert_close(result[row], expected, equal_nan=True)


def test_ewmcovar_offset_stability_float32() -> None:
    """Centering keeps float32 results accurate for offset data."""
    torch.manual_seed(5)
    base = torch.randn(500, dtype=torch.float64)
    noise = torch.randn(500, dtype=torch.float64)
    x32 = (base + 1e6).to(torch.float32)
    y32 = (0.6 * base + 0.8 * noise - 5e5).to(torch.float32)
    for bias in [False, True]:
        c32 = QF.ewmcovar(x32, y32, 0.3, dim=0, bias=bias)
        c64 = QF.ewmcovar(x32.double(), y32.double(), 0.3, dim=0, bias=bias)
        np.testing.assert_allclose(
            c32.numpy().astype(np.float64)[5:],
            c64.numpy()[5:],
            rtol=1e-4,
            atol=1e-5,
        )


def test_ewmcovar_nan_contamination() -> None:
    """A NaN in either input contaminates all subsequent outputs."""
    torch.manual_seed(6)
    x = torch.randn(60, dtype=torch.float64)
    y = torch.randn(60, dtype=torch.float64)
    for nan_in_x, pos in [(True, 15), (False, 25)]:
        xn = x.clone()
        yn = y.clone()
        (xn if nan_in_x else yn)[pos] = math.nan
        result = QF.ewmcovar(xn, yn, 0.3, dim=0).numpy()
        sx = pd.Series(xn.numpy())
        sy = pd.Series(yn.numpy())
        expected = sx.ewm(alpha=0.3).cov(sy).to_numpy()
        assert np.isnan(result[pos:]).all()
        np.testing.assert_allclose(
            result[1:pos], expected[1:pos], rtol=1e-6, atol=1e-8
        )
        # Differential: pandas skips NaNs and keeps producing values.
        assert not np.isnan(expected[pos:]).all()


def test_ewmcovar_multi_dimensional() -> None:
    """Results along any dimension match the equivalent 1D computation."""
    torch.manual_seed(7)
    x = torch.randn(4, 5, 6, dtype=torch.float64)
    y = torch.randn(4, 5, 6, dtype=torch.float64)
    for dim in range(3):
        result = QF.ewmcovar(x, y, 0.5, dim=dim)
        assert result.shape == x.shape
        index: list = [0] * 3
        index[dim] = slice(None)
        slice_result = QF.ewmcovar(x[tuple(index)], y[tuple(index)], 0.5, dim=0)
        np.testing.assert_allclose(
            result[tuple(index)].numpy(),
            slice_result.numpy(),
            equal_nan=True,
        )
    np.testing.assert_allclose(
        QF.ewmcovar(x, y, 0.5, dim=-1).numpy(),
        QF.ewmcovar(x, y, 0.5, dim=2).numpy(),
        equal_nan=True,
    )


def test_ewmcovar_first_element() -> None:
    """The first element must be NaN (bias=False) or 0 (bias=True)."""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([2.0, 1.0, 3.0])
    assert math.isnan(QF.ewmcovar(x, y, 0.5, dim=0)[0].item())
    assert QF.ewmcovar(x, y, 0.5, dim=0, bias=True)[0].item() == 0.0


def test_ewmcovar_dtype_preservation() -> None:
    """The result preserves shape, dtype and device of the input."""
    torch.manual_seed(8)
    for dtype in [torch.float32, torch.float64]:
        x = torch.randn(3, 20, dtype=dtype)
        y = torch.randn(3, 20, dtype=dtype)
        result = QF.ewmcovar(x, y, 0.5, dim=1)
        assert_basic_properties(result, x)
