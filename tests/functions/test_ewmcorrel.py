import math

import numpy as np
import pandas as pd
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties

# Tolerances per alpha for comparisons against pandas (see
# test_ewmvar.py for why the precision depends on alpha).
CORR_TOLS = {0.05: (1e-8, 1e-10), 0.3: (1e-6, 1e-7), 0.7: (1e-4, 1e-5)}


def test_ewmcorrel_pandas_comparison_1d() -> None:
    """Compare 1D results with pandas ewm().corr()."""
    torch.manual_seed(0)
    x = torch.randn(200, dtype=torch.float64)
    y = torch.randn(200, dtype=torch.float64)
    sx = pd.Series(x.numpy())
    sy = pd.Series(y.numpy())
    for alpha, (rtol, atol) in CORR_TOLS.items():
        result = QF.ewmcorrel(x, y, alpha, dim=0)
        expected = sx.ewm(alpha=alpha, adjust=True).corr(sy)
        np.testing.assert_allclose(
            result.numpy(),
            expected.to_numpy(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )


def test_ewmcorrel_pandas_comparison_2d() -> None:
    """Compare 2D results column-wise with pandas ewm().corr()."""
    torch.manual_seed(1)
    x = torch.randn(50, 4, dtype=torch.float64)
    y = torch.randn(50, 4, dtype=torch.float64)
    result = QF.ewmcorrel(x, y, 0.3, dim=0)
    for col in range(x.shape[1]):
        sx = pd.Series(x[:, col].numpy())
        sy = pd.Series(y[:, col].numpy())
        expected = sx.ewm(alpha=0.3, adjust=True).corr(sy)
        np.testing.assert_allclose(
            result[:, col].numpy(),
            expected.to_numpy(),
            rtol=1e-6,
            atol=1e-7,
            equal_nan=True,
        )


def test_ewmcorrel_linear_relationship() -> None:
    """Linear relationships give a correlation of exactly +1 or -1."""
    torch.manual_seed(2)
    x = torch.randn(100, dtype=torch.float64)
    result = QF.ewmcorrel(x, 2 * x + 1, 0.3, dim=0)
    assert math.isnan(result[0].item())
    np.testing.assert_allclose(
        result[1:].numpy(), np.ones(99), rtol=0, atol=1e-12
    )
    result = QF.ewmcorrel(x, -x, 0.3, dim=0)
    np.testing.assert_allclose(
        result[1:].numpy(), -np.ones(99), rtol=0, atol=1e-12
    )


def test_ewmcorrel_bounded() -> None:
    """The correlation never exceeds [-1, 1] beyond rounding error."""
    for seed in range(5):
        torch.manual_seed(seed)
        x = torch.randn(300, dtype=torch.float64)
        y = torch.randn(300, dtype=torch.float64)
        for alpha in [0.01, 0.3, 0.7, 0.99]:
            result = QF.ewmcorrel(x, y, alpha, dim=0)
            valid = result[~result.isnan()]
            assert (valid.abs() <= 1 + 1e-6).all()


def test_ewmcorrel_constant_input() -> None:
    """A constant input has zero variance, so the correlation is NaN."""
    torch.manual_seed(3)
    x = torch.randn(50, dtype=torch.float64)
    y = torch.full((50,), 2.0, dtype=torch.float64)
    assert QF.ewmcorrel(x, y, 0.3, dim=0).isnan().all()
    assert QF.ewmcorrel(y, x, 0.3, dim=0).isnan().all()
    # pandas agrees on all-NaN output for a constant input.
    expected = pd.Series(x.numpy()).ewm(alpha=0.3).corr(pd.Series(y.numpy()))
    assert expected.isna().all()


def test_ewmcorrel_first_element_nan() -> None:
    """The correlation of a single pair is undefined (0 / 0)."""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([2.0, 1.0, 3.0])
    assert math.isnan(QF.ewmcorrel(x, y, 0.5, dim=0)[0].item())


def test_ewmcorrel_offset_stability_float32() -> None:
    """Centering keeps float32 results accurate for offset data."""
    torch.manual_seed(4)
    base = torch.randn(500, dtype=torch.float64)
    noise = torch.randn(500, dtype=torch.float64)
    x32 = (base + 1e6).to(torch.float32)
    y32 = (0.6 * base + 0.8 * noise - 5e5).to(torch.float32)
    r32 = QF.ewmcorrel(x32, y32, 0.3, dim=0)
    r64 = QF.ewmcorrel(x32.double(), y32.double(), 0.3, dim=0)
    np.testing.assert_allclose(
        r32.numpy().astype(np.float64)[5:],
        r64.numpy()[5:],
        rtol=0,
        atol=1e-4,
    )
    # Even in float32, the bound only degrades by float32 rounding.
    valid = r32[~r32.isnan()]
    assert (valid.abs() <= 1 + 1e-4).all()


def test_ewmcorrel_broadcasting() -> None:
    """Inputs of shapes (N,) and (B, N) broadcast to (B, N)."""
    torch.manual_seed(5)
    x = torch.randn(100, dtype=torch.float64)
    y = torch.randn(3, 100, dtype=torch.float64)
    result = QF.ewmcorrel(x, y, 0.3, dim=-1)
    assert result.shape == (3, 100)
    for row in range(3):
        expected = QF.ewmcorrel(x, y[row], 0.3, dim=0)
        torch.testing.assert_close(result[row], expected, equal_nan=True)


def test_ewmcorrel_nan_contamination() -> None:
    """A NaN in either input contaminates all subsequent outputs."""
    torch.manual_seed(6)
    x = torch.randn(60, dtype=torch.float64)
    y = torch.randn(60, dtype=torch.float64)
    for nan_in_x, pos in [(True, 15), (False, 25)]:
        xn = x.clone()
        yn = y.clone()
        (xn if nan_in_x else yn)[pos] = math.nan
        result = QF.ewmcorrel(xn, yn, 0.3, dim=0).numpy()
        sx = pd.Series(xn.numpy())
        sy = pd.Series(yn.numpy())
        expected = sx.ewm(alpha=0.3).corr(sy).to_numpy()
        assert np.isnan(result[pos:]).all()
        np.testing.assert_allclose(
            result[1:pos], expected[1:pos], rtol=1e-6, atol=1e-7
        )
        # Differential: pandas skips NaNs and keeps producing values.
        assert not np.isnan(expected[pos:]).all()


def test_ewmcorrel_multi_dimensional() -> None:
    """Results along any dimension match the equivalent 1D computation."""
    torch.manual_seed(7)
    x = torch.randn(4, 5, 6, dtype=torch.float64)
    y = torch.randn(4, 5, 6, dtype=torch.float64)
    for dim in range(3):
        result = QF.ewmcorrel(x, y, 0.5, dim=dim)
        assert result.shape == x.shape
        index: list = [0] * 3
        index[dim] = slice(None)
        slice_result = QF.ewmcorrel(
            x[tuple(index)], y[tuple(index)], 0.5, dim=0
        )
        np.testing.assert_allclose(
            result[tuple(index)].numpy(),
            slice_result.numpy(),
            equal_nan=True,
        )
    np.testing.assert_allclose(
        QF.ewmcorrel(x, y, 0.5, dim=-1).numpy(),
        QF.ewmcorrel(x, y, 0.5, dim=2).numpy(),
        equal_nan=True,
    )


def test_ewmcorrel_dtype_preservation() -> None:
    """The result preserves shape, dtype and device of the input."""
    torch.manual_seed(8)
    for dtype in [torch.float32, torch.float64]:
        x = torch.randn(3, 20, dtype=dtype)
        y = torch.randn(3, 20, dtype=dtype)
        result = QF.ewmcorrel(x, y, 0.5, dim=1)
        assert_basic_properties(result, x)
