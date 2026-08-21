import math

import numpy as np
import pandas as pd
import pytest
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties

# Tolerances per alpha for comparisons against pandas.  The shared
# O(log n) exponential-sum kernel of QF.ema truncates weights smaller
# than about 1e-8, so the achievable precision degrades as alpha grows
# (pandas accumulates all weights exactly).
VAR_TOLS = {0.05: (1e-10, 1e-13), 0.3: (1e-6, 1e-9), 0.7: (1e-4, 1e-7)}


def test_ewmvar_pandas_comparison_1d() -> None:
    """Compare 1D results with pandas ewm().var() for both bias modes."""
    torch.manual_seed(0)
    x = torch.randn(200, dtype=torch.float64)
    s = pd.Series(x.numpy())
    for alpha, (rtol, atol) in VAR_TOLS.items():
        for bias in [False, True]:
            result = QF.ewmvar(x, alpha, dim=0, bias=bias)
            expected = s.ewm(alpha=alpha, adjust=True).var(bias=bias)
            np.testing.assert_allclose(
                result.numpy(),
                expected.to_numpy(),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )


def test_ewmvar_pandas_comparison_2d() -> None:
    """Compare 2D results with pandas DataFrame.ewm().var()."""
    torch.manual_seed(1)
    x = torch.randn(50, 4, dtype=torch.float64)
    df = pd.DataFrame(x.numpy())
    for alpha, (rtol, atol) in VAR_TOLS.items():
        for bias in [False, True]:
            result = QF.ewmvar(x, alpha, dim=0, bias=bias)
            expected = df.ewm(alpha=alpha, adjust=True).var(bias=bias)
            np.testing.assert_allclose(
                result.numpy(),
                expected.to_numpy(),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )


def test_ewmvar_first_element() -> None:
    """The first element must be NaN (bias=False) or 0 (bias=True)."""
    x = torch.tensor([3.0, 4.0, 5.0])
    assert math.isnan(QF.ewmvar(x, 0.5, dim=0)[0].item())
    assert QF.ewmvar(x, 0.5, dim=0, bias=True)[0].item() == 0.0
    # A single observation behaves like the first element.
    single = torch.tensor([5.0])
    assert math.isnan(QF.ewmvar(single, 0.5, dim=0)[0].item())
    assert QF.ewmvar(single, 0.5, dim=0, bias=True)[0].item() == 0.0


def test_ewmvar_constant_input() -> None:
    """A constant series has zero variance, matching pandas exactly."""
    x = torch.full((10,), 3.0, dtype=torch.float64)
    s = pd.Series(x.numpy())
    for bias in [False, True]:
        result = QF.ewmvar(x, 0.3, dim=0, bias=bias)
        expected = s.ewm(alpha=0.3, adjust=True).var(bias=bias)
        np.testing.assert_allclose(
            result.numpy(), expected.to_numpy(), equal_nan=True
        )
        # From the second element on, the variance is exactly zero.
        assert (result[1:] == 0).all()
    # bias=True defines the variance of a single observation as zero.
    assert QF.ewmvar(x, 0.3, dim=0, bias=True)[0].item() == 0.0


def test_ewmvar_offset_stability_float32() -> None:
    """Centering keeps float32 results accurate for offset data."""
    torch.manual_seed(1)
    x32 = (torch.randn(500, dtype=torch.float64) + 1e6).to(torch.float32)
    for bias in [False, True]:
        v32 = QF.ewmvar(x32, 0.3, dim=0, bias=bias)
        v64 = QF.ewmvar(x32.double(), 0.3, dim=0, bias=bias)
        np.testing.assert_allclose(
            v32.numpy().astype(np.float64)[5:],
            v64.numpy()[5:],
            rtol=1e-4,
        )


def test_ewmvar_nan_contamination() -> None:
    """A NaN contaminates all subsequent outputs, unlike pandas."""
    torch.manual_seed(2)
    x = torch.randn(60, dtype=torch.float64)
    x[20] = math.nan
    result = QF.ewmvar(x, 0.3, dim=0).numpy()
    expected = pd.Series(x.numpy()).ewm(alpha=0.3).var().to_numpy()
    # All outputs from the NaN position on are NaN.
    assert np.isnan(result[20:]).all()
    # Outputs before the NaN position match pandas.
    np.testing.assert_allclose(
        result[1:20], expected[1:20], rtol=1e-6, atol=1e-9
    )
    # Differential: pandas skips NaNs and keeps producing values.
    assert not np.isnan(expected[20:]).all()


def test_ewmvar_multi_dimensional() -> None:
    """Results along any dimension match the equivalent 1D computation."""
    torch.manual_seed(3)
    x = torch.randn(4, 5, 6, dtype=torch.float64)
    for dim in range(3):
        result = QF.ewmvar(x, 0.3, dim=dim)
        assert result.shape == x.shape
        index: list = [0] * 3
        index[dim] = slice(None)
        slice_result = QF.ewmvar(x[tuple(index)], 0.3, dim=0)
        np.testing.assert_allclose(
            result[tuple(index)].numpy(),
            slice_result.numpy(),
            equal_nan=True,
        )
    # Negative dimension indexing.
    np.testing.assert_allclose(
        QF.ewmvar(x, 0.3, dim=-1).numpy(),
        QF.ewmvar(x, 0.3, dim=2).numpy(),
        equal_nan=True,
    )


def test_ewmvar_dtype_preservation() -> None:
    """The result preserves shape, dtype and device of the input."""
    torch.manual_seed(4)
    for dtype in [torch.float32, torch.float64]:
        x = torch.randn(3, 20, dtype=dtype)
        result = QF.ewmvar(x, 0.5, dim=1)
        assert_basic_properties(result, x)


def test_ewmvar_alpha_extremes() -> None:
    """Sanity checks for alpha values close to 0 and 1."""
    torch.manual_seed(5)
    x = torch.randn(100, dtype=torch.float64)
    s = pd.Series(x.numpy())
    for alpha in [0.01, 0.99]:
        for bias in [False, True]:
            result = QF.ewmvar(x, alpha, dim=0, bias=bias)
            # Finite and non-negative everywhere except the leading NaN.
            valid = result[~result.isnan()]
            assert valid.isfinite().all()
            assert (valid >= 0).all()
            expected = s.ewm(alpha=alpha, adjust=True).var(bias=bias)
            np.testing.assert_allclose(
                result.numpy(),
                expected.to_numpy(),
                rtol=1e-4,
                atol=1e-4,
                equal_nan=True,
            )


def test_ewmvar_alpha_one_matches_pandas() -> None:
    """Alpha 1 is a valid boundary value and matches pandas."""
    x = torch.tensor([1.0, 2.0, 3.0])
    expected = pd.Series(x.numpy()).ewm(alpha=1.0)
    for bias in (False, True):
        result = QF.ewmvar(x, 1.0, bias=bias)
        np.testing.assert_allclose(
            result.numpy(),
            expected.var(bias=bias).to_numpy(),
            equal_nan=True,
        )


@pytest.mark.parametrize(
    "alpha", [-math.inf, -1.0, 0.0, 1.01, math.inf, math.nan]
)
def test_ewmvar_invalid_alpha_raises_value_error(alpha: float) -> None:
    """Alpha must satisfy the same ``0 < alpha <= 1`` bound as pandas."""
    with pytest.raises(ValueError, match="alpha must satisfy"):
        QF.ewmvar(torch.ones(3), alpha)


def test_ewmvar_empty_tensor() -> None:
    """An empty input produces an empty output."""
    x = torch.zeros(0)
    result = QF.ewmvar(x, 0.5, dim=0)
    assert result.shape == x.shape
