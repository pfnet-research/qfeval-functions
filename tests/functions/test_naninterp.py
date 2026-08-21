import math

import numpy as np
import pandas as pd
import pytest
import torch

import qfeval_functions.functions as QF
from tests.functions.test_utils import assert_basic_properties


def test_naninterp_matches_pandas_inside_interpolation() -> None:
    """Test naninterp against pandas interpolate(limit_area="inside")."""
    patterns = [
        [1.0, math.nan, 3.0],
        [1.0, math.nan, math.nan, 4.0],
        [math.nan, 1.0, math.nan, 2.0, math.nan],
        [math.nan, math.nan, math.nan],
        [1.0, 2.0, 3.0],
        [math.nan, 5.0, math.nan],
        [5.0],
        [math.nan],
        [1.0, math.nan, 2.0, math.nan, 3.0, math.nan],  # alternating
    ]
    for values in patterns:
        x = torch.tensor(values, dtype=torch.float64)
        expected = (
            pd.Series(values)
            .interpolate(method="linear", limit_area="inside")
            .to_numpy()
        )
        np.testing.assert_allclose(
            QF.naninterp(x).numpy(), expected, equal_nan=True
        )


def test_naninterp_matches_pandas_random() -> None:
    """Test naninterp against pandas on random data with random masks."""
    torch.manual_seed(0)
    for _ in range(20):
        n = int(torch.randint(1, 30, ()).item())
        x = torch.randn(n, dtype=torch.float64)
        x[torch.rand(n) < 0.4] = math.nan
        expected = (
            pd.Series(x.numpy())
            .interpolate(method="linear", limit_area="inside")
            .to_numpy()
        )
        np.testing.assert_allclose(
            QF.naninterp(x).numpy(), expected, equal_nan=True
        )


def test_naninterp_exact_values() -> None:
    """Test exact hand-computed interpolation values."""
    torch.testing.assert_close(
        QF.naninterp(torch.tensor([0.0, math.nan, math.nan, 3.0])),
        torch.tensor([0.0, 1.0, 2.0, 3.0]),
    )
    torch.testing.assert_close(
        QF.naninterp(torch.tensor([10.0, math.nan, 20.0])),
        torch.tensor([10.0, 15.0, 20.0]),
    )


def test_naninterp_leading_trailing_nans_stay_nan() -> None:
    """Test that boundary NaNs remain NaN, unlike pandas' default
    interpolate(), which forward-fills trailing NaNs."""
    values = [math.nan, 1.0, 2.0, math.nan, math.nan]
    x = torch.tensor(values, dtype=torch.float64)
    result = QF.naninterp(x)
    assert math.isnan(result[0].item())
    assert torch.isnan(result[3:]).all()
    np.testing.assert_allclose(result[1:3].numpy(), [1.0, 2.0])

    # pandas' default interpolate() fills the trailing NaNs instead.
    pandas_default = pd.Series(values).interpolate(method="linear")
    assert not pandas_default.iloc[3:].isna().any()


def test_naninterp_valid_values_untouched_and_nan_mask() -> None:
    """Test that valid values are bit-exact and only boundary NaNs stay."""
    torch.manual_seed(1)
    x = torch.randn(8, 30, dtype=torch.float64)
    x[torch.rand(x.shape) < 0.4] = math.nan
    result = QF.naninterp(x, dim=1)

    # Valid values must be returned unchanged (bit-exact).
    valid = ~x.isnan()
    assert torch.equal(result[valid], x[valid])

    # Only NaNs without a valid value on one side remain NaN.
    seen_before = torch.cummax(valid, dim=1).values
    seen_after = torch.cummax(valid.flip(1), dim=1).values.flip(1)
    expected_nan = x.isnan() & ~(seen_before & seen_after)
    assert torch.equal(result.isnan(), expected_nan)


def test_naninterp_multi_dim_and_negative_dim() -> None:
    """Test naninterp on 2D/3D tensors and with negative dimensions."""
    x = torch.tensor(
        [
            [0.0, math.nan, math.nan, 3.0],
            [4.0, math.nan, 8.0, math.nan],
        ]
    )
    torch.testing.assert_close(
        QF.naninterp(x, dim=1),
        torch.tensor([[0.0, 1.0, 2.0, 3.0], [4.0, 6.0, 8.0, math.nan]]),
        equal_nan=True,
    )

    # dim=0 interpolates along columns.
    torch.testing.assert_close(
        QF.naninterp(x.t(), dim=0),
        QF.naninterp(x, dim=1).t(),
        equal_nan=True,
    )

    # 3D tensors and negative dims.
    torch.manual_seed(2)
    x3 = torch.randn(3, 4, 5)
    x3[torch.rand(x3.shape) < 0.3] = math.nan
    for dim in range(3):
        assert QF.naninterp(x3, dim=dim).shape == x3.shape
    torch.testing.assert_close(
        QF.naninterp(x3, dim=-1), QF.naninterp(x3, dim=2), equal_nan=True
    )
    torch.testing.assert_close(
        QF.naninterp(x3, dim=-2), QF.naninterp(x3, dim=1), equal_nan=True
    )

    # 3D computation matches slice-wise 1D computation.
    torch.testing.assert_close(
        QF.naninterp(x3, dim=1)[0, :, 2],
        QF.naninterp(x3[0, :, 2]),
        equal_nan=True,
    )


def test_naninterp_dtype_preservation() -> None:
    """Test dtype/device/shape preservation for float32 and float64."""
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([1.0, math.nan, 3.0], dtype=dtype)
        result = QF.naninterp(x)
        assert_basic_properties(result, x)
        torch.testing.assert_close(
            result, torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
        )


def test_naninterp_type_error_for_non_floating_input() -> None:
    """Test that integer and boolean tensors raise TypeError."""
    with pytest.raises(TypeError, match="floating point"):
        QF.naninterp(torch.tensor([1, 2, 3]))
    with pytest.raises(TypeError, match="floating point"):
        QF.naninterp(torch.tensor([True, False]))


def test_naninterp_infinity_neighbors() -> None:
    """Test that ±inf neighbors follow IEEE 754 arithmetic."""
    x = torch.tensor([1.0, math.nan, math.inf, math.nan, 2.0])
    result = QF.naninterp(x)
    assert result[0].item() == 1.0
    assert torch.isposinf(result[1])  # 1 + (inf - 1) * 0.5 = inf
    assert torch.isposinf(result[2])  # Valid inf values are kept as is.
    assert torch.isnan(result[3])  # inf + (2 - inf) * 0.5 = nan
    assert result[4].item() == 2.0


def test_naninterp_monotonic_within_gaps() -> None:
    """Test that interpolation is monotone between monotone endpoints."""
    result = QF.naninterp(
        torch.tensor([0.0, math.nan, math.nan, math.nan, 10.0])
    )
    assert (result.diff() > 0).all()
    torch.testing.assert_close(result, torch.tensor([0.0, 2.5, 5.0, 7.5, 10.0]))

    # Interpolated values stay within the range of the gap endpoints.
    torch.manual_seed(3)
    x = torch.randn(50, dtype=torch.float64)
    x[torch.rand(50) < 0.5] = math.nan
    result = QF.naninterp(x)
    lower = torch.minimum(QF.ffill(x), QF.bfill(x))
    upper = torch.maximum(QF.ffill(x), QF.bfill(x))
    gap = x.isnan() & ~result.isnan()
    assert (result[gap] >= lower[gap]).all()
    assert (result[gap] <= upper[gap]).all()


def test_naninterp_edge_cases() -> None:
    """Test empty, single-element, all-NaN, and NaN-free tensors."""
    # Empty tensors.
    assert QF.naninterp(torch.empty(0)).shape == (0,)
    assert QF.naninterp(torch.empty(2, 0), dim=1).shape == (2, 0)

    # Single-element tensors.
    assert QF.naninterp(torch.tensor([5.0])).item() == 5.0
    assert math.isnan(QF.naninterp(torch.tensor([math.nan])).item())

    # All-NaN input yields all-NaN output.
    assert torch.isnan(QF.naninterp(torch.full((4,), math.nan))).all()

    # NaN-free input is returned unchanged.
    x = torch.tensor([3.0, 1.0, 2.0])
    assert torch.equal(QF.naninterp(x), x)
