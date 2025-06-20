import math

import numpy as np
import pandas as pd
import torch

import qfeval_functions.functions as QF
from tests.functions.test_utils import generic_test_consistency
from tests.functions.test_utils import generic_test_device_preservation
from tests.functions.test_utils import generic_test_dtype_preservation
from tests.functions.test_utils import generic_test_memory_efficiency
from tests.functions.test_utils import generic_test_single_element
import pytest


def test_ma() -> None:
    x = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0.0])
    np.testing.assert_allclose(
        QF.ma(x, 4).numpy(),
        np.array([math.nan, math.nan, math.nan, 0.25, 0.25, 0.25, 0.25, 0]),
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


def test_ma_basic_functionality() -> None:
    """Test basic moving average functionality."""
    # Simple sequence
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = QF.ma(x, 3)

    # Check shape
    assert result.shape == x.shape

    # First span-1 values should be NaN
    assert torch.isnan(result[:2]).all()

    # Check computed values
    expected_2 = (1.0 + 2.0 + 3.0) / 3.0  # 2.0
    expected_3 = (2.0 + 3.0 + 4.0) / 3.0  # 3.0
    expected_4 = (3.0 + 4.0 + 5.0) / 3.0  # 4.0

    torch.testing.assert_close(result[2], torch.tensor(expected_2))
    torch.testing.assert_close(result[3], torch.tensor(expected_3))
    torch.testing.assert_close(result[4], torch.tensor(expected_4))


@pytest.mark.random
def test_ma_shape_preservation() -> None:
    """Test that moving average preserves tensor shape."""
    # 1D tensor
    x_1d = torch.randn(10)
    result_1d = QF.ma(x_1d, 3)
    assert result_1d.shape == x_1d.shape

    # 2D tensor
    x_2d = torch.randn(5, 8)
    result_2d = QF.ma(x_2d, 4)
    assert result_2d.shape == x_2d.shape

    # 3D tensor
    x_3d = torch.randn(3, 4, 6)
    result_3d = QF.ma(x_3d, 2)
    assert result_3d.shape == x_3d.shape


def test_ma_different_spans() -> None:
    """Test moving average with different span values."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    # span = 1 (should be identity after first NaN)
    result_1 = QF.ma(x, 1)
    torch.testing.assert_close(result_1, x)

    # span = 2
    result_2 = QF.ma(x, 2)
    assert torch.isnan(result_2[0])
    torch.testing.assert_close(result_2[1], torch.tensor(1.5))  # (1+2)/2
    torch.testing.assert_close(result_2[2], torch.tensor(2.5))  # (2+3)/2

    # span = 3
    result_3 = QF.ma(x, 3)
    assert torch.isnan(result_3[:2]).all()
    torch.testing.assert_close(result_3[2], torch.tensor(2.0))  # (1+2+3)/3


def test_ma_dim_parameter() -> None:
    """Test moving average with different dimension parameters."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    # dim=-1 (default, along last dimension)
    result_dim_neg1 = QF.ma(x, 2, dim=-1)

    # dim=1 (same as dim=-1 for 2D tensor)
    result_dim_1 = QF.ma(x, 2, dim=1)
    torch.testing.assert_close(result_dim_neg1, result_dim_1, equal_nan=True)

    # dim=0 (along first dimension)
    result_dim_0 = QF.ma(x, 2, dim=0)
    assert result_dim_0.shape == x.shape

    # Check values for dim=0
    assert torch.isnan(result_dim_0[0]).all()
    torch.testing.assert_close(
        result_dim_0[1], torch.tensor([3.0, 4.0, 5.0, 6.0])
    )


def test_ma_with_nan_values() -> None:
    """Test moving average with NaN input values."""
    x = torch.tensor([1.0, math.nan, 3.0, 4.0, 5.0])
    result = QF.ma(x, 3)

    # Shape should be preserved
    assert result.shape == x.shape

    # First two values should be NaN due to span
    assert torch.isnan(result[:2]).all()

    # Values involving NaN should be NaN
    assert torch.isnan(result[2])  # includes math.nan at index 1
    assert torch.isnan(result[3])  # includes math.nan at index 1

    # Value not involving NaN should be finite
    torch.testing.assert_close(result[4], torch.tensor(4.0))  # (3+4+5)/3


def test_ma_constant_signal() -> None:
    """Test moving average with constant signal."""
    x = torch.full((8,), 3.0)
    result = QF.ma(x, 4)

    # First 3 values should be NaN
    assert torch.isnan(result[:3]).all()

    # Remaining values should be 3.0
    expected = torch.full((5,), 3.0)
    torch.testing.assert_close(result[3:], expected)


def test_ma_span_equals_length() -> None:
    """Test moving average when span equals tensor length."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = QF.ma(x, 4)

    # First 3 values should be NaN
    assert torch.isnan(result[:3]).all()

    # Last value should be the mean of all elements
    expected = x.mean()
    torch.testing.assert_close(result[3], expected)


def test_ma_span_larger_than_length() -> None:
    """Test moving average when span is larger than tensor length."""
    x = torch.tensor([1.0, 2.0, 3.0])
    result = QF.ma(x, 5)

    # All values should be NaN
    assert torch.isnan(result).all()


def test_ma_mathematical_properties() -> None:
    """Test mathematical properties of moving average."""
    x = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
    result = QF.ma(x, 3)

    # Moving average should be bounded by min and max of input
    finite_result = result[~torch.isnan(result)]
    assert torch.all(finite_result >= x.min())
    assert torch.all(finite_result <= x.max())

    # For arithmetic sequence, moving average should also form arithmetic sequence
    # MA at index i: (2i + 2(i+1) + 2(i+2))/3 = 2(3i+3)/3 = 2(i+1)
    expected_3 = 4.0  # (2+4+6)/3
    expected_4 = 6.0  # (4+6+8)/3
    torch.testing.assert_close(result[2], torch.tensor(expected_3))
    torch.testing.assert_close(result[3], torch.tensor(expected_4))


def test_ma_sliding_window_behavior() -> None:
    """Test that moving average correctly implements sliding window."""
    x = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    result = QF.ma(x, 3)

    # Check each window manually
    # Window [10, 20, 30] -> 20
    torch.testing.assert_close(result[2], torch.tensor(20.0))

    # Window [20, 30, 40] -> 30
    torch.testing.assert_close(result[3], torch.tensor(30.0))

    # Window [30, 40, 50] -> 40
    torch.testing.assert_close(result[4], torch.tensor(40.0))

    # Window [40, 50, 60] -> 50
    torch.testing.assert_close(result[5], torch.tensor(50.0))


def test_ma_zero_values() -> None:
    """Test moving average with zero values."""
    x = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
    result = QF.ma(x, 3)

    # First two should be NaN
    assert torch.isnan(result[:2]).all()

    # Check specific values
    torch.testing.assert_close(result[2], torch.tensor(0.0))  # (0+0+0)/3
    torch.testing.assert_close(result[3], torch.tensor(1.0 / 3.0))  # (0+0+1)/3
    torch.testing.assert_close(result[4], torch.tensor(1.0 / 3.0))  # (0+1+0)/3


def test_ma_negative_values() -> None:
    """Test moving average with negative values."""
    x = torch.tensor([-1.0, -2.0, 3.0, -4.0, 5.0])
    result = QF.ma(x, 3)

    # Check computed values
    expected_2 = (-1.0 + -2.0 + 3.0) / 3.0  # 0.0
    expected_3 = (-2.0 + 3.0 + -4.0) / 3.0  # -1.0
    expected_4 = (3.0 + -4.0 + 5.0) / 3.0  # 4/3

    torch.testing.assert_close(result[2], torch.tensor(expected_2))
    torch.testing.assert_close(result[3], torch.tensor(expected_3))
    torch.testing.assert_close(result[4], torch.tensor(expected_4))


def test_ma_large_values() -> None:
    """Test moving average with large values."""
    x = torch.tensor([1e6, 2e6, 3e6, 4e6])
    result = QF.ma(x, 2)

    # Check that large values are handled correctly
    torch.testing.assert_close(result[1], torch.tensor(1.5e6))
    torch.testing.assert_close(result[2], torch.tensor(2.5e6))
    torch.testing.assert_close(result[3], torch.tensor(3.5e6))


def test_ma_small_values() -> None:
    """Test moving average with very small values."""
    x = torch.tensor([1e-6, 2e-6, 3e-6, 4e-6])
    result = QF.ma(x, 2)

    # Check that small values are handled correctly
    torch.testing.assert_close(result[1], torch.tensor(1.5e-6))
    torch.testing.assert_close(result[2], torch.tensor(2.5e-6))
    torch.testing.assert_close(result[3], torch.tensor(3.5e-6))


def test_ma_multidimensional() -> None:
    """Test moving average with multidimensional tensors."""
    # 2D tensor - apply along different dimensions
    x_2d = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    # Along dimension 1 (columns)
    result_dim1 = QF.ma(x_2d, 2, dim=1)
    assert torch.isnan(result_dim1[:, 0]).all()
    torch.testing.assert_close(result_dim1[0, 1], torch.tensor(1.5))
    torch.testing.assert_close(result_dim1[1, 1], torch.tensor(5.5))

    # Along dimension 0 (rows)
    result_dim0 = QF.ma(x_2d, 2, dim=0)
    assert torch.isnan(result_dim0[0, :]).all()
    torch.testing.assert_close(
        result_dim0[1, :], torch.tensor([3.0, 4.0, 5.0, 6.0])
    )


@pytest.mark.random
def test_ma_batch_processing() -> None:
    """Test moving average with batch processing scenarios."""
    batch_size = 3
    seq_length = 10
    x = torch.randn(batch_size, seq_length)

    result = QF.ma(x, 4)
    assert result.shape == (batch_size, seq_length)

    # First 3 values should be NaN for each batch
    assert torch.isnan(result[:, :3]).all()

    # Check that each batch is processed independently
    for i in range(batch_size):
        individual_result = QF.ma(x[i], 4)
        torch.testing.assert_close(result[i], individual_result, equal_nan=True)


def test_ma_numerical_stability() -> None:
    """Test numerical stability of moving average."""
    # Test with values that could cause overflow/underflow
    x_large = torch.tensor([1e10, 1e10, 1e10])
    result_large = QF.ma(x_large, 2)
    assert torch.isfinite(result_large[1:]).all()

    x_small = torch.tensor([1e-10, 1e-10, 1e-10])
    result_small = QF.ma(x_small, 2)
    assert torch.isfinite(result_small[1:]).all()
    assert (result_small[1:] > 0).all()


def test_ma_edge_cases() -> None:
    """Test moving average edge cases."""
    # Empty tensor
    x_empty = torch.empty(0)
    result_empty = QF.ma(x_empty, 1)
    assert result_empty.shape == (0,)

    # Span = 1 (identity operation)
    x = torch.tensor([1.0, 2.0, 3.0])
    result_span1 = QF.ma(x, 1)
    torch.testing.assert_close(result_span1, x)


def test_ma_comparison_with_pandas() -> None:
    """Test consistency with pandas rolling mean."""
    # Create test data
    x = torch.tensor([1.0, 3.0, 2.0, 8.0, 5.0, 4.0, 7.0, 6.0])
    span = 3

    # Compute using our function
    result = QF.ma(x, span)

    # Compute using pandas
    df = pd.DataFrame(x.numpy())
    pandas_result = df.rolling(span).mean().values.flatten()

    # Compare (allowing for NaN differences and dtype differences)
    torch.testing.assert_close(
        result[~torch.isnan(result)],
        torch.tensor(
            pandas_result[~np.isnan(pandas_result)], dtype=result.dtype
        ),
    )


def test_ma_linearity() -> None:
    """Test linearity property of moving average."""
    x1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x2 = torch.tensor([2.0, 3.0, 4.0, 5.0])
    span = 2

    # MA(x1 + x2) should equal MA(x1) + MA(x2)
    result_sum = QF.ma(x1 + x2, span)
    result1 = QF.ma(x1, span)
    result2 = QF.ma(x2, span)

    # Compare non-NaN values
    mask = ~torch.isnan(result_sum)
    torch.testing.assert_close(result_sum[mask], (result1 + result2)[mask])


def test_ma_scaling_property() -> None:
    """Test scaling property of moving average."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    scale = 3.0
    span = 3

    # MA(scale * x) should equal scale * MA(x)
    result_scaled = QF.ma(scale * x, span)
    result_original = QF.ma(x, span)

    # Compare non-NaN values
    mask = ~torch.isnan(result_scaled)
    torch.testing.assert_close(
        result_scaled[mask], (scale * result_original)[mask]
    )


def test_ma_boundary_behavior() -> None:
    """Test boundary behavior of moving average."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    # Test different spans near boundary conditions
    result_2 = QF.ma(x, 2)
    result_5 = QF.ma(x, 5)
    result_6 = QF.ma(x, 6)  # span > length

    # Check NaN patterns
    assert torch.isnan(result_2[:1]).all()
    assert torch.isnan(result_5[:4]).all()
    assert torch.isnan(result_6).all()


def test_ma_signal_smoothing() -> None:
    """Test that moving average actually smooths signals."""
    # Noisy signal
    x_noisy = torch.tensor([1.0, 10.0, 2.0, 9.0, 3.0, 8.0, 4.0])
    result = QF.ma(x_noisy, 3)

    # Calculate total variation (sum of absolute differences)
    def total_variation(signal: torch.Tensor) -> torch.Tensor:
        finite_signal = signal[~torch.isnan(signal)]
        if len(finite_signal) <= 1:
            return torch.tensor(0.0)
        return torch.abs(finite_signal[1:] - finite_signal[:-1]).sum()

    original_tv = total_variation(x_noisy)
    smoothed_tv = total_variation(result)

    # Smoothed signal should have lower total variation
    assert smoothed_tv < original_tv


def test_ma_trend_preservation() -> None:
    """Test that moving average preserves trends."""
    # Increasing trend
    x_increasing = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result_inc = QF.ma(x_increasing, 3)

    # Moving average should also be increasing
    finite_result = result_inc[~torch.isnan(result_inc)]
    diffs = finite_result[1:] - finite_result[:-1]
    assert torch.all(diffs >= 0)  # Non-decreasing

    # For linear trend, moving average should also be linear
    torch.testing.assert_close(
        finite_result, torch.tensor([2.0, 3.0, 4.0, 5.0])
    )


def test_ma_with_mixed_data_types() -> None:
    """Test moving average with mixed positive/negative/zero values."""
    x_mixed = torch.tensor([0.0, -1.0, 2.0, -3.0, 4.0, 0.0])
    result = QF.ma(x_mixed, 3)

    # Check specific calculations
    torch.testing.assert_close(result[2], torch.tensor(1.0 / 3.0))  # (0-1+2)/3
    torch.testing.assert_close(
        result[3], torch.tensor(-2.0 / 3.0)
    )  # (-1+2-3)/3
    torch.testing.assert_close(result[4], torch.tensor(1.0))  # (2-3+4)/3
    torch.testing.assert_close(result[5], torch.tensor(1.0 / 3.0))  # (-3+4+0)/3


def test_ma_dtype_preservation() -> None:
    """Test that moving average preserves input dtype."""

    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    generic_test_dtype_preservation(QF.ma, x, 3)


def test_ma_device_preservation() -> None:
    """Test that moving average preserves input device."""

    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    generic_test_device_preservation(QF.ma, x, 3)


def test_ma_memory_efficiency() -> None:
    """Test memory efficiency of moving average."""

    generic_test_memory_efficiency(QF.ma, span=3)


def test_ma_single_element() -> None:
    """Test moving average with single element tensor."""

    generic_test_single_element(QF.ma, span=1)


def test_ma_consistency() -> None:
    """Test that multiple calls to moving average produce same result."""

    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    generic_test_consistency(QF.ma, x, span=3)
