import math

import numpy as np
import torch

import qfeval_functions.functions as QF
import pytest


def test_skipna() -> None:
    # Test if skipna enables to implement nancumsum.
    x = torch.tensor([math.nan, 1.0, math.nan, 2.0, 3.0, math.nan, 4.0])
    np.testing.assert_allclose(
        QF.skipna(lambda x: x.cumsum(dim=0), x, dim=0).numpy(),  # type: ignore
        np.array([math.nan, 1.0, math.nan, 3.0, 6.0, math.nan, 10.0]),
    )


def test_skipna_basic_functionality() -> None:
    """Test basic skipna functionality."""
    # Simple cumsum with NaNs
    x = torch.tensor([1.0, math.nan, 2.0, 3.0])
    result = QF.skipna(lambda x: x.cumsum(dim=0), x, dim=0)
    expected = torch.tensor([1.0, math.nan, 3.0, 6.0])
    torch.testing.assert_close(result, expected, equal_nan=True)

    # Simple multiplication by 2
    x = torch.tensor([1.0, math.nan, 2.0, math.nan, 3.0])
    result = QF.skipna(lambda x: x * 2, x, dim=0)
    expected = torch.tensor([2.0, math.nan, 4.0, math.nan, 6.0])
    torch.testing.assert_close(result, expected, equal_nan=True)


def test_skipna_no_nans() -> None:
    """Test skipna behavior when there are no NaNs."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = QF.skipna(lambda x: x.cumsum(dim=0), x, dim=0)
    expected = torch.tensor([1.0, 3.0, 6.0, 10.0])
    torch.testing.assert_close(result, expected)


def test_skipna_all_nans() -> None:
    """Test skipna behavior when all values are NaN."""
    x = torch.tensor([math.nan, math.nan, math.nan])
    result = QF.skipna(lambda x: x.cumsum(dim=0), x, dim=0)
    # All should remain NaN
    assert torch.isnan(result).all()


def test_skipna_mixed_nans() -> None:
    """Test skipna with various NaN patterns."""
    # NaNs at beginning
    x1 = torch.tensor([math.nan, math.nan, 1.0, 2.0])
    result1 = QF.skipna(lambda x: x.cumsum(dim=0), x1, dim=0)
    expected1 = torch.tensor([math.nan, math.nan, 1.0, 3.0])
    torch.testing.assert_close(result1, expected1, equal_nan=True)

    # NaNs at end
    x2 = torch.tensor([1.0, 2.0, math.nan, math.nan])
    result2 = QF.skipna(lambda x: x.cumsum(dim=0), x2, dim=0)
    expected2 = torch.tensor([1.0, 3.0, math.nan, math.nan])
    torch.testing.assert_close(result2, expected2, equal_nan=True)

    # NaNs scattered
    x3 = torch.tensor([1.0, math.nan, 2.0, math.nan, 3.0])
    result3 = QF.skipna(lambda x: x.cumsum(dim=0), x3, dim=0)
    expected3 = torch.tensor([1.0, math.nan, 3.0, math.nan, 6.0])
    torch.testing.assert_close(result3, expected3, equal_nan=True)


def test_skipna_multidimensional() -> None:
    """Test skipna with multidimensional tensors."""
    x = torch.tensor(
        [[1.0, math.nan, 3.0], [4.0, 5.0, math.nan], [math.nan, 7.0, 8.0]]
    )

    # Apply along dimension 0 (rows)
    result_dim0 = QF.skipna(lambda x: x.cumsum(dim=0), x, dim=0)
    expected_dim0 = torch.tensor(
        [[1.0, math.nan, 3.0], [5.0, 5.0, math.nan], [math.nan, 12.0, 11.0]]
    )
    torch.testing.assert_close(result_dim0, expected_dim0, equal_nan=True)

    # Apply along dimension 1 (columns)
    result_dim1 = QF.skipna(lambda x: x.cumsum(dim=1), x, dim=1)
    expected_dim1 = torch.tensor(
        [[1.0, math.nan, 4.0], [4.0, 9.0, math.nan], [math.nan, 7.0, 15.0]]
    )
    torch.testing.assert_close(result_dim1, expected_dim1, equal_nan=True)


def test_skipna_multiple_tensors() -> None:
    """Test skipna with multiple input tensors."""
    x1 = torch.tensor([1.0, math.nan, 3.0, 4.0])
    x2 = torch.tensor([2.0, 3.0, math.nan, 5.0])

    # Element-wise addition
    def add_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    result = QF.skipna(add_func, x1, x2, dim=0)
    # Combined NaN mask: [False, True, True, False]
    # Valid values: x1=[1.0, 4.0], x2=[2.0, 5.0] -> [3.0, 9.0]
    expected = torch.tensor([3.0, math.nan, math.nan, 9.0])
    torch.testing.assert_close(result, expected, equal_nan=True)


def test_skipna_different_operations() -> None:
    """Test skipna with different operations."""
    x = torch.tensor([2.0, math.nan, 3.0, math.nan, 4.0])

    # Cumulative product
    result_cumprod = QF.skipna(lambda x: x.cumprod(dim=0), x, dim=0)
    expected_cumprod = torch.tensor([2.0, math.nan, 6.0, math.nan, 24.0])
    torch.testing.assert_close(result_cumprod, expected_cumprod, equal_nan=True)

    # Running maximum
    result_cummax = QF.skipna(lambda x: x.cummax(dim=0).values, x, dim=0)
    expected_cummax = torch.tensor([2.0, math.nan, 3.0, math.nan, 4.0])
    torch.testing.assert_close(result_cummax, expected_cummax, equal_nan=True)

    # Running minimum
    result_cummin = QF.skipna(lambda x: x.cummin(dim=0).values, x, dim=0)
    expected_cummin = torch.tensor([2.0, math.nan, 2.0, math.nan, 2.0])
    torch.testing.assert_close(result_cummin, expected_cummin, equal_nan=True)


def test_skipna_shape_preservation() -> None:
    """Test that skipna preserves tensor shape."""
    shapes = [(5,), (3, 4), (2, 3, 4)]

    for shape in shapes:
        x = torch.randn(shape)
        # Introduce some NaNs
        x_flat = x.view(-1)
        x_flat[0] = math.nan
        if x.numel() > 1:
            x_flat[x.numel() // 2] = math.nan

        result = QF.skipna(lambda x: x.cumsum(dim=0), x, dim=0)
        assert result.shape == x.shape


def test_skipna_negative_dim() -> None:
    """Test skipna with negative dimension indices."""
    x = torch.tensor([[1.0, math.nan, 3.0], [4.0, 5.0, math.nan]])

    # dim=-1 should be same as dim=1
    result_neg1 = QF.skipna(lambda x: x.cumsum(dim=-1), x, dim=-1)
    result_pos1 = QF.skipna(lambda x: x.cumsum(dim=1), x, dim=1)
    torch.testing.assert_close(result_neg1, result_pos1, equal_nan=True)


def test_skipna_edge_cases() -> None:
    """Test skipna with edge cases."""
    # Single element with NaN
    x_single_nan = torch.tensor([math.nan])
    result_single_nan = QF.skipna(
        lambda x: x.cumsum(dim=0), x_single_nan, dim=0
    )
    assert torch.isnan(result_single_nan[0])

    # Single element without NaN
    x_single = torch.tensor([5.0])
    result_single = QF.skipna(lambda x: x.cumsum(dim=0), x_single, dim=0)
    assert result_single[0] == 5.0

    # Empty tensor
    x_empty = torch.empty(0)
    result_empty = QF.skipna(lambda x: x.cumsum(dim=0), x_empty, dim=0)
    assert result_empty.shape == (0,)


def test_skipna_complex_function() -> None:
    """Test skipna with more complex functions."""
    x = torch.tensor([1.0, math.nan, 2.0, math.nan, 3.0, 4.0])

    # Complex function: cumsum then square
    def complex_func(x: torch.Tensor) -> torch.Tensor:
        return x.cumsum(dim=0).square()

    result = QF.skipna(complex_func, x, dim=0)
    # Valid values: [1.0, 2.0, 3.0, 4.0]
    # Cumsum: [1.0, 3.0, 6.0, 10.0]
    # Square: [1.0, 9.0, 36.0, 100.0]
    expected = torch.tensor([1.0, math.nan, 9.0, math.nan, 36.0, 100.0])
    torch.testing.assert_close(result, expected, equal_nan=True)


def test_skipna_order_preservation() -> None:
    """Test that skipna preserves order of non-NaN values."""
    x = torch.tensor([5.0, math.nan, 1.0, math.nan, 3.0, 2.0])

    # The non-NaN values should be processed in order: [5.0, 1.0, 3.0, 2.0]
    result = QF.skipna(lambda x: x.cumsum(dim=0), x, dim=0)
    expected = torch.tensor([5.0, math.nan, 6.0, math.nan, 9.0, 11.0])
    torch.testing.assert_close(result, expected, equal_nan=True)


def test_skipna_batch_processing() -> None:
    """Test skipna with batch processing."""
    batch_size = 3
    seq_length = 5
    x = torch.randn(batch_size, seq_length)

    # Introduce some NaNs
    x[0, 1] = math.nan
    x[1, 3] = math.nan
    x[2, 0] = math.nan
    x[2, 4] = math.nan

    result = QF.skipna(lambda x: x.cumsum(dim=1), x, dim=1)
    assert result.shape == (batch_size, seq_length)

    # NaN positions should be preserved
    assert torch.isnan(result[0, 1])
    assert torch.isnan(result[1, 3])
    assert torch.isnan(result[2, 0])
    assert torch.isnan(result[2, 4])


def test_skipna_with_infinite_values() -> None:
    """Test skipna behavior with infinite values."""
    x = torch.tensor([1.0, math.inf, math.nan, 2.0, -math.inf])
    result = QF.skipna(lambda x: x.cumsum(dim=0), x, dim=0)

    # Infinite values should be processed normally, only NaN is skipped
    expected = torch.tensor([1.0, math.inf, math.nan, math.inf, math.nan])
    torch.testing.assert_close(result, expected, equal_nan=True)


def test_skipna_function_with_keepdim() -> None:
    """Test skipna with functions that preserve shape."""
    x = torch.tensor([1.0, math.nan, 2.0, 3.0])

    # Simple function that preserves shape
    def add_one(x: torch.Tensor) -> torch.Tensor:
        return x + 1

    result = QF.skipna(add_one, x, dim=0)
    # Should add 1 to non-NaN values
    expected = torch.tensor([2.0, math.nan, 3.0, 4.0])
    torch.testing.assert_close(result, expected, equal_nan=True)


def test_skipna_numerical_stability() -> None:
    """Test numerical stability of skipna."""
    # Very small and large numbers
    x = torch.tensor([1e-10, math.nan, 1e10, math.nan, 1e-15])
    result = QF.skipna(lambda x: x.cumsum(dim=0), x, dim=0)

    # Should handle extreme values properly
    assert torch.isfinite(result[0])
    assert torch.isnan(result[1])
    assert torch.isfinite(result[2])
    assert torch.isnan(result[3])
    assert torch.isfinite(result[4])


def test_skipna_gradient_compatibility() -> None:
    """Test that skipna works with gradient computation."""
    # TODO(claude): The skipna function should better handle gradient propagation through
    # NaN positions. Expected behavior: gradients should flow correctly through finite
    # values while ensuring NaN positions get zero gradients, and the function should
    # support higher-order derivatives for functions that require them.
    x = torch.tensor([1.0, math.nan, 2.0, 3.0], requires_grad=True)
    result = QF.skipna(lambda x: x.cumsum(dim=0), x, dim=0)

    # Sum non-NaN values for backward pass
    loss = result[~torch.isnan(result)].sum()
    loss.backward()

    # Gradients should exist for non-NaN positions
    assert x.grad is not None
    assert not torch.isnan(x.grad[0])  # Should have gradient
    assert torch.isnan(x.grad[1]) or x.grad[1] == 0  # NaN position
    assert not torch.isnan(x.grad[2])  # Should have gradient
    assert not torch.isnan(x.grad[3])  # Should have gradient


def test_skipna_high_dimensional() -> None:
    """Test skipna with high-dimensional tensors."""
    x = torch.randn(2, 3, 4, 5)
    # Add some NaNs
    x[0, 1, :, 2] = math.nan
    x[1, :, 2, 3] = math.nan

    # Apply along different dimensions
    for dim in range(4):
        result = QF.skipna(lambda x: x.cumsum(dim=dim), x, dim=dim)
        assert result.shape == x.shape
        # NaN positions should be preserved
        nan_mask = torch.isnan(x)
        assert torch.isnan(result[nan_mask]).all()


def test_skipna_lambda_functions() -> None:
    """Test skipna with various lambda functions."""
    x = torch.tensor([2.0, math.nan, 3.0, math.nan, 4.0])

    # Different lambda functions
    funcs_and_expected = [
        (lambda x: x * 2, torch.tensor([4.0, math.nan, 6.0, math.nan, 8.0])),
        (
            lambda x: x.cumsum(dim=0),
            torch.tensor([2.0, math.nan, 5.0, math.nan, 9.0]),
        ),
        (
            lambda x: x.sqrt(),
            torch.tensor([math.sqrt(2), math.nan, math.sqrt(3), math.nan, 2.0]),
        ),
    ]

    for func, expected in funcs_and_expected:
        result = QF.skipna(func, x, dim=0)
        torch.testing.assert_close(result, expected, equal_nan=True)


def test_skipna_performance() -> None:
    """Test skipna performance with large tensors."""
    for size in [1000, 5000]:
        x = torch.randn(size)
        # Add scattered NaNs
        x[::100] = math.nan

        result = QF.skipna(lambda x: x.cumsum(dim=0), x, dim=0)

        # Verify properties
        assert result.shape == x.shape
        nan_positions = torch.isnan(x)
        assert torch.isnan(result[nan_positions]).all()

        # Clean up
        del x, result


def test_skipna_different_dtypes() -> None:
    """Test skipna with different dtypes."""
    for dtype in [torch.float32, torch.float64]:
        x = torch.tensor([1.0, math.nan, 2.0], dtype=dtype)
        result = QF.skipna(lambda x: x.cumsum(dim=0), x, dim=0)
        assert result.dtype == dtype
        expected = torch.tensor([1.0, math.nan, 3.0], dtype=dtype)
        torch.testing.assert_close(result, expected, equal_nan=True)
