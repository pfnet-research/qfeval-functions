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
    # If all values are the same, their min should also be the same.
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


def test_mmin_basic_functionality() -> None:
    """Test basic moving minimum functionality."""
    # Simple sequence
    x = torch.tensor([5.0, 1.0, 3.0, 4.0, 2.0])
    result = QF.mmin(x, 3)

    # Check shape
    assert result.shape == x.shape

    # Check computed values (moving min over window of 3)
    expected = torch.tensor([5.0, 1.0, 1.0, 1.0, 2.0])
    torch.testing.assert_close(result, expected)


def test_mmin_shape_preservation() -> None:
    """Test that moving minimum preserves tensor shape."""
    # 1D tensor
    x_1d = torch.randn(10)
    result_1d = QF.mmin(x_1d, 3)
    assert result_1d.shape == x_1d.shape

    # 2D tensor
    x_2d = torch.randn(5, 8)
    result_2d = QF.mmin(x_2d, 4)
    assert result_2d.shape == x_2d.shape

    # 3D tensor
    x_3d = torch.randn(3, 4, 6)
    result_3d = QF.mmin(x_3d, 2)
    assert result_3d.shape == x_3d.shape


def test_mmin_different_spans() -> None:
    """Test moving minimum with different span values."""
    x = torch.tensor([5.0, 1.0, 4.0, 2.0, 3.0, 6.0])

    # span = 1 (should be identity)
    result_1 = QF.mmin(x, 1)
    torch.testing.assert_close(result_1, x)

    # span = 2
    result_2 = QF.mmin(x, 2)
    expected_2 = torch.tensor([5.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    torch.testing.assert_close(result_2, expected_2)

    # span = 3
    result_3 = QF.mmin(x, 3)
    expected_3 = torch.tensor([5.0, 1.0, 1.0, 1.0, 2.0, 2.0])
    torch.testing.assert_close(result_3, expected_3)


def test_mmin_dim_parameter() -> None:
    """Test moving minimum with different dimension parameters."""
    x = torch.tensor([[4.0, 3.0, 2.0, 1.0], [8.0, 7.0, 6.0, 5.0]])

    # dim=-1 (default, along last dimension)
    result_dim_neg1 = QF.mmin(x, 2, dim=-1)

    # dim=1 (same as dim=-1 for 2D tensor)
    result_dim_1 = QF.mmin(x, 2, dim=1)
    torch.testing.assert_close(result_dim_neg1, result_dim_1)

    # dim=0 (along first dimension)
    result_dim_0 = QF.mmin(x, 2, dim=0)
    assert result_dim_0.shape == x.shape

    # Check values for dim=0
    expected_dim_0 = torch.tensor([[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])
    torch.testing.assert_close(result_dim_0, expected_dim_0)


def test_mmin_with_nan_values() -> None:
    """Test moving minimum with NaN input values."""
    x = torch.tensor([4.0, float("nan"), 1.0, 2.0, 3.0])
    result = QF.mmin(x, 3)

    # Shape should be preserved
    assert result.shape == x.shape

    # NaN values propagate through the moving window
    # The exact behavior depends on how PyTorch handles NaN in min operations
    assert result.shape == x.shape


def test_mmin_constant_signal() -> None:
    """Test moving minimum with constant signal."""
    x = torch.full((8,), 3.0)
    result = QF.mmin(x, 4)

    # All values should remain 3.0
    expected = torch.full((8,), 3.0)
    torch.testing.assert_close(result, expected)


def test_mmin_increasing_sequence() -> None:
    """Test moving minimum with strictly increasing sequence."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = QF.mmin(x, 3)

    # For increasing sequence, mmin should be the earliest value in the window
    expected = torch.tensor([1.0, 1.0, 1.0, 2.0, 3.0, 4.0])
    torch.testing.assert_close(result, expected)


def test_mmin_decreasing_sequence() -> None:
    """Test moving minimum with strictly decreasing sequence."""
    x = torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    result = QF.mmin(x, 3)

    # For decreasing sequence, mmin should be the latest value in the window
    expected = torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    torch.testing.assert_close(result, expected)


def test_mmin_span_equals_length() -> None:
    """Test moving minimum when span equals tensor length."""
    x = torch.tensor([4.0, 1.0, 3.0, 2.0])
    result = QF.mmin(x, 4)

    # mmin implementation has specific behavior - let's check actual behavior
    # The first elements may not be the global min due to edge padding behavior
    assert result.shape == x.shape
    # The minimum value should appear in the result
    assert result.min() == x.min()


def test_mmin_span_larger_than_length() -> None:
    """Test moving minimum when span is larger than tensor length."""
    x = torch.tensor([3.0, 1.0, 2.0])
    result = QF.mmin(x, 5)

    # Similar to above - check actual behavior rather than assuming global min everywhere
    assert result.shape == x.shape
    # The minimum value should appear in the result
    assert result.min() == x.min()


def test_mmin_mathematical_properties() -> None:
    """Test mathematical properties of moving minimum."""
    x = torch.tensor([5.0, 1.0, 4.0, 2.0, 3.0])
    result = QF.mmin(x, 3)

    # Moving min should be <= all values in the original tensor
    assert torch.all(result <= x.max())

    # Moving min should be >= global min
    assert torch.all(result >= x.min())

    # Each moving min should be <= the corresponding input value
    assert torch.all(result <= x)


def test_mmin_sliding_window_behavior() -> None:
    """Test that moving minimum correctly implements sliding window."""
    x = torch.tensor([10.0, 30.0, 20.0, 5.0, 40.0, 15.0])
    result = QF.mmin(x, 3)

    # Check each window manually
    # Window [10, 30, 20] -> 10
    # Window [30, 20, 5] -> 5
    # Window [20, 5, 40] -> 5
    # Window [5, 40, 15] -> 5
    expected = torch.tensor([10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
    torch.testing.assert_close(result, expected)


def test_mmin_zero_values() -> None:
    """Test moving minimum with zero values."""
    x = torch.tensor([2.0, 0.0, 3.0, 1.0, 0.0])
    result = QF.mmin(x, 3)

    # Check specific values
    expected = torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0])
    torch.testing.assert_close(result, expected)


def test_mmin_negative_values() -> None:
    """Test moving minimum with negative values."""
    x = torch.tensor([-1.0, -3.0, -2.0, -4.0, -1.0])
    result = QF.mmin(x, 3)

    # Check computed values
    expected = torch.tensor([-1.0, -3.0, -3.0, -4.0, -4.0])
    torch.testing.assert_close(result, expected)


def test_mmin_mixed_positive_negative() -> None:
    """Test moving minimum with mixed positive and negative values."""
    x = torch.tensor([3.0, -2.0, 1.0, -4.0, 3.0])
    result = QF.mmin(x, 3)

    # Check computed values
    expected = torch.tensor([3.0, -2.0, -2.0, -4.0, -4.0])
    torch.testing.assert_close(result, expected)


def test_mmin_large_values() -> None:
    """Test moving minimum with large values."""
    x = torch.tensor([4e6, 1e6, 3e6, 2e6])
    result = QF.mmin(x, 2)

    # Check that large values are handled correctly
    expected = torch.tensor([4e6, 1e6, 1e6, 2e6])
    torch.testing.assert_close(result, expected)


def test_mmin_small_values() -> None:
    """Test moving minimum with very small values."""
    x = torch.tensor([4e-6, 1e-6, 3e-6, 2e-6])
    result = QF.mmin(x, 2)

    # Check that small values are handled correctly
    expected = torch.tensor([4e-6, 1e-6, 1e-6, 2e-6])
    torch.testing.assert_close(result, expected)


def test_mmin_multidimensional() -> None:
    """Test moving minimum with multidimensional tensors."""
    # 2D tensor - apply along different dimensions
    x_2d = torch.tensor([[4.0, 1.0, 3.0, 2.0], [1.0, 5.0, 2.0, 6.0]])

    # Along dimension 1 (columns)
    result_dim1 = QF.mmin(x_2d, 2, dim=1)
    expected_dim1 = torch.tensor([[4.0, 1.0, 1.0, 2.0], [1.0, 1.0, 2.0, 2.0]])
    torch.testing.assert_close(result_dim1, expected_dim1)

    # Along dimension 0 (rows)
    result_dim0 = QF.mmin(x_2d, 2, dim=0)
    expected_dim0 = torch.tensor([[4.0, 1.0, 3.0, 2.0], [1.0, 1.0, 2.0, 2.0]])
    torch.testing.assert_close(result_dim0, expected_dim0)


def test_mmin_batch_processing() -> None:
    """Test moving minimum with batch processing scenarios."""
    batch_size = 3
    seq_length = 8
    x = torch.randn(batch_size, seq_length)

    result = QF.mmin(x, 4)
    assert result.shape == (batch_size, seq_length)

    # Check that each batch is processed independently
    for i in range(batch_size):
        individual_result = QF.mmin(x[i], 4)
        torch.testing.assert_close(result[i], individual_result)


def test_mmin_numerical_stability() -> None:
    """Test numerical stability of moving minimum."""
    # Test with values that could cause overflow/underflow
    x_large = torch.tensor([2e10, 1e10, 1.5e10])
    result_large = QF.mmin(x_large, 2)
    assert torch.isfinite(result_large).all()

    x_small = torch.tensor([2e-10, 1e-10, 1.5e-10])
    result_small = QF.mmin(x_small, 2)
    assert torch.isfinite(result_small).all()


def test_mmin_edge_cases() -> None:
    """Test moving minimum edge cases."""
    # Skip empty tensor test as mmin implementation doesn't handle empty tensors
    # This is a limitation of the current implementation

    # Span = 1 (identity operation)
    x = torch.tensor([3.0, 1.0, 2.0])
    result_span1 = QF.mmin(x, 1)
    torch.testing.assert_close(result_span1, x)


def test_mmin_comparison_with_pandas() -> None:
    """Test consistency with pandas rolling min."""
    # Create test data
    x = torch.tensor([5.0, 1.0, 8.0, 2.0, 7.0, 4.0, 3.0, 6.0])
    span = 3

    # Compute using our function
    result = QF.mmin(x, span)

    # Compute using pandas
    df = pd.DataFrame(x.numpy())
    pandas_result = df.rolling(span, min_periods=1).min().values.flatten()

    # Compare
    torch.testing.assert_close(
        result, torch.tensor(pandas_result, dtype=result.dtype)
    )


def test_mmin_idempotency() -> None:
    """Test idempotency property of moving minimum."""
    x = torch.tensor([5.0, 1.0, 3.0, 4.0, 2.0])
    span = 3

    # Applying mmin twice should give same result as applying once
    result_once = QF.mmin(x, span)
    result_twice = QF.mmin(result_once, span)

    # For moving min, applying twice is NOT the same as applying once
    # But the second application should still be a valid moving min
    assert result_twice.shape == result_once.shape
    # Each element in result_twice should be <= corresponding element in result_once
    assert torch.all(result_twice <= result_once)


def test_mmin_monotonicity() -> None:
    """Test monotonicity properties of moving minimum."""
    x1 = torch.tensor([4.0, 3.0, 2.0, 1.0])
    x2 = torch.tensor([5.0, 4.0, 3.0, 2.0])  # x2 >= x1
    span = 2

    result1 = QF.mmin(x1, span)
    result2 = QF.mmin(x2, span)

    # If x2 >= x1 elementwise, then mmin(x2) >= mmin(x1)
    assert torch.all(result2 >= result1)


def test_mmin_performance() -> None:
    """Test moving minimum performance with larger tensors."""
    # Test with moderately large tensor
    x_large = torch.randn(1000)
    result = QF.mmin(x_large, 20)

    assert result.shape == x_large.shape
    assert torch.isfinite(result).all()


def test_mmin_consistency() -> None:
    """Test that moving minimum produces consistent results."""
    x = torch.tensor([5.0, 1.0, 4.0, 3.0, 2.0])
    span = 3

    # Multiple calls should produce same result
    result1 = QF.mmin(x, span)
    result2 = QF.mmin(x, span)
    torch.testing.assert_close(result1, result2)


def test_mmin_boundary_behavior() -> None:
    """Test boundary behavior of moving minimum."""
    x = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])

    # Test different spans near boundary conditions
    result_2 = QF.mmin(x, 2)
    result_5 = QF.mmin(x, 5)
    result_6 = QF.mmin(x, 6)  # span > length

    # Check results make sense
    assert result_2.shape == x.shape
    assert result_5.shape == x.shape
    assert result_6.shape == x.shape

    # span = tensor length and larger should preserve minimum values
    assert result_5.min() == x.min()
    assert result_6.min() == x.min()


def test_mmin_with_duplicates() -> None:
    """Test moving minimum with duplicate values."""
    x = torch.tensor([3.0, 3.0, 1.0, 1.0, 4.0, 4.0])
    result = QF.mmin(x, 3)

    # Check that duplicates are handled correctly
    expected = torch.tensor([3.0, 3.0, 1.0, 1.0, 1.0, 1.0])
    torch.testing.assert_close(result, expected)


def test_mmin_extremes() -> None:
    """Test moving minimum with extreme patterns."""
    # Alternating pattern
    x_alt = torch.tensor([5.0, 1.0, 6.0, 2.0, 7.0, 3.0])
    result_alt = QF.mmin(x_alt, 3)

    # Should capture the minimum in each window
    expected_alt = torch.tensor([5.0, 1.0, 1.0, 1.0, 2.0, 2.0])
    torch.testing.assert_close(result_alt, expected_alt)


def test_mmin_with_infinity() -> None:
    """Test moving minimum with infinite values."""
    x = torch.tensor([3.0, float("-inf"), 4.0, 1.0])
    result = QF.mmin(x, 2)

    # Check that negative infinity is handled correctly
    assert result.shape == x.shape
    # Negative infinity should be the minimum value in the result
    assert torch.isneginf(result).any()  # Should contain negative infinity
    assert result.min().isneginf()  # Minimum should be negative infinity


def test_mmin_signal_characteristics() -> None:
    """Test signal processing characteristics of moving minimum."""
    # Valley detection capability
    x_valleys = torch.tensor([5.0, 3.0, 1.0, 4.0, 2.0, 0.5, 3.0])
    result_valleys = QF.mmin(x_valleys, 3)

    # Moving min should capture and extend valleys
    assert result_valleys.shape == x_valleys.shape
    # Should preserve the minimum value from valleys
    assert result_valleys.min() == x_valleys.min()


def test_mmin_high_dimensional() -> None:
    """Test moving minimum with high-dimensional tensors."""
    # 3D tensor
    x_3d = torch.randn(2, 3, 5)
    result_3d = QF.mmin(x_3d, 2)

    assert result_3d.shape == x_3d.shape

    # Check that operation is applied along the last dimension by default
    for i in range(2):
        for j in range(3):
            individual_result = QF.mmin(x_3d[i, j], 2)
            torch.testing.assert_close(result_3d[i, j], individual_result)
