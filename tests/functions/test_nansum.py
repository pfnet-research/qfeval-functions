import math

import numpy as np
import torch

import qfeval_functions.functions as QF
from tests.functions.test_utils import generic_test_consistency
from tests.functions.test_utils import generic_test_device_preservation
from tests.functions.test_utils import generic_test_dtype_preservation
from tests.functions.test_utils import generic_test_memory_efficiency


def test_nansum() -> None:
    x = torch.tensor(
        [
            [0.0, -1.0, 1.5, math.nan],
            [math.nan, math.nan, math.nan, math.nan],
            [0.0, -1.0, 2.0, -2.0],
        ]
    )
    np.testing.assert_allclose(
        QF.nansum(x, dim=1).numpy(),
        np.array([0.5, math.nan, -1.0]),
    )


def test_nansum_basic_functionality() -> None:
    """Test basic NaN-aware sum functionality."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = QF.nansum(x)
    expected = torch.tensor(10.0)
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nansum_with_nans() -> None:
    """Test NaN-aware sum with NaN values."""
    x = torch.tensor([1.0, math.nan, 3.0, 4.0])
    result = QF.nansum(x)
    expected = torch.tensor(8.0)
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nansum_all_nans() -> None:
    """Test sum with all NaN values."""
    x = torch.tensor([math.nan, math.nan, math.nan])
    result = QF.nansum(x)
    assert torch.isnan(result)


def test_nansum_no_nans() -> None:
    """Test sum without any NaN values."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = QF.nansum(x)
    expected = x.sum()
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nansum_dimensions() -> None:
    """Test NaN-aware sum along specific dimensions."""
    x = torch.tensor(
        [[1.0, math.nan, 3.0], [4.0, 5.0, math.nan], [math.nan, 6.0, 7.0]]
    )

    # Test along dimension 0
    result_dim0 = QF.nansum(x, dim=0)
    expected_dim0 = torch.tensor([5.0, 11.0, 10.0])
    np.testing.assert_allclose(result_dim0.numpy(), expected_dim0.numpy())

    # Test along dimension 1
    result_dim1 = QF.nansum(x, dim=1)
    expected_dim1 = torch.tensor([4.0, 9.0, 13.0])
    np.testing.assert_allclose(result_dim1.numpy(), expected_dim1.numpy())


def test_nansum_keepdim() -> None:
    """Test keepdim parameter functionality."""
    x = torch.tensor([[1.0, math.nan, 3.0], [4.0, 5.0, math.nan]])

    # Test keepdim=True
    result_keepdim = QF.nansum(x, dim=1, keepdim=True)
    assert result_keepdim.shape == (2, 1)
    expected_keepdim = torch.tensor([[4.0], [9.0]])
    np.testing.assert_allclose(result_keepdim.numpy(), expected_keepdim.numpy())

    # Test keepdim=False
    result_no_keepdim = QF.nansum(x, dim=1, keepdim=False)
    assert result_no_keepdim.shape == (2,)
    expected_no_keepdim = torch.tensor([4.0, 9.0])
    np.testing.assert_allclose(
        result_no_keepdim.numpy(), expected_no_keepdim.numpy()
    )


def test_nansum_negative_dimensions() -> None:
    """Test negative dimension indexing."""
    x = torch.tensor([[1.0, math.nan, 3.0], [4.0, 5.0, math.nan]])

    result_neg1 = QF.nansum(x, dim=-1)
    result_pos1 = QF.nansum(x, dim=1)
    np.testing.assert_allclose(result_neg1.numpy(), result_pos1.numpy())

    result_neg2 = QF.nansum(x, dim=-2)
    result_pos0 = QF.nansum(x, dim=0)
    np.testing.assert_allclose(result_neg2.numpy(), result_pos0.numpy())


def test_nansum_multiple_dimensions() -> None:
    """Test sum along multiple dimensions."""
    x = torch.tensor(
        [[[1.0, math.nan], [3.0, 4.0]], [[math.nan, 5.0], [6.0, 7.0]]]
    )

    # Test along dimensions (0, 1)
    result_01 = QF.nansum(x, dim=(0, 1))
    expected_01 = torch.tensor([10.0, 16.0])
    np.testing.assert_allclose(result_01.numpy(), expected_01.numpy())

    # Test along dimensions (1, 2)
    result_12 = QF.nansum(x, dim=(1, 2))
    expected_12 = torch.tensor([8.0, 18.0])
    np.testing.assert_allclose(result_12.numpy(), expected_12.numpy())


def test_nansum_single_nan() -> None:
    """Test sum with single NaN element."""
    x = torch.tensor([math.nan])
    result = QF.nansum(x)
    assert torch.isnan(result)


def test_nansum_zero_values() -> None:
    """Test sum with zero values."""
    x = torch.tensor([1.0, 0.0, math.nan, 3.0])
    result = QF.nansum(x)
    expected = torch.tensor(4.0)
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nansum_negative_values() -> None:
    """Test sum with negative values."""
    x = torch.tensor([-1.0, 2.0, math.nan, -3.0])
    result = QF.nansum(x)
    expected = torch.tensor(-2.0)
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nansum_large_values() -> None:
    """Test sum with large values."""
    x = torch.tensor([1e6, math.nan, 2e6, 3e6])
    result = QF.nansum(x)
    expected = torch.tensor(6e6)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)


def test_nansum_small_values() -> None:
    """Test sum with very small values."""
    x = torch.tensor([1e-6, math.nan, 2e-6, 3e-6])
    result = QF.nansum(x)
    expected = torch.tensor(6e-6)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)


def test_nansum_2d_tensors() -> None:
    """Test NaN-aware sum with 2D tensors."""
    x = torch.tensor(
        [[1.0, 2.0, math.nan], [math.nan, 3.0, 4.0], [5.0, math.nan, 6.0]]
    )

    # Test global sum
    result_global = QF.nansum(x)
    expected_global = torch.tensor(21.0)
    np.testing.assert_allclose(result_global.numpy(), expected_global.numpy())

    # Test along dimension 0
    result_dim0 = QF.nansum(x, dim=0)
    expected_dim0 = torch.tensor([6.0, 5.0, 10.0])
    np.testing.assert_allclose(result_dim0.numpy(), expected_dim0.numpy())

    # Test along dimension 1
    result_dim1 = QF.nansum(x, dim=1)
    expected_dim1 = torch.tensor([3.0, 7.0, 11.0])
    np.testing.assert_allclose(result_dim1.numpy(), expected_dim1.numpy())


def test_nansum_3d_tensors() -> None:
    """Test NaN-aware sum with 3D tensors."""
    x = torch.tensor(
        [[[1.0, 2.0], [math.nan, 4.0]], [[5.0, math.nan], [7.0, 8.0]]]
    )

    result = QF.nansum(x, dim=0)
    expected = torch.tensor([[6.0, 2.0], [7.0, 12.0]])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nansum_infinity() -> None:
    """Test sum with infinity values."""
    x = torch.tensor([1.0, math.inf, 2.0, math.nan])
    result = QF.nansum(x)
    assert torch.isinf(result)

    # Test with negative infinity
    x_neg_inf = torch.tensor([1.0, -math.inf, 2.0, math.nan])
    result_neg_inf = QF.nansum(x_neg_inf)
    assert torch.isinf(result_neg_inf) and result_neg_inf < 0


def test_nansum_mixed_special_values() -> None:
    """Test sum with mixed special values."""
    x = torch.tensor([1.0, 0.0, math.inf, math.nan, -2.0])
    result = QF.nansum(x)
    assert torch.isinf(result)


def test_nansum_different_dimensions() -> None:
    """Test sum along different dimensions."""
    x = torch.randn(3, 4, 5)
    # Add some NaNs
    x[1, 2, 3] = math.nan
    x[0, 1, 4] = math.nan

    for dim in [0, 1, 2, -1, -2, -3]:
        result = QF.nansum(x, dim=dim)
        assert (
            result.shape == x.shape[:dim] + x.shape[dim + 1 :]
            if dim >= 0
            else x.shape[: x.dim() + dim] + x.shape[x.dim() + dim + 1 :]
        )


def test_nansum_numerical_stability() -> None:
    """Test numerical stability with edge cases."""
    # Test with very small numbers
    x = torch.tensor([1e-100, 1e-100, math.nan, 1e-100])
    result = QF.nansum(x)
    expected = torch.tensor(3e-100)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)

    # Test with very large numbers
    x_large = torch.tensor([1e100, math.nan, 1e100])
    result_large = QF.nansum(x_large)
    expected_large = torch.tensor(2e100)
    np.testing.assert_allclose(
        result_large.numpy(), expected_large.numpy(), rtol=1e-6
    )


def test_nansum_batch_processing() -> None:
    """Test sum with batch processing scenarios."""
    batch_size = 3
    seq_length = 5
    features = 4

    x = torch.randn(batch_size, seq_length, features)
    # Add some NaNs
    x[0, 2, 1] = math.nan
    x[1, 0, 3] = math.nan
    x[2, 4, 0] = math.nan

    # Test batch-wise sum (across sequence and features)
    result_batch = QF.nansum(x, dim=(1, 2))
    assert result_batch.shape == (batch_size,)

    # All results should be finite (enough non-NaN values)
    assert torch.isfinite(result_batch).all()


def test_nansum_mathematical_properties() -> None:
    """Test mathematical properties of sum."""
    x = torch.tensor([1.0, math.nan, 3.0, 4.0])

    # Test linearity: nansum(ax) = a * nansum(x) when no NaN pattern changes
    a = 3.0
    x_scaled = a * x
    result1 = QF.nansum(x_scaled)
    result2 = a * QF.nansum(x)
    np.testing.assert_allclose(result1.numpy(), result2.numpy(), rtol=1e-5, atol=1e-5)

    # Test additivity with non-overlapping valid values
    x1 = torch.tensor([1.0, math.nan, 3.0])
    x2 = torch.tensor([math.nan, 2.0, 4.0])
    # Element-wise addition where at least one is valid
    x_combined = torch.where(
        x1.isnan(), x2, torch.where(x2.isnan(), x1, x1 + x2)
    )
    result_combined = QF.nansum(x_combined)
    expected_combined = torch.tensor(10.0)  # 1 + 2 + 7
    np.testing.assert_allclose(
        result_combined.numpy(), expected_combined.numpy()
    )


def test_nansum_edge_case_patterns() -> None:
    """Test edge case patterns."""
    # Pattern: NaN, non-NaN, NaN
    x1 = torch.tensor([math.nan, 5.0, math.nan])
    result1 = QF.nansum(x1)
    expected1 = torch.tensor(5.0)
    np.testing.assert_allclose(result1.numpy(), expected1.numpy())

    # Pattern: All same values with NaN
    x2 = torch.tensor([2.0, 2.0, math.nan, 2.0])
    result2 = QF.nansum(x2)
    expected2 = torch.tensor(6.0)
    np.testing.assert_allclose(result2.numpy(), expected2.numpy())

    # Pattern: Alternating NaN and values
    x3 = torch.tensor([1.0, math.nan, 2.0, math.nan, 3.0])
    result3 = QF.nansum(x3)
    expected3 = torch.tensor(6.0)
    np.testing.assert_allclose(result3.numpy(), expected3.numpy())


def test_nansum_dimension_validation() -> None:
    """Test that function works with various tensor dimensions."""
    # 1D tensor
    x1d = torch.tensor([1.0, math.nan, 3.0])
    result1d = QF.nansum(x1d)
    expected1d = torch.tensor(4.0)
    np.testing.assert_allclose(result1d.numpy(), expected1d.numpy())

    # 2D tensor
    x2d = torch.tensor([[1.0, math.nan], [3.0, 4.0]])
    result2d = QF.nansum(x2d, dim=0)
    expected2d = torch.tensor([4.0, 4.0])
    np.testing.assert_allclose(result2d.numpy(), expected2d.numpy())

    # 3D tensor
    x3d = torch.randn(2, 3, 4)
    x3d[0, 1, 2] = math.nan
    result3d = QF.nansum(x3d, dim=1)
    assert result3d.shape == (2, 4)


def test_nansum_all_dimensions_nan() -> None:
    """Test behavior when entire dimensions contain only NaN."""
    x = torch.tensor([[math.nan, math.nan], [1.0, 2.0], [math.nan, math.nan]])

    # Sum along dimension 0
    result_dim0 = QF.nansum(x, dim=0)
    expected_dim0 = torch.tensor([1.0, 2.0])
    np.testing.assert_allclose(result_dim0.numpy(), expected_dim0.numpy())

    # Sum along dimension 1
    result_dim1 = QF.nansum(x, dim=1)

    assert torch.isnan(result_dim1[0])
    assert result_dim1[1].item() == 3.0
    assert torch.isnan(result_dim1[2])


def test_nansum_precision_preservation() -> None:
    """Test that precision is maintained in sum."""
    x = torch.tensor([0.1, 0.2, math.nan, 0.3], dtype=torch.float64)
    result = QF.nansum(x)
    expected = torch.tensor(0.6, dtype=torch.float64)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-15, atol=1e-15)


def test_nansum_performance_comparison() -> None:
    """Test that nansum provides correct results for performance cases."""
    # Test with larger tensor to ensure functionality scales
    for size in [50, 100]:
        x = torch.randn(size, size)
        # Add some NaNs
        nan_indices = torch.randint(0, size, (size // 10,))
        x[nan_indices, nan_indices] = math.nan

        result = QF.nansum(x)

        # Verify basic properties
        assert torch.isfinite(
            result
        )  # Should be finite since most values are non-NaN

        # Compare with manual calculation
        valid_mask = ~x.isnan()
        manual_sum = x[valid_mask].sum()
        np.testing.assert_allclose(
            result.numpy(), manual_sum.numpy(), rtol=1e-5
        )

        # Clean up
        del x, result


def test_nansum_empty_dimension_tuple() -> None:
    """Test sum with empty dimension tuple (global sum)."""
    x = torch.tensor([[1.0, math.nan], [3.0, 4.0]])

    result = QF.nansum(x, dim=())
    expected = torch.tensor(8.0)
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nansum_complex_nan_patterns() -> None:
    """Test complex NaN patterns in multi-dimensional tensors."""
    # Create a 3D tensor with complex NaN patterns
    x = torch.tensor(
        [
            [[1.0, math.nan, 3.0], [math.nan, 5.0, math.nan]],
            [[math.nan, 2.0, math.nan], [4.0, math.nan, 6.0]],
        ]
    )

    # Test global sum
    result_global = QF.nansum(x)
    expected_global = torch.tensor(21.0)  # 1+3+5+2+4+6
    np.testing.assert_allclose(result_global.numpy(), expected_global.numpy())

    # Test along each dimension
    result_dim0 = QF.nansum(x, dim=0)
    result_dim1 = QF.nansum(x, dim=1)
    result_dim2 = QF.nansum(x, dim=2)

    assert result_dim0.shape == (2, 3)
    assert result_dim1.shape == (2, 3)
    assert result_dim2.shape == (2, 2)


def test_nansum_broadcasting_compatibility() -> None:
    """Test that nansum works correctly with broadcast-compatible shapes."""
    # Test with shapes that would broadcast in other operations
    x = torch.tensor([[1.0, math.nan, 3.0]])  # Shape: (1, 3)

    result = QF.nansum(x, dim=0)

    assert result[0].item() == 1.0
    assert torch.isnan(result[1])
    assert result[2].item() == 3.0


def test_nansum_numerical_edge_cases() -> None:
    """Test numerical edge cases."""
    # Very small numbers that might underflow
    x = torch.tensor([1e-150, 1e-150, math.nan, 1e-150])
    result = QF.nansum(x)
    # Result might be 0 due to underflow in float32, but should be finite
    assert torch.isfinite(result)

    # Very large numbers that might overflow
    x_large = torch.tensor([1e100, math.nan, 1e100, 1e100])
    result_large = QF.nansum(x_large)
    # Result might overflow to inf
    assert torch.isinf(result_large) or torch.isfinite(result_large)


def test_nansum_dtype_preservation() -> None:
    """Test that nansum preserves input dtype."""

    x = torch.tensor([1.0, math.nan, 3.0, 4.0])
    generic_test_dtype_preservation(QF.nansum, x)


def test_nansum_device_preservation() -> None:
    """Test that nansum preserves input device."""

    x = torch.tensor([1.0, math.nan, 3.0, 4.0])
    generic_test_device_preservation(QF.nansum, x)


def test_nansum_memory_efficiency() -> None:
    """Test memory efficiency of nansum."""

    generic_test_memory_efficiency(QF.nansum)


def test_nansum_single_element() -> None:
    """Test nansum with single element tensor."""
    x_single = torch.tensor([42.0])
    result = QF.nansum(x_single)
    # nansum reduces to scalar
    assert result.shape == torch.Size([])
    assert result.item() == 42.0


def test_nansum_empty_tensor() -> None:
    """Test nansum with empty tensor."""
    x_empty = torch.empty(0)
    result = QF.nansum(x_empty)
    # nansum of empty tensor returns NaN (scalar)
    assert result.shape == torch.Size([])
    assert torch.isnan(result)


def test_nansum_consistency() -> None:
    """Test that multiple calls to nansum produce same result."""

    x = torch.tensor([1.0, math.nan, 3.0, 4.0])
    generic_test_consistency(QF.nansum, x)
