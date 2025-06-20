import math

import numpy as np
import torch

import qfeval_functions.functions as QF
import pytest


def test_nancumsum() -> None:
    x = torch.tensor([math.nan, 1.0, math.nan, 2.0, 3.0, math.nan, 4.0])
    np.testing.assert_allclose(
        QF.nancumsum(x, dim=0).numpy(),
        np.array([math.nan, 1.0, math.nan, 3.0, 6.0, math.nan, 10.0]),
    )


def test_nancumsum_basic_functionality() -> None:
    """Test basic NaN-aware cumulative sum functionality."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = QF.nancumsum(x, dim=0)
    expected = torch.tensor([1.0, 3.0, 6.0, 10.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nancumsum_no_nans() -> None:
    """Test cumulative sum without any NaN values."""
    x = torch.tensor([2.0, 3.0, 4.0, 5.0])
    result = QF.nancumsum(x, dim=0)
    expected = x.cumsum(dim=0)
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nancumsum_all_nans() -> None:
    """Test cumulative sum with all NaN values."""
    x = torch.tensor([math.nan, math.nan, math.nan])
    result = QF.nancumsum(x, dim=0)

    # All results should be NaN
    assert torch.isnan(result).all()


def test_nancumsum_leading_nan() -> None:
    """Test cumulative sum starting with NaN."""
    x = torch.tensor([math.nan, 2.0, 3.0, 4.0])
    result = QF.nancumsum(x, dim=0)
    expected = torch.tensor([math.nan, 2.0, 5.0, 9.0])

    # Check NaN at position 0
    assert torch.isnan(result[0])
    # Check non-NaN values
    np.testing.assert_allclose(result[1:].numpy(), expected[1:].numpy())


def test_nancumsum_trailing_nan() -> None:
    """Test cumulative sum ending with NaN."""
    x = torch.tensor([1.0, 2.0, 3.0, math.nan])
    result = QF.nancumsum(x, dim=0)
    expected = torch.tensor([1.0, 3.0, 6.0, math.nan])

    # Check non-NaN values
    np.testing.assert_allclose(result[:3].numpy(), expected[:3].numpy())
    # Check NaN at last position
    assert torch.isnan(result[3])


def test_nancumsum_middle_nan() -> None:
    """Test cumulative sum with NaN in the middle."""
    x = torch.tensor([1.0, 2.0, math.nan, 4.0, 5.0])
    result = QF.nancumsum(x, dim=0)
    expected = torch.tensor([1.0, 3.0, math.nan, 7.0, 12.0])

    # Check non-NaN values
    np.testing.assert_allclose(result[[0, 1]].numpy(), expected[[0, 1]].numpy())
    np.testing.assert_allclose(result[[3, 4]].numpy(), expected[[3, 4]].numpy())
    # Check NaN at position 2
    assert torch.isnan(result[2])


def test_nancumsum_2d_tensors() -> None:
    """Test NaN-aware cumulative sum with 2D tensors."""
    x = torch.tensor(
        [[1.0, 2.0, math.nan], [math.nan, 3.0, 4.0], [5.0, math.nan, 6.0]]
    )

    # Test along dimension 0 (columns)
    result_dim0 = QF.nancumsum(x, dim=0)

    # Check specific non-NaN values
    assert result_dim0[0, 0].item() == 1.0
    assert result_dim0[0, 1].item() == 2.0
    assert result_dim0[1, 1].item() == 5.0  # 2 + 3
    assert (
        result_dim0[2, 0].item() == 6.0
    )  # 1 + 5 (continues sum, treating NaN as 0)
    # Check NaN values
    assert torch.isnan(result_dim0[0, 2])
    assert torch.isnan(result_dim0[1, 0])


def test_nancumsum_dim1() -> None:
    """Test NaN-aware cumulative sum along dimension 1."""
    x = torch.tensor(
        [[1.0, 2.0, 3.0], [math.nan, 4.0, 5.0], [6.0, math.nan, 7.0]]
    )

    result = QF.nancumsum(x, dim=1)

    # First row: [1, 3, 6]
    expected_row0 = torch.tensor([1.0, 3.0, 6.0])
    np.testing.assert_allclose(result[0].numpy(), expected_row0.numpy())

    # Second row: [nan, 4, 9]
    assert torch.isnan(result[1, 0])
    assert result[1, 1].item() == 4.0
    assert result[1, 2].item() == 9.0

    # Third row: [6, nan, 13] (continues from 6, treating NaN as 0)
    assert result[2, 0].item() == 6.0
    assert torch.isnan(result[2, 1])
    assert result[2, 2].item() == 13.0


def test_nancumsum_3d_tensors() -> None:
    """Test NaN-aware cumulative sum with 3D tensors."""
    x = torch.tensor(
        [[[1.0, 2.0], [3.0, math.nan]], [[math.nan, 4.0], [5.0, 6.0]]]
    )

    result = QF.nancumsum(x, dim=0)

    assert result.shape == x.shape
    # Check specific values
    assert result[0, 0, 0].item() == 1.0
    assert result[0, 0, 1].item() == 2.0
    assert result[0, 1, 0].item() == 3.0
    assert torch.isnan(result[0, 1, 1])
    assert torch.isnan(result[1, 0, 0])
    assert result[1, 0, 1].item() == 6.0  # 2 + 4
    assert result[1, 1, 0].item() == 8.0  # 3 + 5
    assert result[1, 1, 1].item() == 6.0  # 0 + 6 (NaN treated as 0)


def test_nancumsum_zeros() -> None:
    """Test cumulative sum with zero values."""
    x = torch.tensor([1.0, 0.0, 3.0, 4.0])
    result = QF.nancumsum(x, dim=0)
    expected = torch.tensor([1.0, 1.0, 4.0, 8.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nancumsum_zeros_and_nans() -> None:
    """Test cumulative sum with both zeros and NaNs."""
    x = torch.tensor([1.0, 0.0, math.nan, 4.0])
    result = QF.nancumsum(x, dim=0)

    # Check non-NaN values
    assert result[0].item() == 1.0
    assert result[1].item() == 1.0
    assert result[3].item() == 5.0
    # Check NaN value
    assert torch.isnan(result[2])


def test_nancumsum_negative_values() -> None:
    """Test cumulative sum with negative values."""
    x = torch.tensor([-1.0, 2.0, -3.0, 4.0])
    result = QF.nancumsum(x, dim=0)
    expected = torch.tensor([-1.0, 1.0, -2.0, 2.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nancumsum_negative_with_nans() -> None:
    """Test cumulative sum with negative values and NaNs."""
    x = torch.tensor([-1.0, math.nan, -3.0, 4.0])
    result = QF.nancumsum(x, dim=0)

    # Check non-NaN values
    assert result[0].item() == -1.0
    assert result[2].item() == -4.0  # -1 + 0 + (-3) = -4 (NaN treated as 0)
    assert result[3].item() == 0.0  # -4 + 4 = 0
    # Check NaN value
    assert torch.isnan(result[1])


def test_nancumsum_single_nan() -> None:
    """Test cumulative sum with single NaN element."""
    x = torch.tensor([math.nan])
    result = QF.nancumsum(x, dim=0)
    assert torch.isnan(result[0])


def test_nancumsum_large_values() -> None:
    """Test cumulative sum with large values."""
    x = torch.tensor([1e6, 1e6, math.nan, 1e6])
    result = QF.nancumsum(x, dim=0)

    assert result[0].item() == 1e6
    assert result[1].item() == 2e6
    assert torch.isnan(result[2])
    assert result[3].item() == 3e6  # 2e6 + 0 + 1e6 = 3e6


def test_nancumsum_small_values() -> None:
    """Test cumulative sum with small values."""
    x = torch.tensor([1e-6, 1e-6, math.nan, 1e-6])
    result = QF.nancumsum(x, dim=0)

    assert abs(result[0].item() - 1e-6) < 1e-10
    assert abs(result[1].item() - 2e-6) < 1e-10
    assert torch.isnan(result[2])
    assert abs(result[3].item() - 3e-6) < 1e-10


def test_nancumsum_infinity() -> None:
    """Test cumulative sum with infinity values."""
    x = torch.tensor([1.0, math.inf, 2.0, math.nan])
    result = QF.nancumsum(x, dim=0)

    assert result[0].item() == 1.0
    # inf gets clamped to max float32 when nan_to_num() is used
    assert result[1].item() > 1e30 or torch.isinf(result[1])
    assert result[2].item() > 1e30 or torch.isinf(result[2])
    assert torch.isnan(result[3])


def test_nancumsum_mixed_special_values() -> None:
    """Test cumulative sum with mixed special values."""
    x = torch.tensor([1.0, 0.0, math.inf, math.nan, 2.0])
    result = QF.nancumsum(x, dim=0)

    assert result[0].item() == 1.0
    assert result[1].item() == 1.0
    # inf gets clamped to max float32 when nan_to_num() is used
    assert result[2].item() > 1e30 or torch.isinf(result[2])
    assert torch.isnan(result[3])
    assert result[4].item() > 1e30 or torch.isinf(
        result[4]
    )  # large + 2 = large


@pytest.mark.random
def test_nancumsum_different_dimensions() -> None:
    """Test cumulative sum along different dimensions."""
    x = torch.randn(3, 4, 5)

    for dim in [0, 1, 2, -1, -2, -3]:
        result = QF.nancumsum(x, dim=dim)
        assert result.shape == x.shape


def test_nancumsum_alternating_nans() -> None:
    """Test cumulative sum with alternating NaN pattern."""
    x = torch.tensor([1.0, math.nan, 2.0, math.nan, 3.0])
    result = QF.nancumsum(x, dim=0)

    assert result[0].item() == 1.0
    assert torch.isnan(result[1])
    assert result[2].item() == 3.0  # 1 + 0 + 2 = 3 (NaN treated as 0)
    assert torch.isnan(result[3])
    assert result[4].item() == 6.0  # 3 + 0 + 3 = 6


def test_nancumsum_numerical_stability() -> None:
    """Test numerical stability with edge cases."""
    # Test with very small numbers
    x = torch.tensor([1e-100, 1e-100, math.nan, 1e-100])
    result = QF.nancumsum(x, dim=0)

    assert result[0].item() >= 0
    assert result[1].item() >= result[0].item()
    assert torch.isnan(result[2])
    assert result[3].item() >= result[1].item()


@pytest.mark.random
def test_nancumsum_batch_processing() -> None:
    """Test cumulative sum with batch processing."""
    batch_size = 3
    seq_length = 5

    x = torch.randn(batch_size, seq_length)
    # Add some NaNs randomly
    x[0, 2] = math.nan
    x[1, 0] = math.nan
    x[2, 4] = math.nan

    result = QF.nancumsum(x, dim=1)

    assert result.shape == (batch_size, seq_length)

    # Check that NaNs are preserved in output
    assert torch.isnan(result[0, 2])
    assert torch.isnan(result[1, 0])
    assert torch.isnan(result[2, 4])


def test_nancumsum_mathematical_properties() -> None:
    """Test mathematical properties of cumulative sum."""
    x = torch.tensor([2.0, 3.0, 4.0])
    result = QF.nancumsum(x, dim=0)

    # Test that cumulative sum grows appropriately
    assert result[0].item() == 2.0
    assert result[1].item() == 5.0
    assert result[2].item() == 9.0

    # Test monotonicity with positive values
    assert result[0] <= result[1] <= result[2]


def test_nancumsum_edge_case_patterns() -> None:
    """Test edge case patterns."""
    # Pattern: NaN, non-NaN, NaN
    x1 = torch.tensor([math.nan, 5.0, math.nan])
    result1 = QF.nancumsum(x1, dim=0)
    assert torch.isnan(result1[0])
    assert result1[1].item() == 5.0
    assert torch.isnan(result1[2])

    # Pattern: All same values with NaN
    x2 = torch.tensor([2.0, 2.0, math.nan, 2.0])
    result2 = QF.nancumsum(x2, dim=0)
    assert result2[0].item() == 2.0
    assert result2[1].item() == 4.0
    assert torch.isnan(result2[2])
    assert result2[3].item() == 6.0


@pytest.mark.random
def test_nancumsum_dimension_validation() -> None:
    """Test that function works with various tensor dimensions."""
    # 1D tensor
    x1d = torch.tensor([1.0, math.nan, 3.0])
    result1d = QF.nancumsum(x1d, dim=0)
    assert result1d.shape == (3,)

    # 2D tensor
    x2d = torch.tensor([[1.0, 2.0], [math.nan, 4.0]])
    result2d = QF.nancumsum(x2d, dim=0)
    assert result2d.shape == (2, 2)

    # 3D tensor
    x3d = torch.randn(2, 3, 4)
    x3d[0, 1, 2] = math.nan
    result3d = QF.nancumsum(x3d, dim=1)
    assert result3d.shape == (2, 3, 4)


def test_nancumsum_accumulation_behavior() -> None:
    """Test specific accumulation behavior with NaNs."""
    # Test that sum continues correctly after NaN positions
    x = torch.tensor([10.0, math.nan, 20.0, 30.0])
    result = QF.nancumsum(x, dim=0)

    assert result[0].item() == 10.0
    assert torch.isnan(result[1])
    assert result[2].item() == 30.0  # 10 + 0 + 20 = 30 (NaN treated as 0)
    assert result[3].item() == 60.0  # 30 + 30 = 60


def test_nancumsum_multiple_consecutive_nans() -> None:
    """Test behavior with multiple consecutive NaNs."""
    x = torch.tensor([1.0, math.nan, math.nan, 2.0, 3.0])
    result = QF.nancumsum(x, dim=0)

    assert result[0].item() == 1.0
    assert torch.isnan(result[1])
    assert torch.isnan(result[2])
    assert result[3].item() == 3.0  # 1 + 0 + 0 + 2 = 3
    assert result[4].item() == 6.0  # 3 + 3 = 6


def test_nancumsum_precision_preservation() -> None:
    """Test that precision is maintained in cumulative sum."""
    x = torch.tensor([0.1, 0.2, math.nan, 0.3], dtype=torch.float64)
    result = QF.nancumsum(x, dim=0)

    assert abs(result[0].item() - 0.1) < 1e-15
    assert abs(result[1].item() - 0.3) < 1e-15
    assert torch.isnan(result[2])
    assert abs(result[3].item() - 0.6) < 1e-15


@pytest.mark.random
def test_nancumsum_performance_comparison() -> None:
    """Test that nancumsum provides correct results for performance cases."""
    # Test with larger tensor to ensure functionality scales
    for size in [50, 100]:
        x = torch.randn(size)
        # Add some NaNs
        nan_indices = torch.randint(0, size, (size // 10,))
        x[nan_indices] = math.nan

        result = QF.nancumsum(x, dim=0)

        # Verify basic properties
        assert result.shape == x.shape
        assert torch.isnan(result[nan_indices]).all()

        # Clean up
        del x, result
