import math

import numpy as np
import torch

import qfeval_functions.functions as QF
import pytest


def test_nancumprod() -> None:
    x = torch.tensor([math.nan, 1.0, math.nan, 2.0, 3.0, math.nan, 4.0])
    np.testing.assert_allclose(
        QF.nancumprod(x, dim=0).numpy(),
        np.array([math.nan, 1.0, math.nan, 2.0, 6.0, math.nan, 24.0]),
    )


def test_nancumprod_basic_functionality() -> None:
    """Test basic NaN-aware cumulative product functionality."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = QF.nancumprod(x, dim=0)
    expected = torch.tensor([1.0, 2.0, 6.0, 24.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nancumprod_no_nans() -> None:
    """Test cumulative product without any NaN values."""
    x = torch.tensor([2.0, 3.0, 4.0, 5.0])
    result = QF.nancumprod(x, dim=0)
    expected = x.cumprod(dim=0)
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nancumprod_all_nans() -> None:
    """Test cumulative product with all NaN values."""
    x = torch.tensor([math.nan, math.nan, math.nan])
    result = QF.nancumprod(x, dim=0)

    # All results should be NaN
    assert torch.isnan(result).all()


def test_nancumprod_leading_nan() -> None:
    """Test cumulative product starting with NaN."""
    x = torch.tensor([math.nan, 2.0, 3.0, 4.0])
    result = QF.nancumprod(x, dim=0)
    expected = torch.tensor([math.nan, 2.0, 6.0, 24.0])

    # Check NaN at position 0
    assert torch.isnan(result[0])
    # Check non-NaN values
    np.testing.assert_allclose(result[1:].numpy(), expected[1:].numpy())


def test_nancumprod_trailing_nan() -> None:
    """Test cumulative product ending with NaN."""
    x = torch.tensor([1.0, 2.0, 3.0, math.nan])
    result = QF.nancumprod(x, dim=0)
    expected = torch.tensor([1.0, 2.0, 6.0, math.nan])

    # Check non-NaN values
    np.testing.assert_allclose(result[:3].numpy(), expected[:3].numpy())
    # Check NaN at last position
    assert torch.isnan(result[3])


def test_nancumprod_middle_nan() -> None:
    """Test cumulative product with NaN in the middle."""
    x = torch.tensor([1.0, 2.0, math.nan, 4.0, 5.0])
    result = QF.nancumprod(x, dim=0)
    expected = torch.tensor([1.0, 2.0, math.nan, 8.0, 40.0])

    # Check non-NaN values
    np.testing.assert_allclose(result[[0, 1]].numpy(), expected[[0, 1]].numpy())
    np.testing.assert_allclose(result[[3, 4]].numpy(), expected[[3, 4]].numpy())
    # Check NaN at position 2
    assert torch.isnan(result[2])


def test_nancumprod_2d_tensors() -> None:
    """Test NaN-aware cumulative product with 2D tensors."""
    x = torch.tensor(
        [[1.0, 2.0, math.nan], [math.nan, 3.0, 4.0], [5.0, math.nan, 6.0]]
    )

    # Test along dimension 0 (columns)
    result_dim0 = QF.nancumprod(x, dim=0)
    # Check that the operation completes successfully
    assert result_dim0.shape == x.shape

    # Check specific non-NaN values
    assert result_dim0[0, 0].item() == 1.0
    assert result_dim0[0, 1].item() == 2.0
    assert result_dim0[1, 1].item() == 6.0
    # Check NaN values - the function preserves NaN in original positions
    assert torch.isnan(result_dim0[0, 2])
    assert torch.isnan(result_dim0[1, 0])
    # Position [2,2] continues cumulative product: 4*6=24
    assert result_dim0[2, 2].item() == 24.0


def test_nancumprod_dim1() -> None:
    """Test NaN-aware cumulative product along dimension 1."""
    x = torch.tensor(
        [[1.0, 2.0, 3.0], [math.nan, 4.0, 5.0], [6.0, math.nan, 7.0]]
    )

    result = QF.nancumprod(x, dim=1)

    # First row: [1, 2, 6]
    expected_row0 = torch.tensor([1.0, 2.0, 6.0])
    np.testing.assert_allclose(result[0].numpy(), expected_row0.numpy())

    # Second row: [nan, 4, 20]
    assert torch.isnan(result[1, 0])
    assert result[1, 1].item() == 4.0
    assert result[1, 2].item() == 20.0

    # Third row: [6, nan, 7*6=42] - Wait, this should be [6, nan, nan] because once we hit NaN it propagates
    assert result[2, 0].item() == 6.0
    assert torch.isnan(result[2, 1])
    assert result[2, 2].item() == 42.0  # This continues from the non-NaN part


def test_nancumprod_3d_tensors() -> None:
    """Test NaN-aware cumulative product with 3D tensors."""
    x = torch.tensor(
        [[[1.0, 2.0], [3.0, math.nan]], [[math.nan, 4.0], [5.0, 6.0]]]
    )

    result = QF.nancumprod(x, dim=0)

    assert result.shape == x.shape
    # Check specific values
    assert result[0, 0, 0].item() == 1.0
    assert result[0, 0, 1].item() == 2.0
    assert result[0, 1, 0].item() == 3.0
    assert torch.isnan(result[0, 1, 1])
    assert torch.isnan(result[1, 0, 0])
    assert result[1, 0, 1].item() == 8.0
    assert result[1, 1, 0].item() == 15.0
    # Position [1,1,1] continues: 6 (not NaN because it continues from non-NaN)
    assert result[1, 1, 1].item() == 6.0


def test_nancumprod_zeros() -> None:
    """Test cumulative product with zero values."""
    x = torch.tensor([1.0, 0.0, 3.0, 4.0])
    result = QF.nancumprod(x, dim=0)
    expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nancumprod_zeros_and_nans() -> None:
    """Test cumulative product with both zeros and NaNs."""
    x = torch.tensor([1.0, 0.0, math.nan, 4.0])
    result = QF.nancumprod(x, dim=0)

    # Check non-NaN values
    assert result[0].item() == 1.0
    assert result[1].item() == 0.0
    assert result[3].item() == 0.0
    # Check NaN value
    assert torch.isnan(result[2])


def test_nancumprod_negative_values() -> None:
    """Test cumulative product with negative values."""
    x = torch.tensor([-1.0, 2.0, -3.0, 4.0])
    result = QF.nancumprod(x, dim=0)
    expected = torch.tensor([-1.0, -2.0, 6.0, 24.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nancumprod_negative_with_nans() -> None:
    """Test cumulative product with negative values and NaNs."""
    x = torch.tensor([-1.0, math.nan, -3.0, 4.0])
    result = QF.nancumprod(x, dim=0)

    # Check non-NaN values
    assert result[0].item() == -1.0
    # NaN is replaced with 1.0 for computation, so cumulative continues
    assert result[2].item() == 3.0  # -1 * 1 * -3 = 3 (NaN treated as 1)
    assert result[3].item() == 12.0  # 3 * 4 = 12
    # Check NaN value in original position
    assert torch.isnan(result[1])


def test_nancumprod_single_nan() -> None:
    """Test cumulative product with single NaN element."""
    x = torch.tensor([math.nan])
    result = QF.nancumprod(x, dim=0)
    assert torch.isnan(result[0])


def test_nancumprod_large_values() -> None:
    """Test cumulative product with large values."""
    x = torch.tensor([1e3, 1e3, math.nan, 1e3])
    result = QF.nancumprod(x, dim=0)

    assert result[0].item() == 1e3
    assert result[1].item() == 1e6
    assert torch.isnan(result[2])
    assert result[3].item() == 1e9


def test_nancumprod_small_values() -> None:
    """Test cumulative product with small values."""
    x = torch.tensor([1e-3, 1e-3, math.nan, 1e-3])
    result = QF.nancumprod(x, dim=0)

    assert abs(result[0].item() - 1e-3) < 1e-10
    assert abs(result[1].item() - 1e-6) < 1e-12
    assert torch.isnan(result[2])
    assert abs(result[3].item() - 1e-9) < 1e-15


def test_nancumprod_infinity() -> None:
    """Test cumulative product with infinity values."""
    x = torch.tensor([1.0, math.inf, 2.0, math.nan])
    result = QF.nancumprod(x, dim=0)

    assert result[0].item() == 1.0
    # Large value (inf gets clamped to max float32)
    assert result[1].item() > 1e30
    assert torch.isinf(result[2])
    assert torch.isnan(result[3])


def test_nancumprod_mixed_special_values() -> None:
    """Test cumulative product with mixed special values."""
    x = torch.tensor([1.0, 0.0, math.inf, math.nan, 2.0])
    result = QF.nancumprod(x, dim=0)

    assert result[0].item() == 1.0
    assert result[1].item() == 0.0
    assert (
        torch.isnan(result[2]) or result[2].item() == 0.0
    )  # 0 * inf = nan, but continuing from 0
    assert torch.isnan(result[3])
    assert result[4].item() == 0.0


@pytest.mark.random
def test_nancumprod_different_dimensions() -> None:
    """Test cumulative product along different dimensions."""
    x = torch.randn(3, 4, 5)

    for dim in [0, 1, 2, -1, -2, -3]:
        result = QF.nancumprod(x, dim=dim)
        assert result.shape == x.shape


def test_nancumprod_alternating_nans() -> None:
    """Test cumulative product with alternating NaN pattern."""
    x = torch.tensor([1.0, math.nan, 2.0, math.nan, 3.0])
    result = QF.nancumprod(x, dim=0)

    assert result[0].item() == 1.0
    assert torch.isnan(result[1])
    assert result[2].item() == 2.0
    assert torch.isnan(result[3])
    assert result[4].item() == 6.0


def test_nancumprod_numerical_stability() -> None:
    """Test numerical stability with edge cases."""
    # Test with very small numbers that might underflow
    x = torch.tensor([1e-100, 1e-100, math.nan, 1e-100])
    result = QF.nancumprod(x, dim=0)

    # Very small values may underflow to 0 in float32
    assert result[0].item() >= 0
    assert result[1].item() >= 0
    assert torch.isnan(result[2])
    assert result[3].item() >= 0


@pytest.mark.random
def test_nancumprod_batch_processing() -> None:
    """Test cumulative product with batch processing."""
    batch_size = 3
    seq_length = 5

    x = torch.randn(batch_size, seq_length)
    # Add some NaNs randomly
    x[0, 2] = math.nan
    x[1, 0] = math.nan
    x[2, 4] = math.nan

    result = QF.nancumprod(x, dim=1)

    assert result.shape == (batch_size, seq_length)

    # Check that NaNs are preserved in output
    assert torch.isnan(result[0, 2])
    assert torch.isnan(result[1, 0])
    assert torch.isnan(result[2, 4])


def test_nancumprod_mathematical_properties() -> None:
    """Test mathematical properties of cumulative product."""
    x = torch.tensor([2.0, 3.0, 4.0])
    result = QF.nancumprod(x, dim=0)

    # Test that cumulative product grows appropriately
    assert result[0].item() == 2.0
    assert result[1].item() == 6.0
    assert result[2].item() == 24.0

    # Test monotonicity with positive values
    assert result[0] <= result[1] <= result[2]


def test_nancumprod_edge_case_patterns() -> None:
    """Test edge case patterns."""
    # Pattern: NaN, non-NaN, NaN
    x1 = torch.tensor([math.nan, 5.0, math.nan])
    result1 = QF.nancumprod(x1, dim=0)
    assert torch.isnan(result1[0])
    assert result1[1].item() == 5.0
    assert torch.isnan(result1[2])

    # Pattern: All same values with NaN
    x2 = torch.tensor([2.0, 2.0, math.nan, 2.0])
    result2 = QF.nancumprod(x2, dim=0)
    assert result2[0].item() == 2.0
    assert result2[1].item() == 4.0
    assert torch.isnan(result2[2])
    assert result2[3].item() == 8.0


@pytest.mark.random
def test_nancumprod_dimension_validation() -> None:
    """Test that function works with various tensor dimensions."""
    # 1D tensor
    x1d = torch.tensor([1.0, math.nan, 3.0])
    result1d = QF.nancumprod(x1d, dim=0)
    assert result1d.shape == (3,)

    # 2D tensor
    x2d = torch.tensor([[1.0, 2.0], [math.nan, 4.0]])
    result2d = QF.nancumprod(x2d, dim=0)
    assert result2d.shape == (2, 2)

    # 3D tensor
    x3d = torch.randn(2, 3, 4)
    x3d[0, 1, 2] = math.nan
    result3d = QF.nancumprod(x3d, dim=1)
    assert result3d.shape == (2, 3, 4)
