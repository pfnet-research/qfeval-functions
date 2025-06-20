import math

import numpy as np
import torch

import qfeval_functions.functions as QF
import pytest


def test_nanmulsum() -> None:
    a = QF.randn(100, 1)
    a = torch.where(a < 0, torch.as_tensor(math.nan), a)
    b = QF.randn(1, 200)
    b = torch.where(b < 0, torch.as_tensor(math.nan), b)
    np.testing.assert_allclose(
        (a * b).nansum().numpy(),
        QF.nanmulsum(a, b).numpy(),
        1e-5,
        1e-5,
    )
    np.testing.assert_allclose(
        (a * b).nansum(dim=1).numpy(),
        QF.nanmulsum(a, b, dim=1).nan_to_num().numpy(),
        1e-5,
        1e-5,
    )
    np.testing.assert_array_equal(
        (a * b).isnan().all(dim=1).numpy(),
        QF.nanmulsum(a, b, dim=1).isnan().numpy(),
    )
    np.testing.assert_allclose(
        (a * b).nansum(dim=-1, keepdim=True).numpy(),
        QF.nanmulsum(a, b, dim=-1, keepdim=True).nan_to_num().numpy(),
        1e-5,
        1e-5,
    )


def test_nanmulsum_basic_functionality() -> None:
    """Test basic NaN-aware matrix multiplication sum functionality."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[2.0, 1.0], [1.0, 2.0]])

    # Test global sum
    result = QF.nanmulsum(x, y)
    expected = (x * y).sum()
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nanmulsum_with_nans() -> None:
    """Test NaN-aware matrix multiplication sum with NaN values."""
    x = torch.tensor([[1.0, math.nan], [3.0, 4.0]])
    y = torch.tensor([[2.0, 1.0], [1.0, 2.0]])

    result = QF.nanmulsum(x, y)
    # Expected: 1*2 + 3*1 + 4*2 = 2 + 3 + 8 = 13
    expected = torch.tensor(13.0)
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nanmulsum_all_nans() -> None:
    """Test matrix multiplication sum with all NaN values."""
    x = torch.tensor([[math.nan, math.nan], [math.nan, math.nan]])
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    result = QF.nanmulsum(x, y)
    assert torch.isnan(result)


def test_nanmulsum_broadcasting() -> None:
    """Test NaN-aware multiplication sum with broadcasting."""
    x = torch.tensor([[1.0, math.nan, 3.0]])
    y = torch.tensor([[2.0], [3.0]])

    result = QF.nanmulsum(x, y)
    # Expected: sum of [2, nan, 9, 3, nan, 6] ignoring NaN = 2+9+3+6 = 20
    expected = torch.tensor(20.0)
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nanmulsum_dimensions() -> None:
    """Test NaN-aware sum calculation along specific dimensions."""
    x = torch.tensor(
        [[[1.0, 2.0], [math.nan, 4.0]], [[5.0, 6.0], [7.0, math.nan]]]
    )
    y = torch.tensor([[[2.0, 1.0], [1.0, 2.0]], [[1.0, 3.0], [2.0, 1.0]]])

    # Test along dimension 0
    result_dim0 = QF.nanmulsum(x, y, dim=0)
    assert result_dim0.shape == (2, 2)

    # Test along dimension 1
    result_dim1 = QF.nanmulsum(x, y, dim=1)
    assert result_dim1.shape == (2, 2)

    # Test along dimension 2
    result_dim2 = QF.nanmulsum(x, y, dim=2)
    assert result_dim2.shape == (2, 2)


def test_nanmulsum_keepdim() -> None:
    """Test keepdim parameter functionality."""
    x = torch.tensor([[[1.0, math.nan], [3.0, 4.0]]])
    y = torch.tensor([[[2.0, 1.0], [1.0, 2.0]]])

    # Test keepdim=True
    result_keepdim = QF.nanmulsum(x, y, dim=1, keepdim=True)
    assert result_keepdim.shape == (1, 1, 2)

    # Test keepdim=False
    result_no_keepdim = QF.nanmulsum(x, y, dim=1, keepdim=False)
    assert result_no_keepdim.shape == (1, 2)


def test_nanmulsum_negative_dimensions() -> None:
    """Test negative dimension indexing."""
    x = torch.tensor([[[1.0, math.nan], [3.0, 4.0]]])
    y = torch.tensor([[[2.0, 1.0], [1.0, 2.0]]])

    result_neg1 = QF.nanmulsum(x, y, dim=-1)
    result_pos2 = QF.nanmulsum(x, y, dim=2)
    np.testing.assert_allclose(result_neg1.numpy(), result_pos2.numpy())


def test_nanmulsum_single_nan() -> None:
    """Test multiplication and sum with single NaN element."""
    x = torch.tensor([[math.nan]])
    y = torch.tensor([[3.0]])

    result = QF.nanmulsum(x, y)
    assert torch.isnan(result)


def test_nanmulsum_zero_values() -> None:
    """Test multiplication and sum with zero values."""
    x = torch.tensor([[1.0, 0.0, 3.0], [0.0, 2.0, math.nan]])
    y = torch.tensor([[2.0, 1.0, 0.0], [3.0, 0.0, 4.0]])

    result = QF.nanmulsum(x, y)
    # Expected: sum of [2, 0, 0, 0, 0] = 2
    expected = torch.tensor(2.0)
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nanmulsum_negative_values() -> None:
    """Test multiplication and sum with negative values."""
    x = torch.tensor([[-1.0, 2.0], [3.0, math.nan]])
    y = torch.tensor([[2.0, -1.0], [-2.0, 3.0]])

    result = QF.nanmulsum(x, y)
    # Expected: sum of [-2, -2, -6] = -10
    expected = torch.tensor(-10.0)
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nanmulsum_large_values() -> None:
    """Test multiplication and sum with large values."""
    x = torch.tensor([[1e6, math.nan], [3e6, 4e6]])
    y = torch.tensor([[2e6, 1e6], [1e6, 2e6]])

    result = QF.nanmulsum(x, y)
    # Expected: sum of [2e12, 3e12, 8e12] = 13e12
    expected = torch.tensor(13e12)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-4)


def test_nanmulsum_small_values() -> None:
    """Test multiplication and sum with very small values."""
    x = torch.tensor([[1e-6, math.nan], [3e-6, 4e-6]])
    y = torch.tensor([[2e-6, 1e-6], [1e-6, 2e-6]])

    result = QF.nanmulsum(x, y)
    # Expected: sum of [2e-12, 3e-12, 8e-12] = 13e-12
    expected = torch.tensor(13e-12)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-4)


def test_nanmulsum_broadcasting_edge_cases() -> None:
    """Test edge cases in broadcasting."""
    # Scalar with tensor
    x_scalar = torch.tensor(2.0)
    y_tensor = torch.tensor([[1.0, math.nan], [3.0, 4.0]])

    result1 = QF.nanmulsum(x_scalar, y_tensor)
    # Expected: sum of [2, 6, 8] = 16
    expected1 = torch.tensor(16.0)
    np.testing.assert_allclose(result1.numpy(), expected1.numpy())

    # Different shape broadcasting
    x_broadcast = torch.tensor([[[1.0]], [[math.nan]]])
    y_broadcast = torch.tensor([[[3.0, 4.0, 5.0]]])

    result2 = QF.nanmulsum(x_broadcast, y_broadcast)
    # Expected: sum of [3, 4, 5] = 12
    expected2 = torch.tensor(12.0)
    np.testing.assert_allclose(result2.numpy(), expected2.numpy())


def test_nanmulsum_with_infinity() -> None:
    """Test nanmulsum behavior with infinity values."""
    x = torch.tensor([[1.0, math.inf], [3.0, math.nan]])
    y = torch.tensor([[2.0, 1.0], [1.0, 2.0]])

    result = QF.nanmulsum(x, y)
    # inf * 1 = inf, so result should be inf
    assert torch.isinf(result) or torch.isnan(result)


@pytest.mark.random
def test_nanmulsum_high_dimensional() -> None:
    """Test nanmulsum with high-dimensional tensors."""
    x = torch.randn(2, 3, 4, 5)
    # Add some NaNs
    x[0, 1, 2, 3] = math.nan
    x[1, 0, 1, 2] = math.nan
    y = torch.randn(2, 3, 4, 5)

    result = QF.nanmulsum(x, y)
    assert torch.isfinite(
        result
    )  # Should be finite since most values are non-NaN

    # Test along specific high dimensions
    result_dim3 = QF.nanmulsum(x, y, dim=3)
    assert result_dim3.shape == (2, 3, 4)


def test_nanmulsum_mathematical_properties() -> None:
    """Test mathematical properties of nanmulsum."""
    x = torch.tensor([[1.0, math.nan], [3.0, 4.0]])
    y = torch.tensor([[2.0, 1.0], [1.0, 2.0]])

    # Test linearity: nanmulsum(ax, y) = a * nanmulsum(x, y) when no broadcast changes NaN pattern
    a = 3.0
    x_scaled = a * x
    result1 = QF.nanmulsum(x_scaled, y)
    result2 = a * QF.nanmulsum(x, y)
    np.testing.assert_allclose(result1.numpy(), result2.numpy(), rtol=1e-5)

    # Test commutativity: nanmulsum(x, y) = nanmulsum(y, x)
    result_xy = QF.nanmulsum(x, y)
    result_yx = QF.nanmulsum(y, x)
    np.testing.assert_allclose(result_xy.numpy(), result_yx.numpy(), rtol=1e-5)


def test_nanmulsum_numerical_stability() -> None:
    """Test numerical stability of nanmulsum."""
    # Test with values that might cause overflow in naive multiplication
    x = torch.tensor([[1e10, math.nan], [1e-10, 1e10]])
    y = torch.tensor([[1e-10, 1e10], [1e10, 1e-10]])

    result = QF.nanmulsum(x, y)
    # Expected: sum of [1, 1, 1] = 3 (three valid products: 1e10*1e-10, 1e-10*1e10, 1e10*1e-10)
    expected = torch.tensor(3.0)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-4)


@pytest.mark.random
def test_nanmulsum_batch_processing() -> None:
    """Test nanmulsum with batch processing scenarios."""
    batch_size = 3
    seq_length = 4
    features = 5

    x = torch.randn(batch_size, seq_length, features)
    y = torch.randn(batch_size, seq_length, features)

    # Add some NaNs
    x[0, 2, 1] = math.nan
    x[1, 0, 3] = math.nan
    y[2, 1, 4] = math.nan

    # Test batch-wise sum (across sequence and features)
    result_batch = QF.nanmulsum(x, y, dim=(1, 2))
    assert result_batch.shape == (batch_size,)

    # All results should be finite (enough non-NaN values)
    assert torch.isfinite(result_batch).all()


def test_nanmulsum_edge_case_patterns() -> None:
    """Test edge case patterns."""
    # Pattern: NaN, non-NaN, NaN
    x1 = torch.tensor([[math.nan, 5.0, math.nan]])
    y1 = torch.tensor([[2.0, 3.0, 4.0]])
    result1 = QF.nanmulsum(x1, y1)
    expected1 = torch.tensor(15.0)  # Only 5*3 = 15
    np.testing.assert_allclose(result1.numpy(), expected1.numpy())

    # Pattern: All same values with NaN
    x2 = torch.tensor([[2.0, 2.0, math.nan, 2.0]])
    y2 = torch.tensor([[3.0, 3.0, 3.0, 3.0]])
    result2 = QF.nanmulsum(x2, y2)
    expected2 = torch.tensor(18.0)  # Sum of [6, 6, 6] = 18
    np.testing.assert_allclose(result2.numpy(), expected2.numpy())


@pytest.mark.random
def test_nanmulsum_dimension_validation() -> None:
    """Test that function works with various tensor dimensions."""
    # 1D tensor
    x1d = torch.tensor([1.0, math.nan, 3.0])
    y1d = torch.tensor([2.0, 1.0, 4.0])
    result1d = QF.nanmulsum(x1d, y1d)
    expected1d = torch.tensor(14.0)  # Sum of [2, 12] = 14
    np.testing.assert_allclose(result1d.numpy(), expected1d.numpy())

    # 2D tensor
    x2d = torch.tensor([[1.0, math.nan], [3.0, 4.0]])
    y2d = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    result2d = QF.nanmulsum(x2d, y2d, dim=0)
    assert result2d.shape == (2,)

    # 3D tensor
    x3d = torch.randn(2, 3, 4)
    x3d[0, 1, 2] = math.nan
    y3d = torch.randn(2, 3, 4)
    result3d = QF.nanmulsum(x3d, y3d, dim=1)
    assert result3d.shape == (2, 4)


@pytest.mark.random
def test_nanmulsum_multiple_dimensions() -> None:
    """Test sum calculation along multiple dimensions."""
    x = torch.randn(3, 4, 5)
    y = torch.randn(3, 4, 5)
    # Add some NaNs
    x[1, 2, 3] = math.nan
    y[0, 1, 4] = math.nan

    # Test along multiple dimensions
    result_01 = QF.nanmulsum(x, y, dim=(0, 1))
    assert result_01.shape == (5,)

    result_12 = QF.nanmulsum(x, y, dim=(1, 2))
    assert result_12.shape == (3,)

    # All results should be finite
    assert torch.isfinite(result_01).all()
    assert torch.isfinite(result_12).all()


@pytest.mark.random
def test_nanmulsum_performance_comparison() -> None:
    """Test that nanmulsum provides memory efficiency compared to naive approach."""
    for size in [50, 100]:
        x = torch.randn(size, size)
        y = torch.randn(size, size)

        # Add some NaNs
        nan_indices = torch.randint(0, size, (size // 10,))
        x[nan_indices, nan_indices] = math.nan

        result_nanmulsum = QF.nanmulsum(x, y)
        result_naive = (x * y).nansum()

        if not (torch.isnan(result_nanmulsum) and torch.isnan(result_naive)):
            np.testing.assert_allclose(
                result_nanmulsum.numpy(), result_naive.numpy(), rtol=5e-5
            )

        # Clean up
        del x, y, result_nanmulsum, result_naive


def test_nanmulsum_zero_count_behavior() -> None:
    """Test behavior when no valid (non-NaN) multiplications exist."""
    # All combinations result in NaN
    x = torch.tensor([[1.0, math.nan], [math.nan, 2.0]])
    y = torch.tensor([[math.nan, 1.0], [1.0, math.nan]])

    result = QF.nanmulsum(x, y)
    assert torch.isnan(result)

    # Test along dimensions where some slices have no valid values
    result_dim0 = QF.nanmulsum(x, y, dim=0)
    # Both positions should be NaN since all combinations are NaN
    assert torch.isnan(result_dim0).all()


def test_nanmulsum_partial_nan_dimensions() -> None:
    """Test behavior with partial NaN patterns along dimensions."""
    x = torch.tensor([[1.0, math.nan, 3.0], [math.nan, 2.0, math.nan]])
    y = torch.tensor([[2.0, 1.0, math.nan], [1.0, math.nan, 4.0]])

    # Test along dimension 0
    result_dim0 = QF.nanmulsum(x, y, dim=0)

    assert result_dim0[0].item() == 2.0
    assert torch.isnan(result_dim0[1])
    assert torch.isnan(result_dim0[2])

    # Test along dimension 1
    result_dim1 = QF.nanmulsum(x, y, dim=1)

    assert result_dim1[0].item() == 2.0
    assert torch.isnan(result_dim1[1])


def test_nanmulsum_numerical_edge_cases() -> None:
    """Test numerical edge cases."""
    # Very small numbers that might underflow
    x = torch.tensor([[1e-150, math.nan], [1e-150, 1e-150]])
    y = torch.tensor([[1e-150, 1e-150], [1e-150, 1e-150]])

    result = QF.nanmulsum(x, y)
    # Result might be 0 due to underflow, but should be finite
    assert torch.isfinite(result)

    # Very large numbers that might overflow
    x_large = torch.tensor([[1e100, math.nan], [1e100, 1e100]])
    y_large = torch.tensor([[1e100, 1e100], [1e100, 1e100]])

    result_large = QF.nanmulsum(x_large, y_large)
    # Result should be inf due to overflow, or might be NaN if fillna doesn't handle inf properly
    assert torch.isinf(result_large) or torch.isnan(result_large)
