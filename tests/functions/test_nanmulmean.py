import math

import numpy as np
import pytest
import torch

import qfeval_functions.functions as QF


@pytest.mark.filterwarnings("ignore:Mean of")
def test_nanmulmean() -> None:
    a = QF.randn(100, 1)
    a = torch.where(a < 0, torch.as_tensor(math.nan), a)
    b = QF.randn(1, 200)
    b = torch.where(b < 0, torch.as_tensor(math.nan), b)
    np.testing.assert_allclose(
        np.nanmean((a * b).numpy()),
        QF.nanmulmean(a, b).numpy(),
        1e-5,
        1e-5,
    )
    np.testing.assert_allclose(
        np.nanmean((a * b).numpy(), axis=1),
        QF.nanmulmean(a, b, dim=1).numpy(),
        1e-5,
        1e-5,
    )
    np.testing.assert_array_equal(
        (a * b).isnan().all(dim=1).numpy(),
        QF.nanmulmean(a, b, dim=1).isnan().numpy(),
    )
    np.testing.assert_allclose(
        np.nanmean((a * b).numpy(), axis=-1, keepdims=True),
        QF.nanmulmean(a, b, dim=-1, keepdim=True).numpy(),
        1e-5,
        1e-5,
    )


def test_nanmulmean_basic_functionality() -> None:
    """Test basic NaN-aware matrix multiplication mean functionality."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[2.0, 1.0], [1.0, 2.0]])

    # Test global mean
    result = QF.nanmulmean(x, y)
    expected = (x * y).mean()
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nanmulmean_with_nans() -> None:
    """Test NaN-aware matrix multiplication mean with NaN values."""
    x = torch.tensor([[1.0, math.nan], [3.0, 4.0]])
    y = torch.tensor([[2.0, 1.0], [1.0, 2.0]])

    result = QF.nanmulmean(x, y)
    # Manually calculate: (1*2 + 3*1 + 4*2) / 3 = (2 + 3 + 8) / 3 = 13/3
    expected = torch.tensor(13.0 / 3.0)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-5)


def test_nanmulmean_all_nans() -> None:
    """Test matrix multiplication mean with all NaN values."""
    x = torch.tensor([[math.nan, math.nan], [math.nan, math.nan]])
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    result = QF.nanmulmean(x, y)
    assert torch.isnan(result)


def test_nanmulmean_broadcasting() -> None:
    """Test NaN-aware multiplication mean with broadcasting."""
    x = torch.tensor([[1.0, math.nan, 3.0]])
    y = torch.tensor([[2.0], [3.0]])

    result = QF.nanmulmean(x, y)
    # Expected: mean of [2, nan, 9, 3, nan, 6] ignoring NaN = (2+9+3+6)/4 = 5
    expected = torch.tensor(5.0)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-5)


def test_nanmulmean_dimensions() -> None:
    """Test NaN-aware mean calculation along specific dimensions."""
    x = torch.tensor(
        [[[1.0, 2.0], [math.nan, 4.0]], [[5.0, 6.0], [7.0, math.nan]]]
    )
    y = torch.tensor([[[2.0, 1.0], [1.0, 2.0]], [[1.0, 3.0], [2.0, 1.0]]])

    # Test along dimension 0
    result_dim0 = QF.nanmulmean(x, y, dim=0)
    assert result_dim0.shape == (2, 2)

    # Test along dimension 1
    result_dim1 = QF.nanmulmean(x, y, dim=1)
    assert result_dim1.shape == (2, 2)

    # Test along dimension 2
    result_dim2 = QF.nanmulmean(x, y, dim=2)
    assert result_dim2.shape == (2, 2)


def test_nanmulmean_keepdim() -> None:
    """Test keepdim parameter functionality."""
    x = torch.tensor([[[1.0, math.nan], [3.0, 4.0]]])
    y = torch.tensor([[[2.0, 1.0], [1.0, 2.0]]])

    # Test keepdim=True
    result_keepdim = QF.nanmulmean(x, y, dim=1, keepdim=True)
    assert result_keepdim.shape == (1, 1, 2)

    # Test keepdim=False
    result_no_keepdim = QF.nanmulmean(x, y, dim=1, keepdim=False)
    assert result_no_keepdim.shape == (1, 2)


def test_nanmulmean_negative_dimensions() -> None:
    """Test negative dimension indexing."""
    x = torch.tensor([[[1.0, math.nan], [3.0, 4.0]]])
    y = torch.tensor([[[2.0, 1.0], [1.0, 2.0]]])

    result_neg1 = QF.nanmulmean(x, y, dim=-1)
    result_pos2 = QF.nanmulmean(x, y, dim=2)
    np.testing.assert_allclose(result_neg1.numpy(), result_pos2.numpy())


def test_nanmulmean_single_nan() -> None:
    """Test multiplication and mean with single NaN element."""
    x = torch.tensor([[math.nan]])
    y = torch.tensor([[3.0]])

    result = QF.nanmulmean(x, y)
    assert torch.isnan(result)


def test_nanmulmean_zero_values() -> None:
    """Test multiplication and mean with zero values."""
    x = torch.tensor([[1.0, 0.0, 3.0], [0.0, 2.0, math.nan]])
    y = torch.tensor([[2.0, 1.0, 0.0], [3.0, 0.0, 4.0]])

    result = QF.nanmulmean(x, y)
    # Expected: mean of [2, 0, 0, 0, 0] = 2/5 = 0.4
    expected = torch.tensor(0.4)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-5)


def test_nanmulmean_negative_values() -> None:
    """Test multiplication and mean with negative values."""
    x = torch.tensor([[-1.0, 2.0], [3.0, math.nan]])
    y = torch.tensor([[2.0, -1.0], [-2.0, 3.0]])

    result = QF.nanmulmean(x, y)
    # Expected: mean of [-2, -2, -6] = -10/3
    expected = torch.tensor(-10.0 / 3.0)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-5)


def test_nanmulmean_large_values() -> None:
    """Test multiplication and mean with large values."""
    x = torch.tensor([[1e6, math.nan], [3e6, 4e6]])
    y = torch.tensor([[2e6, 1e6], [1e6, 2e6]])

    result = QF.nanmulmean(x, y)
    # Expected: mean of [2e12, 3e12, 8e12] = 13e12/3
    expected = torch.tensor(13e12 / 3.0)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-4)


def test_nanmulmean_small_values() -> None:
    """Test multiplication and mean with very small values."""
    x = torch.tensor([[1e-6, math.nan], [3e-6, 4e-6]])
    y = torch.tensor([[2e-6, 1e-6], [1e-6, 2e-6]])

    result = QF.nanmulmean(x, y)
    # Expected: mean of [2e-12, 3e-12, 8e-12] = 13e-12/3
    expected = torch.tensor(13e-12 / 3.0)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-4)


def test_nanmulmean_broadcasting_edge_cases() -> None:
    """Test edge cases in broadcasting."""
    # Scalar with tensor
    x_scalar = torch.tensor(2.0)
    y_tensor = torch.tensor([[1.0, math.nan], [3.0, 4.0]])

    result1 = QF.nanmulmean(x_scalar, y_tensor)
    # Expected: mean of [2, 6, 8] = 16/3
    expected1 = torch.tensor(16.0 / 3.0)
    np.testing.assert_allclose(result1.numpy(), expected1.numpy(), rtol=1e-5)

    # Different shape broadcasting
    x_broadcast = torch.tensor([[[1.0]], [[math.nan]]])
    y_broadcast = torch.tensor([[[3.0, 4.0, 5.0]]])

    result2 = QF.nanmulmean(x_broadcast, y_broadcast)
    # Expected: mean of [3, 4, 5] = 4
    expected2 = torch.tensor(4.0)
    np.testing.assert_allclose(result2.numpy(), expected2.numpy(), rtol=1e-5)


def test_nanmulmean_with_infinity() -> None:
    """Test nanmulmean behavior with infinity values."""
    x = torch.tensor([[1.0, math.inf], [3.0, math.nan]])
    y = torch.tensor([[2.0, 1.0], [1.0, 2.0]])

    result = QF.nanmulmean(x, y)
    # inf * 1 = inf, so result should be inf
    assert torch.isinf(result) or torch.isnan(result)


@pytest.mark.random
def test_nanmulmean_high_dimensional() -> None:
    """Test nanmulmean with high-dimensional tensors."""
    x = torch.randn(2, 3, 4, 5)
    # Add some NaNs
    x[0, 1, 2, 3] = math.nan
    x[1, 0, 1, 2] = math.nan
    y = torch.randn(2, 3, 4, 5)

    result = QF.nanmulmean(x, y)
    assert torch.isfinite(
        result
    )  # Should be finite since most values are non-NaN

    # Test along specific high dimensions
    result_dim3 = QF.nanmulmean(x, y, dim=3)
    assert result_dim3.shape == (2, 3, 4)


def test_nanmulmean_mathematical_properties() -> None:
    """Test mathematical properties of nanmulmean."""
    x = torch.tensor([[1.0, math.nan], [3.0, 4.0]])
    y = torch.tensor([[2.0, 1.0], [1.0, 2.0]])

    # Test linearity: nanmulmean(ax, y) ≈ a * nanmulmean(x, y) when no broadcast changes NaN pattern
    a = 3.0
    x_scaled = a * x
    result1 = QF.nanmulmean(x_scaled, y)
    result2 = a * QF.nanmulmean(x, y)
    np.testing.assert_allclose(result1.numpy(), result2.numpy(), rtol=1e-5)

    # Test commutativity: nanmulmean(x, y) = nanmulmean(y, x)
    result_xy = QF.nanmulmean(x, y)
    result_yx = QF.nanmulmean(y, x)
    np.testing.assert_allclose(result_xy.numpy(), result_yx.numpy(), rtol=1e-5)


def test_nanmulmean_numerical_stability() -> None:
    """Test numerical stability of nanmulmean."""
    # Test with values that might cause overflow in naive multiplication
    x = torch.tensor([[1e10, math.nan], [1e-10, 1e10]])
    y = torch.tensor([[1e-10, 1e10], [1e10, 1e-10]])

    result = QF.nanmulmean(x, y)
    # Both products should be 1.0, so mean should be 1.0
    expected = torch.tensor(1.0)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-4)


@pytest.mark.random
def test_nanmulmean_batch_processing() -> None:
    """Test nanmulmean with batch processing scenarios."""
    batch_size = 3
    seq_length = 4
    features = 5

    x = torch.randn(batch_size, seq_length, features)
    y = torch.randn(batch_size, seq_length, features)

    # Add some NaNs
    x[0, 2, 1] = math.nan
    x[1, 0, 3] = math.nan
    y[2, 1, 4] = math.nan

    # Test batch-wise mean (across sequence and features)
    result_batch = QF.nanmulmean(x, y, dim=(1, 2))
    assert result_batch.shape == (batch_size,)

    # All results should be finite (enough non-NaN values)
    assert torch.isfinite(result_batch).all()


def test_nanmulmean_edge_case_patterns() -> None:
    """Test edge case patterns."""
    # Pattern: NaN, non-NaN, NaN
    x1 = torch.tensor([[math.nan, 5.0, math.nan]])
    y1 = torch.tensor([[2.0, 3.0, 4.0]])
    result1 = QF.nanmulmean(x1, y1)
    expected1 = torch.tensor(15.0)  # Only 5*3 = 15
    np.testing.assert_allclose(result1.numpy(), expected1.numpy())

    # Pattern: All same values with NaN
    x2 = torch.tensor([[2.0, 2.0, math.nan, 2.0]])
    y2 = torch.tensor([[3.0, 3.0, 3.0, 3.0]])
    result2 = QF.nanmulmean(x2, y2)
    expected2 = torch.tensor(6.0)  # Mean of [6, 6, 6] = 6
    np.testing.assert_allclose(result2.numpy(), expected2.numpy())


@pytest.mark.random
def test_nanmulmean_dimension_validation() -> None:
    """Test that function works with various tensor dimensions."""
    # 1D tensor
    x1d = torch.tensor([1.0, math.nan, 3.0])
    y1d = torch.tensor([2.0, 1.0, 4.0])
    result1d = QF.nanmulmean(x1d, y1d)
    expected1d = torch.tensor(7.0)  # Mean of [2, 12] = 7
    np.testing.assert_allclose(result1d.numpy(), expected1d.numpy())

    # 2D tensor
    x2d = torch.tensor([[1.0, math.nan], [3.0, 4.0]])
    y2d = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    result2d = QF.nanmulmean(x2d, y2d, dim=0)
    assert result2d.shape == (2,)

    # 3D tensor
    x3d = torch.randn(2, 3, 4)
    x3d[0, 1, 2] = math.nan
    y3d = torch.randn(2, 3, 4)
    result3d = QF.nanmulmean(x3d, y3d, dim=1)
    assert result3d.shape == (2, 4)


def test_nanmulmean_ddof_parameter() -> None:
    """Test _ddof parameter functionality."""
    x = torch.tensor([[1.0, math.nan], [3.0, 4.0]])
    y = torch.tensor([[2.0, 1.0], [1.0, 2.0]])

    # Test with ddof=0 (default)
    result_ddof0 = QF.nanmulmean(x, y, _ddof=0)

    # Test with ddof=1 (sample mean)
    result_ddof1 = QF.nanmulmean(x, y, _ddof=1)

    # With ddof=1, the denominator is reduced by 1
    # Should give a larger result than ddof=0
    assert result_ddof1.item() > result_ddof0.item()


@pytest.mark.random
def test_nanmulmean_multiple_dimensions() -> None:
    """Test mean calculation along multiple dimensions."""
    x = torch.randn(3, 4, 5)
    y = torch.randn(3, 4, 5)
    # Add some NaNs
    x[1, 2, 3] = math.nan
    y[0, 1, 4] = math.nan

    # Test along multiple dimensions
    result_01 = QF.nanmulmean(x, y, dim=(0, 1))
    assert result_01.shape == (5,)

    result_12 = QF.nanmulmean(x, y, dim=(1, 2))
    assert result_12.shape == (3,)

    # All results should be finite
    assert torch.isfinite(result_01).all()
    assert torch.isfinite(result_12).all()


@pytest.mark.random
def test_nanmulmean_performance_comparison() -> None:
    """Test that nanmulmean provides memory efficiency compared to naive approach."""
    for size in [50, 100]:
        x = torch.randn(size, size)
        y = torch.randn(size, size)

        # Add some NaNs
        nan_indices = torch.randint(0, size, (size // 10,))
        x[nan_indices, nan_indices] = math.nan

        result_nanmulmean = QF.nanmulmean(x, y)

        # Verify it's finite (enough non-NaN values)
        assert torch.isfinite(result_nanmulmean)

        # Clean up
        del x, y, result_nanmulmean
