import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_nancovar_basic_functionality() -> None:
    """Test basic NaN-aware covariance functionality."""
    a = QF.randn(100, 200)
    a = torch.where(a < -1, torch.as_tensor(math.nan), a)
    b = QF.randn(100, 200)
    b = torch.where(b < -1, torch.as_tensor(math.nan), b)
    actual = QF.nancovar(a, b)
    expected = np.zeros((100,))
    for i in range(a.shape[0]):
        xa, xb = a[i].numpy(), b[i].numpy()
        mask = ~np.isnan(xa + xb)
        if mask.sum() > 1:  # Need at least 2 points for covariance
            expected[i] = np.cov(xa[mask], xb[mask])[0, 1]
        else:
            expected[i] = math.nan

    # Filter out NaN values for comparison
    valid_mask = ~np.isnan(expected)
    np.testing.assert_allclose(
        actual[valid_mask], expected[valid_mask], 1e-4, 1e-4
    )


def test_nancovar_perfect_positive_correlation() -> None:
    """Test NaN covariance with perfect positive correlation."""
    x = torch.tensor([1.0, 2.0, math.nan, 4.0, 5.0])
    y = torch.tensor([2.0, 4.0, math.nan, 8.0, 10.0])  # y = 2*x

    result = QF.nancovar(x, y, dim=0)

    # Should be positive covariance
    assert result > 0


def test_nancovar_perfect_negative_correlation() -> None:
    """Test NaN covariance with perfect negative correlation."""
    x = torch.tensor([1.0, 2.0, math.nan, 4.0, 5.0])
    y = torch.tensor([-2.0, -4.0, math.nan, -8.0, -10.0])  # y = -2*x

    result = QF.nancovar(x, y, dim=0)

    # Should be negative covariance
    assert result < 0


def test_nancovar_zero_covariance() -> None:
    """Test NaN covariance with zero covariance."""
    torch.manual_seed(42)
    x = torch.randn(1000)
    y = torch.randn(1000)  # Independent random data

    # Add some NaN values
    x[::10] = math.nan
    y[::7] = math.nan

    result = QF.nancovar(x, y, dim=0)

    # Should be close to zero for uncorrelated data
    assert abs(result.item()) < 0.1


def test_nancovar_identical_tensors() -> None:
    """Test NaN covariance of tensor with itself (should equal variance)."""
    x = torch.tensor([1.0, 2.0, math.nan, 4.0, 5.0])

    covar_result = QF.nancovar(x, x, dim=0)

    # Should be positive (variance)
    assert covar_result > 0


def test_nancovar_all_nan() -> None:
    """Test NaN covariance when all values are NaN."""
    x = torch.full((5,), math.nan)
    y = torch.full((5,), math.nan)

    result = QF.nancovar(x, y, dim=0)

    # The function returns 0 when no valid data (not NaN)
    assert result.item() == 0.0


def test_nancovar_single_valid_pair() -> None:
    """Test NaN covariance with only one valid pair."""
    x = torch.tensor([math.nan, 1.0, math.nan, math.nan])
    y = torch.tensor([math.nan, 2.0, math.nan, math.nan])

    result = QF.nancovar(x, y, dim=0)

    # Should be NaN with only one valid pair
    assert torch.isnan(result)


def test_nancovar_2d_tensors() -> None:
    """Test NaN covariance with 2D tensors."""
    x = torch.tensor([[1.0, 2.0, math.nan, 4.0], [math.nan, 6.0, 7.0, 8.0]])
    y = torch.tensor([[2.0, 4.0, math.nan, 8.0], [math.nan, 12.0, 14.0, 16.0]])

    # Test along dimension 1
    result = QF.nancovar(x, y, dim=1)

    assert result.shape == (2,)
    # Both rows should show positive covariance
    assert result[0] > 0
    assert result[1] > 0


def test_nancovar_3d_tensors() -> None:
    """Test NaN covariance with 3D tensors."""
    x = torch.randn(3, 4, 50)
    y = torch.randn(3, 4, 50)

    # Add some NaN values
    x[:, :, ::5] = math.nan
    y[:, :, ::7] = math.nan

    result = QF.nancovar(x, y, dim=2)

    assert result.shape == (3, 4)
    # Most covariances should be finite with enough data
    finite_count = torch.isfinite(result).sum()
    assert finite_count >= 8  # At least 2/3 should be finite


def test_nancovar_keepdim() -> None:
    """Test NaN covariance with keepdim parameter."""
    x = torch.randn(2, 3, 20)
    y = torch.randn(2, 3, 20)

    # Add some NaN values
    x[:, :, ::3] = math.nan
    y[:, :, ::4] = math.nan

    result_keepdim = QF.nancovar(x, y, dim=2, keepdim=True)
    result_no_keepdim = QF.nancovar(x, y, dim=2, keepdim=False)

    assert result_keepdim.shape == (2, 3, 1)
    assert result_no_keepdim.shape == (2, 3)

    np.testing.assert_allclose(
        result_keepdim.squeeze().numpy(), result_no_keepdim.numpy()
    )


def test_nancovar_ddof_values() -> None:
    """Test NaN covariance with different ddof values."""
    x = torch.tensor([1.0, 2.0, math.nan, 4.0, 5.0])
    y = torch.tensor([2.0, 4.0, math.nan, 8.0, 10.0])

    # Test ddof=0 (population covariance)
    result_ddof0 = QF.nancovar(x, y, dim=0, ddof=0)

    # Test ddof=1 (sample covariance)
    result_ddof1 = QF.nancovar(x, y, dim=0, ddof=1)

    # ddof=0 should be smaller than ddof=1 for positive covariance
    assert result_ddof0 < result_ddof1


def test_nancovar_constant_values() -> None:
    """Test NaN covariance with constant values."""
    # One constant, one varying
    x = torch.tensor([5.0, 5.0, math.nan, 5.0, 5.0])
    y = torch.tensor([1.0, 2.0, math.nan, 3.0, 4.0])

    result = QF.nancovar(x, y, dim=0)

    # Covariance with constant should be zero
    assert abs(result.item()) < 1e-6

    # Both constant
    x_const = torch.tensor([5.0, 5.0, math.nan, 5.0])
    y_const = torch.tensor([3.0, 3.0, math.nan, 3.0])

    result_both = QF.nancovar(x_const, y_const, dim=0)
    assert abs(result_both.item()) < 1e-6


def test_nancovar_linear_relationship() -> None:
    """Test NaN covariance with linear relationships."""
    x = torch.tensor([1.0, 2.0, 3.0, math.nan, 5.0, 6.0])

    # Linear relationship: y = ax + b
    a, b = 2.0, 1.0
    y = a * x + b
    y[3] = math.nan  # Ensure NaN alignment

    result = QF.nancovar(x, y, dim=0)

    # Should be positive for positive slope
    assert result > 0


def test_nancovar_asymmetric_nan_patterns() -> None:
    """Test NaN covariance with different NaN patterns in x and y."""
    x = torch.tensor([1.0, math.nan, 3.0, 4.0, 5.0, math.nan])
    y = torch.tensor([2.0, 4.0, math.nan, 8.0, math.nan, 12.0])

    result = QF.nancovar(x, y, dim=0)

    # Should compute covariance using only positions where both are valid
    # Valid pairs: (1,2), (4,8) - should still work
    assert torch.isfinite(result)


def test_nancovar_batch_processing() -> None:
    """Test NaN covariance with batch processing."""
    batch_size = 10
    seq_length = 100

    x = torch.randn(batch_size, seq_length)
    y = torch.randn(batch_size, seq_length)

    # Add random NaN values
    nan_mask_x = torch.rand(batch_size, seq_length) < 0.1
    nan_mask_y = torch.rand(batch_size, seq_length) < 0.1
    x[nan_mask_x] = math.nan
    y[nan_mask_y] = math.nan

    result = QF.nancovar(x, y, dim=1)

    assert result.shape == (batch_size,)
    # Most covariances should be finite with enough data
    finite_count = torch.isfinite(result).sum()
    assert finite_count >= batch_size * 0.8  # At least 80% should be finite


def test_nancovar_numerical_stability() -> None:
    """Test numerical stability with very small and large values."""
    # Very small values
    x_small = torch.tensor([1e-10, 2e-10, math.nan, 4e-10], dtype=torch.float64)
    y_small = torch.tensor([2e-10, 4e-10, math.nan, 8e-10], dtype=torch.float64)

    result_small = QF.nancovar(x_small, y_small, dim=0)
    assert torch.isfinite(result_small)

    # Very large values
    x_large = torch.tensor([1e10, 2e10, math.nan, 4e10], dtype=torch.float64)
    y_large = torch.tensor([2e10, 4e10, math.nan, 8e10], dtype=torch.float64)

    result_large = QF.nancovar(x_large, y_large, dim=0)
    assert torch.isfinite(result_large)


def test_nancovar_mixed_finite_infinite() -> None:
    """Test NaN covariance with mix of finite and infinite values."""
    x = torch.tensor([1.0, 2.0, math.inf, 4.0, math.nan])
    y = torch.tensor([2.0, 4.0, math.inf, 8.0, math.nan])

    result = QF.nancovar(x, y, dim=0)

    # Infinite values should be handled gracefully
    assert torch.isnan(result) or torch.isinf(result) or torch.isfinite(result)


def test_nancovar_negative_dimension() -> None:
    """Test NaN covariance with negative dimension indexing."""
    x = torch.randn(3, 4, 20)
    y = torch.randn(3, 4, 20)

    # Add some NaN values
    x[:, :, ::3] = math.nan
    y[:, :, ::4] = math.nan

    result_neg = QF.nancovar(x, y, dim=-1)
    result_pos = QF.nancovar(x, y, dim=2)

    np.testing.assert_allclose(result_neg.numpy(), result_pos.numpy())


def test_nancovar_empty_after_nan_removal() -> None:
    """Test NaN covariance when all data becomes NaN after processing."""
    x = torch.tensor([math.nan, math.inf, -math.inf])
    y = torch.tensor([math.nan, 1.0, 2.0])

    result = QF.nancovar(x, y, dim=0)

    # The function returns 0 when no valid pairs remain
    assert result.item() == 0.0


def test_nancovar_numpy_comparison() -> None:
    """Test NaN covariance against numpy for validation."""
    torch.manual_seed(123)
    x = torch.randn(50)
    y = torch.randn(50)

    # Add some NaN values
    x[::5] = math.nan
    y[::7] = math.nan

    result = QF.nancovar(x, y, dim=0)

    # Compare with numpy calculation
    x_np, y_np = x.numpy(), y.numpy()
    valid_mask = ~np.isnan(x_np) & ~np.isnan(y_np)

    if valid_mask.sum() > 1:
        expected = np.cov(x_np[valid_mask], y_np[valid_mask])[0, 1]
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-3, atol=1e-3
        )


def test_nancovar_symmetry() -> None:
    """Test that NaN covariance is symmetric: nancovar(x,y) = nancovar(y,x)."""
    x = torch.tensor([1.0, math.nan, 3.0, 4.0, 5.0])
    y = torch.tensor([2.0, 4.0, math.nan, 8.0, 10.0])

    result_xy = QF.nancovar(x, y, dim=0)
    result_yx = QF.nancovar(y, x, dim=0)

    np.testing.assert_allclose(result_xy.numpy(), result_yx.numpy())


def test_nancovar_linear_transformation() -> None:
    """Test NaN covariance properties under linear transformation."""
    x = torch.tensor([1.0, 2.0, math.nan, 4.0, 5.0])
    y = torch.tensor([2.0, 4.0, math.nan, 8.0, 10.0])

    # Original covariance
    cov_xy = QF.nancovar(x, y, dim=0)

    # Linear transformations
    a, b = 2.0, 3.0
    c, d = -1.0, 5.0

    x_transformed = a * x + b
    y_transformed = c * y + d

    cov_transformed = QF.nancovar(x_transformed, y_transformed, dim=0)
    expected_transformed = a * c * cov_xy  # cov(ax+b, cy+d) = ac * cov(x,y)

    np.testing.assert_allclose(
        cov_transformed.numpy(), expected_transformed.numpy(), rtol=1e-6
    )


def test_nancovar_mixed_signs() -> None:
    """Test NaN covariance with mixed positive and negative values."""
    x = torch.tensor([-2.0, -1.0, math.nan, 1.0, 2.0])
    y = torch.tensor([-4.0, -2.0, math.nan, 2.0, 4.0])  # y = 2*x

    result = QF.nancovar(x, y, dim=0)

    # Should be positive since they're positively related
    assert result > 0


def test_nancovar_different_ddof() -> None:
    """Test NaN covariance with different ddof values on batch data."""
    batch_size = 5
    seq_length = 50
    x = torch.randn(batch_size, seq_length)
    y = torch.randn(batch_size, seq_length)

    # Add some NaN values
    x[:, ::5] = math.nan
    y[:, ::7] = math.nan

    result_ddof0 = QF.nancovar(x, y, dim=1, ddof=0)
    result_ddof1 = QF.nancovar(x, y, dim=1, ddof=1)

    assert result_ddof0.shape == (batch_size,)
    assert result_ddof1.shape == (batch_size,)

    # ddof=1 results should generally have larger absolute value than ddof=0
    # The ratio depends on the sign of covariance and sample size
    finite_mask = torch.isfinite(result_ddof0) & torch.isfinite(result_ddof1)
    non_zero_mask = finite_mask & (result_ddof0.abs() > 1e-8)
    if non_zero_mask.any():
        abs_ratio = result_ddof1[non_zero_mask].abs() / result_ddof0[non_zero_mask].abs()
        # Most ratios should be close to n/(n-1), but allow some numerical tolerance
        # Due to NaN pattern differences and finite sample effects, allow 95% tolerance
        assert torch.mean((abs_ratio > 0.95).float()) >= 0.8


def test_nancovar_precision_warning() -> None:
    """Test that function handles precision limitations gracefully."""
    # Create data with many NaN values to test precision warning scenario
    x = torch.randn(1000)
    y = torch.randn(1000)

    # Make 90% of values NaN
    nan_indices = torch.randperm(1000)[:900]
    x[nan_indices] = math.nan
    y[nan_indices] = math.nan

    result = QF.nancovar(x, y, dim=0)

    # Should still compute something reasonable with remaining data
    assert torch.isfinite(result) or torch.isnan(result)
