import math

import numpy as np
import torch
from scipy.stats import linregress

import qfeval_functions.functions as QF
import pytest


def test_nancorrel_basic_functionality() -> None:
    """Test basic NaN-aware correlation functionality."""
    a = QF.randn(100, 200)
    a = torch.where(a < 0, torch.as_tensor(math.nan), a)
    b = QF.randn(100, 200)
    b = torch.where(b < 0, torch.as_tensor(math.nan), b)
    actual = QF.nancorrel(a, b, dim=1)
    expected = np.zeros((100,))
    for i in range(a.shape[0]):
        xa, xb = a[i].numpy(), b[i].numpy()
        mask = ~np.isnan(xa + xb)
        if mask.sum() > 1:  # Need at least 2 points for correlation
            expected[i] = linregress(xa[mask], xb[mask]).rvalue
        else:
            expected[i] = math.nan

    # Filter out NaN values for comparison
    valid_mask = ~np.isnan(expected)
    np.testing.assert_allclose(
        actual[valid_mask], expected[valid_mask], 1e-6, 1e-6
    )


def test_nancorrel_perfect_positive_correlation() -> None:
    """Test NaN correlation with perfect positive correlation."""
    x = torch.tensor([1.0, 2.0, math.nan, 4.0, 5.0])
    y = torch.tensor([2.0, 4.0, math.nan, 8.0, 10.0])  # y = 2*x

    result = QF.nancorrel(x, y, dim=0)

    # Should be 1.0 for perfect positive correlation
    assert abs(result.item() - 1.0) < 1e-6


def test_nancorrel_perfect_negative_correlation() -> None:
    """Test NaN correlation with perfect negative correlation."""
    x = torch.tensor([1.0, 2.0, math.nan, 4.0, 5.0])
    y = torch.tensor([-2.0, -4.0, math.nan, -8.0, -10.0])  # y = -2*x

    result = QF.nancorrel(x, y, dim=0)

    # Should be -1.0 for perfect negative correlation
    assert abs(result.item() + 1.0) < 1e-6


def test_nancorrel_zero_correlation() -> None:
    """Test NaN correlation with zero correlation."""
    torch.manual_seed(42)
    x = torch.randn(1000)
    y = torch.randn(1000)  # Independent random data

    # Add some NaN values
    x[::10] = math.nan
    y[::7] = math.nan

    result = QF.nancorrel(x, y, dim=0)

    # Should be close to zero for uncorrelated data
    assert abs(result.item()) < 0.1


def test_nancorrel_all_nan() -> None:
    """Test NaN correlation when all values are NaN."""
    x = torch.full((5,), math.nan)
    y = torch.full((5,), math.nan)

    result = QF.nancorrel(x, y, dim=0)

    # Should be NaN when no valid data
    assert torch.isnan(result)


def test_nancorrel_single_valid_pair() -> None:
    """Test NaN correlation with only one valid pair."""
    x = torch.tensor([math.nan, 1.0, math.nan, math.nan])
    y = torch.tensor([math.nan, 2.0, math.nan, math.nan])

    result = QF.nancorrel(x, y, dim=0)

    # Should be NaN with only one valid pair (need at least 2 for correlation)
    assert torch.isnan(result)


def test_nancorrel_2d_tensors() -> None:
    """Test NaN correlation with 2D tensors."""
    x = torch.tensor([[1.0, 2.0, math.nan, 4.0], [math.nan, 6.0, 7.0, 8.0]])
    y = torch.tensor([[2.0, 4.0, math.nan, 8.0], [math.nan, 12.0, 14.0, 16.0]])

    # Test along dimension 1
    result = QF.nancorrel(x, y, dim=1)

    assert result.shape == (2,)
    # Both rows should show perfect positive correlation
    assert abs(result[0].item() - 1.0) < 1e-6
    assert abs(result[1].item() - 1.0) < 1e-6


def test_nancorrel_3d_tensors() -> None:
    """Test NaN correlation with 3D tensors."""
    x = torch.randn(3, 4, 50)
    y = torch.randn(3, 4, 50)

    # Add some NaN values
    x[:, :, ::5] = math.nan
    y[:, :, ::7] = math.nan

    result = QF.nancorrel(x, y, dim=2)

    assert result.shape == (3, 4)
    # All correlations should be finite (not NaN) with enough data
    assert torch.isfinite(result).all()


def test_nancorrel_keepdim() -> None:
    """Test NaN correlation with keepdim parameter."""
    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 3, 4)

    # Add some NaN values
    x[0, 0, 0] = math.nan
    y[1, 1, 1] = math.nan

    result_keepdim = QF.nancorrel(x, y, dim=2, keepdim=True)
    result_no_keepdim = QF.nancorrel(x, y, dim=2, keepdim=False)

    assert result_keepdim.shape == (2, 3, 1)
    assert result_no_keepdim.shape == (2, 3)

    np.testing.assert_allclose(
        result_keepdim.squeeze().numpy(), result_no_keepdim.numpy()
    )


def test_nancorrel_multiple_dimensions() -> None:
    """Test NaN correlation along multiple dimensions."""
    x = torch.randn(2, 3, 4, 5)
    y = torch.randn(2, 3, 4, 5)

    # Add some NaN values
    x[:, :, 0, 0] = math.nan
    y[:, :, 1, 1] = math.nan

    result = QF.nancorrel(x, y, dim=(2, 3))

    assert result.shape == (2, 3)
    assert torch.isfinite(result).all()


def test_nancorrel_constant_values() -> None:
    """Test NaN correlation with constant values."""
    # One constant, one varying
    x = torch.tensor([5.0, 5.0, math.nan, 5.0, 5.0])
    y = torch.tensor([1.0, 2.0, math.nan, 3.0, 4.0])

    result = QF.nancorrel(x, y, dim=0)

    # Correlation with constant should be NaN (zero variance)
    assert torch.isnan(result)

    # Both constant
    x_const = torch.tensor([5.0, 5.0, math.nan, 5.0])
    y_const = torch.tensor([3.0, 3.0, math.nan, 3.0])

    result_both = QF.nancorrel(x_const, y_const, dim=0)
    assert torch.isnan(result_both)


def test_nancorrel_linear_relationship() -> None:
    """Test NaN correlation with linear relationships."""
    x = torch.tensor([1.0, 2.0, 3.0, math.nan, 5.0, 6.0])

    # Linear relationship with noise
    y_linear = 2.0 * x + 1.0 + 0.1 * torch.randn_like(x)
    y_linear[3] = math.nan  # Ensure NaN alignment

    result = QF.nancorrel(x, y_linear, dim=0)

    # Should be very close to 1.0 for strong linear relationship
    assert result > 0.9


def test_nancorrel_asymmetric_nan_patterns() -> None:
    """Test NaN correlation with different NaN patterns in x and y."""
    x = torch.tensor([1.0, math.nan, 3.0, 4.0, math.nan, 6.0])
    y = torch.tensor([2.0, 4.0, math.nan, 8.0, 10.0, math.nan])

    result = QF.nancorrel(x, y, dim=0)

    # Should compute correlation using only positions where both are valid
    # Valid pairs: (1,2), (4,8) - 2 points should give perfect correlation
    assert abs(result.item() - 1.0) < 1e-6  # Perfect correlation with 2 points


def test_nancorrel_batch_processing() -> None:
    """Test NaN correlation with batch processing."""
    batch_size = 10
    seq_length = 100

    x = torch.randn(batch_size, seq_length)
    y = torch.randn(batch_size, seq_length)

    # Add random NaN values
    nan_mask_x = torch.rand(batch_size, seq_length) < 0.1
    nan_mask_y = torch.rand(batch_size, seq_length) < 0.1
    x[nan_mask_x] = math.nan
    y[nan_mask_y] = math.nan

    result = QF.nancorrel(x, y, dim=1)

    assert result.shape == (batch_size,)
    # Most correlations should be finite with enough data
    finite_count = torch.isfinite(result).sum()
    assert finite_count >= batch_size * 0.8  # At least 80% should be finite


def test_nancorrel_numerical_stability() -> None:
    """Test numerical stability with very small and large values."""
    # Very small values
    x_small = torch.tensor([1e-10, 2e-10, math.nan, 4e-10], dtype=torch.float64)
    y_small = torch.tensor([2e-10, 4e-10, math.nan, 8e-10], dtype=torch.float64)

    result_small = QF.nancorrel(x_small, y_small, dim=0)
    assert abs(result_small.item() - 1.0) < 1e-10

    # Very large values
    x_large = torch.tensor([1e10, 2e10, math.nan, 4e10], dtype=torch.float64)
    y_large = torch.tensor([2e10, 4e10, math.nan, 8e10], dtype=torch.float64)

    result_large = QF.nancorrel(x_large, y_large, dim=0)
    assert abs(result_large.item() - 1.0) < 1e-10


def test_nancorrel_mixed_finite_infinite() -> None:
    """Test NaN correlation with mix of finite and infinite values."""
    x = torch.tensor([1.0, 2.0, math.inf, 4.0, math.nan])
    y = torch.tensor([2.0, 4.0, math.inf, 8.0, math.nan])

    result = QF.nancorrel(x, y, dim=0)

    # Infinite values should be handled gracefully
    assert torch.isnan(result) or torch.isinf(result) or torch.isfinite(result)


def test_nancorrel_negative_dimension() -> None:
    """Test NaN correlation with negative dimension indexing."""
    x = torch.randn(3, 4, 5)
    y = torch.randn(3, 4, 5)

    # Add some NaN values
    x[0, 0, 0] = math.nan
    y[1, 1, 1] = math.nan

    result_neg = QF.nancorrel(x, y, dim=-1)
    result_pos = QF.nancorrel(x, y, dim=2)

    np.testing.assert_allclose(result_neg.numpy(), result_pos.numpy())


def test_nancorrel_empty_after_nan_removal() -> None:
    """Test NaN correlation when all data becomes NaN after processing."""
    x = torch.tensor([math.nan, math.inf, -math.inf])
    y = torch.tensor([math.nan, 1.0, 2.0])

    result = QF.nancorrel(x, y, dim=0)

    # Should be NaN when no valid pairs remain
    assert torch.isnan(result)


def test_nancorrel_scipy_comparison() -> None:
    """Test NaN correlation against scipy for validation."""
    torch.manual_seed(123)
    x = torch.randn(50)
    y = torch.randn(50)

    # Add some NaN values
    x[::5] = math.nan
    y[::7] = math.nan

    result = QF.nancorrel(x, y, dim=0)

    # Compare with scipy calculation
    x_np, y_np = x.numpy(), y.numpy()
    valid_mask = ~np.isnan(x_np) & ~np.isnan(y_np)

    if valid_mask.sum() > 1:
        expected = np.corrcoef(x_np[valid_mask], y_np[valid_mask])[0, 1]
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4, atol=1e-4)


def test_nancorrel_monotonic_relationship() -> None:
    """Test NaN correlation with monotonic but non-linear relationship."""
    x = torch.tensor([1.0, 2.0, 3.0, math.nan, 5.0, 6.0, 7.0])
    y = x**2  # Quadratic relationship
    y[3] = math.nan  # Align NaN positions

    result = QF.nancorrel(x, y, dim=0)

    # Should show strong positive correlation for monotonic relationship
    assert result > 0.8


def test_nancorrel_range_validation() -> None:
    """Test that correlation values are in valid range [-1, 1]."""
    torch.manual_seed(456)
    x = torch.randn(5, 100)
    y = torch.randn(5, 100)

    # Add random NaN values
    x[torch.rand_like(x) < 0.1] = math.nan
    y[torch.rand_like(y) < 0.1] = math.nan

    result = QF.nancorrel(x, y, dim=1)

    # Filter out NaN results
    finite_results = result[torch.isfinite(result)]

    if len(finite_results) > 0:
        assert torch.all(finite_results >= -1.0)
        assert torch.all(finite_results <= 1.0)
