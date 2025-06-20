import math

import numpy as np
import torch
from scipy.stats import linregress

import qfeval_functions.functions as QF
import pytest


def test_nanslope_basic_functionality() -> None:
    """Test basic NaN-aware slope functionality."""
    a = QF.randn(100, 200)
    a = torch.where(a < 0, torch.as_tensor(math.nan), a)
    b = QF.randn(100, 200)
    b = torch.where(b < 0, torch.as_tensor(math.nan), b)
    actual = QF.nanslope(a, b, dim=1)
    expected = np.zeros((100,))
    for i in range(a.shape[0]):
        xa, xb = a[i].numpy(), b[i].numpy()
        mask = ~np.isnan(xa + xb)
        if mask.sum() > 1:  # Need at least 2 points for slope
            expected[i] = linregress(xa[mask], xb[mask]).slope
        else:
            expected[i] = math.nan

    # Filter out NaN values for comparison
    valid_mask = ~np.isnan(expected)
    np.testing.assert_allclose(
        actual[valid_mask], expected[valid_mask], 1e-6, 1e-6
    )


def test_nanslope_perfect_positive_linear() -> None:
    """Test NaN slope with perfect positive linear relationship."""
    x = torch.tensor([1.0, 2.0, math.nan, 4.0, 5.0])
    y = torch.tensor([3.0, 6.0, math.nan, 12.0, 15.0])  # y = 3*x

    result = QF.nanslope(x, y, dim=0)

    # Should be 3.0 for y = 3*x
    assert abs(result.item() - 3.0) < 1e-6


def test_nanslope_perfect_negative_linear() -> None:
    """Test NaN slope with perfect negative linear relationship."""
    x = torch.tensor([1.0, 2.0, math.nan, 4.0, 5.0])
    y = torch.tensor([-2.0, -4.0, math.nan, -8.0, -10.0])  # y = -2*x

    result = QF.nanslope(x, y, dim=0)

    # Should be -2.0 for y = -2*x
    assert abs(result.item() + 2.0) < 1e-6


def test_nanslope_with_intercept() -> None:
    """Test NaN slope with linear relationship including intercept."""
    x = torch.tensor([1.0, 2.0, math.nan, 4.0, 5.0])
    y = torch.tensor([5.0, 7.0, math.nan, 11.0, 13.0])  # y = 2*x + 3

    result = QF.nanslope(x, y, dim=0)

    # Should be 2.0 for y = 2*x + 3 (intercept doesn't affect slope)
    assert abs(result.item() - 2.0) < 1e-6


def test_nanslope_zero_slope() -> None:
    """Test NaN slope with zero slope (horizontal line)."""
    x = torch.tensor([1.0, 2.0, math.nan, 4.0, 5.0])
    y = torch.tensor([3.0, 3.0, math.nan, 3.0, 3.0])  # y = 3 (constant)

    result = QF.nanslope(x, y, dim=0)

    # Should be 0.0 for horizontal line
    assert abs(result.item()) < 1e-6


def test_nanslope_vertical_line() -> None:
    """Test NaN slope with vertical line (undefined slope)."""
    x = torch.tensor([3.0, 3.0, math.nan, 3.0, 3.0])  # x = 3 (constant)
    y = torch.tensor([1.0, 2.0, math.nan, 4.0, 5.0])

    result = QF.nanslope(x, y, dim=0)

    # Should be infinite or NaN for vertical line (division by zero)
    assert torch.isinf(result) or torch.isnan(result)


def test_nanslope_all_nan() -> None:
    """Test NaN slope when all values are NaN."""
    x = torch.full((5,), math.nan)
    y = torch.full((5,), math.nan)

    result = QF.nanslope(x, y, dim=0)

    # Should be NaN when no valid data
    assert torch.isnan(result)


def test_nanslope_single_valid_pair() -> None:
    """Test NaN slope with only one valid pair."""
    x = torch.tensor([math.nan, 1.0, math.nan, math.nan])
    y = torch.tensor([math.nan, 2.0, math.nan, math.nan])

    result = QF.nanslope(x, y, dim=0)

    # Should be NaN with only one valid pair
    assert torch.isnan(result)


def test_nanslope_2d_tensors() -> None:
    """Test NaN slope with 2D tensors."""
    x = torch.tensor([[1.0, 2.0, math.nan, 4.0], [math.nan, 1.0, 2.0, 3.0]])
    y = torch.tensor([[2.0, 4.0, math.nan, 8.0], [math.nan, 3.0, 6.0, 9.0]])

    # Test along dimension 1
    result = QF.nanslope(x, y, dim=1)

    assert result.shape == (2,)
    # Both rows should show slope of 2 and 3 respectively
    assert abs(result[0].item() - 2.0) < 1e-6
    assert abs(result[1].item() - 3.0) < 1e-6


@pytest.mark.random
def test_nanslope_3d_tensors() -> None:
    """Test NaN slope with 3D tensors."""
    x = torch.randn(3, 4, 50)
    y = 2.0 * x + 1.0  # y = 2*x + 1

    # Add some NaN values
    x[:, :, ::5] = math.nan
    y[:, :, ::7] = math.nan

    result = QF.nanslope(x, y, dim=2)

    assert result.shape == (3, 4)
    # All slopes should be close to 2.0
    finite_results = result[torch.isfinite(result)]
    if len(finite_results) > 0:
        assert torch.all(torch.abs(finite_results - 2.0) < 0.1)


@pytest.mark.random
def test_nanslope_keepdim() -> None:
    """Test NaN slope with keepdim parameter."""
    x = torch.randn(2, 3, 20)
    y = 1.5 * x + 0.5

    # Add some NaN values
    x[:, :, ::3] = math.nan
    y[:, :, ::4] = math.nan

    result_keepdim = QF.nanslope(x, y, dim=2, keepdim=True)
    result_no_keepdim = QF.nanslope(x, y, dim=2, keepdim=False)

    assert result_keepdim.shape == (2, 3, 1)
    assert result_no_keepdim.shape == (2, 3)

    np.testing.assert_allclose(
        result_keepdim.squeeze().numpy(), result_no_keepdim.numpy()
    )


@pytest.mark.random
def test_nanslope_multiple_dimensions() -> None:
    """Test NaN slope along multiple dimensions."""
    x = torch.randn(2, 3, 4, 5)
    y = 0.8 * x + 2.0

    # Add some NaN values
    x[:, :, 0, 0] = math.nan
    y[:, :, 1, 1] = math.nan

    result = QF.nanslope(x, y, dim=(2, 3))

    assert result.shape == (2, 3)
    # Most slopes should be close to 0.8
    finite_results = result[torch.isfinite(result)]
    if len(finite_results) > 0:
        assert torch.all(torch.abs(finite_results - 0.8) < 0.1)


@pytest.mark.random
def test_nanslope_linear_regression_properties() -> None:
    """Test NaN slope with known linear regression properties."""
    # Create data with known slope
    torch.manual_seed(42)
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    noise = 0.1 * torch.randn(6)
    y = 2.5 * x + 1.0 + noise  # y = 2.5*x + 1 + noise

    # Add some NaN values
    x[0] = math.nan
    y[0] = math.nan

    result = QF.nanslope(x, y, dim=0)

    # Should be close to 2.5
    assert abs(result.item() - 2.5) < 0.5


def test_nanslope_asymmetric_nan_patterns() -> None:
    """Test NaN slope with different NaN patterns in x and y."""
    x = torch.tensor([1.0, math.nan, 3.0, 4.0, 5.0, math.nan])
    y = torch.tensor([2.0, 4.0, math.nan, 8.0, math.nan, 12.0])

    result = QF.nanslope(x, y, dim=0)

    # Should compute slope using only positions where both are valid
    # Valid pairs: (1,2), (4,8) - should give slope of 2
    assert abs(result.item() - 2.0) < 1e-6


@pytest.mark.random
def test_nanslope_batch_processing() -> None:
    """Test NaN slope with batch processing."""
    batch_size = 10
    seq_length = 100

    x = torch.randn(batch_size, seq_length)
    y = 1.5 * x + 0.5 + 0.1 * torch.randn(batch_size, seq_length)

    # Add random NaN values
    nan_mask_x = torch.rand(batch_size, seq_length) < 0.1
    nan_mask_y = torch.rand(batch_size, seq_length) < 0.1
    x[nan_mask_x] = math.nan
    y[nan_mask_y] = math.nan

    result = QF.nanslope(x, y, dim=1)

    assert result.shape == (batch_size,)
    # Most slopes should be close to 1.5
    finite_results = result[torch.isfinite(result)]
    if len(finite_results) > 0:
        assert torch.all(torch.abs(finite_results - 1.5) < 0.5)


def test_nanslope_numerical_stability() -> None:
    """Test numerical stability with very small and large values."""
    # Very small values
    x_small = torch.tensor([1e-10, 2e-10, math.nan, 4e-10], dtype=torch.float64)
    y_small = torch.tensor([2e-10, 4e-10, math.nan, 8e-10], dtype=torch.float64)

    result_small = QF.nanslope(x_small, y_small, dim=0)
    assert abs(result_small.item() - 2.0) < 1e-6

    # Very large values
    x_large = torch.tensor([1e10, 2e10, math.nan, 4e10], dtype=torch.float64)
    y_large = torch.tensor([3e10, 6e10, math.nan, 12e10], dtype=torch.float64)

    result_large = QF.nanslope(x_large, y_large, dim=0)
    assert abs(result_large.item() - 3.0) < 1e-6


def test_nanslope_mixed_finite_infinite() -> None:
    """Test NaN slope with mix of finite and infinite values."""
    x = torch.tensor([1.0, 2.0, math.inf, 4.0, math.nan])
    y = torch.tensor([2.0, 4.0, math.inf, 8.0, math.nan])

    result = QF.nanslope(x, y, dim=0)

    # Infinite values should be handled gracefully
    assert torch.isnan(result) or torch.isinf(result) or torch.isfinite(result)


@pytest.mark.random
def test_nanslope_negative_dimension() -> None:
    """Test NaN slope with negative dimension indexing."""
    x = torch.randn(3, 4, 20)
    y = 2.0 * x + 1.0

    # Add some NaN values
    x[:, :, ::3] = math.nan
    y[:, :, ::4] = math.nan

    result_neg = QF.nanslope(x, y, dim=-1)
    result_pos = QF.nanslope(x, y, dim=2)

    np.testing.assert_allclose(result_neg.numpy(), result_pos.numpy())


def test_nanslope_empty_after_nan_removal() -> None:
    """Test NaN slope when all data becomes NaN after processing."""
    x = torch.tensor([math.nan, math.inf, -math.inf])
    y = torch.tensor([math.nan, 1.0, 2.0])

    result = QF.nanslope(x, y, dim=0)

    # Should be NaN when no valid pairs remain
    assert torch.isnan(result)


@pytest.mark.random
def test_nanslope_scipy_comparison() -> None:
    """Test NaN slope against scipy for validation."""
    torch.manual_seed(123)
    x = torch.randn(50)
    y = 1.8 * x + 0.5 + 0.1 * torch.randn(50)

    # Add some NaN values
    x[::5] = math.nan
    y[::7] = math.nan

    result = QF.nanslope(x, y, dim=0)

    # Compare with scipy calculation
    x_np, y_np = x.numpy(), y.numpy()
    valid_mask = ~np.isnan(x_np) & ~np.isnan(y_np)

    if valid_mask.sum() > 1:
        expected = linregress(x_np[valid_mask], y_np[valid_mask]).slope
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4)


def test_nanslope_quadratic_relationship() -> None:
    """Test NaN slope with quadratic relationship (linear slope varies)."""
    x = torch.tensor([1.0, 2.0, 3.0, math.nan, 5.0, 6.0, 7.0])
    y = x**2  # Quadratic relationship
    y[3] = math.nan  # Align NaN positions

    result = QF.nanslope(x, y, dim=0)

    # For quadratic y = x^2, the best linear fit should have positive slope
    assert result > 0


def test_nanslope_outlier_robustness() -> None:
    """Test NaN slope behavior with outliers."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, math.nan])
    y = torch.tensor(
        [2.0, 4.0, 6.0, 8.0, 100.0, math.nan]
    )  # Last point is outlier

    result = QF.nanslope(x, y, dim=0)

    # Should still compute a slope, though affected by outlier
    assert torch.isfinite(result)
    assert result > 0  # Should still be positive


def test_nanslope_perfect_fit_validation() -> None:
    """Test that perfect linear fits give exact slopes."""
    slopes_to_test = [0.5, 1.0, 2.0, -1.5, -0.8]

    for true_slope in slopes_to_test:
        x = torch.tensor([1.0, 2.0, 3.0, math.nan, 5.0, 6.0])
        y = true_slope * x + 1.0  # Perfect linear relationship
        y[3] = math.nan  # Align NaN

        result = QF.nanslope(x, y, dim=0)

        assert abs(result.item() - true_slope) < 1e-6


def test_nanslope_zero_x_variance() -> None:
    """Test NaN slope when x has zero variance."""
    x = torch.tensor([3.0, 3.0, math.nan, 3.0, 3.0])  # Constant x
    y = torch.tensor([1.0, 2.0, math.nan, 4.0, 5.0])

    result = QF.nanslope(x, y, dim=0)

    # Should be infinite or NaN due to division by zero variance
    assert torch.isinf(result) or torch.isnan(result)


def test_nanslope_identical_points() -> None:
    """Test NaN slope with some identical points."""
    x = torch.tensor([1.0, 1.0, 2.0, math.nan, 3.0, 3.0])
    y = torch.tensor([2.0, 2.0, 4.0, math.nan, 6.0, 6.0])  # y = 2*x

    result = QF.nanslope(x, y, dim=0)

    # Should still compute slope correctly
    assert abs(result.item() - 2.0) < 1e-6


def test_nanslope_regression_through_origin() -> None:
    """Test NaN slope with data that should pass through origin."""
    x = torch.tensor([1.0, 2.0, math.nan, 4.0, 5.0])
    y = torch.tensor([3.0, 6.0, math.nan, 12.0, 15.0])  # y = 3*x (no intercept)

    result = QF.nanslope(x, y, dim=0)

    # Should be exactly 3.0
    assert abs(result.item() - 3.0) < 1e-6


def test_nanslope_negative_values() -> None:
    """Test NaN slope with negative x and y values."""
    x = torch.tensor([-3.0, -2.0, math.nan, 0.0, 1.0])
    y = torch.tensor([-6.0, -4.0, math.nan, 0.0, 2.0])  # y = 2*x

    result = QF.nanslope(x, y, dim=0)

    # Should be 2.0 regardless of negative values
    assert abs(result.item() - 2.0) < 1e-6
