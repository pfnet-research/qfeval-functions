import math

import numpy as np
import pytest
import torch

import qfeval_functions.functions as QF


@pytest.mark.parametrize("ddof", [0, 1])
def test_covar_basic_numpy_comparison(ddof: int) -> None:
    """Test basic covariance functionality against numpy implementation."""
    a = QF.randn(10, 1, 200)
    b = QF.randn(1, 20, 200)
    actual = QF.covar(a, b, ddof=ddof)
    expected = np.zeros((10, 20))
    for i in range(10):
        for j in range(20):
            xa, xb = a[i, 0].numpy(), b[0, j].numpy()
            expected[i, j] = np.cov(xa, xb, ddof=ddof)[0, 1]
    np.testing.assert_allclose(actual, expected, 1e-4, 1e-4)


def test_covar_1d_tensors() -> None:
    """Test covariance with 1D tensors."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])  # Perfect positive correlation

    result = QF.covar(x, y, dim=0)
    expected = np.cov(x.numpy(), y.numpy())[0, 1]

    np.testing.assert_allclose(result.numpy(), expected)


def test_covar_identical_tensors() -> None:
    """Test covariance of tensor with itself (should equal variance)."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    covar_result = QF.covar(x, x, dim=0, ddof=1)
    var_result = torch.var(x, dim=0, unbiased=True)

    np.testing.assert_allclose(
        covar_result.numpy(), var_result.numpy()
    )


def test_covar_2d_tensors() -> None:
    """Test covariance on 2D tensors along different dimensions."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = torch.tensor([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])

    # Test along dim=1
    result = QF.covar(x, y, dim=1)
    assert result.shape == (2,)

    # Manually verify first row
    expected_0 = np.cov(x[0].numpy(), y[0].numpy())[0, 1]
    np.testing.assert_allclose(result[0].numpy(), expected_0)


def test_covar_3d_tensors() -> None:
    """Test covariance on 3D tensors."""
    x = torch.randn(5, 4, 10)
    y = torch.randn(5, 4, 10)

    # Test along last dimension
    result = QF.covar(x, y, dim=2)
    assert result.shape == (5, 4)

    # Verify a specific element
    expected = np.cov(x[0, 0].numpy(), y[0, 0].numpy())[0, 1]
    # Use appropriate tolerance for float32 precision
    np.testing.assert_allclose(result[0, 0].numpy(), expected, rtol=1e-4, atol=1e-4)


def test_covar_negative_dim() -> None:
    """Test covariance with negative dimension indexing."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = torch.tensor([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])

    result_neg = QF.covar(x, y, dim=-1)
    result_pos = QF.covar(x, y, dim=1)

    np.testing.assert_allclose(result_neg.numpy(), result_pos.numpy())


def test_covar_keepdim() -> None:
    """Test covariance with keepdim parameter."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = torch.tensor([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])

    result_keepdim = QF.covar(x, y, dim=1, keepdim=True)
    result_no_keepdim = QF.covar(x, y, dim=1, keepdim=False)

    assert result_keepdim.shape == (2, 1)
    assert result_no_keepdim.shape == (2,)
    np.testing.assert_allclose(
        result_keepdim.squeeze().numpy(), result_no_keepdim.numpy()
    )


def test_covar_ddof_values() -> None:
    """Test covariance with different ddof values."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])

    # Test ddof=0 (population covariance)
    result_ddof0 = QF.covar(x, y, dim=0, ddof=0)
    expected_ddof0 = np.cov(x.numpy(), y.numpy(), ddof=0)[0, 1]
    np.testing.assert_allclose(result_ddof0.numpy(), expected_ddof0)

    # Test ddof=1 (sample covariance)
    result_ddof1 = QF.covar(x, y, dim=0, ddof=1)
    expected_ddof1 = np.cov(x.numpy(), y.numpy(), ddof=1)[0, 1]
    np.testing.assert_allclose(result_ddof1.numpy(), expected_ddof1)

    # ddof=0 should be smaller than ddof=1 for positive covariance
    assert result_ddof0 < result_ddof1


def test_covar_zero_variance() -> None:
    """Test covariance when one or both tensors have zero variance."""
    # Constant tensor (zero variance)
    x_const = torch.tensor([5.0, 5.0, 5.0, 5.0])
    x_varying = torch.tensor([1.0, 2.0, 3.0, 4.0])

    result = QF.covar(x_const, x_varying, dim=0)

    # Covariance with constant should be zero
    assert abs(result.item()) < 1e-10

    # Both constant
    y_const = torch.tensor([3.0, 3.0, 3.0, 3.0])
    result_both_const = QF.covar(x_const, y_const, dim=0)
    assert abs(result_both_const.item()) < 1e-10


def test_covar_perfect_correlation() -> None:
    """Test covariance with perfectly correlated data."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    # Perfect positive correlation
    y_pos = 2.0 * x + 1.0  # y = 2x + 1
    result_pos = QF.covar(x, y_pos, dim=0)

    # Perfect negative correlation
    y_neg = -2.0 * x + 10.0  # y = -2x + 10
    result_neg = QF.covar(x, y_neg, dim=0)

    # Should have opposite signs
    assert result_pos > 0
    assert result_neg < 0
    assert abs(result_pos + result_neg) < 1e-10  # Should be symmetric


def test_covar_broadcasting() -> None:
    """Test covariance with broadcasting tensors."""
    x = torch.randn(5, 1, 20)  # (5, 1, 20)
    y = torch.randn(1, 3, 20)  # (1, 3, 20)

    result = QF.covar(x, y, dim=2)  # Should broadcast to (5, 3)
    assert result.shape == (5, 3)

    # Verify specific element matches numpy
    expected = np.cov(x[0, 0].numpy(), y[0, 0].numpy())[0, 1]
    # Use appropriate tolerance for float32 precision
    np.testing.assert_allclose(
        result[0, 0].numpy(), expected, rtol=1e-4, atol=1e-4
    )


def test_covar_with_nan_values() -> None:
    """Test covariance behavior with NaN values."""
    x = torch.tensor([1.0, math.nan, 3.0, 4.0])
    y = torch.tensor([2.0, 4.0, 6.0, 8.0])

    result = QF.covar(x, y, dim=0)

    # Result should be NaN when input contains NaN
    assert torch.isnan(result)


def test_covar_with_infinity() -> None:
    """Test covariance behavior with infinity values."""
    x = torch.tensor([1.0, 2.0, math.inf, 4.0])
    y = torch.tensor([2.0, 4.0, 6.0, 8.0])

    result = QF.covar(x, y, dim=0)

    # Result should be NaN when input contains infinity
    assert torch.isnan(result) or torch.isinf(result)


def test_covar_two_elements() -> None:
    """Test covariance with two element tensors."""
    x = torch.tensor([1.0, 3.0])
    y = torch.tensor([2.0, 6.0])

    result = QF.covar(x, y, dim=0, ddof=1)
    expected = np.cov(x.numpy(), y.numpy(), ddof=1)[0, 1]

    np.testing.assert_allclose(result.numpy(), expected)


def test_covar_large_tensors() -> None:
    """Test covariance with large tensors for performance verification."""
    x = torch.randn(1000)
    y = torch.randn(1000)

    result = QF.covar(x, y, dim=0)

    # Should complete without error and be finite
    assert torch.isfinite(result)


def test_covar_numerical_precision() -> None:
    """Test covariance with values requiring high numerical precision."""
    x = torch.tensor([1.0000001, 1.0000002, 1.0000003], dtype=torch.float64)
    y = torch.tensor([2.0000002, 2.0000004, 2.0000006], dtype=torch.float64)

    result = QF.covar(x, y, dim=0)
    expected = np.cov(x.numpy(), y.numpy())[0, 1]

    np.testing.assert_allclose(result.numpy(), expected)


def test_covar_very_small_values() -> None:
    """Test covariance with very small values."""
    x = torch.tensor([1e-10, 2e-10, 3e-10], dtype=torch.float64)
    y = torch.tensor([2e-10, 4e-10, 6e-10], dtype=torch.float64)

    result = QF.covar(x, y, dim=0)
    expected = np.cov(x.numpy(), y.numpy())[0, 1]

    np.testing.assert_allclose(result.numpy(), expected)


def test_covar_very_large_values() -> None:
    """Test covariance with very large values."""
    x = torch.tensor([1e10, 2e10, 3e10], dtype=torch.float64)
    y = torch.tensor([2e10, 4e10, 6e10], dtype=torch.float64)

    result = QF.covar(x, y, dim=0)
    expected = np.cov(x.numpy(), y.numpy())[0, 1]

    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4, atol=1e-4)


def test_covar_uncorrelated_data() -> None:
    """Test covariance with uncorrelated random data."""
    torch.manual_seed(42)  # For reproducibility
    x = torch.randn(1000)
    y = torch.randn(1000)  # Independent random data

    result = QF.covar(x, y, dim=0)

    # Should be close to zero for large uncorrelated samples
    assert abs(result.item()) < 0.1


def test_covar_batch_different_ddof() -> None:
    """Test covariance with batch processing and different ddof values."""
    batch_size = 5
    seq_length = 100
    x = torch.randn(batch_size, seq_length)
    y = torch.randn(batch_size, seq_length)

    result_ddof0 = QF.covar(x, y, dim=1, ddof=0)
    result_ddof1 = QF.covar(x, y, dim=1, ddof=1)

    assert result_ddof0.shape == (batch_size,)
    assert result_ddof1.shape == (batch_size,)

    # For large samples, difference should be small but ddof=1 should be larger
    # (assuming positive covariance)
    diff_ratio = (result_ddof1 - result_ddof0) / result_ddof0
    # The ratio should be approximately 1/(n-1) where n is seq_length
    expected_ratio = 1.0 / (seq_length - 1)
    # Allow some tolerance for negative covariances and numerical precision
    assert torch.allclose(
        diff_ratio.abs(), torch.tensor(expected_ratio), rtol=0.5
    )


def test_covar_symmetry() -> None:
    """Test that covariance is symmetric: cov(x,y) = cov(y,x)."""
    x = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0])
    y = torch.tensor([2.0, 6.0, 4.0, 10.0, 8.0])

    result_xy = QF.covar(x, y, dim=0)
    result_yx = QF.covar(y, x, dim=0)

    np.testing.assert_allclose(result_xy.numpy(), result_yx.numpy())


def test_covar_linear_transformation() -> None:
    """Test covariance properties under linear transformation."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])

    # Original covariance
    cov_xy = QF.covar(x, y, dim=0)

    # Linear transformations
    a, b = 2.0, 3.0
    c, d = -1.0, 5.0

    x_transformed = a * x + b
    y_transformed = c * y + d

    cov_transformed = QF.covar(x_transformed, y_transformed, dim=0)
    expected_transformed = a * c * cov_xy  # cov(ax+b, cy+d) = ac * cov(x,y)

    np.testing.assert_allclose(
        cov_transformed.numpy(), expected_transformed.numpy()
    )


def test_covar_mixed_signs() -> None:
    """Test covariance with mixed positive and negative values."""
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = torch.tensor([-4.0, -2.0, 0.0, 2.0, 4.0])  # y = 2*x

    result = QF.covar(x, y, dim=0)
    expected = np.cov(x.numpy(), y.numpy())[0, 1]

    np.testing.assert_allclose(result.numpy(), expected)
    # Should be positive since they're positively correlated
    assert result > 0


def test_covar_high_dimensional() -> None:
    """Test covariance with high-dimensional data."""
    x = torch.randn(2, 3, 4, 50)
    y = torch.randn(2, 3, 4, 50)

    result = QF.covar(x, y, dim=3)
    assert result.shape == (2, 3, 4)

    # Verify one element
    expected = np.cov(x[0, 0, 0].numpy(), y[0, 0, 0].numpy())[0, 1]
    # Use appropriate tolerance for float32 precision
    np.testing.assert_allclose(result[0, 0, 0].numpy(), expected, rtol=1e-4, atol=1e-4)
