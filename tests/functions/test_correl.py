import math

import numpy as np
import torch
from scipy.stats import linregress

import qfeval_functions.functions as QF


def test_correl_basic_scipy_comparison() -> None:
    """Test basic correlation functionality against scipy implementation."""
    a = QF.randn(100, 200)
    b = QF.randn(100, 200)
    actual = QF.correl(a, b, dim=1)
    expected = np.zeros((100,))
    for i in range(a.shape[0]):
        expected[i] = linregress(a[i].numpy(), b[i].numpy()).rvalue
    np.testing.assert_allclose(actual, expected, 1e-6, 1e-6)


def test_correl_perfect_correlation() -> None:
    """Test correlation with perfectly correlated data."""
    # Perfect positive correlation
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])  # y = 2*x
    result = QF.correl(x, y, dim=0)
    np.testing.assert_allclose(result.numpy(), 1.0, atol=1e-10)

    # Perfect negative correlation
    y_neg = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0])  # y = -2*x + 12
    result_neg = QF.correl(x, y_neg, dim=0)
    np.testing.assert_allclose(result_neg.numpy(), -1.0, atol=1e-10)


def test_correl_zero_correlation() -> None:
    """Test correlation with uncorrelated data."""
    # Create truly uncorrelated data
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = torch.tensor(
        [2.0, 4.0, 1.0, 5.0, 3.0]
    )  # Different permutation with lower correlation
    result = QF.correl(x, y, dim=0)
    # Should be close to zero but allow for some correlation due to small sample
    assert abs(result.numpy()) < 0.6


def test_correl_identical_arrays() -> None:
    """Test correlation of array with itself."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = QF.correl(x, x, dim=0)
    np.testing.assert_allclose(result.numpy(), 1.0, atol=1e-10)


def test_correl_2d_tensors() -> None:
    """Test correlation on 2D tensors along different dimensions."""
    # Create test data with known correlations
    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], [1.0, 3.0, 2.0, 4.0]]
    )
    y = torch.tensor(
        [
            [2.0, 4.0, 6.0, 8.0],  # Perfect positive correlation with row 0
            [1.0, 2.0, 3.0, 4.0],  # Perfect positive correlation with row 1
            [4.0, 2.0, 3.0, 1.0],  # Some correlation with row 2
        ]
    )

    # Test along dim=1 (correlate along columns for each row)
    result = QF.correl(x, y, dim=1)
    assert result.shape == (3,)

    # First two rows should have perfect correlation
    np.testing.assert_allclose(result[0].numpy(), 1.0, atol=1e-10)
    np.testing.assert_allclose(result[1].numpy(), 1.0, atol=1e-10)


def test_correl_3d_tensors() -> None:
    """Test correlation on 3D tensors."""
    x = torch.randn(5, 10, 20)
    y = torch.randn(5, 10, 20)

    # Test along last dimension
    result = QF.correl(x, y, dim=2)
    assert result.shape == (5, 10)

    # Test along middle dimension
    result = QF.correl(x, y, dim=1)
    assert result.shape == (5, 20)


def test_correl_negative_dim() -> None:
    """Test correlation with negative dimension indexing."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = torch.tensor([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])

    result = QF.correl(x, y, dim=-1)
    expected = QF.correl(x, y, dim=1)
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_correl_constant_arrays() -> None:
    """Test correlation with constant arrays."""
    # Both constant - correlation undefined
    x = torch.tensor([2.0, 2.0, 2.0, 2.0])
    y = torch.tensor([3.0, 3.0, 3.0, 3.0])
    result = QF.correl(x, y, dim=0)
    assert torch.isnan(result)

    # One constant, one varying - correlation undefined
    x_const = torch.tensor([2.0, 2.0, 2.0, 2.0])
    y_vary = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = QF.correl(x_const, y_vary, dim=0)
    assert torch.isnan(result)


def test_correl_with_nan_values() -> None:
    """Test correlation behavior with NaN values."""
    x = torch.tensor([1.0, 2.0, math.nan, 4.0, 5.0])
    y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
    result = QF.correl(x, y, dim=0)
    # Result should be NaN when input contains NaN
    assert torch.isnan(result)


def test_correl_with_infinity() -> None:
    """Test correlation behavior with infinity values."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, math.inf])
    y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
    result = QF.correl(x, y, dim=0)
    # Result should be NaN when input contains infinity
    assert torch.isnan(result)


def test_correl_numerical_precision() -> None:
    """Test correlation with high precision requirements."""
    # Use double precision for better numerical accuracy
    x = torch.tensor(
        [1.0000001, 1.0000002, 1.0000003, 1.0000004], dtype=torch.float64
    )
    y = torch.tensor(
        [2.0000002, 2.0000004, 2.0000006, 2.0000008], dtype=torch.float64
    )
    result = QF.correl(x, y, dim=0)
    # Should be very close to 1.0 despite small differences
    np.testing.assert_allclose(result.numpy(), 1.0, atol=1e-10)


def test_correl_very_small_values() -> None:
    """Test correlation with very small values."""
    x = torch.tensor([1e-20, 2e-20, 3e-20, 4e-20], dtype=torch.float64)
    y = torch.tensor([2e-20, 4e-20, 6e-20, 8e-20], dtype=torch.float64)
    result = QF.correl(x, y, dim=0)
    np.testing.assert_allclose(result.numpy(), 1.0, atol=1e-10)


def test_correl_very_large_values() -> None:
    """Test correlation with very large values."""
    x = torch.tensor([1e20, 2e20, 3e20, 4e20], dtype=torch.float64)
    y = torch.tensor([2e20, 4e20, 6e20, 8e20], dtype=torch.float64)
    result = QF.correl(x, y, dim=0)
    np.testing.assert_allclose(result.numpy(), 1.0, atol=1e-10)


def test_correl_large_tensors() -> None:
    """Test correlation with large tensors."""
    size = 10000
    # Create linearly related data
    x = torch.randn(size, dtype=torch.float64)
    y = 2.0 * x + 0.1 * torch.randn(
        size, dtype=torch.float64
    )  # Strong correlation

    result = QF.correl(x, y, dim=0)
    # Should be strongly correlated (close to 1)
    assert result > 0.95


def test_correl_batch_processing() -> None:
    """Test correlation with multiple batches."""
    # Create multiple correlation scenarios
    batch_size = 50
    seq_length = 100

    x = torch.randn(batch_size, seq_length)
    y = torch.randn(batch_size, seq_length)

    result = QF.correl(x, y, dim=1)
    assert result.shape == (batch_size,)

    # All correlations should be finite (not NaN unless degenerate case)
    finite_mask = torch.isfinite(result)
    # Most should be finite
    assert torch.sum(finite_mask) > batch_size * 0.9


def test_correl_mixed_positive_negative() -> None:
    """Test correlation with mixed positive and negative values."""
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = torch.tensor([-4.0, -2.0, 0.0, 2.0, 4.0])  # y = 2*x
    result = QF.correl(x, y, dim=0)
    np.testing.assert_allclose(result.numpy(), 1.0, atol=1e-10)


def test_correl_asymmetric_relationship() -> None:
    """Test correlation is symmetric: correl(x,y) = correl(y,x)."""
    x = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0])
    y = torch.tensor([2.0, 6.0, 4.0, 10.0, 8.0])

    result_xy = QF.correl(x, y, dim=0)
    result_yx = QF.correl(y, x, dim=0)

    np.testing.assert_allclose(result_xy.numpy(), result_yx.numpy(), atol=1e-10)
