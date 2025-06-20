import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_mulmean() -> None:
    a = QF.randn(100, 1)
    b = QF.randn(1, 200)
    np.testing.assert_allclose(
        (a * b).mean().numpy(),
        QF.mulmean(a, b).numpy(),
        1e-5,
        1e-5,
    )
    np.testing.assert_allclose(
        (a * b).mean(dim=1).numpy(),
        QF.mulmean(a, b, dim=1).numpy(),
        1e-5,
        1e-5,
    )
    np.testing.assert_allclose(
        (a * b).mean(dim=-1, keepdim=True).numpy(),
        QF.mulmean(a, b, dim=-1, keepdim=True).numpy(),
        1e-5,
        1e-5,
    )


def test_mulmean_basic_functionality() -> None:
    """Test basic multiplication and mean functionality."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[2.0, 1.0], [1.0, 2.0]])

    # Test global mean
    result = QF.mulmean(x, y)
    expected = (x * y).mean()
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_mulmean_broadcasting() -> None:
    """Test multiplication with broadcasting."""
    # Test various broadcasting scenarios
    x1 = torch.tensor([[1.0, 2.0, 3.0]])
    y1 = torch.tensor([[2.0], [3.0]])

    result1 = QF.mulmean(x1, y1)
    expected1 = (x1 * y1).mean()
    np.testing.assert_allclose(result1.numpy(), expected1.numpy())

    # Test 1D broadcasting
    x2 = torch.tensor([1.0, 2.0, 3.0])
    y2 = torch.tensor([2.0, 1.0, 3.0])

    result2 = QF.mulmean(x2, y2)
    expected2 = (x2 * y2).mean()
    np.testing.assert_allclose(result2.numpy(), expected2.numpy())


def test_mulmean_dimensions() -> None:
    """Test mean calculation along specific dimensions."""
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    y = torch.tensor([[[2.0, 1.0], [1.0, 2.0]], [[1.0, 3.0], [2.0, 1.0]]])

    # Test along dimension 0
    result_dim0 = QF.mulmean(x, y, dim=0)
    expected_dim0 = (x * y).mean(dim=0)
    np.testing.assert_allclose(result_dim0.numpy(), expected_dim0.numpy())

    # Test along dimension 1
    result_dim1 = QF.mulmean(x, y, dim=1)
    expected_dim1 = (x * y).mean(dim=1)
    np.testing.assert_allclose(result_dim1.numpy(), expected_dim1.numpy())

    # Test along dimension 2
    result_dim2 = QF.mulmean(x, y, dim=2)
    expected_dim2 = (x * y).mean(dim=2)
    np.testing.assert_allclose(result_dim2.numpy(), expected_dim2.numpy())


def test_mulmean_multiple_dimensions() -> None:
    """Test mean calculation along multiple dimensions."""
    x = torch.randn(3, 4, 5)
    y = torch.randn(3, 4, 5)

    # Test along multiple dimensions
    result_01 = QF.mulmean(x, y, dim=(0, 1))
    expected_01 = (x * y).mean(dim=(0, 1))
    np.testing.assert_allclose(
        result_01.numpy(), expected_01.numpy(), rtol=1e-4, atol=1e-4
    )

    result_12 = QF.mulmean(x, y, dim=(1, 2))
    expected_12 = (x * y).mean(dim=(1, 2))
    np.testing.assert_allclose(
        result_12.numpy(), expected_12.numpy(), rtol=1e-4, atol=1e-4
    )


def test_mulmean_keepdim() -> None:
    """Test keepdim parameter functionality."""
    x = torch.randn(3, 4, 5)
    y = torch.randn(3, 4, 5)

    # Test keepdim=True
    result_keepdim = QF.mulmean(x, y, dim=1, keepdim=True)
    expected_keepdim = (x * y).mean(dim=1, keepdim=True)
    np.testing.assert_allclose(
        result_keepdim.numpy(), expected_keepdim.numpy(), rtol=1e-4, atol=1e-4
    )

    # Test keepdim=False
    result_no_keepdim = QF.mulmean(x, y, dim=1, keepdim=False)
    expected_no_keepdim = (x * y).mean(dim=1, keepdim=False)
    np.testing.assert_allclose(
        result_no_keepdim.numpy(), expected_no_keepdim.numpy(), rtol=1e-4, atol=1e-4
    )


def test_mulmean_negative_dimensions() -> None:
    """Test negative dimension indexing."""
    x = torch.randn(3, 4, 5)
    y = torch.randn(3, 4, 5)

    result_neg1 = QF.mulmean(x, y, dim=-1)
    result_pos2 = QF.mulmean(x, y, dim=2)
    np.testing.assert_allclose(result_neg1.numpy(), result_pos2.numpy())

    result_neg2 = QF.mulmean(x, y, dim=-2)
    result_pos1 = QF.mulmean(x, y, dim=1)
    np.testing.assert_allclose(result_neg2.numpy(), result_pos1.numpy())


def test_mulmean_zero_values() -> None:
    """Test multiplication and mean with zero values."""
    x = torch.tensor([[1.0, 0.0, 3.0], [0.0, 2.0, 1.0]])
    y = torch.tensor([[2.0, 1.0, 0.0], [3.0, 0.0, 4.0]])

    result = QF.mulmean(x, y)
    expected = (x * y).mean()
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_mulmean_negative_values() -> None:
    """Test multiplication and mean with negative values."""
    x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
    y = torch.tensor([[2.0, -1.0], [-2.0, 3.0]])

    result = QF.mulmean(x, y)
    expected = (x * y).mean()
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_mulmean_large_values() -> None:
    """Test multiplication and mean with large values."""
    x = torch.tensor([[1e6, 2e6], [3e6, 4e6]])
    y = torch.tensor([[2e6, 1e6], [1e6, 2e6]])

    result = QF.mulmean(x, y)
    expected = (x * y).mean()
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)


def test_mulmean_small_values() -> None:
    """Test multiplication and mean with very small values."""
    x = torch.tensor([[1e-6, 2e-6], [3e-6, 4e-6]])
    y = torch.tensor([[2e-6, 1e-6], [1e-6, 2e-6]])

    result = QF.mulmean(x, y)
    expected = (x * y).mean()
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)


def test_mulmean_broadcasting_edge_cases() -> None:
    """Test edge cases in broadcasting."""
    # Scalar with tensor
    x_scalar = torch.tensor(2.0)
    y_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    result1 = QF.mulmean(x_scalar, y_tensor)
    expected1 = (x_scalar * y_tensor).mean()
    np.testing.assert_allclose(result1.numpy(), expected1.numpy())

    # Different shape broadcasting
    x_broadcast = torch.tensor([[[1.0]], [[2.0]]])
    y_broadcast = torch.tensor([[[3.0, 4.0, 5.0]]])

    result2 = QF.mulmean(x_broadcast, y_broadcast)
    expected2 = (x_broadcast * y_broadcast).mean()
    np.testing.assert_allclose(result2.numpy(), expected2.numpy())


def test_mulmean_with_nan_values() -> None:
    """Test mulmean behavior with NaN values."""
    x = torch.tensor([[1.0, math.nan], [3.0, 4.0]])
    y = torch.tensor([[2.0, 1.0], [1.0, 2.0]])

    result = QF.mulmean(x, y)
    expected = (x * y).mean()

    # Both should be NaN
    assert torch.isnan(result)
    assert torch.isnan(expected)


def test_mulmean_with_infinity() -> None:
    """Test mulmean behavior with infinity values."""
    x = torch.tensor([[1.0, math.inf], [3.0, 4.0]])
    y = torch.tensor([[2.0, 1.0], [1.0, 2.0]])

    result = QF.mulmean(x, y)
    expected = (x * y).mean()

    # Both should handle infinity consistently
    assert torch.isinf(result) or torch.isnan(result)
    assert torch.isinf(expected) or torch.isnan(expected)


def test_mulmean_high_dimensional() -> None:
    """Test mulmean with high-dimensional tensors."""
    x = torch.randn(2, 3, 4, 5, 6)
    y = torch.randn(2, 3, 4, 5, 6)

    result = QF.mulmean(x, y)
    expected = (x * y).mean()
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)

    # Test along specific high dimensions
    result_dim3 = QF.mulmean(x, y, dim=3)
    expected_dim3 = (x * y).mean(dim=3)
    np.testing.assert_allclose(
        result_dim3.numpy(), expected_dim3.numpy(), rtol=1e-4, atol=1e-4
    )


def test_mulmean_empty_dimension() -> None:
    """Test mulmean with empty dimension tuple."""
    x = torch.randn(3, 4)
    y = torch.randn(3, 4)

    # Empty dimension tuple should compute global mean
    result = QF.mulmean(x, y, dim=())
    expected = (x * y).mean()
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)


def test_mulmean_mathematical_properties() -> None:
    """Test mathematical properties of mulmean."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[2.0, 1.0], [1.0, 2.0]])

    # Test linearity: mulmean(ax, y) = a * mulmean(x, y)
    a = 3.0
    result1 = QF.mulmean(a * x, y)
    result2 = a * QF.mulmean(x, y)
    np.testing.assert_allclose(result1.numpy(), result2.numpy())

    # Test commutativity: mulmean(x, y) = mulmean(y, x)
    result_xy = QF.mulmean(x, y)
    result_yx = QF.mulmean(y, x)
    np.testing.assert_allclose(result_xy.numpy(), result_yx.numpy())


def test_mulmean_numerical_stability() -> None:
    """Test numerical stability of mulmean."""
    # Test with values that might cause overflow in naive multiplication
    x = torch.tensor([[1e10, 1e-10], [1e10, 1e-10]])
    y = torch.tensor([[1e-10, 1e10], [1e-10, 1e10]])

    result = QF.mulmean(x, y)
    expected = (x * y).mean()

    # Both should be finite and equal
    assert torch.isfinite(result)
    assert torch.isfinite(expected)
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_mulmean_batch_processing() -> None:
    """Test mulmean with batch processing scenarios."""
    batch_size = 4
    seq_length = 10
    features = 8

    x = torch.randn(batch_size, seq_length, features)
    y = torch.randn(batch_size, seq_length, features)

    # Test batch-wise mean (across sequence and features)
    result_batch = QF.mulmean(x, y, dim=(1, 2))
    expected_batch = (x * y).mean(dim=(1, 2))
    np.testing.assert_allclose(
        result_batch.numpy(), expected_batch.numpy(), rtol=1e-4, atol=1e-4
    )

    assert result_batch.shape == (batch_size,)


def test_mulmean_zero_size_tensor() -> None:
    """Test mulmean with zero-size tensor."""
    x = torch.empty(0, 3)
    y = torch.empty(0, 3)

    result = QF.mulmean(x, y)
    expected = (x * y).mean()

    # Both should handle empty tensors consistently
    assert torch.isnan(result) == torch.isnan(expected)
