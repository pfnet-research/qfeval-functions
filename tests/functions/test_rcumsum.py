import math

import numpy as np
import torch

import qfeval_functions.functions as QF
import pytest


def test_rcumsum_basic_1d() -> None:
    """Test basic reverse cumulative sum on 1D tensor."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = QF.rcumsum(x, dim=0)
    expected = torch.tensor([10.0, 9.0, 7.0, 4.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_rcumsum_2d_tensor_dim0() -> None:
    """Test reverse cumulative sum on 2D tensor along axis 0."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = QF.rcumsum(x, dim=0)
    expected = torch.tensor([[9.0, 12.0], [8.0, 10.0], [5.0, 6.0]])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_rcumsum_2d_tensor_dim1() -> None:
    """Test reverse cumulative sum on 2D tensor along axis 1."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = QF.rcumsum(x, dim=1)
    expected = torch.tensor([[6.0, 5.0, 3.0], [15.0, 11.0, 6.0]])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_rcumsum_negative_dim() -> None:
    """Test reverse cumulative sum with negative dimension indexing."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = QF.rcumsum(x, dim=-1)
    expected = torch.tensor([[3.0, 2.0], [7.0, 4.0]])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_rcumsum_zeros() -> None:
    """Test reverse cumulative sum with tensor containing only zeros."""
    x = torch.tensor([0.0, 0.0, 0.0, 0.0])
    result = QF.rcumsum(x, dim=0)
    expected = torch.tensor([0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_rcumsum_negative_values() -> None:
    """Test reverse cumulative sum with all negative values."""
    x = torch.tensor([-1.0, -2.0, -3.0])
    result = QF.rcumsum(x, dim=0)
    expected = torch.tensor([-6.0, -5.0, -3.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_rcumsum_mixed_values() -> None:
    """Test reverse cumulative sum with mixed positive and negative values."""
    x = torch.tensor([1.0, -2.0, 3.0, -4.0])
    result = QF.rcumsum(x, dim=0)
    expected = torch.tensor([-2.0, -3.0, -1.0, -4.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_rcumsum_3d_tensor() -> None:
    """Test reverse cumulative sum on 3D tensor along axis 0."""
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    result = QF.rcumsum(x, dim=0)
    expected = torch.tensor(
        [[[6.0, 8.0], [10.0, 12.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_rcumsum_3d_tensor_dim1() -> None:
    """Test reverse cumulative sum on 3D tensor along axis 1."""
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    result = QF.rcumsum(x, dim=1)
    expected = torch.tensor(
        [[[4.0, 6.0], [3.0, 4.0]], [[12.0, 14.0], [7.0, 8.0]]]
    )
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_rcumsum_3d_tensor_dim2() -> None:
    """Test reverse cumulative sum on 3D tensor along axis 2."""
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    result = QF.rcumsum(x, dim=2)
    expected = torch.tensor(
        [[[3.0, 2.0], [7.0, 4.0]], [[11.0, 6.0], [15.0, 8.0]]]
    )
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_rcumsum_integer_tensor() -> None:
    """Test reverse cumulative sum with integer tensor."""
    x = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    result = QF.rcumsum(x, dim=0)
    expected = torch.tensor([10, 9, 7, 4], dtype=torch.int32)
    np.testing.assert_array_equal(result.numpy(), expected.numpy())


@pytest.mark.random
def test_rcumsum_large_tensor() -> None:
    """Test reverse cumulative sum with large tensor to verify scalability."""
    x = torch.randn(100, 50)
    result = QF.rcumsum(x, dim=0)

    # Check that the shape is preserved
    assert result.shape == x.shape

    # Check that the last element is the same as the input
    np.testing.assert_allclose(result[-1].numpy(), x[-1].numpy())


def test_rcumsum_with_nan_values() -> None:
    """Test reverse cumulative sum behavior with NaN values."""
    x = torch.tensor([1.0, math.nan, 3.0, 4.0])
    result = QF.rcumsum(x, dim=0)

    # NaN should propagate
    assert torch.isnan(result[1])
    # Elements after NaN in reverse order might not be NaN
    assert not torch.isnan(result[2]) and not torch.isnan(result[3])


def test_rcumsum_with_infinity() -> None:
    """Test reverse cumulative sum with infinity values."""
    x = torch.tensor([1.0, math.inf, 3.0, 4.0])
    result = QF.rcumsum(x, dim=0)

    # Infinity should propagate
    assert torch.isinf(result[0])
    assert torch.isinf(result[1])
    assert not torch.isinf(result[2]) and not torch.isinf(result[3])


def test_rcumsum_alternating_large_values() -> None:
    """Test reverse cumulative sum with alternating large values."""
    x = torch.tensor([1e10, -1e10, 1e10, -1e10], dtype=torch.float64)
    result = QF.rcumsum(x, dim=0)

    # Should handle large values without overflow
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


def test_rcumsum_numerical_precision() -> None:
    """Test reverse cumulative sum with values requiring high precision."""
    x = torch.tensor([1e-15, 2e-15, 3e-15, 4e-15], dtype=torch.float64)
    result = QF.rcumsum(x, dim=0)

    # Should preserve precision for small values
    expected = torch.tensor([10e-15, 9e-15, 7e-15, 4e-15], dtype=torch.float64)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-20)
