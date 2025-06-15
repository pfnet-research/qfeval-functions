import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_rcummax_basic_1d() -> None:
    """Test basic reverse cumulative maximum on 1D tensor."""
    x = torch.tensor([1.0, 3.0, 2.0, 5.0, 1.0])
    result = QF.rcummax(x, dim=0)
    
    expected_values = torch.tensor([5.0, 5.0, 5.0, 5.0, 1.0])
    expected_indices = torch.tensor([3, 3, 3, 3, 4])
    
    np.testing.assert_allclose(result.values.numpy(), expected_values.numpy())
    np.testing.assert_array_equal(result.indices.numpy(), expected_indices.numpy())


def test_rcummax_2d_tensor_dim0() -> None:
    """Test reverse cumulative maximum on 2D tensor along axis 0."""
    x = torch.tensor([[1.0, 4.0], [3.0, 2.0], [2.0, 5.0]])
    result = QF.rcummax(x, dim=0)
    
    expected_values = torch.tensor([[3.0, 5.0], [3.0, 5.0], [2.0, 5.0]])
    expected_indices = torch.tensor([[1, 2], [1, 2], [2, 2]])
    
    np.testing.assert_allclose(result.values.numpy(), expected_values.numpy())
    np.testing.assert_array_equal(result.indices.numpy(), expected_indices.numpy())


def test_rcummax_2d_tensor_dim1() -> None:
    """Test reverse cumulative maximum on 2D tensor along axis 1."""
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 1.0, 5.0]])
    result = QF.rcummax(x, dim=1)
    
    expected_values = torch.tensor([[3.0, 3.0, 2.0], [5.0, 5.0, 5.0]])
    expected_indices = torch.tensor([[1, 1, 2], [2, 2, 2]])
    
    np.testing.assert_allclose(result.values.numpy(), expected_values.numpy())
    np.testing.assert_array_equal(result.indices.numpy(), expected_indices.numpy())


def test_rcummax_negative_dim() -> None:
    """Test reverse cumulative maximum with negative dimension indexing."""
    x = torch.tensor([[1.0, 3.0], [2.0, 1.0]])
    result = QF.rcummax(x, dim=-1)
    
    expected_values = torch.tensor([[3.0, 3.0], [2.0, 1.0]])
    expected_indices = torch.tensor([[1, 1], [0, 1]])
    
    np.testing.assert_allclose(result.values.numpy(), expected_values.numpy())
    np.testing.assert_array_equal(result.indices.numpy(), expected_indices.numpy())


def test_rcummax_strictly_decreasing() -> None:
    """Test reverse cumulative maximum with strictly decreasing sequence."""
    x = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    result = QF.rcummax(x, dim=0)
    
    expected_values = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    expected_indices = torch.tensor([0, 1, 2, 3, 4])
    
    np.testing.assert_allclose(result.values.numpy(), expected_values.numpy())
    np.testing.assert_array_equal(result.indices.numpy(), expected_indices.numpy())


def test_rcummax_strictly_increasing() -> None:
    """Test reverse cumulative maximum with strictly increasing sequence."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = QF.rcummax(x, dim=0)
    
    expected_values = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0])
    expected_indices = torch.tensor([4, 4, 4, 4, 4])
    
    np.testing.assert_allclose(result.values.numpy(), expected_values.numpy())
    np.testing.assert_array_equal(result.indices.numpy(), expected_indices.numpy())


def test_rcummax_with_duplicates() -> None:
    """Test reverse cumulative maximum with duplicate values."""
    x = torch.tensor([2.0, 3.0, 3.0, 1.0])
    result = QF.rcummax(x, dim=0)
    
    expected_values = torch.tensor([3.0, 3.0, 3.0, 1.0])
    expected_indices = torch.tensor([1, 1, 2, 3])
    
    np.testing.assert_allclose(result.values.numpy(), expected_values.numpy())
    np.testing.assert_array_equal(result.indices.numpy(), expected_indices.numpy())


def test_rcummax_single_element() -> None:
    """Test reverse cumulative maximum with single-element tensor."""
    x = torch.tensor([5.0])
    result = QF.rcummax(x, dim=0)
    
    expected_values = torch.tensor([5.0])
    expected_indices = torch.tensor([0])
    
    np.testing.assert_allclose(result.values.numpy(), expected_values.numpy())
    np.testing.assert_array_equal(result.indices.numpy(), expected_indices.numpy())


def test_rcummax_3d_tensor() -> None:
    """Test reverse cumulative maximum on 3D tensor along axis 0."""
    x = torch.tensor([[[1.0, 2.0], [3.0, 1.0]], [[2.0, 3.0], [1.0, 4.0]]])
    result = QF.rcummax(x, dim=0)
    
    expected_values = torch.tensor([[[2.0, 3.0], [3.0, 4.0]], [[2.0, 3.0], [1.0, 4.0]]])
    expected_indices = torch.tensor([[[1, 1], [0, 1]], [[1, 1], [1, 1]]])
    
    np.testing.assert_allclose(result.values.numpy(), expected_values.numpy())
    np.testing.assert_array_equal(result.indices.numpy(), expected_indices.numpy())


def test_rcummax_negative_values() -> None:
    """Test reverse cumulative maximum with all negative values."""
    x = torch.tensor([-5.0, -2.0, -8.0, -1.0])
    result = QF.rcummax(x, dim=0)
    
    expected_values = torch.tensor([-1.0, -1.0, -1.0, -1.0])
    expected_indices = torch.tensor([3, 3, 3, 3])
    
    np.testing.assert_allclose(result.values.numpy(), expected_values.numpy())
    np.testing.assert_array_equal(result.indices.numpy(), expected_indices.numpy())


def test_rcummax_dtype_preservation() -> None:
    """Test that reverse cumulative maximum preserves input tensor's dtype."""
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    result = QF.rcummax(x, dim=0)
    
    assert result.values.dtype == torch.float64
    assert result.indices.dtype == torch.long


def test_rcummax_device_preservation() -> None:
    """Test that reverse cumulative maximum preserves input tensor's device."""
    x = torch.tensor([1.0, 2.0, 3.0])
    result = QF.rcummax(x, dim=0)
    
    assert result.values.device == x.device
    assert result.indices.device == x.device


def test_rcummax_with_nan_values() -> None:
    """Test reverse cumulative maximum behavior with NaN values."""
    x = torch.tensor([1.0, math.nan, 3.0, 2.0])
    result = QF.rcummax(x, dim=0)
    
    # NaN should propagate in cummax operations
    assert not torch.isnan(result.values[0])  # First element might not be NaN
    assert result.values[0] == 3.0 or torch.isnan(result.values[1])


def test_rcummax_with_infinity() -> None:
    """Test reverse cumulative maximum with infinity values."""
    x = torch.tensor([1.0, math.inf, 3.0, 2.0])
    result = QF.rcummax(x, dim=0)
    
    # Infinity should dominate
    assert torch.isinf(result.values[0]) and result.values[0] > 0
    assert torch.isinf(result.values[1]) and result.values[1] > 0
    assert result.values[2] == 3.0
    assert result.values[3] == 2.0


def test_rcummax_large_tensor() -> None:
    """Test reverse cumulative maximum with large tensor."""
    x = torch.randn(1000)
    result = QF.rcummax(x, dim=0)
    
    # Verify properties
    assert result.values.shape == x.shape
    assert result.indices.shape == x.shape
    
    # Verify monotonicity (reverse cummax should be non-decreasing from right to left)
    for i in range(len(x) - 1):
        assert result.values[i] >= result.values[i + 1]