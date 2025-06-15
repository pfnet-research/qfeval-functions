import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_bfill_1d_basic() -> None:
    """Test basic backward fill functionality on 1D tensor."""
    x = torch.tensor(
        [math.nan, 1.0, math.nan, 2.0, 3.0, math.nan, 4.0, math.nan]
    )
    result = QF.bfill(x, dim=0)
    expected = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0, math.nan])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_bfill_2d_tensor_dim0() -> None:
    """Test backward fill on 2D tensor along axis 0."""
    x = torch.tensor(
        [
            [math.nan, 1.0, 2.0],
            [3.0, math.nan, math.nan],
            [math.nan, 4.0, 5.0],
            [6.0, 7.0, math.nan],
        ]
    )
    result = QF.bfill(x, dim=0)
    expected = torch.tensor(
        [
            [3.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 4.0, 5.0],
            [6.0, 7.0, math.nan],
        ]
    )
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_bfill_2d_tensor_dim1() -> None:
    """Test backward fill on 2D tensor along axis 1."""
    x = torch.tensor(
        [
            [math.nan, 1.0, math.nan, 2.0],
            [3.0, math.nan, 4.0, math.nan],
            [math.nan, math.nan, 5.0, 6.0],
        ]
    )
    result = QF.bfill(x, dim=1)
    expected = torch.tensor(
        [[1.0, 1.0, 2.0, 2.0], [3.0, 4.0, 4.0, math.nan], [5.0, 5.0, 5.0, 6.0]]
    )
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_bfill_3d_tensor() -> None:
    """Test backward fill on 3D tensor."""
    x = torch.tensor(
        [
            [[math.nan, 1.0], [2.0, math.nan]],
            [[3.0, math.nan], [math.nan, 4.0]],
            [[math.nan, 5.0], [6.0, math.nan]],
        ]
    )
    result = QF.bfill(x, dim=0)
    expected = torch.tensor(
        [
            [[3.0, 1.0], [2.0, 4.0]],
            [[3.0, 5.0], [6.0, 4.0]],
            [[math.nan, 5.0], [6.0, math.nan]],
        ]
    )
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_bfill_negative_dim() -> None:
    """Test backward fill with negative dimension indexing."""
    x = torch.tensor([[math.nan, 1.0, math.nan], [2.0, math.nan, 3.0]])
    result = QF.bfill(x, dim=-1)
    expected = torch.tensor([[1.0, 1.0, math.nan], [2.0, 3.0, 3.0]])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_bfill_all_nan() -> None:
    """Test backward fill with tensor containing only NaN values."""
    x = torch.tensor([math.nan, math.nan, math.nan, math.nan])
    result = QF.bfill(x, dim=0)
    expected = torch.tensor([math.nan, math.nan, math.nan, math.nan])
    assert torch.isnan(result).all()


def test_bfill_no_nan() -> None:
    """Test backward fill with tensor containing no NaN values."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = QF.bfill(x, dim=0)
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_bfill_single_element() -> None:
    """Test backward fill with single-element tensor."""
    # Single NaN
    x1 = torch.tensor([math.nan])
    result1 = QF.bfill(x1, dim=0)
    assert torch.isnan(result1[0])

    # Single value
    x2 = torch.tensor([5.0])
    result2 = QF.bfill(x2, dim=0)
    assert result2[0] == 5.0


def test_bfill_leading_nans() -> None:
    """Test backward fill with leading NaN values."""
    x = torch.tensor([math.nan, math.nan, 1.0, 2.0, math.nan, 3.0])
    result = QF.bfill(x, dim=0)
    expected = torch.tensor([1.0, 1.0, 1.0, 2.0, 3.0, 3.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_bfill_trailing_nans() -> None:
    """Test backward fill with trailing NaN values."""
    x = torch.tensor([1.0, 2.0, math.nan, math.nan, math.nan])
    result = QF.bfill(x, dim=0)
    expected = torch.tensor([1.0, 2.0, math.nan, math.nan, math.nan])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_bfill_dtype_preservation() -> None:
    """Test that backward fill preserves input tensor's dtype."""
    # Test float64
    x_double = torch.tensor([math.nan, 1.0, math.nan], dtype=torch.float64)
    result_double = QF.bfill(x_double, dim=0)
    assert result_double.dtype == torch.float64

    # Test float32
    x_float = torch.tensor([math.nan, 1.0, math.nan], dtype=torch.float32)
    result_float = QF.bfill(x_float, dim=0)
    assert result_float.dtype == torch.float32


def test_bfill_device_preservation() -> None:
    """Test that backward fill preserves input tensor's device."""
    x = torch.tensor([math.nan, 1.0, math.nan])
    result = QF.bfill(x, dim=0)
    assert result.device == x.device


def test_bfill_mixed_nan_and_inf() -> None:
    """Test backward fill with mixed NaN and infinity values."""
    x = torch.tensor([math.nan, math.inf, math.nan, 1.0, math.nan, -math.inf])
    result = QF.bfill(x, dim=0)
    expected = torch.tensor(
        [math.inf, math.inf, 1.0, 1.0, -math.inf, -math.inf]
    )
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_bfill_large_tensor() -> None:
    """Test backward fill with large tensor to verify performance."""
    size = 1000
    x = torch.randn(size)
    # Randomly place some NaN values
    nan_indices = torch.randperm(size)[:100]
    x[nan_indices] = math.nan

    result = QF.bfill(x, dim=0)

    # Verify shape preservation
    assert result.shape == x.shape
    # Verify some non-NaN values remain unchanged
    non_nan_mask = ~torch.isnan(x)
    if non_nan_mask.any():
        assert not torch.isnan(result[non_nan_mask]).any()


def test_bfill_empty_tensor() -> None:
    """Test backward fill with empty tensor."""
    x = torch.empty(0)
    result = QF.bfill(x, dim=0)
    assert result.shape == x.shape


def test_bfill_alternating_pattern() -> None:
    """Test backward fill with alternating NaN pattern."""
    x = torch.tensor([math.nan, 1.0, math.nan, 2.0, math.nan, 3.0, math.nan])
    result = QF.bfill(x, dim=0)
    expected = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, math.nan])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_bfill_stress_test_complex_pattern() -> None:
    """Test backward fill with complex NaN patterns across multiple dimensions."""
    x = torch.tensor(
        [
            [[math.nan, 1.0, math.nan], [2.0, math.nan, 3.0]],
            [[math.nan, math.nan, 4.0], [math.nan, 5.0, math.nan]],
            [[6.0, math.nan, math.nan], [7.0, 8.0, 9.0]],
        ]
    )

    # Test along each dimension
    for dim in range(3):
        result = QF.bfill(x, dim=dim)
        assert result.shape == x.shape
        assert result.dtype == x.dtype
