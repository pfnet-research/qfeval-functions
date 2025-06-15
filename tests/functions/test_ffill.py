import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_ffill_1d_basic() -> None:
    """Test basic forward fill functionality on 1D tensor."""
    x = torch.tensor(
        [math.nan, 1.0, math.nan, 2.0, 3.0, math.nan, 4.0, math.nan]
    )
    result = QF.ffill(x, dim=0)
    expected = torch.tensor([math.nan, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 4.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_ffill_2d_tensor_dim0() -> None:
    """Test forward fill on 2D tensor along axis 0."""
    x = torch.tensor(
        [
            [math.nan, 1.0, 2.0],
            [3.0, math.nan, math.nan],
            [math.nan, 4.0, 5.0],
            [6.0, 7.0, math.nan],
        ]
    )
    result = QF.ffill(x, dim=0)
    expected = torch.tensor(
        [
            [math.nan, 1.0, 2.0],
            [3.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 5.0],
        ]
    )
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_ffill_2d_tensor_dim1() -> None:
    """Test forward fill on 2D tensor along axis 1."""
    x = torch.tensor(
        [
            [math.nan, 1.0, math.nan, 2.0],
            [3.0, math.nan, 4.0, math.nan],
            [math.nan, math.nan, 5.0, 6.0],
        ]
    )
    result = QF.ffill(x, dim=1)
    expected = torch.tensor(
        [
            [math.nan, 1.0, 1.0, 2.0],
            [3.0, 3.0, 4.0, 4.0],
            [math.nan, math.nan, 5.0, 6.0],
        ]
    )
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_ffill_3d_tensor() -> None:
    """Test forward fill on 3D tensor."""
    x = torch.tensor(
        [
            [[math.nan, 1.0], [2.0, math.nan]],
            [[3.0, math.nan], [math.nan, 4.0]],
            [[math.nan, 5.0], [6.0, math.nan]],
        ]
    )
    result = QF.ffill(x, dim=0)
    expected = torch.tensor(
        [
            [[math.nan, 1.0], [2.0, math.nan]],
            [[3.0, 1.0], [2.0, 4.0]],
            [[3.0, 5.0], [6.0, 4.0]],
        ]
    )
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_ffill_negative_dim() -> None:
    """Test forward fill with negative dimension indexing."""
    x = torch.tensor([[math.nan, 1.0, math.nan], [2.0, math.nan, 3.0]])
    result = QF.ffill(x, dim=-1)
    expected = torch.tensor([[math.nan, 1.0, 1.0], [2.0, 2.0, 3.0]])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_ffill_all_nan() -> None:
    """Test forward fill with tensor containing only NaN values."""
    x = torch.tensor([math.nan, math.nan, math.nan, math.nan])
    result = QF.ffill(x, dim=0)
    assert torch.isnan(result).all()


def test_ffill_no_nan() -> None:
    """Test forward fill with tensor containing no NaN values."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = QF.ffill(x, dim=0)
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_ffill_single_element() -> None:
    """Test forward fill with single-element tensor."""
    # Single NaN
    x1 = torch.tensor([math.nan])
    result1 = QF.ffill(x1, dim=0)
    assert torch.isnan(result1[0])

    # Single value
    x2 = torch.tensor([5.0])
    result2 = QF.ffill(x2, dim=0)
    assert result2[0] == 5.0


def test_ffill_leading_nans() -> None:
    """Test forward fill with leading NaN values."""
    x = torch.tensor([math.nan, math.nan, 1.0, 2.0, math.nan, 3.0])
    result = QF.ffill(x, dim=0)
    expected = torch.tensor([math.nan, math.nan, 1.0, 2.0, 2.0, 3.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_ffill_trailing_nans() -> None:
    """Test forward fill with trailing NaN values."""
    x = torch.tensor([1.0, 2.0, math.nan, math.nan, math.nan])
    result = QF.ffill(x, dim=0)
    expected = torch.tensor([1.0, 2.0, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_ffill_dtype_preservation() -> None:
    """Test that forward fill preserves input tensor's dtype."""
    # Test float64
    x_double = torch.tensor([math.nan, 1.0, math.nan], dtype=torch.float64)
    result_double = QF.ffill(x_double, dim=0)
    assert result_double.dtype == torch.float64

    # Test float32
    x_float = torch.tensor([math.nan, 1.0, math.nan], dtype=torch.float32)
    result_float = QF.ffill(x_float, dim=0)
    assert result_float.dtype == torch.float32


def test_ffill_device_preservation() -> None:
    """Test that forward fill preserves input tensor's device."""
    x = torch.tensor([math.nan, 1.0, math.nan])
    result = QF.ffill(x, dim=0)
    assert result.device == x.device


def test_ffill_mixed_nan_and_inf() -> None:
    """Test forward fill with mixed NaN and infinity values."""
    x = torch.tensor([math.nan, math.inf, math.nan, 1.0, math.nan, -math.inf])
    result = QF.ffill(x, dim=0)
    expected = torch.tensor([math.nan, math.inf, math.inf, 1.0, 1.0, -math.inf])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_ffill_large_tensor() -> None:
    """Test forward fill with large tensor to verify performance."""
    size = 1000
    x = torch.randn(size)
    # Randomly place some NaN values
    nan_indices = torch.randperm(size)[:100]
    x[nan_indices] = math.nan

    result = QF.ffill(x, dim=0)

    # Verify shape preservation
    assert result.shape == x.shape
    # Verify some non-NaN values remain unchanged
    non_nan_mask = ~torch.isnan(x)
    if non_nan_mask.any():
        # Non-NaN values should either remain the same or be filled by previous values
        assert result.shape == x.shape


def test_ffill_empty_tensor() -> None:
    """Test forward fill with empty tensor."""
    x = torch.empty(0)
    result = QF.ffill(x, dim=0)
    assert result.shape == x.shape


def test_ffill_alternating_pattern() -> None:
    """Test forward fill with alternating NaN pattern."""
    x = torch.tensor([math.nan, 1.0, math.nan, 2.0, math.nan, 3.0, math.nan])
    result = QF.ffill(x, dim=0)
    expected = torch.tensor([math.nan, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_ffill_stress_test_complex_pattern() -> None:
    """Test forward fill with complex NaN patterns across multiple dimensions."""
    x = torch.tensor(
        [
            [[math.nan, 1.0, math.nan], [2.0, math.nan, 3.0]],
            [[math.nan, math.nan, 4.0], [math.nan, 5.0, math.nan]],
            [[6.0, math.nan, math.nan], [7.0, 8.0, 9.0]],
        ]
    )

    # Test along each dimension
    for dim in range(3):
        result = QF.ffill(x, dim=dim)
        assert result.shape == x.shape
        assert result.dtype == x.dtype


def test_ffill_comparison_with_bfill() -> None:
    """Test that forward fill and backward fill give different results for same input."""
    x = torch.tensor([math.nan, 1.0, math.nan, 2.0, math.nan])

    ffill_result = QF.ffill(x, dim=0)
    bfill_result = QF.bfill(x, dim=0)

    # They should not be identical for this input
    assert not torch.equal(ffill_result, bfill_result)

    # But shapes should match
    assert ffill_result.shape == bfill_result.shape


def test_ffill_propagation_through_multiple_nans() -> None:
    """Test that forward fill properly propagates values through multiple consecutive NaNs."""
    x = torch.tensor(
        [1.0, math.nan, math.nan, math.nan, 2.0, math.nan, math.nan, 3.0]
    )
    result = QF.ffill(x, dim=0)
    expected = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())
