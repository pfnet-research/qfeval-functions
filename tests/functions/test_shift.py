import math

import numpy as np
import torch

import qfeval_functions.functions as QF
import pytest


def test_shift() -> None:
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    np.testing.assert_allclose(
        QF.shift(x, 0, 0).numpy(),
        x.numpy(),
    )
    np.testing.assert_allclose(
        QF.shift(x, 1, 0).numpy(),
        np.array(
            [
                [math.nan, math.nan, math.nan],
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ),
    )
    np.testing.assert_allclose(
        QF.shift(x, (-1, 1), (0, 1)).numpy(),
        np.array(
            [
                [math.nan, 4.0, 5.0],
                [math.nan, 7.0, 8.0],
                [math.nan, math.nan, math.nan],
            ]
        ),
    )
    np.testing.assert_allclose(
        QF.shift(x, -3, 1).numpy(),
        x.numpy() * math.nan,
    )
    np.testing.assert_allclose(
        QF.shift(x, 3, 1).numpy(),
        x.numpy() * math.nan,
    )
    np.testing.assert_allclose(
        QF.shift(x, -100, 1).numpy(),
        x.numpy() * math.nan,
    )
    np.testing.assert_allclose(
        QF.shift(x, 100, 1).numpy(),
        x.numpy() * math.nan,
    )


def test_shift_basic_functionality() -> None:
    """Test basic shift functionality."""
    # 1D tensor
    x_1d = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    # Shift right by 1
    result_right = QF.shift(x_1d, 1, 0)
    expected_right = torch.tensor([math.nan, 1.0, 2.0, 3.0, 4.0])
    torch.testing.assert_close(result_right, expected_right, equal_nan=True)

    # Shift left by 1
    result_left = QF.shift(x_1d, -1, 0)
    expected_left = torch.tensor([2.0, 3.0, 4.0, 5.0, math.nan])
    torch.testing.assert_close(result_left, expected_left, equal_nan=True)

    # No shift
    result_no_shift = QF.shift(x_1d, 0, 0)
    torch.testing.assert_close(result_no_shift, x_1d)


def test_shift_2d_single_dimension() -> None:
    """Test shift operations on 2D tensors along single dimensions."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    # Shift along dimension 0 (rows)
    result_dim0_pos = QF.shift(x, 1, 0)
    expected_dim0_pos = torch.tensor(
        [[math.nan, math.nan, math.nan], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    )
    torch.testing.assert_close(
        result_dim0_pos, expected_dim0_pos, equal_nan=True
    )

    result_dim0_neg = QF.shift(x, -1, 0)
    expected_dim0_neg = torch.tensor(
        [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [math.nan, math.nan, math.nan]]
    )
    torch.testing.assert_close(
        result_dim0_neg, expected_dim0_neg, equal_nan=True
    )

    # Shift along dimension 1 (columns)
    result_dim1_pos = QF.shift(x, 1, 1)
    expected_dim1_pos = torch.tensor(
        [[math.nan, 1.0, 2.0], [math.nan, 4.0, 5.0], [math.nan, 7.0, 8.0]]
    )
    torch.testing.assert_close(
        result_dim1_pos, expected_dim1_pos, equal_nan=True
    )

    result_dim1_neg = QF.shift(x, -1, 1)
    expected_dim1_neg = torch.tensor(
        [[2.0, 3.0, math.nan], [5.0, 6.0, math.nan], [8.0, 9.0, math.nan]]
    )
    torch.testing.assert_close(
        result_dim1_neg, expected_dim1_neg, equal_nan=True
    )


def test_shift_multiple_dimensions() -> None:
    """Test shift operations on multiple dimensions simultaneously."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    # Shift both dimensions positively
    result_both_pos = QF.shift(x, (1, 1), (0, 1))
    expected_both_pos = torch.tensor(
        [
            [math.nan, math.nan, math.nan],
            [math.nan, 1.0, 2.0],
            [math.nan, 4.0, 5.0],
        ]
    )
    torch.testing.assert_close(
        result_both_pos, expected_both_pos, equal_nan=True
    )

    # Shift both dimensions negatively
    result_both_neg = QF.shift(x, (-1, -1), (0, 1))
    expected_both_neg = torch.tensor(
        [
            [5.0, 6.0, math.nan],
            [8.0, 9.0, math.nan],
            [math.nan, math.nan, math.nan],
        ]
    )
    torch.testing.assert_close(
        result_both_neg, expected_both_neg, equal_nan=True
    )

    # Mixed shifts
    result_mixed = QF.shift(x, (-1, 1), (0, 1))
    expected_mixed = torch.tensor(
        [
            [math.nan, 4.0, 5.0],
            [math.nan, 7.0, 8.0],
            [math.nan, math.nan, math.nan],
        ]
    )
    torch.testing.assert_close(result_mixed, expected_mixed, equal_nan=True)


def test_shift_edge_cases() -> None:
    """Test shift with edge cases."""
    # Single element tensor
    x_single = torch.tensor([42.0])
    result_single_pos = QF.shift(x_single, 1, 0)
    assert torch.isnan(result_single_pos[0])

    result_single_neg = QF.shift(x_single, -1, 0)
    assert torch.isnan(result_single_neg[0])

    result_single_zero = QF.shift(x_single, 0, 0)
    assert result_single_zero[0] == 42.0

    # Empty tensor
    x_empty = torch.empty(0)
    result_empty = QF.shift(x_empty, 1, 0)
    assert result_empty.shape == (0,)


def test_shift_large_shifts() -> None:
    """Test shift with shifts larger than tensor dimensions."""
    x = torch.tensor([1.0, 2.0, 3.0])

    # Shift larger than tensor size
    result_large_pos = QF.shift(x, 10, 0)
    assert torch.isnan(result_large_pos).all()

    result_large_neg = QF.shift(x, -10, 0)
    assert torch.isnan(result_large_neg).all()

    # Shift equal to tensor size
    result_equal_size = QF.shift(x, 3, 0)
    assert torch.isnan(result_equal_size).all()

    result_equal_size_neg = QF.shift(x, -3, 0)
    assert torch.isnan(result_equal_size_neg).all()


def test_shift_shape_preservation() -> None:
    """Test that shift preserves tensor shape."""
    shapes = [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5)]

    for shape in shapes:
        x = torch.randn(shape)
        result = QF.shift(x, 1, 0)
        assert result.shape == x.shape


def test_shift_negative_dimensions() -> None:
    """Test shift with negative dimension indices."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Test that negative dimensions work (though they may not be equivalent to positive ones
    # depending on implementation)
    result_neg1 = QF.shift(x, 1, -1)
    assert result_neg1.shape == x.shape

    result_neg2 = QF.shift(x, 1, -2)
    assert result_neg2.shape == x.shape

    # Both should contain some NaN values due to shifting
    assert torch.isnan(result_neg1).any()
    assert torch.isnan(result_neg2).any()


def test_shift_high_dimensional() -> None:
    """Test shift with high-dimensional tensors."""
    x = torch.randn(2, 3, 4, 5)

    # Shift along different dimensions
    result_dim0 = QF.shift(x, 1, 0)
    assert result_dim0.shape == x.shape

    result_dim3 = QF.shift(x, -2, 3)
    assert result_dim3.shape == x.shape

    # Multiple dimension shifts
    result_multi = QF.shift(x, (1, -1), (0, 2))
    assert result_multi.shape == x.shape


def test_shift_with_existing_nans() -> None:
    """Test shift behavior with tensors that already contain NaN values."""
    x = torch.tensor([1.0, math.nan, 3.0, 4.0])

    result = QF.shift(x, 1, 0)

    # Check non-NaN values
    assert result[1] == 1.0
    assert result[3] == 3.0
    # Check NaN values
    assert torch.isnan(result[0])
    assert torch.isnan(result[2])


def test_shift_with_infinite_values() -> None:
    """Test shift behavior with infinite values."""
    x = torch.tensor([1.0, math.inf, -math.inf, 4.0])

    result = QF.shift(x, 1, 0)

    assert torch.isnan(result[0])
    assert result[1] == 1.0
    assert torch.isinf(result[2]) and result[2] > 0
    assert torch.isinf(result[3]) and result[3] < 0


def test_shift_batch_processing() -> None:
    """Test shift with batch processing."""
    batch_size = 3
    seq_length = 5
    x = torch.randn(batch_size, seq_length)

    # Shift along sequence dimension
    result = QF.shift(x, 1, 1)
    assert result.shape == (batch_size, seq_length)

    # First column should be NaN
    assert torch.isnan(result[:, 0]).all()

    # Remaining columns should match shifted original
    torch.testing.assert_close(result[:, 1:], x[:, :-1], equal_nan=True)


def test_shift_single_vs_multiple_calls() -> None:
    """Test equivalence between single and multiple shift calls."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    # Single call with multiple dimensions
    result_single = QF.shift(x, (1, 1), (0, 1))

    # Multiple calls (order matters due to NaN filling)
    result_step1 = QF.shift(x, 1, 0)
    result_multi = QF.shift(result_step1, 1, 1)

    torch.testing.assert_close(result_single, result_multi, equal_nan=True)


def test_shift_parameter_validation() -> None:
    """Test shift parameter validation and edge cases."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    # Test that matching shifts and dims work properly
    result = QF.shift(x, (1, 1), (0, 1))  # Shift both dimensions
    expected_shape = x.shape
    assert result.shape == expected_shape

    # Should contain NaN values due to shifting
    assert torch.isnan(result).any()


def test_shift_different_magnitudes() -> None:
    """Test shift with different magnitude values."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    # Small shift
    result_small = QF.shift(x, 1, 0)
    assert torch.isnan(result_small[0])
    assert result_small[1] == 1.0

    # Medium shift
    result_medium = QF.shift(x, 3, 0)
    assert torch.isnan(result_medium[:3]).all()
    assert result_medium[3] == 1.0
    assert result_medium[4] == 2.0

    # Large shift (equal to size)
    result_large = QF.shift(x, 5, 0)
    assert torch.isnan(result_large).all()


def test_shift_preserve_gradient() -> None:
    """Test that shift operations preserve gradient computation."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    result = QF.shift(x, 1, 0)

    # Non-NaN values should maintain gradient
    loss = result[1:].sum()  # Skip NaN value
    loss.backward()

    # Gradients should exist for non-shifted values
    assert x.grad is not None
    assert x.grad[0] == 1.0  # This value was shifted to position 1
    assert x.grad[1] == 1.0  # This value was shifted to position 2
    assert x.grad[2] == 1.0  # This value was shifted to position 3


def test_shift_symmetry() -> None:
    """Test symmetry properties of shift operations."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    # For shifts that don't exceed bounds, opposite shifts should be related
    right_shift = QF.shift(x, 2, 0)
    left_shift = QF.shift(x, -2, 0)

    # Check that the non-NaN parts are correctly shifted
    assert torch.isnan(right_shift[:2]).all()
    assert torch.isnan(left_shift[-2:]).all()

    torch.testing.assert_close(right_shift[2:], x[:-2])
    torch.testing.assert_close(left_shift[:-2], x[2:])


def test_shift_boundary_conditions() -> None:
    """Test shift at boundary conditions."""
    x = torch.tensor([1.0, 2.0, 3.0])

    # Shift by exactly the size of the tensor
    result_exact = QF.shift(x, 3, 0)
    assert torch.isnan(result_exact).all()

    result_exact_neg = QF.shift(x, -3, 0)
    assert torch.isnan(result_exact_neg).all()

    # Shift by size - 1
    result_almost = QF.shift(x, 2, 0)
    assert torch.isnan(result_almost[:2]).all()
    assert result_almost[2] == 1.0


def test_shift_multidimensional_combinations() -> None:
    """Test various combinations of multidimensional shifts."""
    x = torch.arange(24).reshape(2, 3, 4).float()

    # Test different shift combinations
    combinations = [
        ((1, 1, 1), (0, 1, 2)),
        ((-1, 0, 1), (0, 1, 2)),
        ((0, -1, -1), (0, 1, 2)),
        ((1, -1), (0, 2)),
        ((-1, 1), (1, 2)),
    ]

    for shifts, dims in combinations:
        result = QF.shift(x, shifts, dims)
        assert result.shape == x.shape
        # Should contain some NaN values due to shifting
        assert torch.isnan(result).any()


def test_shift_performance() -> None:
    """Test shift performance with large tensors."""
    for size in [1000, 5000]:
        x = torch.randn(size)
        result = QF.shift(x, 100, 0)

        # Verify properties
        assert result.shape == x.shape
        assert torch.isnan(result[:100]).all()
        torch.testing.assert_close(result[100:], x[:-100])

        # Clean up
        del x, result


def test_shift_zero_shift_identity() -> None:
    """Test that zero shift is identity operation."""
    shapes = [(5,), (3, 4), (2, 3, 4)]

    for shape in shapes:
        x = torch.randn(shape)
        result = QF.shift(x, 0, 0)
        torch.testing.assert_close(result, x)

        # Test zero shift on multiple dimensions
        if len(shape) > 1:
            zeros = tuple([0] * len(shape))
            dims = tuple(range(len(shape)))
            result_multi = QF.shift(x, zeros, dims)
            torch.testing.assert_close(result_multi, x)
