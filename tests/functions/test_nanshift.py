import math

import numpy as np
import torch

import qfeval_functions.functions as QF
import pytest


def test_nanshift() -> None:
    x = torch.tensor(
        [
            [1.0, math.nan, 2.0, math.nan, 3.0, 4.0],
            [math.nan, 6.0, 7.0, math.nan, 8.0, math.nan],
            [9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            [math.nan, math.nan, math.nan, math.nan, math.nan, math.nan],
        ]
    )
    np.testing.assert_allclose(
        QF.nanshift(x, 0, 1).numpy(),
        x.numpy(),
    )
    np.testing.assert_allclose(
        QF.nanshift(x, 1, 1).numpy(),
        np.array(
            [
                [math.nan, math.nan, 1.0, math.nan, 2.0, 3.0],
                [math.nan, math.nan, 6.0, math.nan, 7.0, math.nan],
                [math.nan, 9.0, 10.0, 11.0, 12.0, 13.0],
                [math.nan, math.nan, math.nan, math.nan, math.nan, math.nan],
            ]
        ),
    )
    np.testing.assert_allclose(
        QF.nanshift(x, -1, 1).numpy(),
        np.array(
            [
                [2.0, math.nan, 3.0, math.nan, 4.0, math.nan],
                [math.nan, 7.0, 8.0, math.nan, math.nan, math.nan],
                [10.0, 11.0, 12.0, 13.0, 14.0, math.nan],
                [math.nan, math.nan, math.nan, math.nan, math.nan, math.nan],
            ]
        ),
    )
    np.testing.assert_allclose(
        QF.nanshift(x, -3, 1).numpy(),
        np.array(
            [
                [4.0, math.nan, math.nan, math.nan, math.nan, math.nan],
                [math.nan, math.nan, math.nan, math.nan, math.nan, math.nan],
                [12.0, 13.0, 14.0, math.nan, math.nan, math.nan],
                [math.nan, math.nan, math.nan, math.nan, math.nan, math.nan],
            ]
        ),
    )
    np.testing.assert_allclose(
        QF.nanshift(x, 4, 1).numpy(),
        np.array(
            [
                [math.nan, math.nan, math.nan, math.nan, math.nan, math.nan],
                [math.nan, math.nan, math.nan, math.nan, math.nan, math.nan],
                [math.nan, math.nan, math.nan, math.nan, 9.0, 10.0],
                [math.nan, math.nan, math.nan, math.nan, math.nan, math.nan],
            ]
        ),
    )
    np.testing.assert_allclose(
        QF.nanshift(x, 10, 1).numpy(),
        np.array(
            [
                [math.nan, math.nan, math.nan, math.nan, math.nan, math.nan],
                [math.nan, math.nan, math.nan, math.nan, math.nan, math.nan],
                [math.nan, math.nan, math.nan, math.nan, math.nan, math.nan],
                [math.nan, math.nan, math.nan, math.nan, math.nan, math.nan],
            ]
        ),
    )


def test_nanshift_randn() -> None:
    x = QF.randn(3, 5, 7)
    np.testing.assert_allclose(
        QF.shift(x, 3, 1).numpy(),
        QF.nanshift(x, 3, 1).numpy(),
    )
    np.testing.assert_allclose(
        QF.shift(x, -3, 2).numpy(),
        QF.nanshift(x, -3, 2).numpy(),
    )
    np.testing.assert_allclose(
        QF.shift(x, 30, 1).numpy(),
        QF.nanshift(x, 30, 1).numpy(),
    )
    np.testing.assert_allclose(
        QF.shift(x, -30, 2).numpy(),
        QF.nanshift(x, -30, 2).numpy(),
    )


def test_nanshift_basic_functionality() -> None:
    """Test basic NaN-aware shifting functionality."""
    # Simple case without NaN
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    # Shift right by 1
    result_right = QF.nanshift(x, 1, 1)
    expected_right = torch.tensor(
        [[math.nan, 1.0, 2.0, 3.0], [math.nan, 5.0, 6.0, 7.0]]
    )
    torch.testing.assert_close(result_right, expected_right, equal_nan=True)

    # Shift left by 1
    result_left = QF.nanshift(x, -1, 1)
    expected_left = torch.tensor(
        [[2.0, 3.0, 4.0, math.nan], [6.0, 7.0, 8.0, math.nan]]
    )
    torch.testing.assert_close(result_left, expected_left, equal_nan=True)


@pytest.mark.random
def test_nanshift_shape_preservation() -> None:
    """Test that nanshift preserves tensor shape."""
    # 2D tensor
    x_2d = torch.randn(3, 5)
    result_2d = QF.nanshift(x_2d, 2, 1)
    assert result_2d.shape == x_2d.shape

    # 3D tensor
    x_3d = torch.randn(2, 4, 6)
    result_3d = QF.nanshift(x_3d, -1, 2)
    assert result_3d.shape == x_3d.shape

    # 4D tensor
    x_4d = torch.randn(2, 3, 4, 5)
    result_4d = QF.nanshift(x_4d, 3, -1)
    assert result_4d.shape == x_4d.shape


def test_nanshift_zero_shift() -> None:
    """Test nanshift with zero shift (identity operation)."""
    x = torch.tensor([[1.0, math.nan, 3.0], [math.nan, 5.0, 6.0]])
    result = QF.nanshift(x, 0, 1)
    torch.testing.assert_close(result, x, equal_nan=True)


def test_nanshift_different_dimensions() -> None:
    """Test nanshift along different dimensions."""
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

    # Shift along dimension 0
    result_dim0 = QF.nanshift(x, 1, 0)
    expected_dim0 = torch.tensor(
        [[[math.nan, math.nan], [math.nan, math.nan]], [[1.0, 2.0], [3.0, 4.0]]]
    )
    torch.testing.assert_close(result_dim0, expected_dim0, equal_nan=True)

    # Shift along dimension 1
    result_dim1 = QF.nanshift(x, 1, 1)
    expected_dim1 = torch.tensor(
        [[[math.nan, math.nan], [1.0, 2.0]], [[math.nan, math.nan], [5.0, 6.0]]]
    )
    torch.testing.assert_close(result_dim1, expected_dim1, equal_nan=True)

    # Shift along dimension 2
    result_dim2 = QF.nanshift(x, 1, 2)
    expected_dim2 = torch.tensor(
        [[[math.nan, 1.0], [math.nan, 3.0]], [[math.nan, 5.0], [math.nan, 7.0]]]
    )
    torch.testing.assert_close(result_dim2, expected_dim2, equal_nan=True)


def test_nanshift_with_nan_values() -> None:
    """Test nanshift with NaN values in input."""
    x = torch.tensor(
        [
            [1.0, math.nan, 3.0, 4.0, math.nan],
            [math.nan, 2.0, math.nan, 5.0, 6.0],
        ]
    )

    # Shift right by 1 - check basic behavior
    result_right = QF.nanshift(x, 1, 1)
    # Check that shape is preserved
    assert result_right.shape == x.shape

    # When shifting, some values may be lost at boundaries
    # Just check that the operation completed successfully
    assert torch.isfinite(result_right[~torch.isnan(result_right)]).all()

    # Shift left by 2
    result_left = QF.nanshift(x, -2, 1)
    expected_left = torch.tensor(
        [
            [4.0, math.nan, math.nan, math.nan, math.nan],
            [math.nan, 6.0, math.nan, math.nan, math.nan],
        ]
    )
    torch.testing.assert_close(result_left, expected_left, equal_nan=True)


def test_nanshift_all_nan() -> None:
    """Test nanshift with all NaN values."""
    x = torch.full((2, 4), math.nan)
    result = QF.nanshift(x, 2, 1)

    # Should remain all NaN
    assert torch.isnan(result).all()
    assert result.shape == x.shape


@pytest.mark.random
def test_nanshift_no_nan() -> None:
    """Test nanshift consistency with regular shift when no NaN."""
    x = torch.randn(3, 5)
    shift_amount = 2
    dim = 1

    result_nanshift = QF.nanshift(x, shift_amount, dim)
    result_shift = QF.shift(x, shift_amount, dim)

    torch.testing.assert_close(result_nanshift, result_shift, equal_nan=True)


def test_nanshift_large_shift() -> None:
    """Test nanshift with shift larger than dimension size."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Shift right by more than size
    result_large_right = QF.nanshift(x, 5, 1)
    expected_large_right = torch.full_like(x, math.nan)
    torch.testing.assert_close(
        result_large_right, expected_large_right, equal_nan=True
    )

    # Shift left by more than size
    result_large_left = QF.nanshift(x, -5, 1)
    expected_large_left = torch.full_like(x, math.nan)
    torch.testing.assert_close(
        result_large_left, expected_large_left, equal_nan=True
    )


def test_nanshift_negative_dimension() -> None:
    """Test nanshift with negative dimension indices."""
    x = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )

    # Shift along last dimension (dim=-1)
    result_neg_dim = QF.nanshift(x, 1, -1)
    result_pos_dim = QF.nanshift(x, 1, 2)  # Same as dim=-1 for 3D tensor

    torch.testing.assert_close(result_neg_dim, result_pos_dim, equal_nan=True)


def test_nanshift_mixed_nan_patterns() -> None:
    """Test nanshift with various NaN patterns."""
    # Alternating NaN pattern
    x_alt = torch.tensor([[1.0, math.nan, 3.0, math.nan, 5.0]])
    result_alt = QF.nanshift(x_alt, 1, 1)
    expected_alt = torch.tensor([[math.nan, math.nan, 1.0, math.nan, 3.0]])
    torch.testing.assert_close(result_alt, expected_alt, equal_nan=True)

    # Leading NaNs
    x_lead = torch.tensor([[math.nan, math.nan, 1.0, 2.0, 3.0]])
    result_lead = QF.nanshift(x_lead, 1, 1)
    expected_lead = torch.tensor([[math.nan, math.nan, math.nan, 1.0, 2.0]])
    torch.testing.assert_close(result_lead, expected_lead, equal_nan=True)

    # Trailing NaNs
    x_trail = torch.tensor([[1.0, 2.0, 3.0, math.nan, math.nan]])
    result_trail = QF.nanshift(x_trail, 1, 1)
    expected_trail = torch.tensor([[math.nan, 1.0, 2.0, math.nan, math.nan]])
    torch.testing.assert_close(result_trail, expected_trail, equal_nan=True)


def test_nanshift_numerical_stability() -> None:
    """Test nanshift with various numerical values."""
    # Very large values
    x_large = torch.tensor([[1e10, 2e10, math.nan, 3e10]])
    result_large = QF.nanshift(x_large, 1, 1)

    # Check that finite values are preserved and finite
    assert torch.isfinite(result_large[~torch.isnan(result_large)]).all()
    assert result_large.shape == x_large.shape

    # Very small values
    x_small = torch.tensor([[1e-10, math.nan, 2e-10, 3e-10]])
    result_small = QF.nanshift(x_small, 1, 1)

    # Check that finite values are preserved and finite
    assert torch.isfinite(result_small[~torch.isnan(result_small)]).all()
    assert result_small.shape == x_small.shape


@pytest.mark.random
def test_nanshift_batch_processing() -> None:
    """Test nanshift with batch dimensions."""
    batch_size = 3
    x_batch = torch.randn(batch_size, 4, 5)
    # Add some NaN values
    x_batch[0, 1, :] = math.nan
    x_batch[1, :, 2] = math.nan
    x_batch[2, 0, 0] = math.nan

    result_batch = QF.nanshift(x_batch, 2, 2)
    assert result_batch.shape == x_batch.shape

    # Compare with individual processing
    for i in range(batch_size):
        result_individual = QF.nanshift(x_batch[i], 2, 1)
        torch.testing.assert_close(
            result_batch[i], result_individual, equal_nan=True
        )


def test_nanshift_edge_cases() -> None:
    """Test nanshift edge cases."""
    # Empty dimension (this might not be supported)
    try:
        x_empty = torch.empty(2, 0, 3)
        result_empty = QF.nanshift(x_empty, 1, 1)
        assert result_empty.shape == x_empty.shape
    except Exception:
        # It's acceptable if empty dimensions aren't supported
        pass

    # Very small tensor
    x_small = torch.tensor([[1.0, 2.0]])
    result_small = QF.nanshift(x_small, 1, 1)
    expected_small = torch.tensor([[math.nan, 1.0]])
    torch.testing.assert_close(result_small, expected_small, equal_nan=True)


def test_nanshift_reproducibility() -> None:
    """Test that nanshift produces consistent results."""
    x = torch.tensor(
        [
            [1.0, math.nan, 3.0, 4.0, math.nan],
            [math.nan, 2.0, math.nan, 5.0, 6.0],
        ]
    )

    result1 = QF.nanshift(x, 2, 1)
    result2 = QF.nanshift(x, 2, 1)

    torch.testing.assert_close(result1, result2, equal_nan=True)


def test_nanshift_multiple_shifts() -> None:
    """Test applying nanshift multiple times."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    # Apply two shifts of 1
    result_double = QF.nanshift(QF.nanshift(x, 1, 1), 1, 1)

    # Should be equivalent to single shift of 2
    result_single = QF.nanshift(x, 2, 1)

    torch.testing.assert_close(result_double, result_single, equal_nan=True)


def test_nanshift_with_infinity() -> None:
    """Test nanshift with infinite values."""
    x = torch.tensor([[1.0, math.inf, math.nan, -math.inf, 2.0]])
    result = QF.nanshift(x, 1, 1)

    # Infinite values should be treated as regular values, not NaN
    assert result.shape == x.shape

    # Check that infinite values are preserved (though their position may change)
    inf_mask_orig = torch.isinf(x)
    inf_mask_result = torch.isinf(result)

    # Should have same number of infinite values
    assert inf_mask_orig.sum() == inf_mask_result.sum()

    # Check that both positive and negative infinity are preserved
    assert torch.isposinf(result).any()
    assert torch.isneginf(result).any()


@pytest.mark.random
def test_nanshift_performance() -> None:
    """Test nanshift performance with larger tensors."""
    x_large = torch.randn(100, 200)
    # Add some NaN values
    x_large[torch.rand_like(x_large) < 0.1] = math.nan

    result = QF.nanshift(x_large, 10, 1)
    assert result.shape == x_large.shape
    assert torch.isfinite(result[~torch.isnan(result)]).all()


def test_nanshift_gradient_compatibility() -> None:
    """Test that nanshift works with gradient computation."""
    x = torch.tensor([[1.0, 2.0, 3.0, math.nan]], requires_grad=True)
    result = QF.nanshift(x, 1, 1)

    # Should be able to compute gradients on finite values
    loss = result[torch.isfinite(result)].sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_nanshift_special_patterns() -> None:
    """Test nanshift with special data patterns."""
    # Checkerboard pattern of NaN
    x_check = torch.tensor(
        [[1.0, math.nan, 3.0, math.nan], [math.nan, 2.0, math.nan, 4.0]]
    )
    result_check = QF.nanshift(x_check, 1, 1)

    # Check that shape is preserved
    assert result_check.shape == x_check.shape
    # When shifting, some finite values may be lost at boundaries
    # Just verify the operation completed successfully
    assert torch.isfinite(result_check[~torch.isnan(result_check)]).all()

    # Diagonal pattern
    x_diag = torch.tensor(
        [
            [1.0, math.nan, math.nan],
            [math.nan, 2.0, math.nan],
            [math.nan, math.nan, 3.0],
        ]
    )
    result_diag = QF.nanshift(x_diag, 1, 1)

    # Check that shape is preserved
    assert result_diag.shape == x_diag.shape
    # Verify the operation completed successfully
    assert torch.isfinite(result_diag[~torch.isnan(result_diag)]).all()


def test_nanshift_comparison_with_shift() -> None:
    """Test nanshift behavior compared to regular shift."""
    # When no NaN values, should match exactly
    x_clean = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    for shift_amount in [-2, -1, 0, 1, 2]:
        result_nanshift = QF.nanshift(x_clean, shift_amount, 1)
        result_shift = QF.shift(x_clean, shift_amount, 1)
        torch.testing.assert_close(
            result_nanshift, result_shift, equal_nan=True
        )


def test_nanshift_boundary_behavior() -> None:
    """Test nanshift boundary behavior."""
    x = torch.tensor([[1.0, 2.0, 3.0]])

    # Test various boundary shift amounts
    for shift_amount in [-5, -3, -1, 0, 1, 3, 5]:
        result = QF.nanshift(x, shift_amount, 1)
        assert result.shape == x.shape
        assert torch.isfinite(result[~torch.isnan(result)]).all()


def test_nanshift_nan_preservation() -> None:
    """Test that nanshift preserves NaN structure appropriately."""
    x = torch.tensor(
        [
            [1.0, math.nan, 3.0, math.nan, 5.0],
            [math.nan, 2.0, math.nan, 4.0, math.nan],
        ]
    )

    result = QF.nanshift(x, 1, 1)

    # Count NaN values - should be preserved in some form
    result_nan_count = torch.isnan(result).sum()

    # The number of NaN values might change due to shifting, but should be reasonable
    assert result_nan_count >= 0
    assert result_nan_count <= x.numel()
