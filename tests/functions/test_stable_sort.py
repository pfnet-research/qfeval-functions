import itertools
import math

import numpy as np
import torch

import qfeval_functions.functions as QF
import pytest


def test_stable_sort() -> None:
    a = (QF.rand(5, 5, 5) * 10.0 - 5).round()
    a = torch.where(torch.eq(a, 1.0), torch.as_tensor(math.inf), a)
    a = torch.where(torch.eq(a, 2.0), torch.as_tensor(math.nan), a)
    a = torch.where(torch.eq(a, 3.0), torch.as_tensor(-math.inf), a)
    for dim in range(3):
        np.testing.assert_array_almost_equal(
            QF.stable_sort(a, dim=dim).values.numpy(),
            np.sort(a.numpy(), axis=dim, kind="stable"),
        )
        np.testing.assert_array_almost_equal(
            QF.stable_sort(a, dim=dim).indices.numpy(),
            np.argsort(a.numpy(), axis=dim, kind="stable"),
        )


def test_stable_sort_with_ints() -> None:
    a = (QF.rand(5, 5, 5) * 10.0 - 5).round().int()
    for dim in range(3):
        np.testing.assert_array_almost_equal(
            QF.stable_sort(a, dim=dim).values.numpy(),
            np.sort(a.numpy(), axis=dim, kind="stable"),
        )
        np.testing.assert_array_almost_equal(
            QF.stable_sort(a, dim=dim).indices.numpy(),
            np.argsort(a.numpy(), axis=dim, kind="stable"),
        )


def test_stable_sort_basic_functionality() -> None:
    """Test basic stable sort functionality."""
    # Simple 1D case
    x = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0], dtype=torch.float64)
    result = QF.stable_sort(x, dim=0)

    # Check sorted values
    expected_values = torch.tensor(
        [1.0, 1.0, 3.0, 4.0, 5.0], dtype=torch.float64
    )
    torch.testing.assert_close(result.values, expected_values)

    # Check indices - should preserve original order for equal elements
    expected_indices = torch.tensor(
        [1, 3, 0, 2, 4], dtype=torch.long
    )  # First 1.0 at index 1, second at 3
    torch.testing.assert_close(result.indices, expected_indices)


def test_stable_sort_stability_property() -> None:
    """Test that stable sort preserves order of equal elements."""
    # Create array with multiple equal elements
    x = torch.tensor([2.0, 1.0, 2.0, 3.0, 1.0, 2.0], dtype=torch.float64)
    result = QF.stable_sort(x, dim=0)

    # Values should be sorted
    expected_values = torch.tensor(
        [1.0, 1.0, 2.0, 2.0, 2.0, 3.0], dtype=torch.float64
    )
    torch.testing.assert_close(result.values, expected_values)

    # For equal values, original order should be preserved
    # Original 1.0s at indices [1, 4], 2.0s at indices [0, 2, 5]
    expected_indices = torch.tensor([1, 4, 0, 2, 5, 3], dtype=torch.long)
    torch.testing.assert_close(result.indices, expected_indices)


@pytest.mark.random
def test_stable_sort_shape_preservation() -> None:
    """Test that stable sort preserves tensor shape."""
    # 2D tensor
    x_2d = torch.randn(3, 5, dtype=torch.float64)

    result_dim0 = QF.stable_sort(x_2d, dim=0)
    assert result_dim0.values.shape == x_2d.shape
    assert result_dim0.indices.shape == x_2d.shape

    result_dim1 = QF.stable_sort(x_2d, dim=1)
    assert result_dim1.values.shape == x_2d.shape
    assert result_dim1.indices.shape == x_2d.shape

    # 3D tensor
    x_3d = torch.randn(2, 4, 6, dtype=torch.float64)
    result_3d = QF.stable_sort(x_3d, dim=2)
    assert result_3d.values.shape == x_3d.shape
    assert result_3d.indices.shape == x_3d.shape


def test_stable_sort_different_dimensions() -> None:
    """Test stable sort along different dimensions."""
    x = torch.tensor(
        [
            [[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]],
            [[9.0, 7.0, 8.0], [12.0, 10.0, 11.0]],
        ],
        dtype=torch.float64,
    )

    # Test each dimension
    for dim in range(x.ndim):
        result = QF.stable_sort(x, dim=dim)
        assert result.values.shape == x.shape
        assert result.indices.shape == x.shape

        # Verify sorting along the specified dimension
        # Create all possible indices manually
        for idx in itertools.product(*[range(s) for s in x.shape]):
            if idx[dim] > 0:
                prev_idx_list = list(idx)
                prev_idx_list[dim] -= 1
                prev_idx = tuple(prev_idx_list)
                assert result.values[prev_idx] <= result.values[idx]


@pytest.mark.random
def test_stable_sort_negative_dimension() -> None:
    """Test stable sort with negative dimension indices."""
    x = torch.randn(3, 4, 5, dtype=torch.float64)

    # Test negative dimension
    result_neg = QF.stable_sort(x, dim=-1)
    result_pos = QF.stable_sort(x, dim=2)

    torch.testing.assert_close(result_neg.values, result_pos.values)
    torch.testing.assert_close(result_neg.indices, result_pos.indices)


def test_stable_sort_with_nan_values() -> None:
    """Test stable sort with NaN values."""
    # TODO(claude): The stable_sort function should provide configurable NaN handling
    # options. Expected behavior: add parameters like nan_policy='last'|'first'|'raise'
    # to control where NaN values are placed in the sorted result, and nan_comparison
    # to specify how NaN values compare to each other for stability.
    x = torch.tensor([3.0, math.nan, 1.0, math.nan, 2.0], dtype=torch.float64)
    result = QF.stable_sort(x, dim=0)

    # Non-NaN values should be sorted at the beginning
    assert torch.isfinite(result.values[:3]).all()
    assert result.values[0] <= result.values[1] <= result.values[2]

    # NaN values should be at the end
    assert torch.isnan(result.values[3:]).all()

    # Check that indices are valid
    assert result.indices.dtype == torch.long
    assert torch.all(result.indices >= 0) and torch.all(result.indices < len(x))


def test_stable_sort_with_infinity() -> None:
    """Test stable sort with infinite values."""
    x = torch.tensor([3.0, math.inf, 1.0, -math.inf, 2.0], dtype=torch.float64)
    result = QF.stable_sort(x, dim=0)

    # Check ordering: -inf < finite values < inf
    assert torch.isneginf(result.values[0])
    assert torch.isfinite(result.values[1:4]).all()
    assert result.values[1] <= result.values[2] <= result.values[3]
    assert torch.isposinf(result.values[4])


def test_stable_sort_mixed_special_values() -> None:
    """Test stable sort with mixed NaN, infinity, and finite values."""
    x = torch.tensor(
        [
            3.0,
            math.nan,
            math.inf,
            1.0,
            -math.inf,
            math.nan,
            2.0,
        ],
        dtype=torch.float64,
    )
    result = QF.stable_sort(x, dim=0)

    # Expected order: -inf, finite values (sorted), inf, NaN values
    assert torch.isneginf(result.values[0])
    assert torch.isfinite(result.values[1:4]).all()
    assert (
        result.values[1] <= result.values[2] <= result.values[3]
    )  # 1.0, 2.0, 3.0
    assert torch.isposinf(result.values[4])
    assert torch.isnan(result.values[5:]).all()


def test_stable_sort_identical_values() -> None:
    """Test stable sort with all identical values."""
    x = torch.full((5,), 3.0, dtype=torch.float64)
    result = QF.stable_sort(x, dim=0)

    # Values should remain the same
    torch.testing.assert_close(result.values, x)

    # Indices should be in original order (stable)
    expected_indices = torch.arange(5, dtype=torch.long)
    torch.testing.assert_close(result.indices, expected_indices)


def test_stable_sort_already_sorted() -> None:
    """Test stable sort with already sorted input."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    result = QF.stable_sort(x, dim=0)

    # Should remain the same
    torch.testing.assert_close(result.values, x)
    expected_indices = torch.arange(5, dtype=torch.long)
    torch.testing.assert_close(result.indices, expected_indices)


def test_stable_sort_reverse_sorted() -> None:
    """Test stable sort with reverse sorted input."""
    x = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], dtype=torch.float64)
    result = QF.stable_sort(x, dim=0)

    expected_values = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64
    )
    torch.testing.assert_close(result.values, expected_values)

    expected_indices = torch.tensor([4, 3, 2, 1, 0], dtype=torch.long)
    torch.testing.assert_close(result.indices, expected_indices)


@pytest.mark.random
def test_stable_sort_large_tensor() -> None:
    """Test stable sort with larger tensors."""
    x = torch.randn(100, dtype=torch.float64)
    result = QF.stable_sort(x, dim=0)

    # Check that result is sorted
    for i in range(len(result.values) - 1):
        assert result.values[i] <= result.values[i + 1]

    # Check that indices are valid
    assert result.indices.dtype == torch.long
    assert torch.all(result.indices >= 0) and torch.all(result.indices < len(x))

    # Check that indices correctly map to values
    reconstructed = x[result.indices]
    torch.testing.assert_close(result.values, reconstructed)


def test_stable_sort_multi_dimensional() -> None:
    """Test stable sort with multi-dimensional tensors."""
    x = torch.tensor(
        [
            [[3.0, 1.0, 4.0], [1.0, 5.0, 9.0]],
            [[2.0, 6.0, 5.0], [3.0, 5.0, 8.0]],
        ],
        dtype=torch.float64,
    )

    # Sort along each dimension
    for dim in range(x.ndim):
        result = QF.stable_sort(x, dim=dim)

        # Verify shape preservation
        assert result.values.shape == x.shape
        assert result.indices.shape == x.shape

        # Verify sorting along the specified dimension
        sorted_slices = torch.moveaxis(result.values, dim, -1)
        for flat_idx in range(sorted_slices.numel() // sorted_slices.shape[-1]):
            slice_data = sorted_slices.reshape(-1, sorted_slices.shape[-1])[
                flat_idx
            ]
            for i in range(len(slice_data) - 1):
                assert slice_data[i] <= slice_data[i + 1]


def test_stable_sort_comparison_with_numpy() -> None:
    """Test stable sort consistency with NumPy."""
    # Create test data without NaN (NumPy handles NaN differently)
    x_np = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    x_torch = torch.from_numpy(x_np)

    result_torch = QF.stable_sort(x_torch, dim=0)
    result_numpy_values = np.sort(x_np, kind="stable")
    result_numpy_indices = np.argsort(x_np, kind="stable")

    # Compare values
    np.testing.assert_array_almost_equal(
        result_torch.values.numpy(), result_numpy_values
    )

    # Compare indices
    np.testing.assert_array_equal(
        result_torch.indices.numpy(), result_numpy_indices
    )


@pytest.mark.random
def test_stable_sort_batch_processing() -> None:
    """Test stable sort with batch dimensions."""
    batch_size = 5
    x_batch = torch.randn(batch_size, 10, dtype=torch.float64)

    result_batch = QF.stable_sort(x_batch, dim=1)
    assert result_batch.values.shape == x_batch.shape
    assert result_batch.indices.shape == x_batch.shape

    # Verify each batch is sorted independently
    for i in range(batch_size):
        for j in range(x_batch.shape[1] - 1):
            assert result_batch.values[i, j] <= result_batch.values[i, j + 1]

        # Verify indices correctly map to original values
        reconstructed = x_batch[i][result_batch.indices[i]]
        torch.testing.assert_close(result_batch.values[i], reconstructed)


def test_stable_sort_numerical_stability() -> None:
    """Test stable sort with various numerical scales."""
    # Very large values
    x_large = torch.tensor([1e10, 2e10, 1e10, 3e10], dtype=torch.float64)
    result_large = QF.stable_sort(x_large, dim=0)

    # Check that large equal values preserve order
    assert (
        result_large.indices[0] < result_large.indices[1]
    )  # First 1e10 before second

    # Very small values
    x_small = torch.tensor([1e-10, 2e-10, 1e-10, 3e-10], dtype=torch.float64)
    result_small = QF.stable_sort(x_small, dim=0)

    # Check that small equal values preserve order
    assert (
        result_small.indices[0] < result_small.indices[1]
    )  # First 1e-10 before second


def test_stable_sort_edge_cases() -> None:
    """Test stable sort edge cases."""
    # All zeros
    x_zeros = torch.zeros(5, dtype=torch.float64)
    result_zeros = QF.stable_sort(x_zeros, dim=0)
    torch.testing.assert_close(result_zeros.values, x_zeros)
    expected_indices = torch.arange(5, dtype=torch.long)
    torch.testing.assert_close(result_zeros.indices, expected_indices)

    # Mix of positive and negative
    x_mixed = torch.tensor([-2.0, 1.0, -1.0, 2.0, 0.0], dtype=torch.float64)
    result_mixed = QF.stable_sort(x_mixed, dim=0)
    expected_values = torch.tensor(
        [-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float64
    )
    torch.testing.assert_close(result_mixed.values, expected_values)


def test_stable_sort_reproducibility() -> None:
    """Test that stable sort produces consistent results."""
    x = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0], dtype=torch.float64)

    result1 = QF.stable_sort(x, dim=0)
    result2 = QF.stable_sort(x, dim=0)

    torch.testing.assert_close(result1.values, result2.values)
    torch.testing.assert_close(result1.indices, result2.indices)


@pytest.mark.random
def test_stable_sort_performance() -> None:
    """Test stable sort performance with larger tensors."""
    x_large = torch.randn(500, 200, dtype=torch.float64)

    result = QF.stable_sort(x_large, dim=1)
    assert result.values.shape == x_large.shape
    assert result.indices.shape == x_large.shape

    # Verify sorting is correct for a few random rows
    for i in [0, 100, 250, 400, 499]:
        row_values = result.values[i]
        for j in range(len(row_values) - 1):
            assert row_values[j] <= row_values[j + 1]


def test_stable_sort_index_mapping() -> None:
    """Test that indices correctly map back to original values."""
    x = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0], dtype=torch.float64)
    result = QF.stable_sort(x, dim=0)

    # Reconstruct sorted values using indices
    reconstructed = x[result.indices]
    torch.testing.assert_close(result.values, reconstructed)

    # Check that all original indices are present
    sorted_indices = torch.sort(result.indices).values
    expected_indices = torch.arange(len(x), dtype=torch.long)
    torch.testing.assert_close(sorted_indices, expected_indices)


def test_stable_sort_gradient_compatibility() -> None:
    """Test that stable sort works with gradient computation."""
    x = torch.tensor(
        [3.0, 1.0, 4.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True
    )
    result = QF.stable_sort(x, dim=0)

    # Values should be differentiable
    loss = result.values.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    # All gradients should be 1 since we're just summing
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_stable_sort_special_float_values() -> None:
    """Test stable sort with special float values."""
    # Test with very specific arrangements of special values
    x = torch.tensor(
        [0.0, -0.0, math.inf, -math.inf, math.nan, 1.0, -1.0],
        dtype=torch.float64,
    )
    result = QF.stable_sort(x, dim=0)

    # Check that result has correct shape and types
    assert result.values.shape == x.shape
    assert result.indices.shape == x.shape
    assert result.values.dtype == x.dtype
    assert result.indices.dtype == torch.long

    # Check ordering properties
    values = result.values
    assert torch.isneginf(values[0])  # -inf first
    assert torch.isfinite(values[1:5]).all()  # finite values in middle
    assert torch.isposinf(values[5])  # +inf before NaN
    assert torch.isnan(values[6])  # NaN last


def test_stable_sort_mathematical_properties() -> None:
    """Test mathematical properties of stable sort."""
    x = torch.tensor([5.0, 2.0, 8.0, 2.0, 1.0, 9.0], dtype=torch.float64)
    result = QF.stable_sort(x, dim=0)

    # Property 1: Result should be non-decreasing
    for i in range(len(result.values) - 1):
        assert result.values[i] <= result.values[i + 1]

    # Property 2: Result should be a permutation of input
    sorted_original = torch.sort(x).values
    torch.testing.assert_close(result.values, sorted_original)

    # Property 3: Indices should be a valid permutation
    assert len(torch.unique(result.indices)) == len(x)
    assert torch.all(result.indices >= 0) and torch.all(result.indices < len(x))

    # Property 4: Stable property - equal elements maintain relative order
    # Find positions of equal elements (2.0 appears at indices 1 and 3)
    equal_val = 2.0
    result_positions = torch.where(result.values == equal_val)[0]

    # The indices in the sorted result should maintain original order
    original_indices_of_equal = result.indices[result_positions]
    assert original_indices_of_equal[0] < original_indices_of_equal[1]  # 1 < 3
