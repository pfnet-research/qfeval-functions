import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_apply_for_axis_1d() -> None:
    """Test apply_for_axis with a simple 1D tensor and square function."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    def square_func(t: torch.Tensor) -> torch.Tensor:
        return t**2

    result = QF.apply_for_axis(square_func, x, dim=0)
    expected = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_apply_for_axis_2d() -> None:
    """Test apply_for_axis with a 2D tensor applying mean along axis 1."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def mean_func(t: torch.Tensor) -> torch.Tensor:
        return t.mean(dim=1, keepdim=True)

    result = QF.apply_for_axis(mean_func, x, dim=1)
    expected = torch.tensor([[2.0], [5.0]])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_apply_for_axis_3d() -> None:
    """Test apply_for_axis with a 3D tensor applying sum along axis 0."""
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

    def sum_func(t: torch.Tensor) -> torch.Tensor:
        return t.sum(dim=1, keepdim=True)

    result = QF.apply_for_axis(sum_func, x, dim=0)
    expected = torch.tensor([[[6.0, 8.0], [10.0, 12.0]]])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_apply_for_axis_negative_dim() -> None:
    """Test apply_for_axis with negative dimension indexing (last dimension)."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def double_func(t: torch.Tensor) -> torch.Tensor:
        return t * 2

    result = QF.apply_for_axis(double_func, x, dim=-1)
    expected = torch.tensor([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_apply_for_axis_edge_case_empty() -> None:
    """Test apply_for_axis with empty tensor to ensure it handles zero-length axes."""
    x = torch.empty((2, 0, 3))

    def identity_func(t: torch.Tensor) -> torch.Tensor:
        return t

    result = QF.apply_for_axis(identity_func, x, dim=1)
    expected = torch.empty((2, 0, 3))
    assert result.shape == expected.shape


def test_apply_for_axis_4d_tensor() -> None:
    """Test apply_for_axis with 4D tensor to verify high-dimensional handling."""
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])

    def cumsum_func(t: torch.Tensor) -> torch.Tensor:
        return t.cumsum(dim=1)

    result = QF.apply_for_axis(cumsum_func, x, dim=3)
    expected = torch.tensor(
        [[[[1.0, 3.0], [3.0, 7.0]], [[5.0, 11.0], [7.0, 15.0]]]]
    )
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_apply_for_axis_function_changes_shape() -> None:
    """Test apply_for_axis with function that changes the output shape."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def expand_func(t: torch.Tensor) -> torch.Tensor:
        # Concatenate the tensor with itself doubled to expand the dimension
        return torch.cat([t, t * 2], dim=1)

    result = QF.apply_for_axis(expand_func, x, dim=1)
    expected = torch.tensor(
        [[1.0, 2.0, 3.0, 2.0, 4.0, 6.0], [4.0, 5.0, 6.0, 8.0, 10.0, 12.0]]
    )
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_apply_for_axis_function_reduces_dimension() -> None:
    """Test apply_for_axis with function that reduces each row to a single value."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def mean_func(t: torch.Tensor) -> torch.Tensor:
        # Apply mean along dim=1 to reduce each row to a single value
        return t.mean(dim=1, keepdim=True)

    result = QF.apply_for_axis(mean_func, x, dim=1)
    # Each row is reduced to its mean value
    expected = torch.tensor([[2.0], [5.0]])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_apply_for_axis_with_nan_values() -> None:
    """Test apply_for_axis behavior with NaN values in input tensor."""
    x = torch.tensor([[1.0, math.nan, 3.0], [4.0, 5.0, math.nan]])

    def nanmean_func(t: torch.Tensor) -> torch.Tensor:
        return torch.nanmean(t, dim=1, keepdim=True)

    result = QF.apply_for_axis(nanmean_func, x, dim=1)
    expected = torch.tensor([[2.0], [4.5]])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_apply_for_axis_different_dtypes() -> None:
    """Test apply_for_axis with different tensor dtypes."""
    # Test with integer tensor
    x_int = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)

    def add_one(t: torch.Tensor) -> torch.Tensor:
        return t + 1

    result_int = QF.apply_for_axis(add_one, x_int, dim=1)
    expected_int = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.int32)
    np.testing.assert_array_equal(result_int.numpy(), expected_int.numpy())
    assert result_int.dtype == torch.int32

    # Test with double precision
    x_double = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    result_double = QF.apply_for_axis(add_one, x_double, dim=0)
    expected_double = torch.tensor(
        [[2.0, 3.0], [4.0, 5.0]], dtype=torch.float64
    )
    np.testing.assert_allclose(result_double.numpy(), expected_double.numpy())
    assert result_double.dtype == torch.float64


def test_apply_for_axis_middle_dimensions() -> None:
    """Test apply_for_axis on middle dimensions of multi-dimensional tensors."""
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

    def scale_func(t: torch.Tensor) -> torch.Tensor:
        # Scale each element by 2 - maintains dimensions
        return t * 2

    result = QF.apply_for_axis(scale_func, x, dim=1)
    # Each element should be doubled
    expected = torch.tensor(
        [[[2.0, 4.0], [6.0, 8.0]], [[10.0, 12.0], [14.0, 16.0]]]
    )
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_apply_for_axis_zero_dimension_axis() -> None:
    """Test apply_for_axis with dimension 0 on various tensor shapes."""
    # Test 2D case
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    def normalize_func(t: torch.Tensor) -> torch.Tensor:
        return t / t.sum(dim=1, keepdim=True)

    result = QF.apply_for_axis(normalize_func, x, dim=0)
    # When applying along dim=0, the tensor is already in the right position
    # After transposing: [[1,3,5],[2,4,6]] (shape 2x3)
    # normalize_func divides each row by its sum
    # Row 0: [1,3,5], sum=9, normalized: [1/9, 3/9, 5/9]
    # Row 1: [2,4,6], sum=12, normalized: [2/12, 4/12, 6/12]
    # After transposing back: [[1/9, 2/12], [3/9, 4/12], [5/9, 6/12]]
    expected = torch.tensor(
        [
            [1.0 / 9.0, 2.0 / 12.0],
            [3.0 / 9.0, 4.0 / 12.0],
            [5.0 / 9.0, 6.0 / 12.0],
        ]
    )
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_apply_for_axis_complex_function() -> None:
    """Test apply_for_axis with more complex mathematical operations."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    def complex_func(t: torch.Tensor) -> torch.Tensor:
        # Apply softmax followed by log
        softmax = torch.softmax(t, dim=1)
        return torch.log(softmax + 1e-8)

    result = QF.apply_for_axis(complex_func, x, dim=1)

    # Verify shape is preserved
    assert result.shape == x.shape

    # Verify each row sums to approximately log(1) = 0 after softmax
    row_sums = torch.exp(result).sum(dim=1)
    expected_sums = torch.ones(2)
    np.testing.assert_allclose(
        row_sums.numpy(), expected_sums.numpy(), atol=1e-4
    )


def test_apply_for_axis_broadcasting_behavior() -> None:
    """Test apply_for_axis with functions that involve broadcasting."""
    x = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])

    def broadcast_func(t: torch.Tensor) -> torch.Tensor:
        # Add a constant that will broadcast
        constant = torch.tensor([10.0, 20.0])
        return t + constant.unsqueeze(0)

    result = QF.apply_for_axis(broadcast_func, x, dim=2)
    expected = torch.tensor(
        [[[11.0, 21.0], [12.0, 22.0]], [[13.0, 23.0], [14.0, 24.0]]]
    )
    np.testing.assert_allclose(result.numpy(), expected.numpy())
