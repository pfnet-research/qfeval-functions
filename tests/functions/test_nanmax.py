import math

import numpy as np
import pytest
import torch

import qfeval_functions.functions as QF


def test_nanmax() -> None:
    x = torch.tensor(
        [
            [0.0, -1.0, 1.0, math.nan],
            [math.nan, math.nan, math.nan, math.nan],
            [0.0, -1.0, 2.0, -2.0],
        ]
    )
    np.testing.assert_allclose(
        QF.nanmax(x, dim=1).values.numpy(),
        np.array([1.0, math.nan, 2.0]),
    )
    np.testing.assert_allclose(
        QF.nanmax(x, dim=1).indices.numpy(),
        np.array([2, 0, 2]),
    )


def test_nanmax_with_nan_and_neginf() -> None:
    x = torch.tensor(
        [
            [-math.inf, -1.0, 1.0, math.nan],
            [math.inf, -1.0, 1.0, math.nan],
            [math.nan, math.nan, math.nan, math.nan],
            [-math.inf, -math.inf, -math.inf, -math.inf],
            [math.nan, -math.inf, math.nan, math.nan],
            [math.nan, math.nan, math.inf, math.nan],
            [0.0, -1.0, 2.0, -2.0],
        ]
    )
    np.testing.assert_allclose(
        QF.nanmax(x, dim=1).values.numpy(),
        np.array(
            [1.0, math.inf, math.nan, -math.inf, -math.inf, math.inf, 2.0]
        ),
    )
    np.testing.assert_allclose(
        QF.nanmax(x, dim=1).indices.numpy(),
        np.array([2, 0, 0, 0, 1, 2, 2]),
    )


def test_nanmax_basic_functionality() -> None:
    """Test basic NaN-aware maximum functionality."""
    # Simple case with no NaN
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 1.0, 5.0]])
    result = QF.nanmax(x, dim=1)

    expected_values = torch.tensor([3.0, 5.0])
    expected_indices = torch.tensor([1, 2])
    torch.testing.assert_close(result.values, expected_values)
    torch.testing.assert_close(result.indices, expected_indices)


def test_nanmax_return_type() -> None:
    """Test that nanmax returns a proper NamedTuple."""
    x = torch.tensor([[1.0, 2.0, 3.0]])
    result = QF.nanmax(x, dim=1)

    # Check it's a named tuple with correct attributes
    assert hasattr(result, "values")
    assert hasattr(result, "indices")
    assert isinstance(result.values, torch.Tensor)
    assert isinstance(result.indices, torch.Tensor)


@pytest.mark.random
def test_nanmax_shape_preservation() -> None:
    """Test that nanmax preserves correct output shapes."""
    # 2D tensor
    x_2d = torch.randn(3, 5)
    result_2d = QF.nanmax(x_2d, dim=1)
    assert result_2d.values.shape == (3,)
    assert result_2d.indices.shape == (3,)

    # 3D tensor
    x_3d = torch.randn(2, 4, 6)
    result_3d_dim1 = QF.nanmax(x_3d, dim=1)
    assert result_3d_dim1.values.shape == (2, 6)
    assert result_3d_dim1.indices.shape == (2, 6)

    result_3d_dim2 = QF.nanmax(x_3d, dim=2)
    assert result_3d_dim2.values.shape == (2, 4)
    assert result_3d_dim2.indices.shape == (2, 4)


def test_nanmax_keepdim_parameter() -> None:
    """Test nanmax with keepdim parameter."""
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 1.0, 5.0]])

    # keepdim=False (default)
    result_no_keepdim = QF.nanmax(x, dim=1, keepdim=False)
    assert result_no_keepdim.values.shape == (2,)
    assert result_no_keepdim.indices.shape == (2,)

    # keepdim=True
    result_keepdim = QF.nanmax(x, dim=1, keepdim=True)
    assert result_keepdim.values.shape == (2, 1)
    assert result_keepdim.indices.shape == (2, 1)

    # Values should be the same
    torch.testing.assert_close(
        result_no_keepdim.values, result_keepdim.values.squeeze()
    )
    torch.testing.assert_close(
        result_no_keepdim.indices, result_keepdim.indices.squeeze()
    )


def test_nanmax_nan_handling() -> None:
    """Test that nanmax correctly ignores NaN values."""
    # Mixed NaN and regular values
    x = torch.tensor(
        [
            [1.0, math.nan, 3.0, 2.0],
            [math.nan, 4.0, math.nan, 1.0],
            [2.0, 3.0, 1.0, math.nan],
        ]
    )
    result = QF.nanmax(x, dim=1)

    expected_values = torch.tensor([3.0, 4.0, 3.0])
    expected_indices = torch.tensor([2, 1, 1])
    torch.testing.assert_close(result.values, expected_values)
    torch.testing.assert_close(result.indices, expected_indices)


def test_nanmax_all_nan() -> None:
    """Test nanmax behavior when all values are NaN."""
    x = torch.tensor([[math.nan, math.nan, math.nan], [1.0, 2.0, 3.0]])
    result = QF.nanmax(x, dim=1)

    # All NaN row should return NaN
    assert torch.isnan(result.values[0])
    # Regular row should work normally
    torch.testing.assert_close(result.values[1], torch.tensor(3.0))
    torch.testing.assert_close(result.indices[1], torch.tensor(2))


def test_nanmax_with_infinity() -> None:
    """Test nanmax with positive and negative infinity."""
    x = torch.tensor(
        [
            [1.0, math.inf, 2.0],
            [-math.inf, 3.0, 1.0],
            [math.inf, -math.inf, math.nan],
        ]
    )
    result = QF.nanmax(x, dim=1)

    # Positive infinity should be maximum
    assert torch.isinf(result.values[0]) and result.values[0] > 0
    torch.testing.assert_close(result.indices[0], torch.tensor(1))

    # Regular value should be max when competing with negative infinity
    torch.testing.assert_close(result.values[1], torch.tensor(3.0))
    torch.testing.assert_close(result.indices[1], torch.tensor(1))

    # Positive infinity should win over negative infinity and NaN
    assert torch.isinf(result.values[2]) and result.values[2] > 0
    torch.testing.assert_close(result.indices[2], torch.tensor(0))


def test_nanmax_negative_infinity_only() -> None:
    """Test nanmax when only negative infinity values are present."""
    x = torch.tensor(
        [
            [-math.inf, -math.inf, -math.inf],
            [-math.inf, math.nan, -math.inf],
        ]
    )
    result = QF.nanmax(x, dim=1)

    # Should return negative infinity
    assert torch.isneginf(result.values[0])
    assert torch.isneginf(result.values[1])


def test_nanmax_different_dimensions() -> None:
    """Test nanmax along different dimensions."""
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

    # Along dimension 0
    result_dim0 = QF.nanmax(x, dim=0)
    expected_values_dim0 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    torch.testing.assert_close(result_dim0.values, expected_values_dim0)

    # Along dimension 1
    result_dim1 = QF.nanmax(x, dim=1)
    expected_values_dim1 = torch.tensor([[3.0, 4.0], [7.0, 8.0]])
    torch.testing.assert_close(result_dim1.values, expected_values_dim1)

    # Along dimension 2
    result_dim2 = QF.nanmax(x, dim=2)
    expected_values_dim2 = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    torch.testing.assert_close(result_dim2.values, expected_values_dim2)


def test_nanmax_index_correctness() -> None:
    """Test that nanmax returns correct indices."""
    x = torch.tensor(
        [
            [1.0, 5.0, 3.0, 2.0],
            [math.nan, 4.0, math.nan, 6.0],
            [2.0, math.nan, 7.0, 1.0],
        ]
    )
    result = QF.nanmax(x, dim=1)

    # Check indices point to correct maximum values
    expected_indices = torch.tensor([1, 3, 2])  # positions of 5.0, 6.0, 7.0
    torch.testing.assert_close(result.indices, expected_indices)

    # Verify values at those indices
    for i, idx in enumerate(expected_indices):
        assert result.values[i] == x[i, idx] or (
            torch.isnan(result.values[i]) and torch.isnan(x[i, idx])
        )


@pytest.mark.random
def test_nanmax_large_tensors() -> None:
    """Test nanmax with larger tensors for performance."""
    x = torch.randn(100, 1000)
    # Add some NaN values
    x[10:15, 100:110] = math.nan

    result = QF.nanmax(x, dim=1)
    assert result.values.shape == (100,)
    assert result.indices.shape == (100,)
    assert torch.isfinite(result.values[0:10]).all()  # No NaN in first 10 rows
    assert torch.isfinite(result.values[15:]).all()  # No NaN after row 15


def test_nanmax_mathematical_properties() -> None:
    """Test mathematical properties of nanmax."""
    x = torch.tensor([[1.0, 2.0, 3.0, math.nan], [4.0, 5.0, math.nan, 6.0]])
    result = QF.nanmax(x, dim=1)

    # Result should be >= all finite values in each row
    for i in range(x.shape[0]):
        finite_values = x[i][torch.isfinite(x[i])]
        if len(finite_values) > 0:
            assert result.values[i] >= finite_values.max()

    # Values should be finite unless all input values are NaN
    assert torch.isfinite(result.values).all()


def test_nanmax_edge_cases() -> None:
    """Test nanmax edge cases."""
    # All zeros
    x_zeros = torch.zeros(2, 5)
    result_zeros = QF.nanmax(x_zeros, dim=1)
    torch.testing.assert_close(result_zeros.values, torch.zeros(2))

    # All same values
    x_same = torch.full((3, 4), 7.0)
    result_same = QF.nanmax(x_same, dim=1)
    torch.testing.assert_close(result_same.values, torch.full((3,), 7.0))
    torch.testing.assert_close(
        result_same.indices, torch.zeros(3, dtype=torch.int64)
    )


def test_nanmax_negative_values() -> None:
    """Test nanmax with negative values."""
    x = torch.tensor(
        [
            [-5.0, -2.0, -8.0, math.nan],
            [math.nan, -3.0, -1.0, -4.0],
            [-10.0, -6.0, -9.0, -7.0],
        ]
    )
    result = QF.nanmax(x, dim=1)

    expected_values = torch.tensor([-2.0, -1.0, -6.0])
    expected_indices = torch.tensor([1, 2, 1])
    torch.testing.assert_close(result.values, expected_values)
    torch.testing.assert_close(result.indices, expected_indices)


def test_nanmax_mixed_signs() -> None:
    """Test nanmax with mixed positive and negative values."""
    x = torch.tensor(
        [
            [-2.0, 3.0, -1.0, math.nan],
            [math.nan, -4.0, 5.0, -3.0],
            [0.0, -1.0, 1.0, math.nan],
        ]
    )
    result = QF.nanmax(x, dim=1)

    expected_values = torch.tensor([3.0, 5.0, 1.0])
    expected_indices = torch.tensor([1, 2, 2])
    torch.testing.assert_close(result.values, expected_values)
    torch.testing.assert_close(result.indices, expected_indices)


def test_nanmax_numerical_stability() -> None:
    """Test numerical stability with very large and small values."""
    x = torch.tensor(
        [
            [1e10, 2e10, math.nan, 1.5e10],
            [1e-10, math.nan, 2e-10, 1.5e-10],
            [math.nan, -1e15, 1e15, -5e14],
        ]
    )
    result = QF.nanmax(x, dim=1)

    # Should handle large values correctly
    torch.testing.assert_close(result.values[0], torch.tensor(2e10))
    torch.testing.assert_close(result.values[1], torch.tensor(2e-10))
    torch.testing.assert_close(result.values[2], torch.tensor(1e15))

    assert torch.isfinite(result.values).all()


def test_nanmax_special_float_values() -> None:
    """Test nanmax with special float values."""
    x = torch.tensor(
        [
            [0.0, -0.0, 1.0, math.nan],
            [math.inf, -math.inf, math.nan, 5.0],
            [math.nan, math.nan, math.inf, -math.inf],
        ]
    )
    result = QF.nanmax(x, dim=1)

    # Regular case with zeros
    torch.testing.assert_close(result.values[0], torch.tensor(1.0))

    # Positive infinity should win
    assert torch.isinf(result.values[1]) and result.values[1] > 0

    # Positive infinity should win over negative infinity and NaN
    assert torch.isinf(result.values[2]) and result.values[2] > 0


def test_nanmax_reproducibility() -> None:
    """Test that nanmax produces consistent results."""
    x = torch.tensor(
        [[1.0, math.nan, 3.0, 2.0], [math.nan, 4.0, math.nan, 5.0]]
    )

    # Multiple calls should produce same result
    result1 = QF.nanmax(x, dim=1)
    result2 = QF.nanmax(x, dim=1)

    torch.testing.assert_close(result1.values, result2.values, equal_nan=True)
    torch.testing.assert_close(result1.indices, result2.indices)


def test_nanmax_comparison_with_numpy() -> None:
    """Test consistency with numpy.nanmax where applicable."""
    # Create test data without infinities (numpy nanmax behaves differently with inf)
    x_np = np.array(
        [
            [1.0, np.nan, 3.0, 2.0],
            [np.nan, 4.0, np.nan, 5.0],
            [2.0, 3.0, 1.0, np.nan],
        ]
    )
    x_torch = torch.tensor(x_np)

    # Our result
    result = QF.nanmax(x_torch, dim=1)

    # Numpy result
    numpy_result = np.nanmax(x_np, axis=1)

    # Compare values (indices may differ due to tie-breaking)
    torch.testing.assert_close(
        result.values, torch.tensor(numpy_result), equal_nan=True
    )


def test_nanmax_gradient_compatibility() -> None:
    """Test that nanmax works with gradient computation."""
    x = torch.tensor(
        [[1.0, 3.0, 2.0, math.nan], [4.0, math.nan, 5.0, 1.0]],
        requires_grad=True,
    )

    result = QF.nanmax(x, dim=1)

    # Should be able to compute gradients on the values
    loss = result.values.sum()
    loss.backward()

    # Gradient should be computed (non-zero where maximum values are)
    assert x.grad is not None
    assert x.grad.shape == x.shape
