import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_nanmin() -> None:
    x = torch.tensor(
        [
            [0.0, -1.0, 1.0, math.nan],
            [math.nan, math.nan, math.nan, math.nan],
            [0.0, -1.0, 2.0, -2.0],
        ]
    )
    np.testing.assert_allclose(
        QF.nanmin(x, dim=1).values.numpy(),
        np.array([-1.0, math.nan, -2.0]),
    )
    np.testing.assert_allclose(
        QF.nanmin(x, dim=1).indices.numpy(),
        np.array([1, 0, 3]),
    )


def test_nanmin_with_nan_and_neginf() -> None:
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
        QF.nanmin(x, dim=1).values.numpy(),
        np.array([-math.inf, -1, math.nan, -math.inf, -math.inf, math.inf, -2]),
    )
    np.testing.assert_allclose(
        QF.nanmin(x, dim=1).indices.numpy(),
        np.array([0, 1, 0, 0, 1, 2, 3]),
    )


def test_nanmin_basic_functionality() -> None:
    """Test basic NaN-aware minimum functionality."""
    # Simple case with no NaN
    x = torch.tensor([[3.0, 1.0, 2.0], [5.0, 4.0, 1.0]])
    result = QF.nanmin(x, dim=1)

    expected_values = torch.tensor([1.0, 1.0])
    expected_indices = torch.tensor([1, 2])
    torch.testing.assert_close(result.values, expected_values)
    torch.testing.assert_close(result.indices, expected_indices)


def test_nanmin_return_type() -> None:
    """Test that nanmin returns a proper NamedTuple."""
    x = torch.tensor([[3.0, 2.0, 1.0]])
    result = QF.nanmin(x, dim=1)

    # Check it's a named tuple with correct attributes
    assert hasattr(result, "values")
    assert hasattr(result, "indices")
    assert isinstance(result.values, torch.Tensor)
    assert isinstance(result.indices, torch.Tensor)


def test_nanmin_shape_preservation() -> None:
    """Test that nanmin preserves correct output shapes."""
    # 2D tensor
    x_2d = torch.randn(3, 5)
    result_2d = QF.nanmin(x_2d, dim=1)
    assert result_2d.values.shape == (3,)
    assert result_2d.indices.shape == (3,)

    # 3D tensor
    x_3d = torch.randn(2, 4, 6)
    result_3d_dim1 = QF.nanmin(x_3d, dim=1)
    assert result_3d_dim1.values.shape == (2, 6)
    assert result_3d_dim1.indices.shape == (2, 6)

    result_3d_dim2 = QF.nanmin(x_3d, dim=2)
    assert result_3d_dim2.values.shape == (2, 4)
    assert result_3d_dim2.indices.shape == (2, 4)


def test_nanmin_keepdim_parameter() -> None:
    """Test nanmin with keepdim parameter."""
    x = torch.tensor([[3.0, 1.0, 2.0], [5.0, 4.0, 1.0]])

    # keepdim=False (default)
    result_no_keepdim = QF.nanmin(x, dim=1, keepdim=False)
    assert result_no_keepdim.values.shape == (2,)
    assert result_no_keepdim.indices.shape == (2,)

    # keepdim=True
    result_keepdim = QF.nanmin(x, dim=1, keepdim=True)
    assert result_keepdim.values.shape == (2, 1)
    assert result_keepdim.indices.shape == (2, 1)

    # Values should be the same
    torch.testing.assert_close(
        result_no_keepdim.values, result_keepdim.values.squeeze()
    )
    torch.testing.assert_close(
        result_no_keepdim.indices, result_keepdim.indices.squeeze()
    )


def test_nanmin_nan_handling() -> None:
    """Test that nanmin correctly ignores NaN values."""
    # Mixed NaN and regular values
    x = torch.tensor(
        [
            [3.0, float("nan"), 1.0, 2.0],
            [float("nan"), 4.0, float("nan"), 5.0],
            [2.0, 1.0, 3.0, float("nan")],
        ]
    )
    result = QF.nanmin(x, dim=1)

    expected_values = torch.tensor([1.0, 4.0, 1.0])
    expected_indices = torch.tensor([2, 1, 1])
    torch.testing.assert_close(result.values, expected_values)
    torch.testing.assert_close(result.indices, expected_indices)


def test_nanmin_all_nan() -> None:
    """Test nanmin behavior when all values are NaN."""
    x = torch.tensor(
        [[float("nan"), float("nan"), float("nan")], [3.0, 2.0, 1.0]]
    )
    result = QF.nanmin(x, dim=1)

    # All NaN row should return NaN
    assert torch.isnan(result.values[0])
    # Regular row should work normally
    torch.testing.assert_close(result.values[1], torch.tensor(1.0))
    torch.testing.assert_close(result.indices[1], torch.tensor(2))


def test_nanmin_with_infinity() -> None:
    """Test nanmin with positive and negative infinity."""
    x = torch.tensor(
        [
            [2.0, float("-inf"), 1.0],
            [float("inf"), 1.0, 3.0],
            [float("-inf"), float("inf"), float("nan")],
        ]
    )
    result = QF.nanmin(x, dim=1)

    # Negative infinity should be minimum
    assert torch.isneginf(result.values[0])
    torch.testing.assert_close(result.indices[0], torch.tensor(1))

    # Regular value should be min when competing with positive infinity
    torch.testing.assert_close(result.values[1], torch.tensor(1.0))
    torch.testing.assert_close(result.indices[1], torch.tensor(1))

    # Negative infinity should win over positive infinity and NaN
    assert torch.isneginf(result.values[2])
    torch.testing.assert_close(result.indices[2], torch.tensor(0))


def test_nanmin_positive_infinity_only() -> None:
    """Test nanmin when only positive infinity values are present."""
    x = torch.tensor(
        [
            [float("inf"), float("inf"), float("inf")],
            [float("inf"), float("nan"), float("inf")],
        ]
    )
    result = QF.nanmin(x, dim=1)

    # Should return positive infinity
    assert torch.isinf(result.values[0]) and result.values[0] > 0
    assert torch.isinf(result.values[1]) and result.values[1] > 0


def test_nanmin_different_dimensions() -> None:
    """Test nanmin along different dimensions."""
    x = torch.tensor([[[4.0, 3.0], [2.0, 1.0]], [[8.0, 7.0], [6.0, 5.0]]])

    # Along dimension 0
    result_dim0 = QF.nanmin(x, dim=0)
    expected_values_dim0 = torch.tensor([[4.0, 3.0], [2.0, 1.0]])
    torch.testing.assert_close(result_dim0.values, expected_values_dim0)

    # Along dimension 1
    result_dim1 = QF.nanmin(x, dim=1)
    expected_values_dim1 = torch.tensor([[2.0, 1.0], [6.0, 5.0]])
    torch.testing.assert_close(result_dim1.values, expected_values_dim1)

    # Along dimension 2
    result_dim2 = QF.nanmin(x, dim=2)
    expected_values_dim2 = torch.tensor([[3.0, 1.0], [7.0, 5.0]])
    torch.testing.assert_close(result_dim2.values, expected_values_dim2)


def test_nanmin_index_correctness() -> None:
    """Test that nanmin returns correct indices."""
    x = torch.tensor(
        [
            [5.0, 1.0, 3.0, 2.0],
            [float("nan"), 4.0, float("nan"), 2.0],
            [3.0, float("nan"), 1.0, 4.0],
        ]
    )
    result = QF.nanmin(x, dim=1)

    # Check indices point to correct minimum values
    expected_indices = torch.tensor([1, 3, 2])  # positions of 1.0, 2.0, 1.0
    torch.testing.assert_close(result.indices, expected_indices)

    # Verify values at those indices
    for i, idx in enumerate(expected_indices):
        assert result.values[i] == x[i, idx] or (
            torch.isnan(result.values[i]) and torch.isnan(x[i, idx])
        )


def test_nanmin_large_tensors() -> None:
    """Test nanmin with larger tensors for performance."""
    x = torch.randn(100, 1000)
    # Add some NaN values
    x[10:15, 100:110] = float("nan")

    result = QF.nanmin(x, dim=1)
    assert result.values.shape == (100,)
    assert result.indices.shape == (100,)
    assert torch.isfinite(result.values[0:10]).all()  # No NaN in first 10 rows
    assert torch.isfinite(result.values[15:]).all()  # No NaN after row 15


def test_nanmin_mathematical_properties() -> None:
    """Test mathematical properties of nanmin."""
    x = torch.tensor(
        [[3.0, 2.0, 1.0, float("nan")], [6.0, 5.0, float("nan"), 4.0]]
    )
    result = QF.nanmin(x, dim=1)

    # Result should be <= all finite values in each row
    for i in range(x.shape[0]):
        finite_values = x[i][torch.isfinite(x[i])]
        if len(finite_values) > 0:
            assert result.values[i] <= finite_values.min()

    # Values should be finite unless all input values are NaN
    assert torch.isfinite(result.values).all()


def test_nanmin_edge_cases() -> None:
    """Test nanmin edge cases."""
    # All zeros
    x_zeros = torch.zeros(2, 5)
    result_zeros = QF.nanmin(x_zeros, dim=1)
    torch.testing.assert_close(result_zeros.values, torch.zeros(2))

    # All same values
    x_same = torch.full((3, 4), 7.0)
    result_same = QF.nanmin(x_same, dim=1)
    torch.testing.assert_close(result_same.values, torch.full((3,), 7.0))
    torch.testing.assert_close(
        result_same.indices, torch.zeros(3, dtype=torch.int64)
    )


def test_nanmin_negative_values() -> None:
    """Test nanmin with negative values."""
    x = torch.tensor(
        [
            [-2.0, -5.0, -1.0, float("nan")],
            [float("nan"), -3.0, -6.0, -4.0],
            [-10.0, -6.0, -9.0, -7.0],
        ]
    )
    result = QF.nanmin(x, dim=1)

    expected_values = torch.tensor([-5.0, -6.0, -10.0])
    expected_indices = torch.tensor([1, 2, 0])
    torch.testing.assert_close(result.values, expected_values)
    torch.testing.assert_close(result.indices, expected_indices)


def test_nanmin_mixed_signs() -> None:
    """Test nanmin with mixed positive and negative values."""
    x = torch.tensor(
        [
            [3.0, -2.0, 1.0, float("nan")],
            [float("nan"), 4.0, -5.0, 3.0],
            [0.0, 1.0, -1.0, float("nan")],
        ]
    )
    result = QF.nanmin(x, dim=1)

    expected_values = torch.tensor([-2.0, -5.0, -1.0])
    expected_indices = torch.tensor([1, 2, 2])
    torch.testing.assert_close(result.values, expected_values)
    torch.testing.assert_close(result.indices, expected_indices)


def test_nanmin_numerical_stability() -> None:
    """Test numerical stability with very large and small values."""
    x = torch.tensor(
        [
            [2e10, 1e10, float("nan"), 1.5e10],
            [2e-10, float("nan"), 1e-10, 1.5e-10],
            [float("nan"), 1e15, -1e15, 5e14],
        ]
    )
    result = QF.nanmin(x, dim=1)

    # Should handle large values correctly
    torch.testing.assert_close(result.values[0], torch.tensor(1e10))
    torch.testing.assert_close(result.values[1], torch.tensor(1e-10))
    torch.testing.assert_close(result.values[2], torch.tensor(-1e15))

    assert torch.isfinite(result.values).all()


def test_nanmin_special_float_values() -> None:
    """Test nanmin with special float values."""
    x = torch.tensor(
        [
            [1.0, 0.0, -0.0, float("nan")],
            [float("-inf"), float("inf"), float("nan"), 5.0],
            [float("nan"), float("nan"), float("-inf"), float("inf")],
        ]
    )
    result = QF.nanmin(x, dim=1)

    # Zero should be minimum (0.0 and -0.0 are equal)
    torch.testing.assert_close(result.values[0], torch.tensor(0.0))

    # Negative infinity should win
    assert torch.isneginf(result.values[1])

    # Negative infinity should win over positive infinity and NaN
    assert torch.isneginf(result.values[2])


def test_nanmin_reproducibility() -> None:
    """Test that nanmin produces consistent results."""
    x = torch.tensor(
        [[3.0, float("nan"), 1.0, 2.0], [float("nan"), 4.0, float("nan"), 1.0]]
    )

    # Multiple calls should produce same result
    result1 = QF.nanmin(x, dim=1)
    result2 = QF.nanmin(x, dim=1)

    torch.testing.assert_close(result1.values, result2.values, equal_nan=True)
    torch.testing.assert_close(result1.indices, result2.indices)


def test_nanmin_comparison_with_numpy() -> None:
    """Test consistency with numpy.nanmin where applicable."""
    # Create test data without infinities (numpy nanmin behaves differently with inf)
    x_np = np.array(
        [
            [3.0, np.nan, 1.0, 2.0],
            [np.nan, 4.0, np.nan, 1.0],
            [2.0, 1.0, 3.0, np.nan],
        ]
    )
    x_torch = torch.tensor(x_np)

    # Our result
    result = QF.nanmin(x_torch, dim=1)

    # Numpy result
    numpy_result = np.nanmin(x_np, axis=1)

    # Compare values (indices may differ due to tie-breaking)
    torch.testing.assert_close(
        result.values, torch.tensor(numpy_result), equal_nan=True
    )


def test_nanmin_gradient_compatibility() -> None:
    """Test that nanmin works with gradient computation."""
    x = torch.tensor(
        [[3.0, 1.0, 2.0, float("nan")], [4.0, float("nan"), 1.0, 5.0]],
        requires_grad=True,
    )

    result = QF.nanmin(x, dim=1)

    # Should be able to compute gradients on the values
    loss = result.values.sum()
    loss.backward()

    # Gradient should be computed (non-zero where minimum values are)
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_nanmin_consistency_with_nanmax() -> None:
    """Test mathematical relationship between nanmin and nanmax."""
    x = torch.tensor(
        [[1.0, 3.0, 2.0, float("nan")], [float("nan"), 4.0, 1.0, 5.0]]
    )

    min_result = QF.nanmin(x, dim=1)
    max_result = QF.nanmax(x, dim=1)

    # Min should be <= max for each row (when both are finite)
    for i in range(x.shape[0]):
        if torch.isfinite(min_result.values[i]) and torch.isfinite(
            max_result.values[i]
        ):
            assert min_result.values[i] <= max_result.values[i]


def test_nanmin_extreme_patterns() -> None:
    """Test nanmin with extreme data patterns."""
    # Alternating high-low pattern
    x_alt = torch.tensor(
        [
            [10.0, 1.0, 9.0, 2.0, 8.0, float("nan")],
            [float("nan"), 5.0, 1.0, 6.0, 2.0, 7.0],
        ]
    )
    result_alt = QF.nanmin(x_alt, dim=1)

    # Should find the minimum values
    expected_values = torch.tensor([1.0, 1.0])
    torch.testing.assert_close(result_alt.values, expected_values)


def test_nanmin_with_duplicates() -> None:
    """Test nanmin with duplicate minimum values."""
    x = torch.tensor(
        [[3.0, 1.0, 2.0, 1.0, float("nan")], [float("nan"), 2.0, 2.0, 3.0, 2.0]]
    )
    result = QF.nanmin(x, dim=1)

    # Should find minimum values (indices may vary due to tie-breaking)
    expected_values = torch.tensor([1.0, 2.0])
    torch.testing.assert_close(result.values, expected_values)

    # Indices should point to first occurrence of minimum
    assert result.indices[0] == 1  # First 1.0
    assert result.indices[1] == 1  # First 2.0


def test_nanmin_performance() -> None:
    """Test nanmin performance with moderately large tensors."""
    x_large = torch.randn(500, 200)
    # Add some NaN values
    x_large[x_large > 2] = float("nan")

    result = QF.nanmin(x_large, dim=1)
    assert result.values.shape == (500,)
    assert result.indices.shape == (500,)
    assert torch.isfinite(result.values).sum() > 400  # Most should be finite


def test_nanmin_boundary_conditions() -> None:
    """Test nanmin with boundary conditions."""
    # Very small tensor
    x_small = torch.tensor([[1.0, 2.0]])
    result_small = QF.nanmin(x_small, dim=1)
    torch.testing.assert_close(result_small.values, torch.tensor([1.0]))
    torch.testing.assert_close(result_small.indices, torch.tensor([0]))

    # Mix of finite and infinite values
    x_mixed = torch.tensor(
        [
            [float("inf"), 1.0, float("-inf"), float("nan")],
            [float("nan"), float("inf"), 2.0, float("-inf")],
        ]
    )
    result_mixed = QF.nanmin(x_mixed, dim=1)

    # Negative infinity should be minimum where present
    assert torch.isneginf(result_mixed.values[0])
    assert torch.isneginf(result_mixed.values[1])
