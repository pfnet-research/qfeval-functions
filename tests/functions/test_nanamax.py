import math

import numpy as np
import torch

import qfeval_functions.functions as QF
from tests.functions.test_utils import generic_test_consistency
from tests.functions.test_utils import generic_test_device_preservation
from tests.functions.test_utils import generic_test_dtype_preservation
from tests.functions.test_utils import generic_test_memory_efficiency


def test_nanamax() -> None:
    x = torch.tensor(
        [
            [0.0, -1.0, 1.0, math.nan],
            [math.nan, math.nan, math.nan, math.nan],
            [0.0, -1.0, 2.0, -2.0],
        ]
    )
    np.testing.assert_allclose(
        QF.nanamax(x, dim=1).numpy(),
        np.array([1.0, math.nan, 2.0]),
    )


def test_nanamax_basic_functionality() -> None:
    """Test basic NaN-aware maximum functionality."""
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 1.0, 5.0]])
    result = QF.nanamax(x, dim=1)

    expected = torch.tensor([3.0, 5.0])
    torch.testing.assert_close(result, expected)


def test_nanamax_return_type() -> None:
    """Test that nanamax returns a plain Tensor, not a NamedTuple."""
    x = torch.tensor([[1.0, 2.0, 3.0]])
    result = QF.nanamax(x, dim=1)

    assert isinstance(result, torch.Tensor)
    # Unlike nanmax, nanamax does not return a NamedTuple
    assert not hasattr(result, "values") or not hasattr(result, "_fields")


def test_nanamax_nan_handling() -> None:
    """Test that nanamax correctly ignores NaN values."""
    x = torch.tensor(
        [
            [1.0, math.nan, 3.0, 2.0],
            [math.nan, 4.0, math.nan, 1.0],
            [2.0, 3.0, 1.0, math.nan],
        ]
    )
    result = QF.nanamax(x, dim=1)

    expected = torch.tensor([3.0, 4.0, 3.0])
    torch.testing.assert_close(result, expected)


def test_nanamax_all_nan() -> None:
    """Test nanamax behavior when all values are NaN."""
    x = torch.tensor([[math.nan, math.nan, math.nan], [1.0, 2.0, 3.0]])
    result = QF.nanamax(x, dim=1)

    # All NaN row should return NaN
    assert torch.isnan(result[0])
    # Regular row should work normally
    torch.testing.assert_close(result[1], torch.tensor(3.0))


def test_nanamax_all_nan_global() -> None:
    """Test nanamax with an entirely NaN tensor."""
    x = torch.tensor([math.nan, math.nan, math.nan])
    result = QF.nanamax(x)
    assert torch.isnan(result)


def test_nanamax_with_nan_and_neginf() -> None:
    """Test nanamax with mixed NaN and ±inf values."""
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
        QF.nanamax(x, dim=1).numpy(),
        np.array(
            [1.0, math.inf, math.nan, -math.inf, -math.inf, math.inf, 2.0]
        ),
    )


def test_nanamax_with_infinity() -> None:
    """Test nanamax with positive and negative infinity."""
    x = torch.tensor(
        [
            [1.0, math.inf, 2.0],
            [-math.inf, 3.0, 1.0],
            [math.inf, -math.inf, math.nan],
        ]
    )
    result = QF.nanamax(x, dim=1)

    # Positive infinity should be maximum
    assert torch.isinf(result[0]) and result[0] > 0
    # Regular value should win over negative infinity
    torch.testing.assert_close(result[1], torch.tensor(3.0))
    # Positive infinity should win over negative infinity and NaN
    assert torch.isinf(result[2]) and result[2] > 0


def test_nanamax_negative_infinity_only() -> None:
    """Test nanamax when only negative infinity values are present."""
    x = torch.tensor(
        [
            [-math.inf, -math.inf, -math.inf],
            [-math.inf, math.nan, -math.inf],
        ]
    )
    result = QF.nanamax(x, dim=1)

    assert torch.isneginf(result[0])
    assert torch.isneginf(result[1])


def test_nanamax_shape_preservation() -> None:
    """Test that nanamax preserves correct output shapes."""
    # 2D tensor
    x_2d = torch.randn(3, 5)
    result_2d = QF.nanamax(x_2d, dim=1)
    assert result_2d.shape == (3,)

    # 3D tensor
    x_3d = torch.randn(2, 4, 6)
    result_3d_dim1 = QF.nanamax(x_3d, dim=1)
    assert result_3d_dim1.shape == (2, 6)

    result_3d_dim2 = QF.nanamax(x_3d, dim=2)
    assert result_3d_dim2.shape == (2, 4)


def test_nanamax_keepdim_parameter() -> None:
    """Test nanamax with keepdim parameter."""
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 1.0, 5.0]])

    # keepdim=False (default)
    result_no_keepdim = QF.nanamax(x, dim=1, keepdim=False)
    assert result_no_keepdim.shape == (2,)

    # keepdim=True
    result_keepdim = QF.nanamax(x, dim=1, keepdim=True)
    assert result_keepdim.shape == (2, 1)

    # Values should be the same
    torch.testing.assert_close(result_no_keepdim, result_keepdim.squeeze())


def test_nanamax_different_dimensions() -> None:
    """Test nanamax along different dimensions."""
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

    # Along dimension 0
    result_dim0 = QF.nanamax(x, dim=0)
    expected_dim0 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    torch.testing.assert_close(result_dim0, expected_dim0)

    # Along dimension 1
    result_dim1 = QF.nanamax(x, dim=1)
    expected_dim1 = torch.tensor([[3.0, 4.0], [7.0, 8.0]])
    torch.testing.assert_close(result_dim1, expected_dim1)

    # Along dimension 2
    result_dim2 = QF.nanamax(x, dim=2)
    expected_dim2 = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    torch.testing.assert_close(result_dim2, expected_dim2)


def test_nanamax_multiple_dimensions() -> None:
    """Test nanamax with multiple dimensions reduced simultaneously."""
    x = torch.tensor(
        [[[1.0, math.nan], [3.0, 4.0]], [[math.nan, 6.0], [7.0, math.nan]]]
    )

    # dim=(0, 1)
    result_01 = QF.nanamax(x, dim=(0, 1))
    expected_01 = torch.tensor([7.0, 6.0])
    torch.testing.assert_close(result_01, expected_01)

    # dim=(1, 2)
    result_12 = QF.nanamax(x, dim=(1, 2))
    expected_12 = torch.tensor([4.0, 7.0])
    torch.testing.assert_close(result_12, expected_12)


def test_nanamax_empty_dimension_tuple() -> None:
    """Test nanamax with dim=() reducing all dimensions."""
    x = torch.tensor([[1.0, math.nan], [3.0, 4.0]])

    result = QF.nanamax(x, dim=())
    expected = torch.tensor(4.0)
    torch.testing.assert_close(result, expected)


def test_nanamax_negative_dimensions() -> None:
    """Test that negative dim indices match positive ones."""
    x = torch.tensor([[1.0, math.nan, 3.0], [4.0, 5.0, math.nan]])

    result_neg1 = QF.nanamax(x, dim=-1)
    result_pos1 = QF.nanamax(x, dim=1)
    torch.testing.assert_close(result_neg1, result_pos1)

    result_neg2 = QF.nanamax(x, dim=-2)
    result_pos0 = QF.nanamax(x, dim=0)
    torch.testing.assert_close(result_neg2, result_pos0)


def test_nanamax_batch_processing() -> None:
    """Test batch processing pattern with dim=(1, 2) on 3D tensor."""
    batch_size = 3
    seq_length = 5
    features = 4

    x = torch.randn(batch_size, seq_length, features)
    # Add some NaN values
    x[0, 2, 1] = math.nan
    x[1, 0, 3] = math.nan
    x[2, 4, 0] = math.nan

    # Per-batch maximum (reduce sequence x features)
    result_batch = QF.nanamax(x, dim=(1, 2))
    assert result_batch.shape == (batch_size,)

    # Few NaN values, so results should be finite
    assert torch.isfinite(result_batch).all()


def test_nanamax_negative_values() -> None:
    """Test nanamax with negative values."""
    x = torch.tensor(
        [
            [-5.0, -2.0, -8.0, math.nan],
            [math.nan, -3.0, -1.0, -4.0],
            [-10.0, -6.0, -9.0, -7.0],
        ]
    )
    result = QF.nanamax(x, dim=1)

    expected = torch.tensor([-2.0, -1.0, -6.0])
    torch.testing.assert_close(result, expected)


def test_nanamax_mixed_signs() -> None:
    """Test nanamax with mixed positive and negative values."""
    x = torch.tensor(
        [
            [-2.0, 3.0, -1.0, math.nan],
            [math.nan, -4.0, 5.0, -3.0],
            [0.0, -1.0, 1.0, math.nan],
        ]
    )
    result = QF.nanamax(x, dim=1)

    expected = torch.tensor([3.0, 5.0, 1.0])
    torch.testing.assert_close(result, expected)


def test_nanamax_edge_cases() -> None:
    """Test nanamax edge cases."""
    # All zeros
    x_zeros = torch.zeros(2, 5)
    result_zeros = QF.nanamax(x_zeros, dim=1)
    torch.testing.assert_close(result_zeros, torch.zeros(2))

    # All same values
    x_same = torch.full((3, 4), 7.0)
    result_same = QF.nanamax(x_same, dim=1)
    torch.testing.assert_close(result_same, torch.full((3,), 7.0))


def test_nanamax_numerical_stability() -> None:
    """Test numerical stability with very large and small values."""
    x = torch.tensor(
        [
            [1e10, 2e10, math.nan, 1.5e10],
            [1e-10, math.nan, 2e-10, 1.5e-10],
            [math.nan, -1e15, 1e15, -5e14],
        ]
    )
    result = QF.nanamax(x, dim=1)

    torch.testing.assert_close(result[0], torch.tensor(2e10))
    torch.testing.assert_close(result[1], torch.tensor(2e-10))
    torch.testing.assert_close(result[2], torch.tensor(1e15))

    assert torch.isfinite(result).all()


def test_nanamax_special_float_values() -> None:
    """Test nanamax with special float values."""
    x = torch.tensor(
        [
            [0.0, -0.0, 1.0, math.nan],
            [math.inf, -math.inf, math.nan, 5.0],
            [math.nan, math.nan, math.inf, -math.inf],
        ]
    )
    result = QF.nanamax(x, dim=1)

    torch.testing.assert_close(result[0], torch.tensor(1.0))
    assert torch.isinf(result[1]) and result[1] > 0
    assert torch.isinf(result[2]) and result[2] > 0


def test_nanamax_mathematical_properties() -> None:
    """Test mathematical properties of nanamax."""
    x = torch.tensor([[1.0, 2.0, 3.0, math.nan], [4.0, 5.0, math.nan, 6.0]])
    result = QF.nanamax(x, dim=1)

    # Result should be >= all finite values in each row
    for i in range(x.shape[0]):
        finite_values = x[i][torch.isfinite(x[i])]
        if len(finite_values) > 0:
            assert result[i] >= finite_values.max()

    assert torch.isfinite(result).all()


def test_nanamax_reproducibility() -> None:
    """Test that nanamax produces consistent results."""
    x = torch.tensor(
        [[1.0, math.nan, 3.0, 2.0], [math.nan, 4.0, math.nan, 5.0]]
    )

    result1 = QF.nanamax(x, dim=1)
    result2 = QF.nanamax(x, dim=1)

    torch.testing.assert_close(result1, result2, equal_nan=True)


def test_nanamax_comparison_with_numpy() -> None:
    """Test consistency with numpy.nanmax."""
    x_np = np.array(
        [
            [1.0, np.nan, 3.0, 2.0],
            [np.nan, 4.0, np.nan, 5.0],
            [2.0, 3.0, 1.0, np.nan],
        ]
    )
    x_torch = torch.tensor(x_np)

    # Single dimension
    result = QF.nanamax(x_torch, dim=1)
    numpy_result = np.nanmax(x_np, axis=1)
    torch.testing.assert_close(
        result, torch.tensor(numpy_result), equal_nan=True
    )


def test_nanamax_comparison_with_numpy_multiple_dims() -> None:
    """Test consistency with numpy.nanmax over multiple dimensions."""
    x_np = np.array(
        [[[1.0, np.nan], [3.0, 4.0]], [[np.nan, 6.0], [7.0, np.nan]]]
    )
    x_torch = torch.tensor(x_np)

    # Multiple dimensions
    result = QF.nanamax(x_torch, dim=(1, 2))
    numpy_result = np.nanmax(x_np, axis=(1, 2))
    torch.testing.assert_close(
        result, torch.tensor(numpy_result), equal_nan=True
    )


def test_nanamax_gradient_compatibility() -> None:
    """Test that nanamax works with gradient computation."""
    x = torch.tensor(
        [[1.0, 3.0, 2.0, math.nan], [4.0, math.nan, 5.0, 1.0]],
        requires_grad=True,
    )

    result = QF.nanamax(x, dim=1)

    loss = result.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_nanamax_dtype_preservation() -> None:
    """Test that input dtype is preserved."""
    x = torch.tensor([1.0, math.nan, 3.0, 4.0])
    generic_test_dtype_preservation(QF.nanamax, x)


def test_nanamax_device_preservation() -> None:
    """Test that input device is preserved."""
    x = torch.tensor([1.0, math.nan, 3.0, 4.0])
    generic_test_device_preservation(QF.nanamax, x)


def test_nanamax_memory_efficiency() -> None:
    """Test memory efficiency."""
    generic_test_memory_efficiency(QF.nanamax)


def test_nanamax_consistency() -> None:
    """Test consistency across multiple calls."""
    x = torch.tensor([1.0, math.nan, 3.0, 4.0])
    generic_test_consistency(QF.nanamax, x)


def test_nanamax_single_element() -> None:
    """Test nanamax with a single-element tensor."""
    x = torch.tensor([42.0])
    result = QF.nanamax(x)
    assert result.shape == torch.Size([])
    assert result.item() == 42.0


def test_nanamax_single_nan() -> None:
    """Test nanamax with a single NaN element."""
    x = torch.tensor([math.nan])
    result = QF.nanamax(x)
    assert torch.isnan(result)


def test_nanamax_large_tensors() -> None:
    """Test nanamax with larger tensors."""
    x = torch.randn(100, 1000)
    x[10:15, 100:110] = math.nan

    result = QF.nanamax(x, dim=1)
    assert result.shape == (100,)
    assert torch.isfinite(result[0:10]).all()
    assert torch.isfinite(result[15:]).all()


def test_nanamax_all_dimensions_nan() -> None:
    """Test nanamax with some rows entirely NaN."""
    x = torch.tensor([[math.nan, math.nan], [1.0, 2.0], [math.nan, math.nan]])

    result_dim1 = QF.nanamax(x, dim=1)
    assert torch.isnan(result_dim1[0])
    assert result_dim1[1].item() == 2.0
    assert torch.isnan(result_dim1[2])


def test_nanamax_empty_tensor() -> None:
    """Test nanamax with empty tensors respects dim and keepdim."""
    # 1D empty tensor
    x = torch.empty(0)
    result = QF.nanamax(x)
    assert result.shape == torch.Size([])
    assert torch.isnan(result)

    # 2D empty tensor: dim=0 collapses the empty axis, leaves size-3 axis
    x = torch.empty(0, 3)
    result = QF.nanamax(x, dim=0)
    assert result.shape == torch.Size([3])
    assert torch.all(torch.isnan(result))

    # 2D empty tensor: dim=1 collapses the size-3 axis, leaves empty axis
    result = QF.nanamax(x, dim=1)
    assert result.shape == torch.Size([0])

    # 2D empty tensor: keepdim=True
    result = QF.nanamax(x, dim=0, keepdim=True)
    assert result.shape == torch.Size([1, 3])
    assert torch.all(torch.isnan(result))

    # 2D empty tensor (transposed shape)
    x = torch.empty(2, 0)
    result = QF.nanamax(x, dim=1)
    assert result.shape == torch.Size([2])
    assert torch.all(torch.isnan(result))

    result = QF.nanamax(x, dim=1, keepdim=True)
    assert result.shape == torch.Size([2, 1])
    assert torch.all(torch.isnan(result))


def test_nanamax_precision_preservation() -> None:
    """Test that float64 precision is preserved."""
    x = torch.tensor([0.1, 0.2, math.nan, 0.3], dtype=torch.float64)
    result = QF.nanamax(x)
    expected = torch.tensor(0.3, dtype=torch.float64)
    torch.testing.assert_close(result, expected)


def test_nanamax_edge_case_patterns() -> None:
    """Test various NaN placement patterns."""
    # Pattern: NaN, value, NaN
    x1 = torch.tensor([math.nan, 5.0, math.nan])
    result1 = QF.nanamax(x1)
    torch.testing.assert_close(result1, torch.tensor(5.0))

    # Pattern: same values + NaN
    x2 = torch.tensor([2.0, 2.0, math.nan, 2.0])
    result2 = QF.nanamax(x2)
    torch.testing.assert_close(result2, torch.tensor(2.0))

    # Pattern: alternating values and NaN
    x3 = torch.tensor([1.0, math.nan, 3.0, math.nan, 2.0])
    result3 = QF.nanamax(x3)
    torch.testing.assert_close(result3, torch.tensor(3.0))


def test_nanamax_complex_nan_patterns() -> None:
    """Test nanamax with complex NaN patterns in 3D tensor."""
    x = torch.tensor(
        [
            [[1.0, math.nan, 3.0], [math.nan, 5.0, math.nan]],
            [[math.nan, 2.0, math.nan], [4.0, math.nan, 6.0]],
        ]
    )

    # Global maximum
    result_global = QF.nanamax(x)
    torch.testing.assert_close(result_global, torch.tensor(6.0))

    # Verify shapes along each dimension
    result_dim0 = QF.nanamax(x, dim=0)
    result_dim1 = QF.nanamax(x, dim=1)
    result_dim2 = QF.nanamax(x, dim=2)

    assert result_dim0.shape == (2, 3)
    assert result_dim1.shape == (2, 3)
    assert result_dim2.shape == (2, 2)


def test_nanamax_broadcasting_compatibility() -> None:
    """Test nanamax with broadcast-compatible shapes."""
    x = torch.tensor([[1.0, math.nan, 3.0]])  # Shape: (1, 3)

    result = QF.nanamax(x, dim=0)

    assert result[0].item() == 1.0
    assert torch.isnan(result[1])
    assert result[2].item() == 3.0


def test_nanamax_extreme_patterns() -> None:
    """Test nanamax with extreme value patterns."""
    x_alt = torch.tensor(
        [
            [1.0, 10.0, 2.0, 9.0, 3.0, math.nan],
            [math.nan, 7.0, 1.0, 6.0, 2.0, 5.0],
        ]
    )
    result_alt = QF.nanamax(x_alt, dim=1)

    expected = torch.tensor([10.0, 7.0])
    torch.testing.assert_close(result_alt, expected)


def test_nanamax_with_duplicates() -> None:
    """Test nanamax with duplicate maximum values."""
    x = torch.tensor(
        [[1.0, 3.0, 2.0, 3.0, math.nan], [math.nan, 5.0, 5.0, 1.0, 5.0]]
    )
    result = QF.nanamax(x, dim=1)

    expected = torch.tensor([3.0, 5.0])
    torch.testing.assert_close(result, expected)


def test_nanamax_performance() -> None:
    """Test nanamax with medium-sized tensors."""
    x_large = torch.randn(500, 200)
    x_large[x_large > 2] = math.nan

    result = QF.nanamax(x_large, dim=1)
    assert result.shape == (500,)
    assert torch.isfinite(result).sum() > 400


def test_nanamax_boundary_conditions() -> None:
    """Test boundary conditions."""
    # Very small tensor
    x_small = torch.tensor([[1.0, 2.0]])
    result_small = QF.nanamax(x_small, dim=1)
    torch.testing.assert_close(result_small, torch.tensor([2.0]))

    # Mixed finite and infinite values
    x_mixed = torch.tensor(
        [
            [-math.inf, 1.0, math.inf, math.nan],
            [math.nan, -math.inf, 2.0, math.inf],
        ]
    )
    result_mixed = QF.nanamax(x_mixed, dim=1)

    # Positive infinity should be maximum
    assert torch.isinf(result_mixed[0]) and result_mixed[0] > 0
    assert torch.isinf(result_mixed[1]) and result_mixed[1] > 0


def test_nanamax_gradient_with_duplicate_max() -> None:
    """Test gradient distribution when multiple elements share the max value.

    Unlike torch.max (which sends gradient to the first max element only),
    torch.amax distributes gradient equally among all max elements
    (subgradient). Since nanamax uses amax internally, this behavior
    must be verified.
    """
    # 1D: two elements share the max value
    x = torch.tensor([3.0, 3.0, 1.0], requires_grad=True)
    y = QF.nanamax(x)
    y.backward()
    expected_grad = torch.tensor(
        [0.5, 0.5, 0.0], dtype=x.dtype, device=x.device
    )
    torch.testing.assert_close(x.grad, expected_grad)

    # 2D: duplicate max along dim=1
    x = torch.tensor([[2.0, 2.0, 1.0], [5.0, 3.0, 5.0]], requires_grad=True)
    y = QF.nanamax(x, dim=1)
    loss = y.sum()
    loss.backward()
    expected_grad = torch.tensor(
        [[0.5, 0.5, 0.0], [0.5, 0.0, 0.5]], dtype=x.dtype, device=x.device
    )
    torch.testing.assert_close(x.grad, expected_grad)

    # With NaN: NaN elements should receive zero gradient
    x = torch.tensor([3.0, math.nan, 3.0, 1.0], requires_grad=True)
    y = QF.nanamax(x)
    y.backward()
    expected_grad = torch.tensor(
        [0.5, 0.0, 0.5, 0.0], dtype=x.dtype, device=x.device
    )
    torch.testing.assert_close(x.grad, expected_grad)
