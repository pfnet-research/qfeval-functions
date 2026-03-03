import math

import numpy as np
import torch

import qfeval_functions.functions as QF
from tests.functions.test_utils import generic_test_consistency
from tests.functions.test_utils import generic_test_device_preservation
from tests.functions.test_utils import generic_test_dtype_preservation
from tests.functions.test_utils import generic_test_memory_efficiency


def test_nanamin() -> None:
    x = torch.tensor(
        [
            [0.0, -1.0, 1.0, math.nan],
            [math.nan, math.nan, math.nan, math.nan],
            [0.0, -1.0, 2.0, -2.0],
        ]
    )
    np.testing.assert_allclose(
        QF.nanamin(x, dim=1).numpy(),
        np.array([-1.0, math.nan, -2.0]),
    )


def test_nanamin_basic_functionality() -> None:
    """Test basic NaN-aware minimum functionality."""
    x = torch.tensor([[3.0, 1.0, 2.0], [5.0, 4.0, 1.0]])
    result = QF.nanamin(x, dim=1)

    expected = torch.tensor([1.0, 1.0])
    torch.testing.assert_close(result, expected)


def test_nanamin_return_type() -> None:
    """Test that nanamin returns a plain Tensor, not a NamedTuple."""
    x = torch.tensor([[3.0, 2.0, 1.0]])
    result = QF.nanamin(x, dim=1)

    assert isinstance(result, torch.Tensor)
    # Unlike nanmin, nanamin does not return a NamedTuple
    assert not hasattr(result, "values") or not hasattr(result, "_fields")


def test_nanamin_nan_handling() -> None:
    """Test that nanamin correctly ignores NaN values."""
    x = torch.tensor(
        [
            [3.0, math.nan, 1.0, 2.0],
            [math.nan, 4.0, math.nan, 5.0],
            [2.0, 1.0, 3.0, math.nan],
        ]
    )
    result = QF.nanamin(x, dim=1)

    expected = torch.tensor([1.0, 4.0, 1.0])
    torch.testing.assert_close(result, expected)


def test_nanamin_all_nan() -> None:
    """Test nanamin behavior when all values are NaN."""
    x = torch.tensor([[math.nan, math.nan, math.nan], [3.0, 2.0, 1.0]])
    result = QF.nanamin(x, dim=1)

    # All NaN row should return NaN
    assert torch.isnan(result[0])
    # Regular row should work normally
    torch.testing.assert_close(result[1], torch.tensor(1.0))


def test_nanamin_all_nan_global() -> None:
    """Test nanamin with an entirely NaN tensor."""
    x = torch.tensor([math.nan, math.nan, math.nan])
    result = QF.nanamin(x)
    assert torch.isnan(result)


def test_nanamin_with_nan_and_neginf() -> None:
    """Test nanamin with mixed NaN and ±inf values."""
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
        QF.nanamin(x, dim=1).numpy(),
        np.array([-math.inf, -1, math.nan, -math.inf, -math.inf, math.inf, -2]),
    )


def test_nanamin_with_infinity() -> None:
    """Test nanamin with positive and negative infinity."""
    x = torch.tensor(
        [
            [2.0, -math.inf, 1.0],
            [math.inf, 1.0, 3.0],
            [-math.inf, math.inf, math.nan],
        ]
    )
    result = QF.nanamin(x, dim=1)

    # Negative infinity should be minimum
    assert torch.isneginf(result[0])
    # Regular value should win over positive infinity
    torch.testing.assert_close(result[1], torch.tensor(1.0))
    # Negative infinity should win over positive infinity and NaN
    assert torch.isneginf(result[2])


def test_nanamin_positive_infinity_only() -> None:
    """Test nanamin when only positive infinity values are present."""
    x = torch.tensor(
        [
            [math.inf, math.inf, math.inf],
            [math.inf, math.nan, math.inf],
        ]
    )
    result = QF.nanamin(x, dim=1)

    assert torch.isinf(result[0]) and result[0] > 0
    assert torch.isinf(result[1]) and result[1] > 0


def test_nanamin_shape_preservation() -> None:
    """Test that nanamin preserves correct output shapes."""
    # 2D tensor
    x_2d = torch.randn(3, 5)
    result_2d = QF.nanamin(x_2d, dim=1)
    assert result_2d.shape == (3,)

    # 3D tensor
    x_3d = torch.randn(2, 4, 6)
    result_3d_dim1 = QF.nanamin(x_3d, dim=1)
    assert result_3d_dim1.shape == (2, 6)

    result_3d_dim2 = QF.nanamin(x_3d, dim=2)
    assert result_3d_dim2.shape == (2, 4)


def test_nanamin_keepdim_parameter() -> None:
    """Test nanamin with keepdim parameter."""
    x = torch.tensor([[3.0, 1.0, 2.0], [5.0, 4.0, 1.0]])

    # keepdim=False (default)
    result_no_keepdim = QF.nanamin(x, dim=1, keepdim=False)
    assert result_no_keepdim.shape == (2,)

    # keepdim=True
    result_keepdim = QF.nanamin(x, dim=1, keepdim=True)
    assert result_keepdim.shape == (2, 1)

    # Values should be the same
    torch.testing.assert_close(result_no_keepdim, result_keepdim.squeeze())


def test_nanamin_different_dimensions() -> None:
    """Test nanamin along different dimensions."""
    x = torch.tensor([[[4.0, 3.0], [2.0, 1.0]], [[8.0, 7.0], [6.0, 5.0]]])

    # Along dimension 0
    result_dim0 = QF.nanamin(x, dim=0)
    expected_dim0 = torch.tensor([[4.0, 3.0], [2.0, 1.0]])
    torch.testing.assert_close(result_dim0, expected_dim0)

    # Along dimension 1
    result_dim1 = QF.nanamin(x, dim=1)
    expected_dim1 = torch.tensor([[2.0, 1.0], [6.0, 5.0]])
    torch.testing.assert_close(result_dim1, expected_dim1)

    # Along dimension 2
    result_dim2 = QF.nanamin(x, dim=2)
    expected_dim2 = torch.tensor([[3.0, 1.0], [7.0, 5.0]])
    torch.testing.assert_close(result_dim2, expected_dim2)


def test_nanamin_multiple_dimensions() -> None:
    """Test nanamin with multiple dimensions reduced simultaneously."""
    x = torch.tensor(
        [[[1.0, math.nan], [3.0, 4.0]], [[math.nan, 6.0], [7.0, math.nan]]]
    )

    # dim=(0, 1)
    result_01 = QF.nanamin(x, dim=(0, 1))
    expected_01 = torch.tensor([1.0, 4.0])
    torch.testing.assert_close(result_01, expected_01)

    # dim=(1, 2)
    result_12 = QF.nanamin(x, dim=(1, 2))
    expected_12 = torch.tensor([1.0, 6.0])
    torch.testing.assert_close(result_12, expected_12)


def test_nanamin_empty_dimension_tuple() -> None:
    """Test nanamin with dim=() reducing all dimensions."""
    x = torch.tensor([[1.0, math.nan], [3.0, 4.0]])

    result = QF.nanamin(x, dim=())
    expected = torch.tensor(1.0)
    torch.testing.assert_close(result, expected)


def test_nanamin_negative_dimensions() -> None:
    """Test that negative dim indices match positive ones."""
    x = torch.tensor([[1.0, math.nan, 3.0], [4.0, 5.0, math.nan]])

    result_neg1 = QF.nanamin(x, dim=-1)
    result_pos1 = QF.nanamin(x, dim=1)
    torch.testing.assert_close(result_neg1, result_pos1)

    result_neg2 = QF.nanamin(x, dim=-2)
    result_pos0 = QF.nanamin(x, dim=0)
    torch.testing.assert_close(result_neg2, result_pos0)


def test_nanamin_batch_processing() -> None:
    """Test batch processing pattern with dim=(1, 2) on 3D tensor."""
    batch_size = 3
    seq_length = 5
    features = 4

    x = torch.randn(batch_size, seq_length, features)
    # Add some NaN values
    x[0, 2, 1] = math.nan
    x[1, 0, 3] = math.nan
    x[2, 4, 0] = math.nan

    # Per-batch minimum (reduce sequence x features)
    result_batch = QF.nanamin(x, dim=(1, 2))
    assert result_batch.shape == (batch_size,)

    # Few NaN values, so results should be finite
    assert torch.isfinite(result_batch).all()


def test_nanamin_negative_values() -> None:
    """Test nanamin with negative values."""
    x = torch.tensor(
        [
            [-2.0, -5.0, -1.0, math.nan],
            [math.nan, -3.0, -6.0, -4.0],
            [-10.0, -6.0, -9.0, -7.0],
        ]
    )
    result = QF.nanamin(x, dim=1)

    expected = torch.tensor([-5.0, -6.0, -10.0])
    torch.testing.assert_close(result, expected)


def test_nanamin_mixed_signs() -> None:
    """Test nanamin with mixed positive and negative values."""
    x = torch.tensor(
        [
            [3.0, -2.0, 1.0, math.nan],
            [math.nan, 4.0, -5.0, 3.0],
            [0.0, 1.0, -1.0, math.nan],
        ]
    )
    result = QF.nanamin(x, dim=1)

    expected = torch.tensor([-2.0, -5.0, -1.0])
    torch.testing.assert_close(result, expected)


def test_nanamin_edge_cases() -> None:
    """Test nanamin edge cases."""
    # All zeros
    x_zeros = torch.zeros(2, 5)
    result_zeros = QF.nanamin(x_zeros, dim=1)
    torch.testing.assert_close(result_zeros, torch.zeros(2))

    # All same values
    x_same = torch.full((3, 4), 7.0)
    result_same = QF.nanamin(x_same, dim=1)
    torch.testing.assert_close(result_same, torch.full((3,), 7.0))


def test_nanamin_numerical_stability() -> None:
    """Test numerical stability with very large and small values."""
    x = torch.tensor(
        [
            [2e10, 1e10, math.nan, 1.5e10],
            [2e-10, math.nan, 1e-10, 1.5e-10],
            [math.nan, 1e15, -1e15, 5e14],
        ]
    )
    result = QF.nanamin(x, dim=1)

    torch.testing.assert_close(result[0], torch.tensor(1e10))
    torch.testing.assert_close(result[1], torch.tensor(1e-10))
    torch.testing.assert_close(result[2], torch.tensor(-1e15))

    assert torch.isfinite(result).all()


def test_nanamin_special_float_values() -> None:
    """Test nanamin with special float values."""
    x = torch.tensor(
        [
            [1.0, 0.0, -0.0, math.nan],
            [-math.inf, math.inf, math.nan, 5.0],
            [math.nan, math.nan, -math.inf, math.inf],
        ]
    )
    result = QF.nanamin(x, dim=1)

    # 0 is the minimum (0.0 and -0.0 are equal)
    torch.testing.assert_close(result[0], torch.tensor(0.0))
    # Negative infinity should win
    assert torch.isneginf(result[1])
    # Negative infinity should win over positive infinity and NaN
    assert torch.isneginf(result[2])


def test_nanamin_mathematical_properties() -> None:
    """Test mathematical properties of nanamin."""
    x = torch.tensor([[3.0, 2.0, 1.0, math.nan], [6.0, 5.0, math.nan, 4.0]])
    result = QF.nanamin(x, dim=1)

    # Result should be <= all finite values in each row
    for i in range(x.shape[0]):
        finite_values = x[i][torch.isfinite(x[i])]
        if len(finite_values) > 0:
            assert result[i] <= finite_values.min()

    assert torch.isfinite(result).all()


def test_nanamin_consistency_with_nanamax() -> None:
    """Test that nanamin <= nanamax always holds."""
    x = torch.tensor([[1.0, 3.0, 2.0, math.nan], [math.nan, 4.0, 1.0, 5.0]])

    min_result = QF.nanamin(x, dim=1)
    max_result = QF.nanamax(x, dim=1)

    for i in range(x.shape[0]):
        if torch.isfinite(min_result[i]) and torch.isfinite(max_result[i]):
            assert min_result[i] <= max_result[i]


def test_nanamin_reproducibility() -> None:
    """Test that nanamin produces consistent results."""
    x = torch.tensor(
        [[3.0, math.nan, 1.0, 2.0], [math.nan, 4.0, math.nan, 1.0]]
    )

    result1 = QF.nanamin(x, dim=1)
    result2 = QF.nanamin(x, dim=1)

    torch.testing.assert_close(result1, result2, equal_nan=True)


def test_nanamin_comparison_with_numpy() -> None:
    """Test consistency with numpy.nanmin."""
    x_np = np.array(
        [
            [3.0, np.nan, 1.0, 2.0],
            [np.nan, 4.0, np.nan, 1.0],
            [2.0, 1.0, 3.0, np.nan],
        ]
    )
    x_torch = torch.tensor(x_np)

    # Single dimension
    result = QF.nanamin(x_torch, dim=1)
    numpy_result = np.nanmin(x_np, axis=1)
    torch.testing.assert_close(
        result, torch.tensor(numpy_result), equal_nan=True
    )


def test_nanamin_comparison_with_numpy_multiple_dims() -> None:
    """Test consistency with numpy.nanmin over multiple dimensions."""
    x_np = np.array(
        [[[1.0, np.nan], [3.0, 4.0]], [[np.nan, 6.0], [7.0, np.nan]]]
    )
    x_torch = torch.tensor(x_np)

    # Multiple dimensions
    result = QF.nanamin(x_torch, dim=(1, 2))
    numpy_result = np.nanmin(x_np, axis=(1, 2))
    torch.testing.assert_close(
        result, torch.tensor(numpy_result), equal_nan=True
    )


def test_nanamin_gradient_compatibility() -> None:
    """Test that nanamin works with gradient computation."""
    x = torch.tensor(
        [[3.0, 1.0, 2.0, math.nan], [4.0, math.nan, 1.0, 5.0]],
        requires_grad=True,
    )

    result = QF.nanamin(x, dim=1)

    loss = result.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_nanamin_dtype_preservation() -> None:
    """Test that input dtype is preserved."""
    x = torch.tensor([1.0, math.nan, 3.0, 4.0])
    generic_test_dtype_preservation(QF.nanamin, x)


def test_nanamin_device_preservation() -> None:
    """Test that input device is preserved."""
    x = torch.tensor([1.0, math.nan, 3.0, 4.0])
    generic_test_device_preservation(QF.nanamin, x)


def test_nanamin_memory_efficiency() -> None:
    """Test memory efficiency."""
    generic_test_memory_efficiency(QF.nanamin)


def test_nanamin_consistency() -> None:
    """Test consistency across multiple calls."""
    x = torch.tensor([1.0, math.nan, 3.0, 4.0])
    generic_test_consistency(QF.nanamin, x)


def test_nanamin_single_element() -> None:
    """Test nanamin with a single-element tensor."""
    x = torch.tensor([42.0])
    result = QF.nanamin(x)
    assert result.shape == torch.Size([])
    assert result.item() == 42.0


def test_nanamin_single_nan() -> None:
    """Test nanamin with a single NaN element."""
    x = torch.tensor([math.nan])
    result = QF.nanamin(x)
    assert torch.isnan(result)


def test_nanamin_large_tensors() -> None:
    """Test nanamin with larger tensors."""
    x = torch.randn(100, 1000)
    x[10:15, 100:110] = math.nan

    result = QF.nanamin(x, dim=1)
    assert result.shape == (100,)
    assert torch.isfinite(result[0:10]).all()
    assert torch.isfinite(result[15:]).all()


def test_nanamin_all_dimensions_nan() -> None:
    """Test nanamin with some rows entirely NaN."""
    x = torch.tensor([[math.nan, math.nan], [1.0, 2.0], [math.nan, math.nan]])

    result_dim1 = QF.nanamin(x, dim=1)
    assert torch.isnan(result_dim1[0])
    assert result_dim1[1].item() == 1.0
    assert torch.isnan(result_dim1[2])


def test_nanamin_empty_tensor() -> None:
    """Test nanamin with empty tensors respects dim and keepdim."""
    # 1D empty tensor
    x = torch.empty(0)
    result = QF.nanamin(x)
    assert result.shape == torch.Size([])
    assert torch.isnan(result)

    # 2D empty tensor: dim=0 collapses the empty axis, leaves size-3 axis
    x = torch.empty(0, 3)
    result = QF.nanamin(x, dim=0)
    assert result.shape == torch.Size([3])
    assert torch.all(torch.isnan(result))

    # 2D empty tensor: dim=1 collapses the size-3 axis, leaves empty axis
    result = QF.nanamin(x, dim=1)
    assert result.shape == torch.Size([0])

    # 2D empty tensor: keepdim=True
    result = QF.nanamin(x, dim=0, keepdim=True)
    assert result.shape == torch.Size([1, 3])
    assert torch.all(torch.isnan(result))

    # 2D empty tensor (transposed shape)
    x = torch.empty(2, 0)
    result = QF.nanamin(x, dim=1)
    assert result.shape == torch.Size([2])
    assert torch.all(torch.isnan(result))

    result = QF.nanamin(x, dim=1, keepdim=True)
    assert result.shape == torch.Size([2, 1])
    assert torch.all(torch.isnan(result))


def test_nanamin_precision_preservation() -> None:
    """Test that float64 precision is preserved."""
    x = torch.tensor([0.3, 0.2, math.nan, 0.1], dtype=torch.float64)
    result = QF.nanamin(x)
    expected = torch.tensor(0.1, dtype=torch.float64)
    torch.testing.assert_close(result, expected)


def test_nanamin_edge_case_patterns() -> None:
    """Test various NaN placement patterns."""
    # Pattern: NaN, value, NaN
    x1 = torch.tensor([math.nan, 5.0, math.nan])
    result1 = QF.nanamin(x1)
    torch.testing.assert_close(result1, torch.tensor(5.0))

    # Pattern: same values + NaN
    x2 = torch.tensor([2.0, 2.0, math.nan, 2.0])
    result2 = QF.nanamin(x2)
    torch.testing.assert_close(result2, torch.tensor(2.0))

    # Pattern: alternating values and NaN
    x3 = torch.tensor([3.0, math.nan, 1.0, math.nan, 2.0])
    result3 = QF.nanamin(x3)
    torch.testing.assert_close(result3, torch.tensor(1.0))


def test_nanamin_complex_nan_patterns() -> None:
    """Test nanamin with complex NaN patterns in 3D tensor."""
    x = torch.tensor(
        [
            [[1.0, math.nan, 3.0], [math.nan, 5.0, math.nan]],
            [[math.nan, 2.0, math.nan], [4.0, math.nan, 6.0]],
        ]
    )

    # Global minimum
    result_global = QF.nanamin(x)
    torch.testing.assert_close(result_global, torch.tensor(1.0))

    # Verify shapes along each dimension
    result_dim0 = QF.nanamin(x, dim=0)
    result_dim1 = QF.nanamin(x, dim=1)
    result_dim2 = QF.nanamin(x, dim=2)

    assert result_dim0.shape == (2, 3)
    assert result_dim1.shape == (2, 3)
    assert result_dim2.shape == (2, 2)


def test_nanamin_broadcasting_compatibility() -> None:
    """Test nanamin with broadcast-compatible shapes."""
    x = torch.tensor([[1.0, math.nan, 3.0]])  # Shape: (1, 3)

    result = QF.nanamin(x, dim=0)

    assert result[0].item() == 1.0
    assert torch.isnan(result[1])
    assert result[2].item() == 3.0


def test_nanamin_extreme_patterns() -> None:
    """Test nanamin with extreme value patterns."""
    x_alt = torch.tensor(
        [
            [10.0, 1.0, 9.0, 2.0, 8.0, math.nan],
            [math.nan, 5.0, 1.0, 6.0, 2.0, 7.0],
        ]
    )
    result_alt = QF.nanamin(x_alt, dim=1)

    expected = torch.tensor([1.0, 1.0])
    torch.testing.assert_close(result_alt, expected)


def test_nanamin_with_duplicates() -> None:
    """Test nanamin with duplicate minimum values."""
    x = torch.tensor(
        [[3.0, 1.0, 2.0, 1.0, math.nan], [math.nan, 2.0, 2.0, 3.0, 2.0]]
    )
    result = QF.nanamin(x, dim=1)

    expected = torch.tensor([1.0, 2.0])
    torch.testing.assert_close(result, expected)


def test_nanamin_performance() -> None:
    """Test nanamin with medium-sized tensors."""
    x_large = torch.randn(500, 200)
    x_large[x_large > 2] = math.nan

    result = QF.nanamin(x_large, dim=1)
    assert result.shape == (500,)
    assert torch.isfinite(result).sum() > 400


def test_nanamin_boundary_conditions() -> None:
    """Test boundary conditions."""
    # Very small tensor
    x_small = torch.tensor([[1.0, 2.0]])
    result_small = QF.nanamin(x_small, dim=1)
    torch.testing.assert_close(result_small, torch.tensor([1.0]))

    # Mixed finite and infinite values
    x_mixed = torch.tensor(
        [
            [math.inf, 1.0, -math.inf, math.nan],
            [math.nan, math.inf, 2.0, -math.inf],
        ]
    )
    result_mixed = QF.nanamin(x_mixed, dim=1)

    # Negative infinity should be minimum
    assert torch.isneginf(result_mixed[0])
    assert torch.isneginf(result_mixed[1])


def test_nanamin_gradient_with_duplicate_min() -> None:
    """Test gradient distribution when multiple elements share the min value.

    nanamin delegates to nanamax via -nanamax(-x), so the gradient of amin
    (equal distribution among duplicate min elements) must propagate
    correctly through the sign negation.
    """
    # 1D: two elements share the min value
    x = torch.tensor([1.0, 1.0, 3.0], requires_grad=True)
    y = QF.nanamin(x)
    y.backward()
    expected_grad = torch.tensor(
        [0.5, 0.5, 0.0], dtype=x.dtype, device=x.device
    )
    torch.testing.assert_close(x.grad, expected_grad)

    # 2D: duplicate min along dim=1
    x = torch.tensor([[2.0, 2.0, 5.0], [1.0, 3.0, 1.0]], requires_grad=True)
    y = QF.nanamin(x, dim=1)
    loss = y.sum()
    loss.backward()
    expected_grad = torch.tensor(
        [[0.5, 0.5, 0.0], [0.5, 0.0, 0.5]], dtype=x.dtype, device=x.device
    )
    torch.testing.assert_close(x.grad, expected_grad)

    # With NaN: NaN elements should receive zero gradient
    x = torch.tensor([1.0, math.nan, 1.0, 3.0], requires_grad=True)
    y = QF.nanamin(x)
    y.backward()
    expected_grad = torch.tensor(
        [0.5, 0.0, 0.5, 0.0], dtype=x.dtype, device=x.device
    )
    torch.testing.assert_close(x.grad, expected_grad)
