import math

import numpy as np
import torch

import qfeval_functions.functions as QF
from tests.functions.test_utils import generic_test_consistency
from tests.functions.test_utils import generic_test_memory_efficiency


def test_mulsum() -> None:
    a = QF.randn(100, 1).double()
    b = QF.randn(1, 200).double()
    np.testing.assert_allclose(
        (a * b).sum().numpy(),
        QF.mulsum(a, b).numpy(),
        1e-5,
        1e-5,
    )
    np.testing.assert_allclose(
        (a * b).sum(dim=1).numpy(),
        QF.mulsum(a, b, dim=1).numpy(),
        1e-5,
        1e-5,
    )
    np.testing.assert_allclose(
        (a * b).sum(dim=-1, keepdim=True).numpy(),
        QF.mulsum(a, b, dim=-1, keepdim=True).numpy(),
        1e-5,
        1e-5,
    )


def test_mulsum_basic_functionality() -> None:
    """Test basic multiplication and sum functionality."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[2.0, 1.0], [1.0, 2.0]])

    # Test global sum
    result = QF.mulsum(x, y)
    expected = (x * y).sum()
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_mulsum_broadcasting() -> None:
    """Test multiplication with broadcasting."""
    # Test various broadcasting scenarios
    x1 = torch.tensor([[1.0, 2.0, 3.0]])
    y1 = torch.tensor([[2.0], [3.0]])

    result1 = QF.mulsum(x1, y1)
    expected1 = (x1 * y1).sum()
    np.testing.assert_allclose(result1.numpy(), expected1.numpy())

    # Test 1D broadcasting
    x2 = torch.tensor([1.0, 2.0, 3.0])
    y2 = torch.tensor([2.0, 1.0, 3.0])

    result2 = QF.mulsum(x2, y2)
    expected2 = (x2 * y2).sum()
    np.testing.assert_allclose(result2.numpy(), expected2.numpy())


def test_mulsum_dimensions() -> None:
    """Test sum calculation along specific dimensions."""
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    y = torch.tensor([[[2.0, 1.0], [1.0, 2.0]], [[1.0, 3.0], [2.0, 1.0]]])

    # Test along dimension 0
    result_dim0 = QF.mulsum(x, y, dim=0)
    expected_dim0 = (x * y).sum(dim=0)
    np.testing.assert_allclose(result_dim0.numpy(), expected_dim0.numpy())

    # Test along dimension 1
    result_dim1 = QF.mulsum(x, y, dim=1)
    expected_dim1 = (x * y).sum(dim=1)
    np.testing.assert_allclose(result_dim1.numpy(), expected_dim1.numpy())

    # Test along dimension 2
    result_dim2 = QF.mulsum(x, y, dim=2)
    expected_dim2 = (x * y).sum(dim=2)
    np.testing.assert_allclose(result_dim2.numpy(), expected_dim2.numpy())


def test_mulsum_multiple_dimensions() -> None:
    """Test sum calculation along multiple dimensions."""
    x = torch.randn(3, 4, 5)
    y = torch.randn(3, 4, 5)

    # Test along multiple dimensions
    result_01 = QF.mulsum(x, y, dim=(0, 1))
    expected_01 = (x * y).sum(dim=(0, 1))
    np.testing.assert_allclose(
        result_01.numpy(), expected_01.numpy(), rtol=1e-5
    )

    result_12 = QF.mulsum(x, y, dim=(1, 2))
    expected_12 = (x * y).sum(dim=(1, 2))
    np.testing.assert_allclose(
        result_12.numpy(), expected_12.numpy(), rtol=1e-5
    )


def test_mulsum_keepdim() -> None:
    """Test keepdim parameter functionality."""
    x = torch.randn(3, 4, 5)
    y = torch.randn(3, 4, 5)

    # Test keepdim=True
    result_keepdim = QF.mulsum(x, y, dim=1, keepdim=True)
    expected_keepdim = (x * y).sum(dim=1, keepdim=True)
    np.testing.assert_allclose(
        result_keepdim.numpy(), expected_keepdim.numpy(), rtol=1e-5
    )

    # Test keepdim=False
    result_no_keepdim = QF.mulsum(x, y, dim=1, keepdim=False)
    expected_no_keepdim = (x * y).sum(dim=1, keepdim=False)
    np.testing.assert_allclose(
        result_no_keepdim.numpy(), expected_no_keepdim.numpy(), rtol=1e-5
    )


def test_mulsum_mean_mode() -> None:
    """Test mean mode functionality."""
    x = torch.randn(3, 4, 5)
    y = torch.randn(3, 4, 5)

    # Test mean=True
    result_mean = QF.mulsum(x, y, dim=1, mean=True)
    expected_mean = (x * y).mean(dim=1)
    np.testing.assert_allclose(
        result_mean.numpy(), expected_mean.numpy(), rtol=1e-5
    )

    # Test mean=False (default)
    result_sum = QF.mulsum(x, y, dim=1, mean=False)
    expected_sum = (x * y).sum(dim=1)
    np.testing.assert_allclose(
        result_sum.numpy(), expected_sum.numpy(), rtol=1e-5
    )


def test_mulsum_negative_dimensions() -> None:
    """Test negative dimension indexing."""
    x = torch.randn(3, 4, 5)
    y = torch.randn(3, 4, 5)

    result_neg1 = QF.mulsum(x, y, dim=-1)
    result_pos2 = QF.mulsum(x, y, dim=2)
    np.testing.assert_allclose(result_neg1.numpy(), result_pos2.numpy())

    result_neg2 = QF.mulsum(x, y, dim=-2)
    result_pos1 = QF.mulsum(x, y, dim=1)
    np.testing.assert_allclose(result_neg2.numpy(), result_pos1.numpy())


def test_mulsum_zero_values() -> None:
    """Test multiplication and sum with zero values."""
    x = torch.tensor([[1.0, 0.0, 3.0], [0.0, 2.0, 1.0]])
    y = torch.tensor([[2.0, 1.0, 0.0], [3.0, 0.0, 4.0]])

    result = QF.mulsum(x, y)
    expected = (x * y).sum()
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_mulsum_negative_values() -> None:
    """Test multiplication and sum with negative values."""
    x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
    y = torch.tensor([[2.0, -1.0], [-2.0, 3.0]])

    result = QF.mulsum(x, y)
    expected = (x * y).sum()
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_mulsum_large_values() -> None:
    """Test multiplication and sum with large values."""
    x = torch.tensor([[1e6, 2e6], [3e6, 4e6]])
    y = torch.tensor([[2e6, 1e6], [1e6, 2e6]])

    result = QF.mulsum(x, y)
    expected = (x * y).sum()
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-6)


def test_mulsum_small_values() -> None:
    """Test multiplication and sum with very small values."""
    x = torch.tensor([[1e-6, 2e-6], [3e-6, 4e-6]])
    y = torch.tensor([[2e-6, 1e-6], [1e-6, 2e-6]])

    result = QF.mulsum(x, y)
    expected = (x * y).sum()
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-6)


def test_mulsum_broadcasting_edge_cases() -> None:
    """Test edge cases in broadcasting."""
    # Scalar with tensor
    x_scalar = torch.tensor(2.0)
    y_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    result1 = QF.mulsum(x_scalar, y_tensor)
    expected1 = (x_scalar * y_tensor).sum()
    np.testing.assert_allclose(result1.numpy(), expected1.numpy())

    # Different shape broadcasting
    x_broadcast = torch.tensor([[[1.0]], [[2.0]]])
    y_broadcast = torch.tensor([[[3.0, 4.0, 5.0]]])

    result2 = QF.mulsum(x_broadcast, y_broadcast)
    expected2 = (x_broadcast * y_broadcast).sum()
    np.testing.assert_allclose(result2.numpy(), expected2.numpy())


def test_mulsum_with_nan_values() -> None:
    """Test mulsum behavior with NaN values."""
    # TODO(claude): Consider implementing NaN-aware version of mulsum that can skip NaN values
    # when computing element-wise multiplication and sum, similar to nansum functionality.
    # Expected behavior: QF.mulsum(x_with_nans, y, skipna=True) should compute sum over
    # only the finite element pairs.
    x = torch.tensor([[1.0, math.nan], [3.0, 4.0]])
    y = torch.tensor([[2.0, 1.0], [1.0, 2.0]])

    result = QF.mulsum(x, y)
    expected = (x * y).sum()

    # Both should be NaN
    assert torch.isnan(result)
    assert torch.isnan(expected)


def test_mulsum_with_infinity() -> None:
    """Test mulsum behavior with infinity values."""
    x = torch.tensor([[1.0, math.inf], [3.0, 4.0]])
    y = torch.tensor([[2.0, 1.0], [1.0, 2.0]])

    result = QF.mulsum(x, y)
    expected = (x * y).sum()

    # Both should handle infinity consistently
    assert torch.isinf(result) or torch.isnan(result)
    assert torch.isinf(expected) or torch.isnan(expected)


def test_mulsum_high_dimensional() -> None:
    """Test mulsum with high-dimensional tensors."""
    x = torch.randn(2, 3, 4, 5, 6)
    y = torch.randn(2, 3, 4, 5, 6)

    result = QF.mulsum(x, y)
    expected = (x * y).sum()
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-6)

    # Test along specific high dimensions
    result_dim3 = QF.mulsum(x, y, dim=3)
    expected_dim3 = (x * y).sum(dim=3)
    np.testing.assert_allclose(
        result_dim3.numpy(), expected_dim3.numpy(), rtol=1e-5
    )


def test_mulsum_empty_dimension() -> None:
    """Test mulsum with empty dimension tuple."""
    x = torch.randn(3, 4)
    y = torch.randn(3, 4)

    # Empty dimension tuple should compute global sum
    result = QF.mulsum(x, y, dim=())
    expected = (x * y).sum()
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-6)


def test_mulsum_mathematical_properties() -> None:
    """Test mathematical properties of mulsum."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[2.0, 1.0], [1.0, 2.0]])

    # Test linearity: mulsum(ax, y) = a * mulsum(x, y)
    a = 3.0
    result1 = QF.mulsum(a * x, y)
    result2 = a * QF.mulsum(x, y)
    np.testing.assert_allclose(result1.numpy(), result2.numpy())

    # Test commutativity: mulsum(x, y) = mulsum(y, x)
    result_xy = QF.mulsum(x, y)
    result_yx = QF.mulsum(y, x)
    np.testing.assert_allclose(result_xy.numpy(), result_yx.numpy())


def test_mulsum_numerical_stability() -> None:
    """Test numerical stability of mulsum."""
    # Test with values that might cause overflow in naive multiplication
    x = torch.tensor([[1e10, 1e-10], [1e10, 1e-10]])
    y = torch.tensor([[1e-10, 1e10], [1e-10, 1e10]])

    result = QF.mulsum(x, y)
    expected = (x * y).sum()

    # Both should be finite and equal
    assert torch.isfinite(result)
    assert torch.isfinite(expected)
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_mulsum_batch_processing() -> None:
    """Test mulsum with batch processing scenarios."""
    batch_size = 4
    seq_length = 10
    features = 8

    x = torch.randn(batch_size, seq_length, features)
    y = torch.randn(batch_size, seq_length, features)

    # Test batch-wise sum (across sequence and features)
    result_batch = QF.mulsum(x, y, dim=(1, 2))
    expected_batch = (x * y).sum(dim=(1, 2))
    np.testing.assert_allclose(
        result_batch.numpy(), expected_batch.numpy(), rtol=1e-5
    )

    assert result_batch.shape == (batch_size,)


def test_mulsum_zero_size_tensor() -> None:
    """Test mulsum with zero-size tensor."""
    # TODO(claude): The mulsum function should provide more informative handling of
    # empty tensors and edge cases. Expected behavior: add input validation to check
    # for compatible tensor shapes early and provide clear error messages when
    # broadcasting fails or when operations on empty tensors might be ambiguous.
    x = torch.empty(0, 3)
    y = torch.empty(0, 3)

    result = QF.mulsum(x, y)
    expected = (x * y).sum()

    # Both should handle empty tensors consistently
    assert result.item() == expected.item() == 0


def test_mulsum_ddof_parameter() -> None:
    """Test _ddof parameter in mean mode."""
    x = torch.randn(3, 4)
    y = torch.randn(3, 4)

    # Test with ddof=0 (default)
    result_ddof0 = QF.mulsum(x, y, dim=1, mean=True, _ddof=0)
    expected_ddof0 = (x * y).mean(dim=1)
    np.testing.assert_allclose(
        result_ddof0.numpy(), expected_ddof0.numpy(), rtol=1e-5
    )

    # Test with ddof=1 (sample mean)
    result_ddof1 = QF.mulsum(x, y, dim=1, mean=True, _ddof=1)
    # Manual calculation: sum / (n - ddof)
    expected_ddof1 = (x * y).sum(dim=1) / (x.shape[1] - 1)
    np.testing.assert_allclose(
        result_ddof1.numpy(), expected_ddof1.numpy(), rtol=1e-5
    )


def test_mulsum_complex_broadcasting() -> None:
    """Test complex broadcasting scenarios."""
    # Test with compatible shapes that can broadcast
    x1 = torch.randn(2, 1, 4)
    y1 = torch.randn(1, 5, 4)  # Changed to be compatible with x1

    result1 = QF.mulsum(x1, y1, dim=2)
    expected1 = (x1 * y1).sum(dim=2)
    np.testing.assert_allclose(result1.numpy(), expected1.numpy(), rtol=1e-5)

    # Test with 1D vs 3D
    x2 = torch.randn(5)
    y2 = torch.randn(2, 3, 5)

    result2 = QF.mulsum(x2, y2, dim=-1)
    expected2 = (x2 * y2).sum(dim=-1)
    np.testing.assert_allclose(result2.numpy(), expected2.numpy(), rtol=1e-5)


def test_mulsum_einsum_integration() -> None:
    """Test that mulsum properly integrates with einsum."""
    # Test case that exercises the einsum logic
    x = torch.randn(3, 4, 5)
    y = torch.randn(3, 4, 5)

    # Test various dimension combinations
    dims_to_test: list[int | tuple[int, ...]] = [
        0,
        1,
        2,
        (0, 1),
        (1, 2),
        (0, 2),
        (0, 1, 2),
    ]

    for dim in dims_to_test:
        result = QF.mulsum(x, y, dim=dim)
        expected = (x * y).sum(dim=dim)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-4)


def test_mulsum_performance_comparison() -> None:
    """Test that mulsum provides memory efficiency compared to naive approach."""
    # This test verifies the function works correctly for larger tensors
    # without explicitly measuring memory (which would be complex in tests)
    for size in [50, 100]:
        x = torch.randn(size, size)
        y = torch.randn(size, size)

        result_mulsum = QF.mulsum(x, y)
        result_naive = (x * y).sum()

        np.testing.assert_allclose(
            result_mulsum.numpy(), result_naive.numpy(), rtol=1e-5
        )

        # Clean up
        del x, y, result_mulsum, result_naive


def test_mulsum_dtype_preservation() -> None:
    """Test that mulsum preserves input dtype."""
    test_dtypes = [torch.float32, torch.float64]

    for dtype in test_dtypes:
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
        y = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=dtype)
        result = QF.mulsum(x, y)
        assert result.dtype == dtype


def test_mulsum_device_preservation() -> None:
    """Test that mulsum preserves input device."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    result = QF.mulsum(x, y)
    assert result.device == x.device


def test_mulsum_memory_efficiency() -> None:
    """Test memory efficiency of mulsum."""

    def mulsum_wrapper(x: torch.Tensor) -> torch.Tensor:
        return QF.mulsum(x, x)

    generic_test_memory_efficiency(mulsum_wrapper)


def test_mulsum_single_element() -> None:
    """Test mulsum with single element tensors."""
    x_single = torch.tensor([42.0])
    y_single = torch.tensor([2.0])
    result = QF.mulsum(x_single, y_single)
    expected = (x_single * y_single).sum()
    torch.testing.assert_close(result, expected)


def test_mulsum_consistency() -> None:
    """Test that multiple calls to mulsum produce same result."""

    def mulsum_wrapper(x: torch.Tensor) -> torch.Tensor:
        return QF.mulsum(x, x)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    generic_test_consistency(mulsum_wrapper, x)
