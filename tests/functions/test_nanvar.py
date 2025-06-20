import math
import warnings

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_nanvar() -> None:
    x = torch.tensor(
        [
            [1.0, math.nan, 2.0, math.nan, 3.0, 4.0],
            [math.nan, 3.0, 7.0, math.nan, 8.0, math.nan],
            [9.0, 10.0, 11.0, 12.0, 13.0, 20.0],
            [math.nan, math.nan, math.nan, math.nan, math.nan, math.nan],
        ],
        dtype=torch.float64,
    )
    # NOTE: Suppress a warning of calculation of NaNs:
    # RuntimeWarning: Degrees of freedom <= 0 for slice.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = np.nanvar(x, axis=1, ddof=0)
    np.testing.assert_allclose(
        QF.nanvar(x, dim=1, unbiased=False).numpy(),
        expected,
    )


def test_nanvar_random_data_comparison() -> None:
    """Test nanvar with random data against numpy implementation."""
    x = torch.randn((73, 83), dtype=torch.float64)
    np.testing.assert_allclose(
        QF.nanvar(x, dim=1, unbiased=False).numpy(),
        np.nanvar(x, axis=1, ddof=0),
    )
    np.testing.assert_allclose(
        QF.nanvar(x, dim=1, unbiased=True).numpy(),
        np.nanvar(x, axis=1, ddof=1),
    )


def test_nanvar_basic_functionality() -> None:
    """Test basic NaN-aware variance functionality."""
    # Simple case without NaN
    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float64
    )

    # Unbiased variance (default)
    result_unbiased = QF.nanvar(x, dim=1, unbiased=True)
    expected_unbiased = torch.tensor(
        [5.0 / 3.0, 5.0 / 3.0], dtype=torch.float64
    )
    torch.testing.assert_close(result_unbiased, expected_unbiased)

    # Biased variance
    result_biased = QF.nanvar(x, dim=1, unbiased=False)
    expected_biased = torch.tensor([1.25, 1.25], dtype=torch.float64)
    torch.testing.assert_close(result_biased, expected_biased)


def test_nanvar_shape_preservation() -> None:
    """Test that nanvar preserves correct output shapes."""
    # 2D tensor
    x_2d = torch.randn(3, 5)
    result_2d = QF.nanvar(x_2d, dim=1)
    assert result_2d.shape == (3,)

    # 3D tensor
    x_3d = torch.randn(2, 4, 6)
    result_3d_dim1 = QF.nanvar(x_3d, dim=1)
    assert result_3d_dim1.shape == (2, 6)

    result_3d_dim2 = QF.nanvar(x_3d, dim=2)
    assert result_3d_dim2.shape == (2, 4)


def test_nanvar_keepdim_parameter() -> None:
    """Test nanvar with keepdim parameter."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)

    # keepdim=False (default)
    result_no_keepdim = QF.nanvar(x, dim=1, keepdim=False)
    assert result_no_keepdim.shape == (2,)

    # keepdim=True
    result_keepdim = QF.nanvar(x, dim=1, keepdim=True)
    assert result_keepdim.shape == (2, 1)

    # Values should be the same
    torch.testing.assert_close(result_no_keepdim, result_keepdim.squeeze())


def test_nanvar_nan_handling() -> None:
    """Test that nanvar correctly ignores NaN values."""
    # Mixed NaN and regular values
    x = torch.tensor(
        [
            [1.0, math.nan, 3.0, 5.0],
            [math.nan, 4.0, math.nan, 6.0],
            [2.0, 3.0, 4.0, math.nan],
        ],
        dtype=torch.float64,
    )

    result = QF.nanvar(x, dim=1, unbiased=False)

    # Manual calculation for first row: [1, 3, 5] -> mean=3, var=((1-3)²+(3-3)²+(5-3)²)/3 = 8/3
    expected_0 = ((1 - 3) ** 2 + (3 - 3) ** 2 + (5 - 3) ** 2) / 3
    # Manual calculation for second row: [4, 6] -> mean=5, var=((4-5)²+(6-5)²)/2 = 1
    expected_1 = ((4 - 5) ** 2 + (6 - 5) ** 2) / 2
    # Manual calculation for third row: [2, 3, 4] -> mean=3, var=((2-3)²+(3-3)²+(4-3)²)/3 = 2/3
    expected_2 = ((2 - 3) ** 2 + (3 - 3) ** 2 + (4 - 3) ** 2) / 3

    expected = torch.tensor(
        [expected_0, expected_1, expected_2], dtype=torch.float64
    )
    torch.testing.assert_close(result, expected)


def test_nanvar_all_nan() -> None:
    """Test nanvar behavior when all values are NaN."""
    x = torch.tensor(
        [[math.nan, math.nan, math.nan], [1.0, 2.0, 3.0]],
        dtype=torch.float64,
    )

    result = QF.nanvar(x, dim=1, unbiased=False)

    # All NaN row should return NaN
    assert torch.isnan(result[0])
    # Regular row should work normally
    expected_1 = ((1 - 2) ** 2 + (2 - 2) ** 2 + (3 - 2) ** 2) / 3  # = 2/3
    torch.testing.assert_close(
        result[1], torch.tensor(expected_1, dtype=torch.float64)
    )


def test_nanvar_single_value() -> None:
    """Test nanvar with single values."""
    # Single value per row
    x = torch.tensor([[5.0], [3.0]], dtype=torch.float64)

    # Unbiased should return NaN (division by n-1=0)
    result_unbiased = QF.nanvar(x, dim=1, unbiased=True)
    assert torch.isnan(result_unbiased).all()

    # Biased should return 0 (no variation)
    result_biased = QF.nanvar(x, dim=1, unbiased=False)
    expected_biased = torch.zeros(2, dtype=torch.float64)
    torch.testing.assert_close(result_biased, expected_biased)


def test_nanvar_different_dimensions() -> None:
    """Test nanvar along different dimensions."""
    x = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        dtype=torch.float64,
    )

    # Along dimension 0
    result_dim0 = QF.nanvar(x, dim=0, unbiased=False)
    expected_dim0 = torch.tensor([[4.0, 4.0], [4.0, 4.0]], dtype=torch.float64)
    torch.testing.assert_close(result_dim0, expected_dim0)

    # Along dimension 1
    result_dim1 = QF.nanvar(x, dim=1, unbiased=False)
    expected_dim1 = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    torch.testing.assert_close(result_dim1, expected_dim1)

    # Along dimension 2
    result_dim2 = QF.nanvar(x, dim=2, unbiased=False)
    expected_dim2 = torch.tensor(
        [[0.25, 0.25], [0.25, 0.25]], dtype=torch.float64
    )
    torch.testing.assert_close(result_dim2, expected_dim2)


def test_nanvar_multiple_dimensions() -> None:
    """Test nanvar along multiple dimensions."""
    x = torch.randn(3, 4, 5, dtype=torch.float64)

    # Along multiple dimensions
    result_multi = QF.nanvar(x, dim=(1, 2), unbiased=False)
    assert result_multi.shape == (3,)

    # Should be equivalent to flattening those dimensions
    x_flat = x.reshape(3, -1)
    result_flat = QF.nanvar(x_flat, dim=1, unbiased=False)
    torch.testing.assert_close(result_multi, result_flat)


def test_nanvar_unbiased_vs_biased() -> None:
    """Test difference between biased and unbiased variance."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float64)

    result_biased = QF.nanvar(x, dim=1, unbiased=False)
    result_unbiased = QF.nanvar(x, dim=1, unbiased=True)

    # Unbiased should be larger than biased (n/(n-1) factor)
    assert result_unbiased > result_biased

    # Relationship: unbiased = biased * n/(n-1)
    n = 5
    expected_ratio = n / (n - 1)
    actual_ratio = result_unbiased / result_biased
    torch.testing.assert_close(
        actual_ratio, torch.tensor([expected_ratio], dtype=torch.float64)
    )


def test_nanvar_constant_values() -> None:
    """Test nanvar with constant values."""
    x = torch.full((2, 5), 3.0, dtype=torch.float64)

    result_biased = QF.nanvar(x, dim=1, unbiased=False)
    result_unbiased = QF.nanvar(x, dim=1, unbiased=True)

    # Variance of constant values should be zero
    expected = torch.zeros(2, dtype=torch.float64)
    torch.testing.assert_close(result_biased, expected)
    torch.testing.assert_close(result_unbiased, expected)


def test_nanvar_mathematical_properties() -> None:
    """Test mathematical properties of variance."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)

    result = QF.nanvar(x, dim=1, unbiased=False)

    # Variance should be non-negative
    assert torch.all(result >= 0)

    # Manual calculation: mean = 2.5, var = ((1-2.5)² + (2-2.5)² + (3-2.5)² + (4-2.5)²)/4 = 1.25
    expected = (
        (1 - 2.5) ** 2 + (2 - 2.5) ** 2 + (3 - 2.5) ** 2 + (4 - 2.5) ** 2
    ) / 4
    torch.testing.assert_close(
        result, torch.tensor([expected], dtype=torch.float64)
    )


def test_nanvar_edge_cases() -> None:
    """Test nanvar edge cases."""
    # Very small tensor
    x_small = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    result_small = QF.nanvar(x_small, dim=1, unbiased=False)
    expected_small = torch.tensor(
        [0.25], dtype=torch.float64
    )  # ((1-1.5)² + (2-1.5)²)/2 = 0.25
    torch.testing.assert_close(result_small, expected_small)


def test_nanvar_numerical_stability() -> None:
    """Test numerical stability with various scales."""
    # Very large values
    x_large = torch.tensor([[1e6, 2e6, 3e6]], dtype=torch.float64)
    result_large = QF.nanvar(x_large, dim=1, unbiased=False)
    assert torch.isfinite(result_large).all()

    # Very small values
    x_small = torch.tensor([[1e-6, 2e-6, 3e-6]], dtype=torch.float64)
    result_small = QF.nanvar(x_small, dim=1, unbiased=False)
    assert torch.isfinite(result_small).all()


def test_nanvar_reproducibility() -> None:
    """Test that nanvar produces consistent results."""
    x = torch.tensor([[1.0, math.nan, 3.0, 4.0]], dtype=torch.float64)

    result1 = QF.nanvar(x, dim=1, unbiased=True)
    result2 = QF.nanvar(x, dim=1, unbiased=True)

    torch.testing.assert_close(result1, result2, equal_nan=True)


def test_nanvar_comparison_with_numpy() -> None:
    """Test consistency with numpy.nanvar."""
    x_np = np.array(
        [
            [1.0, np.nan, 3.0, 4.0],
            [np.nan, 2.0, np.nan, 5.0],
            [2.0, 3.0, 4.0, np.nan],
        ]
    )
    x_torch = torch.tensor(x_np, dtype=torch.float64)

    # Compare biased variance
    result_biased = QF.nanvar(x_torch, dim=1, unbiased=False)
    numpy_biased = np.nanvar(x_np, axis=1, ddof=0)
    torch.testing.assert_close(
        result_biased,
        torch.tensor(numpy_biased, dtype=torch.float64),
    )

    # Compare unbiased variance
    result_unbiased = QF.nanvar(x_torch, dim=1, unbiased=True)
    numpy_unbiased = np.nanvar(x_np, axis=1, ddof=1)
    torch.testing.assert_close(
        result_unbiased,
        torch.tensor(numpy_unbiased, dtype=torch.float64),
        equal_nan=True,
    )


def test_nanvar_gradient_compatibility() -> None:
    """Test that nanvar works with gradient computation."""
    x = torch.tensor(
        [[1.0, 2.0, 3.0, math.nan]], dtype=torch.float64, requires_grad=True
    )
    result = QF.nanvar(x, dim=1, unbiased=False)

    # Should be able to compute gradients
    loss = result.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_nanvar_batch_processing() -> None:
    """Test nanvar with batch dimensions."""
    batch_size = 3
    x_batch = torch.randn(batch_size, 20, dtype=torch.float64)
    # Add some NaN values
    x_batch[x_batch < -1.5] = math.nan

    result_batch = QF.nanvar(x_batch, dim=1)
    assert result_batch.shape == (batch_size,)

    # Compare with individual processing
    for i in range(batch_size):
        result_individual = QF.nanvar(x_batch[i : i + 1], dim=1)
        torch.testing.assert_close(result_batch[i : i + 1], result_individual)


def test_nanvar_special_values() -> None:
    """Test nanvar with special float values."""
    # Test with zeros
    x_zeros = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    result_zeros = QF.nanvar(x_zeros, dim=1, unbiased=False)
    expected_zeros = torch.zeros(1, dtype=torch.float64)
    torch.testing.assert_close(result_zeros, expected_zeros)

    # Test with mixed signs
    x_mixed = torch.tensor([[-2.0, 0.0, 2.0]], dtype=torch.float64)
    result_mixed = QF.nanvar(x_mixed, dim=1, unbiased=False)
    expected_mixed = torch.tensor(
        [8.0 / 3.0], dtype=torch.float64
    )  # (((-2-0)² + (0-0)² + (2-0)²)/3 = 8/3
    torch.testing.assert_close(result_mixed, expected_mixed)


def test_nanvar_with_infinity() -> None:
    """Test nanvar with infinite values."""
    x = torch.tensor([[1.0, math.inf, 2.0]], dtype=torch.float64)
    result = QF.nanvar(x, dim=1, unbiased=False)

    # Variance involving infinity should be infinity or NaN
    assert not torch.isfinite(result[0])


def test_nanvar_performance() -> None:
    """Test nanvar performance with larger tensors."""
    x_large = torch.randn(500, 200, dtype=torch.float64)
    # Add some NaN values
    x_large[torch.rand_like(x_large) < 0.05] = math.nan

    result = QF.nanvar(x_large, dim=1)
    assert result.shape == (500,)
    assert torch.isfinite(result[~torch.isnan(result)]).all()


def test_nanvar_boundary_conditions() -> None:
    """Test nanvar with boundary conditions."""
    # Two values (minimum for unbiased variance)
    x_two = torch.tensor([[1.0, 3.0]], dtype=torch.float64)
    result_two_biased = QF.nanvar(x_two, dim=1, unbiased=False)
    result_two_unbiased = QF.nanvar(x_two, dim=1, unbiased=True)

    expected_biased = torch.tensor(
        [1.0], dtype=torch.float64
    )  # ((1-2)² + (3-2)²)/2 = 1
    expected_unbiased = torch.tensor(
        [2.0], dtype=torch.float64
    )  # ((1-2)² + (3-2)²)/(2-1) = 2

    torch.testing.assert_close(result_two_biased, expected_biased)
    torch.testing.assert_close(result_two_unbiased, expected_unbiased)
