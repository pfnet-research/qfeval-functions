import math

import numpy as np
import torch

import qfeval_functions.functions as QF
import pytest


def test_nanmean_basic_2d() -> None:
    """Test basic nanmean functionality on 2D tensor."""
    x = torch.tensor(
        [
            [0.0, -1.0, 1.5, math.nan],
            [math.nan, math.nan, math.nan, math.nan],
            [0.0, -1.0, 2.0, -2.0],
        ]
    )
    result = QF.nanmean(x, dim=1)
    expected = torch.tensor([0.5 / 3, math.nan, -0.25])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nanmean_1d_tensor() -> None:
    """Test nanmean on 1D tensor with various NaN patterns."""
    # Mixed values with NaN
    x1 = torch.tensor([1.0, math.nan, 3.0, math.nan, 5.0])
    result1 = QF.nanmean(x1, dim=0)
    expected1 = (1.0 + 3.0 + 5.0) / 3.0
    np.testing.assert_allclose(result1.numpy(), expected1)

    # All NaN
    x2 = torch.tensor([math.nan, math.nan, math.nan])
    result2 = QF.nanmean(x2, dim=0)
    assert torch.isnan(result2)

    # No NaN
    x3 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result3 = QF.nanmean(x3, dim=0)
    expected3 = 2.5
    np.testing.assert_allclose(result3.numpy(), expected3)


def test_nanmean_3d_tensor() -> None:
    """Test nanmean on 3D tensor along different axes."""
    x = torch.tensor(
        [
            [[1.0, math.nan], [3.0, 4.0]],
            [[math.nan, 2.0], [math.nan, math.nan]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )

    # Along axis 0
    result_dim0 = QF.nanmean(x, dim=0)
    expected_dim0 = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    np.testing.assert_allclose(result_dim0.numpy(), expected_dim0.numpy())

    # Along axis 1
    result_dim1 = QF.nanmean(x, dim=1)
    expected_dim1 = torch.tensor([[2.0, 4.0], [math.nan, 2.0], [6.0, 7.0]])
    np.testing.assert_allclose(result_dim1.numpy(), expected_dim1.numpy())

    # Along axis 2
    result_dim2 = QF.nanmean(x, dim=2)
    expected_dim2 = torch.tensor([[1.0, 3.5], [2.0, math.nan], [5.5, 7.5]])
    np.testing.assert_allclose(result_dim2.numpy(), expected_dim2.numpy())


def test_nanmean_negative_dim() -> None:
    """Test nanmean with negative dimension indexing."""
    x = torch.tensor([[1.0, math.nan, 3.0], [4.0, 5.0, math.nan]])
    result = QF.nanmean(x, dim=-1)
    expected = torch.tensor([2.0, 4.5])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nanmean_with_infinity() -> None:
    """Test nanmean behavior with infinity values."""
    x = torch.tensor([1.0, math.inf, math.nan, 3.0, -math.inf])
    result = QF.nanmean(x, dim=0)
    # Should handle infinity appropriately
    assert torch.isnan(result) or torch.isinf(result)


def test_nanmean_empty_after_nan_removal() -> None:
    """Test nanmean when all values are NaN."""
    x = torch.tensor([[math.nan, math.nan], [math.nan, math.nan]])
    result = QF.nanmean(x, dim=1)
    assert torch.isnan(result).all()


def test_nanmean_large_tensor() -> None:
    """Test nanmean with large tensor."""
    size = 1000
    x = torch.randn(size)
    # Add some NaN values
    nan_indices = torch.randperm(size)[:100]
    x[nan_indices] = math.nan

    result = QF.nanmean(x, dim=0)

    # Result should not be NaN if there are valid values
    non_nan_count = (~torch.isnan(x)).sum()
    if non_nan_count > 0:
        assert not torch.isnan(result)


def test_nanmean_keepdim() -> None:
    """Test nanmean with keepdim parameter if supported."""
    x = torch.tensor([[1.0, math.nan, 3.0], [4.0, 5.0, math.nan]])

    # Test if keepdim is supported
    try:
        result = QF.nanmean(x, dim=1, keepdim=True)
        expected_shape = (2, 1)
        assert result.shape == expected_shape
    except TypeError:
        # If keepdim is not supported, that's fine
        pass


def test_nanmean_numerical_precision() -> None:
    """Test nanmean with values requiring high numerical precision."""
    x = torch.tensor(
        [1.0000001, math.nan, 1.0000002, 1.0000003], dtype=torch.float64
    )

    result = QF.nanmean(x, dim=0)
    expected = (1.0000001 + 1.0000002 + 1.0000003) / 3.0
    np.testing.assert_allclose(result.numpy(), expected)


def test_nanmean_very_small_values() -> None:
    """Test nanmean with very small values."""
    x = torch.tensor([1e-20, math.nan, 2e-20, 3e-20], dtype=torch.float64)
    result = QF.nanmean(x, dim=0)
    expected = (1e-20 + 2e-20 + 3e-20) / 3.0
    np.testing.assert_allclose(result.numpy(), expected)


def test_nanmean_very_large_values() -> None:
    """Test nanmean with very large values."""
    x = torch.tensor([1e20, math.nan, 2e20, 3e20], dtype=torch.float64)
    result = QF.nanmean(x, dim=0)
    expected = (1e20 + 2e20 + 3e20) / 3.0
    np.testing.assert_allclose(result.numpy(), expected)


def test_nanmean_edge_case_single_valid_value() -> None:
    """Test nanmean when only one valid value exists."""
    x = torch.tensor([math.nan, math.nan, 5.0, math.nan])
    result = QF.nanmean(x, dim=0)
    expected = 5.0
    np.testing.assert_allclose(result.numpy(), expected)


def test_nanmean_comparison_with_torch_nanmean() -> None:
    """Test nanmean against PyTorch's built-in nanmean if available."""
    x = torch.tensor(
        [
            [1.0, math.nan, 3.0, 4.0],
            [math.nan, 2.0, math.nan, 5.0],
            [6.0, 7.0, 8.0, math.nan],
        ]
    )

    result = QF.nanmean(x, dim=1)

    # Compare with torch.nanmean if available
    try:
        torch_result = torch.nanmean(x, dim=1)
        np.testing.assert_allclose(result.numpy(), torch_result.numpy())
    except AttributeError:
        # torch.nanmean might not be available in all versions
        pass
