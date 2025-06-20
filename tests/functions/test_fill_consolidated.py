"""Consolidated tests for fill functions (ffill, bfill, fillna)."""

import math
from typing import Callable

import numpy as np
import pytest
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


@pytest.mark.parametrize("fill_func", [QF.ffill, QF.bfill])
def test_fill_1d_basic(fill_func: Callable) -> None:
    """Test basic fill functionality on 1D tensor."""
    x = torch.tensor(
        [math.nan, 1.0, math.nan, 2.0, 3.0, math.nan, 4.0, math.nan]
    )
    result = fill_func(x, dim=0)

    # Common properties all fill functions should satisfy
    assert_basic_properties(result, x)

    # Non-NaN values should be preserved
    non_nan_mask = ~torch.isnan(x)
    assert torch.equal(result[non_nan_mask], x[non_nan_mask])


@pytest.mark.parametrize("fill_func", [QF.ffill, QF.bfill])
def test_fill_2d_tensor(fill_func: Callable) -> None:
    """Test fill on 2D tensor along different dimensions."""
    x = torch.tensor(
        [
            [math.nan, 1.0, 2.0],
            [3.0, math.nan, math.nan],
            [math.nan, 4.0, 5.0],
            [6.0, 7.0, math.nan],
        ]
    )

    # Test both dimensions
    for dim in [0, 1]:
        result = fill_func(x, dim=dim)
        assert_basic_properties(result, x)


@pytest.mark.parametrize("fill_func", [QF.ffill, QF.bfill])
def test_fill_no_nan(fill_func: Callable) -> None:
    """Test fill with tensor containing no NaN values."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = fill_func(x, dim=0)
    torch.testing.assert_close(result, x)


@pytest.mark.parametrize("fill_func", [QF.ffill, QF.bfill])
def test_fill_all_nan(fill_func: Callable) -> None:
    """Test fill with tensor containing all NaN values."""
    x = torch.tensor([math.nan, math.nan, math.nan])
    result = fill_func(x, dim=0)
    assert torch.isnan(result).all()


@pytest.mark.parametrize("fill_func", [QF.ffill, QF.bfill])
def test_fill_mixed_special_values(fill_func: Callable) -> None:
    """Test fill with mixed NaN and infinity values."""
    x = torch.tensor([math.nan, math.inf, math.nan, 1.0, math.nan, -math.inf])
    result = fill_func(x, dim=0)

    # Should preserve infinity values
    assert torch.isinf(result[1])
    assert torch.isinf(result[5])

    # Should handle NaN appropriately
    assert_basic_properties(result, x)


def test_fillna_basic() -> None:
    """Test basic fillna functionality."""
    x = torch.tensor([1.0, math.nan, 3.0, math.inf, -math.inf, math.nan])
    result = QF.fillna(x, nan=-1.0, posinf=-2.0, neginf=-3.0)
    expected = torch.tensor([1.0, -1.0, 3.0, -2.0, -3.0, -1.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_fillna_selective_replacement() -> None:
    """Test fillna with selective replacement."""
    x = torch.tensor([math.nan, math.inf, -math.inf, 1.0])

    # Only replace NaN - this is the default behavior
    result_nan_only = QF.fillna(x, nan=0.0)
    expected_nan_only = torch.tensor([0.0, math.inf, -math.inf, 1.0])
    np.testing.assert_allclose(
        result_nan_only.numpy(), expected_nan_only.numpy()
    )

    # Replace both NaN and positive infinity explicitly
    result_mixed = QF.fillna(x, nan=-1.0, posinf=999.0)
    expected_mixed = torch.tensor([-1.0, 999.0, -math.inf, 1.0])
    np.testing.assert_allclose(result_mixed.numpy(), expected_mixed.numpy())


def test_fillna_3d_tensor() -> None:
    """Test fillna with 3D tensor containing various special values."""
    x = torch.tensor(
        [
            [[1.0, math.nan], [math.inf, 2.0]],
            [[math.nan, -math.inf], [3.0, 4.0]],
        ]
    )
    result = QF.fillna(x, nan=-1.0, posinf=-2.0, neginf=-3.0)
    expected = torch.tensor(
        [
            [[1.0, -1.0], [-2.0, 2.0]],
            [[-1.0, -3.0], [3.0, 4.0]],
        ]
    )
    np.testing.assert_allclose(result.numpy(), expected.numpy())


@pytest.mark.parametrize("fill_func", [QF.ffill, QF.bfill])
def test_fill_edge_cases(fill_func: Callable) -> None:
    """Test fill functions with edge cases."""
    # Single element
    x_single = torch.tensor([math.nan])
    result_single = fill_func(x_single, dim=0)
    assert torch.isnan(result_single).all()

    # Empty tensor
    x_empty = torch.empty(0)
    result_empty = fill_func(x_empty, dim=0)
    assert result_empty.shape == (0,)


@pytest.mark.parametrize("fill_func", [QF.ffill, QF.bfill])
def test_fill_negative_dimensions(fill_func: Callable) -> None:
    """Test fill with negative dimension indices."""
    x = torch.tensor([[math.nan, 1.0], [2.0, math.nan]])

    result_neg1 = fill_func(x, dim=-1)
    result_pos1 = fill_func(x, dim=1)
    torch.testing.assert_close(result_neg1, result_pos1, equal_nan=True)


@pytest.mark.parametrize("fill_func", [QF.ffill, QF.bfill])
@pytest.mark.random
def test_fill_large_tensors(fill_func: Callable) -> None:
    """Test fill with larger tensors."""
    x = torch.randn(100, 50)
    # Add some NaN values
    x[torch.rand_like(x) < 0.1] = math.nan

    result = fill_func(x, dim=0)
    assert_basic_properties(result, x)
