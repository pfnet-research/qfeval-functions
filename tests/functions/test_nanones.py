import math

import numpy as np
import torch

import qfeval_functions.functions as QF
import pytest


def test_nanones_basic() -> None:
    """Test basic nanones functionality with tensor containing no NaN values."""
    x = torch.tensor([1.0, 2.0, 3.0])
    result = QF.nanones(x)
    expected = torch.tensor([1.0, 1.0, 1.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nanones_with_nan() -> None:
    """Test nanones with tensor containing NaN values - should preserve NaN positions."""
    x = torch.tensor([1.0, math.nan, 3.0])
    result = QF.nanones(x)
    expected = torch.tensor([1.0, math.nan, 1.0])

    assert result[0] == expected[0]
    assert torch.isnan(result[1]) and torch.isnan(expected[1])
    assert result[2] == expected[2]


def test_nanones_all_nan() -> None:
    """Test nanones with tensor containing only NaN values."""
    x = torch.tensor([math.nan, math.nan, math.nan])
    result = QF.nanones(x)

    assert torch.isnan(result).all()


def test_nanones_mixed_values() -> None:
    """Test nanones with mixed regular values and NaN values."""
    x = torch.tensor([0.0, math.nan, -5.0, 10.0, math.nan])
    result = QF.nanones(x)
    expected = torch.tensor([1.0, math.nan, 1.0, 1.0, math.nan])

    assert result[0] == expected[0]
    assert torch.isnan(result[1]) and torch.isnan(expected[1])
    assert result[2] == expected[2]
    assert result[3] == expected[3]
    assert torch.isnan(result[4]) and torch.isnan(expected[4])


def test_nanones_2d_tensor() -> None:
    """Test nanones with 2D tensor containing NaN in various positions."""
    x = torch.tensor([[1.0, math.nan], [math.nan, 2.0]])
    result = QF.nanones(x)
    expected = torch.tensor([[1.0, math.nan], [math.nan, 1.0]])

    assert result[0, 0] == expected[0, 0]
    assert torch.isnan(result[0, 1]) and torch.isnan(expected[0, 1])
    assert torch.isnan(result[1, 0]) and torch.isnan(expected[1, 0])
    assert result[1, 1] == expected[1, 1]


def test_nanones_3d_tensor() -> None:
    """Test nanones with 3D tensor to verify multidimensional handling."""
    x = torch.tensor([[[math.nan, 1.0], [2.0, math.nan]]])
    result = QF.nanones(x)

    assert torch.isnan(result[0, 0, 0])
    assert result[0, 0, 1] == 1.0
    assert result[0, 1, 0] == 1.0
    assert torch.isnan(result[0, 1, 1])


def test_nanones_single_element_value() -> None:
    """Test nanones with single-element tensor containing regular value."""
    x = torch.tensor([42.0])
    result = QF.nanones(x)
    expected = torch.tensor([1.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_nanones_with_infinity() -> None:
    """Test nanones with infinity values - should treat them as regular values, not NaN."""
    x = torch.tensor([math.inf, -math.inf, math.nan, 1.0])
    result = QF.nanones(x)
    expected = torch.tensor([1.0, 1.0, math.nan, 1.0])

    assert result[0] == expected[0]
    assert result[1] == expected[1]
    assert torch.isnan(result[2]) and torch.isnan(expected[2])
    assert result[3] == expected[3]


def test_nanones_large_tensor() -> None:
    """Test nanones with large tensor containing scattered NaN values."""
    x = torch.randn(100, 50)
    x[::10, ::10] = math.nan
    result = QF.nanones(x)

    for i in range(0, 100, 10):
        for j in range(0, 50, 10):
            assert torch.isnan(result[i, j])

    for i in range(100):
        for j in range(50):
            if i % 10 != 0 or j % 10 != 0:
                assert result[i, j] == 1.0


def test_nanones_complex_pattern() -> None:
    """Test nanones with complex NaN pattern."""
    x = torch.tensor(
        [[1.0, math.nan, 3.0], [math.nan, 5.0, math.nan], [7.0, math.nan, 9.0]]
    )
    result = QF.nanones(x)
    expected = torch.tensor(
        [[1.0, math.nan, 1.0], [math.nan, 1.0, math.nan], [1.0, math.nan, 1.0]]
    )

    assert result[0, 0] == expected[0, 0]
    assert torch.isnan(result[0, 1]) and torch.isnan(expected[0, 1])
    assert result[0, 2] == expected[0, 2]
    assert torch.isnan(result[1, 0]) and torch.isnan(expected[1, 0])
    assert result[1, 1] == expected[1, 1]
    assert torch.isnan(result[1, 2]) and torch.isnan(expected[1, 2])
    assert result[2, 0] == expected[2, 0]
    assert torch.isnan(result[2, 1]) and torch.isnan(expected[2, 1])
    assert result[2, 2] == expected[2, 2]


def test_nanones_extreme_values() -> None:
    """Test nanones with extreme finite values."""
    x = torch.tensor([1e20, -1e20, 1e-20, -1e-20, math.nan])
    result = QF.nanones(x)

    assert result[0] == 1.0
    assert result[1] == 1.0
    assert result[2] == 1.0
    assert result[3] == 1.0
    assert torch.isnan(result[4])


def test_nanones_edge_case_all_finite() -> None:
    """Test nanones with tensor containing no NaN values."""
    x = torch.tensor([[-5.0, 0.0, 10.0], [math.inf, -math.inf, 1e-10]])
    result = QF.nanones(x)
    expected = torch.ones_like(x)

    np.testing.assert_allclose(result.numpy(), expected.numpy())
