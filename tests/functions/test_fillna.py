import math

import numpy as np
import pytest
import torch

import qfeval_functions.functions as QF


def test_fillna_basic() -> None:
    """Test basic fillna functionality with default replacement value (0.0) for NaN."""
    x = torch.tensor([1.0, math.nan, 3.0, math.nan, 5.0])
    result = QF.fillna(x)
    expected = torch.tensor([1.0, 0.0, 3.0, 0.0, 5.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_fillna_custom_nan_value() -> None:
    """Test fillna with custom replacement value for NaN values."""
    x = torch.tensor([1.0, math.nan, 3.0, math.nan, 5.0])
    result = QF.fillna(x, nan=-999.0)
    expected = torch.tensor([1.0, -999.0, 3.0, -999.0, 5.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_fillna_positive_infinity() -> None:
    """Test fillna replacement of positive infinity values."""
    x = torch.tensor([1.0, math.inf, 3.0, math.inf, 5.0])
    result = QF.fillna(x, posinf=100.0)
    expected = torch.tensor([1.0, 100.0, 3.0, 100.0, 5.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_fillna_negative_infinity() -> None:
    """Test fillna replacement of negative infinity values."""
    x = torch.tensor([1.0, -math.inf, 3.0, -math.inf, 5.0])
    result = QF.fillna(x, neginf=-100.0)
    expected = torch.tensor([1.0, -100.0, 3.0, -100.0, 5.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_fillna_all_special_values() -> None:
    """Test fillna with all special values (NaN, +inf, -inf) replaced simultaneously."""
    x = torch.tensor([math.nan, math.inf, -math.inf, 1.0])
    result = QF.fillna(x, nan=0.0, posinf=99.0, neginf=-99.0)
    expected = torch.tensor([0.0, 99.0, -99.0, 1.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_fillna_preserve_infinity_default() -> None:
    """Test that fillna preserves infinity values by default (only replaces NaN)."""
    x = torch.tensor([1.0, math.inf, -math.inf, math.nan])
    result = QF.fillna(x)
    expected = torch.tensor([1.0, math.inf, -math.inf, 0.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_fillna_2d_tensor() -> None:
    """Test fillna with 2D tensor containing various special values."""
    x = torch.tensor([[1.0, math.nan], [math.inf, -math.inf]])
    result = QF.fillna(x, nan=0.0, posinf=10.0, neginf=-10.0)
    expected = torch.tensor([[1.0, 0.0], [10.0, -10.0]])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_fillna_no_special_values() -> None:
    """Test fillna with tensor containing no special values - should remain unchanged."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = QF.fillna(x, nan=-1.0, posinf=-2.0, neginf=-3.0)
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_fillna_3d_tensor() -> None:
    """Test fillna with 3D tensor containing various special values."""
    x = torch.tensor(
        [
            [[1.0, math.nan], [math.inf, 2.0]],
            [[math.nan, -math.inf], [3.0, 4.0]],
        ]
    )
    result = QF.fillna(x, nan=-1.0, posinf=999.0, neginf=-999.0)
    expected = torch.tensor(
        [[[1.0, -1.0], [999.0, 2.0]], [[-1.0, -999.0], [3.0, 4.0]]]
    )
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_fillna_mixed_nan_and_infinity() -> None:
    """Test fillna with tensor containing both NaN and infinity in same positions."""
    x = torch.tensor([math.nan, math.inf, -math.inf, math.nan, 1.0])
    result = QF.fillna(x, nan=0.0, posinf=100.0, neginf=-100.0)
    expected = torch.tensor([0.0, 100.0, -100.0, 0.0, 1.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_fillna_large_values() -> None:
    """Test fillna with very large replacement values."""
    x = torch.tensor([math.nan, math.inf, -math.inf])
    result = QF.fillna(x, nan=1e10, posinf=1e20, neginf=-1e20)
    expected = torch.tensor([1e10, 1e20, -1e20])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_fillna_zero_replacement() -> None:
    """Test fillna with zero as replacement value for all special values."""
    x = torch.tensor([math.nan, math.inf, -math.inf, 5.0])
    result = QF.fillna(x, nan=0.0, posinf=0.0, neginf=0.0)
    expected = torch.tensor([0.0, 0.0, 0.0, 5.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_fillna_negative_replacements() -> None:
    """Test fillna with negative replacement values."""
    x = torch.tensor([math.nan, math.inf, -math.inf, 1.0])
    result = QF.fillna(x, nan=-10.0, posinf=-20.0, neginf=-30.0)
    expected = torch.tensor([-10.0, -20.0, -30.0, 1.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_fillna_integer_tensor() -> None:
    """Test fillna behavior when called on integer tensor with special float values."""
    # Note: This tests the edge case of what happens with integer tensors
    x = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    result = QF.fillna(x, nan=-1, posinf=999, neginf=-999)
    expected = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    np.testing.assert_array_equal(result.numpy(), expected.numpy())
    assert result.dtype == torch.int32


def test_fillna_very_small_values() -> None:
    """Test fillna with very small replacement values."""
    x = torch.tensor([math.nan, math.inf, -math.inf])
    result = QF.fillna(x, nan=1e-20, posinf=1e-15, neginf=-1e-15)
    expected = torch.tensor([1e-20, 1e-15, -1e-15])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_fillna_batch_processing() -> None:
    """Test fillna with multiple tensors to verify consistent behavior."""
    tensors = [
        torch.tensor([math.nan, 1.0, math.inf]),
        torch.tensor([[math.nan, 2.0], [-math.inf, 3.0]]),
        torch.tensor([[[math.inf, math.nan]]]),
    ]

    for i, x in enumerate(tensors):
        result = QF.fillna(x, nan=0.0, posinf=100.0, neginf=-100.0)

        # Verify no NaN or infinity remains
        assert not torch.any(torch.isnan(result)), f"NaN found in tensor {i}"
        assert not torch.any(
            torch.isinf(result)
        ), f"Infinity found in tensor {i}"

        # Verify shape preservation
        assert result.shape == x.shape, f"Shape mismatch in tensor {i}"


def test_fillna_edge_case_all_special_values() -> None:
    """Test fillna with tensor containing only special values."""
    x = torch.tensor([math.nan, math.inf, -math.inf, math.nan, math.inf])
    result = QF.fillna(x, nan=1.0, posinf=2.0, neginf=3.0)
    expected = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_fillna_partial_replacement() -> None:
    """Test fillna with only some parameters specified (using defaults for others)."""
    x = torch.tensor([math.nan, math.inf, -math.inf, 1.0])

    # Only replace NaN
    result1 = QF.fillna(x, nan=42.0)
    assert result1[0] == 42.0
    assert torch.isinf(result1[1]) and result1[1] > 0
    assert torch.isinf(result1[2]) and result1[2] < 0
    assert result1[3] == 1.0


def test_fillna_same_replacement_values() -> None:
    """Test fillna when all replacement values are the same."""
    x = torch.tensor([math.nan, math.inf, -math.inf, 5.0])
    result = QF.fillna(x, nan=99.0, posinf=99.0, neginf=99.0)
    expected = torch.tensor([99.0, 99.0, 99.0, 5.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_fillna_replacement_with_infinity() -> None:
    """Test fillna using infinity as replacement value."""
    x = torch.tensor([math.nan, 1.0, 2.0])
    result = QF.fillna(x, nan=math.inf)
    assert torch.isinf(result[0]) and result[0] > 0
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_fillna_replacement_with_nan() -> None:
    """Test fillna using NaN as replacement for infinity (edge case)."""
    x = torch.tensor([math.inf, -math.inf, 1.0])
    result = QF.fillna(x, posinf=math.nan, neginf=math.nan)

    assert torch.isnan(result[0])
    assert torch.isnan(result[1])
    assert result[2] == 1.0


@pytest.mark.random
def test_fillna_stress_test_large_tensor() -> None:
    """Test fillna with large tensor containing scattered special values."""
    size = 1000
    x = torch.randn(size)

    # Randomly place some special values
    special_indices = torch.randperm(size)[:100]
    x[special_indices[:33]] = math.nan
    x[special_indices[33:66]] = math.inf
    x[special_indices[66:]] = -math.inf

    result = QF.fillna(x, nan=0.0, posinf=1e6, neginf=-1e6)

    # Verify no special values remain
    assert not torch.any(torch.isnan(result))
    assert not torch.any(torch.isinf(result))

    # Verify replacements are correct
    assert torch.sum(result == 0.0) == 33  # NaN replacements
    assert torch.sum(result == 1e6) == 33  # +inf replacements
    assert torch.sum(result == -1e6) == 34  # -inf replacements
