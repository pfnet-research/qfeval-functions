import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_ffill_leading_nans() -> None:
    """Test forward fill with leading NaN values."""
    x = torch.tensor([math.nan, math.nan, 1.0, 2.0, math.nan, 3.0])
    result = QF.ffill(x, dim=0)
    expected = torch.tensor([math.nan, math.nan, 1.0, 2.0, 2.0, 3.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_ffill_trailing_nans() -> None:
    """Test forward fill with trailing NaN values."""
    x = torch.tensor([1.0, 2.0, math.nan, math.nan, math.nan])
    result = QF.ffill(x, dim=0)
    expected = torch.tensor([1.0, 2.0, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_ffill_alternating_pattern() -> None:
    """Test forward fill with alternating NaN pattern."""
    x = torch.tensor([math.nan, 1.0, math.nan, 2.0, math.nan, 3.0, math.nan])
    result = QF.ffill(x, dim=0)
    expected = torch.tensor([math.nan, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_ffill_stress_test_complex_pattern() -> None:
    """Test forward fill with complex NaN patterns across multiple dimensions."""
    x = torch.tensor(
        [
            [[math.nan, 1.0, math.nan], [2.0, math.nan, 3.0]],
            [[math.nan, math.nan, 4.0], [math.nan, 5.0, math.nan]],
            [[6.0, math.nan, math.nan], [7.0, 8.0, 9.0]],
        ]
    )

    # Test along each dimension
    for dim in range(3):
        result = QF.ffill(x, dim=dim)
        assert result.shape == x.shape
        assert result.dtype == x.dtype


def test_ffill_comparison_with_bfill() -> None:
    """Test that forward fill and backward fill give different results for same input."""
    x = torch.tensor([math.nan, 1.0, math.nan, 2.0, math.nan])

    ffill_result = QF.ffill(x, dim=0)
    bfill_result = QF.bfill(x, dim=0)

    # They should not be identical for this input
    assert not torch.equal(ffill_result, bfill_result)

    # But shapes should match
    assert ffill_result.shape == bfill_result.shape


def test_ffill_propagation_through_multiple_nans() -> None:
    """Test that forward fill properly propagates values through multiple consecutive NaNs."""
    x = torch.tensor(
        [1.0, math.nan, math.nan, math.nan, 2.0, math.nan, math.nan, 3.0]
    )
    result = QF.ffill(x, dim=0)
    expected = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())
