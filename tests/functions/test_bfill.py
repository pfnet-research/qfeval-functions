import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_bfill_leading_nans() -> None:
    """Test backward fill with leading NaN values."""
    x = torch.tensor([math.nan, math.nan, 1.0, 2.0, math.nan, 3.0])
    result = QF.bfill(x, dim=0)
    expected = torch.tensor([1.0, 1.0, 1.0, 2.0, 3.0, 3.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_bfill_trailing_nans() -> None:
    """Test backward fill with trailing NaN values."""
    x = torch.tensor([1.0, 2.0, math.nan, math.nan, math.nan])
    result = QF.bfill(x, dim=0)
    expected = torch.tensor([1.0, 2.0, math.nan, math.nan, math.nan])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_bfill_alternating_pattern() -> None:
    """Test backward fill with alternating NaN pattern."""
    x = torch.tensor([math.nan, 1.0, math.nan, 2.0, math.nan, 3.0, math.nan])
    result = QF.bfill(x, dim=0)
    expected = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, math.nan])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_bfill_stress_test_complex_pattern() -> None:
    """Test backward fill with complex NaN patterns across multiple dimensions."""
    x = torch.tensor(
        [
            [[math.nan, 1.0, math.nan], [2.0, math.nan, 3.0]],
            [[math.nan, math.nan, 4.0], [math.nan, 5.0, math.nan]],
            [[6.0, math.nan, math.nan], [7.0, 8.0, 9.0]],
        ]
    )

    # Test along each dimension
    for dim in range(3):
        result = QF.bfill(x, dim=dim)
        assert result.shape == x.shape
        assert result.dtype == x.dtype
