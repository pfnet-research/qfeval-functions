from math import nan

import numpy as np
import scipy.stats
import torch

import qfeval_functions.functions as QF


def test_nankurtosis() -> None:
    """Test nankurtosis with sufficient data points after NaN omission."""
    # Create test data with enough non-NaN values per row for reliable kurtosis computation
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, nan, nan],  # 8 values
            [nan, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, nan],  # 8 values
            [
                2.0,
                4.0,
                6.0,
                8.0,
                10.0,
                12.0,
                14.0,
                16.0,
                18.0,
                20.0,
            ],  # 10 values
            [1.1, 2.2, nan, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],  # 9 values
        ],
        dtype=torch.float64,
    )

    expected = scipy.stats.kurtosis(x, axis=1, bias=True, nan_policy="omit")
    result = QF.nankurtosis(x, dim=1, unbiased=False).numpy()

    np.testing.assert_allclose(result, expected)


def test_nankurtosis_edge_cases() -> None:
    """Test nankurtosis edge cases with small samples."""
    # Test cases with known behaviors - test our function's specific responses

    # Single non-NaN value should return NaN (insufficient data for kurtosis)
    x1 = torch.tensor([1.0, nan, nan, nan], dtype=torch.float64)
    result1 = QF.nankurtosis(x1, dim=0, unbiased=False)
    assert torch.isnan(
        result1
    ), f"Single value should return NaN, got {result1}"

    # All NaN values should return NaN
    x2 = torch.tensor([nan, nan, nan, nan], dtype=torch.float64)
    result2 = QF.nankurtosis(x2, dim=0, unbiased=False)
    assert torch.isnan(result2), f"All NaN should return NaN, got {result2}"

    # Two values: should return -2.0 (known mathematical result for 2-point kurtosis)
    x3 = torch.tensor([1.0, 2.0, nan, nan], dtype=torch.float64)
    result3 = QF.nankurtosis(x3, dim=0, unbiased=False)
    expected3 = -2.0  # Known mathematical result for 2 equal-weighted points
    np.testing.assert_allclose(result3.numpy(), expected3, rtol=1e-10, atol=1e-10)

    # Three values: test a known case
    x4 = torch.tensor([1.0, 2.0, 3.0, nan], dtype=torch.float64)
    result4 = QF.nankurtosis(x4, dim=0, unbiased=False)
    expected4 = -1.5  # Known mathematical result for 3 equally-spaced points
    np.testing.assert_allclose(result4.numpy(), expected4, rtol=1e-10, atol=1e-10)


def test_nankurtosis_with_randn() -> None:
    x = QF.randn(73, 83, dtype=torch.float64)
    np.testing.assert_allclose(
        QF.nankurtosis(x, dim=1, unbiased=False).numpy(),
        scipy.stats.kurtosis(x, axis=1, bias=True),
    )
    np.testing.assert_allclose(
        QF.nankurtosis(x, dim=1, unbiased=True).numpy(),
        scipy.stats.kurtosis(x, axis=1, bias=False),
    )
