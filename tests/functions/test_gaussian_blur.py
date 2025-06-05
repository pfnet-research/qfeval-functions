import numpy as np
import pytest
import torch

import qfeval_functions.functions as QF
from qfeval_functions.functions.gaussian_blur import _gaussian_filter


@pytest.mark.parametrize(
    "n",
    [101, 1000, 100001],
)
@pytest.mark.parametrize(
    "sigma",
    [0.01, 0.1, 1, 10],
)
def test_gaussian_filter(n: int, sigma: float) -> None:
    # Test that the weighting window sums to 1.
    a = _gaussian_filter(n, sigma)
    assert a.shape == (n,)
    assert a.sum() == pytest.approx(1.0)

    # Even a result with a narrow window width should contain the same values.
    b = _gaussian_filter(10 + n % 2, sigma)
    np.testing.assert_array_almost_equal(
        b,
        a[(a.size(0) - b.size(0)) // 2 :][: b.size(0)],
    )


def test_gaussian_blur() -> None:
    # Test a regular case.
    a = torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=torch.float32)
    np.testing.assert_array_almost_equal(
        QF.gaussian_blur(a, 1),
        [0.009, 0.065, 0.243, 0.383, 0.243, 0.065, 0.009],
        decimal=3,
    )

    # Values outside the bounds of a tensor should not not weighted, so
    # larger values than the previous test case should be returned.
    a = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32)
    np.testing.assert_array_almost_equal(
        QF.gaussian_blur(a, 1),
        [0.088, 0.259, 0.388, 0.259, 0.088],
        decimal=3,
    )

    # Even a small sigma should blur a little.
    a = torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=torch.float32)
    np.testing.assert_array_almost_equal(
        QF.gaussian_blur(a, 0.2),
        [0, 0, 0.0062, 0.9876, 0.0062, 0, 0],
        decimal=4,
    )

    # Test if gaussian_blur works even with an integer tensor.
    assert QF.gaussian_blur(
        torch.tensor([0, 0, 0, 10000, 0, 0, 0, 0, 0, 0, 0]), 3
    ).tolist() == [1425, 1537, 1575, 1517, 1364, 1138, 878, 629, 420, 264, 157]
