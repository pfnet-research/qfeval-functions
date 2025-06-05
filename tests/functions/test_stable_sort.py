import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_stable_sort() -> None:
    a = (QF.rand(5, 5, 5) * 10.0 - 5).round()
    a = torch.where(torch.eq(a, 1.0), torch.as_tensor(math.inf), a)
    a = torch.where(torch.eq(a, 2.0), torch.as_tensor(math.nan), a)
    a = torch.where(torch.eq(a, 3.0), torch.as_tensor(-math.inf), a)
    for dim in range(3):
        np.testing.assert_array_almost_equal(
            QF.stable_sort(a, dim=dim).values.numpy(),
            np.sort(a.numpy(), axis=dim, kind="stable"),
        )
        np.testing.assert_array_almost_equal(
            QF.stable_sort(a, dim=dim).indices.numpy(),
            np.argsort(a.numpy(), axis=dim, kind="stable"),
        )


def test_stable_sort_with_ints() -> None:
    a = (QF.rand(5, 5, 5) * 10.0 - 5).round().int()
    for dim in range(3):
        np.testing.assert_array_almost_equal(
            QF.stable_sort(a, dim=dim).values.numpy(),
            np.sort(a.numpy(), axis=dim, kind="stable"),
        )
        np.testing.assert_array_almost_equal(
            QF.stable_sort(a, dim=dim).indices.numpy(),
            np.argsort(a.numpy(), axis=dim, kind="stable"),
        )
