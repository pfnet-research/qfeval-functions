import numpy as np
import torch

import qfeval_functions.functions as QF


def manual_apply(f, x, dim):
    x_move = x.movedim(dim, -1)
    batch_shape = x_move.shape[:-1]
    flat = x_move.reshape(int(np.prod(batch_shape)), x.shape[dim])
    out = f(flat)
    return out.reshape(batch_shape + (x.shape[dim],)).movedim(-1, dim)


def test_apply_for_axis_basic() -> None:
    x = QF.randn(2, 3, 4)
    f = lambda t: t**2 + 1
    for dim in range(x.ndim):
        actual = QF.apply_for_axis(f, x, dim=dim)
        expected = manual_apply(f, x, dim)
        assert torch.allclose(actual, expected)


def test_apply_for_axis_zero_length() -> None:
    x = torch.empty(2, 0, 3)
    f = lambda t: t + 1
    actual = QF.apply_for_axis(f, x, dim=1)
    expected = manual_apply(f, x, dim=1)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected)
