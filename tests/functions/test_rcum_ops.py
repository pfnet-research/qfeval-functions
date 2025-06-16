import numpy as np
import torch

import qfeval_functions.functions as QF


def test_rcummax() -> None:
    x = torch.tensor([[1, 3, 2], [4, 1, 0]], dtype=torch.float32)
    result = QF.rcummax(x, dim=1)
    expected_values, expected_indices = torch.cummax(torch.flip(x, [1]), 1)
    expected_values = torch.flip(expected_values, [1])
    expected_indices = x.shape[1] - 1 - torch.flip(expected_indices, [1])
    assert torch.allclose(result.values, expected_values)
    assert torch.equal(result.indices, expected_indices)


def test_rcumsum() -> None:
    x = QF.randn(5, 4)
    actual = QF.rcumsum(x, dim=1)
    expected = torch.flip(torch.cumsum(torch.flip(x, [1]), 1), [1])
    assert torch.allclose(actual, expected)
