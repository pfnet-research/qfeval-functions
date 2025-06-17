import numpy as np
import torch

import qfeval_functions.functions as QF


def test_fillna_default() -> None:
    x = torch.tensor([float("nan"), float("inf"), float("-inf")])
    actual = QF.fillna(x)
    expected = torch.tensor([0.0, float("inf"), float("-inf")])
    assert torch.all(torch.eq(actual, expected))


def test_fillna_custom() -> None:
    x = torch.tensor([float("nan"), float("inf"), float("-inf")])
    actual = QF.fillna(x, nan=1.0, posinf=2.0, neginf=-2.0)
    expected = torch.tensor([1.0, 2.0, -2.0])
    assert torch.all(torch.eq(actual, expected))
