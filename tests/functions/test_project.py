import numpy as np
import torch

import qfeval_functions.functions as QF


def test_project() -> None:
    a = torch.tensor([[1, 1, 0], [1, 0, 1]])
    x = torch.tensor([[1, 10, 200], [2, 30, 100], [3, 20, 300]])
    np.testing.assert_allclose(
        QF.project(a, x), np.array([[11, 201], [32, 102], [23, 303]])
    )
