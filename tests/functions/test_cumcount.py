import numpy as np
import torch

import qfeval_functions.functions as QF


def test_cumcount() -> None:
    a = torch.tensor([0, 1, 3, 2, 1, 0, 3, 1, 2])
    np.testing.assert_array_almost_equal(
        QF.cumcount(a).numpy(), np.array([0, 0, 0, 0, 1, 1, 1, 2, 1])
    )
    a = torch.tensor([[1, 1, 1, 0, 0, 0, 1, 1, 1]])
    np.testing.assert_array_almost_equal(
        QF.cumcount(a, dim=1).numpy(),
        np.array([[0, 1, 2, 0, 1, 2, 3, 4, 5]]),
    )
