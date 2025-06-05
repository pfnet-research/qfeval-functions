import numpy as np
import torch

import qfeval_functions
import qfeval_functions.functions as QF


def test_rand_like() -> None:
    with qfeval_functions.random.seed(1):
        v1a = QF.rand(3, 4, 5)
    with qfeval_functions.random.seed(1):
        v1b = QF.rand_like(torch.zeros(3, 4, 5))
    np.testing.assert_array_equal(v1a.numpy(), v1b.numpy())
