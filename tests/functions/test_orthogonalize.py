import numpy as np
import torch

import qfeval_functions
import qfeval_functions.functions as QF


def test_orthogonalize() -> None:
    qfeval_functions.random.seed(1)
    for _ in range(10):
        x = QF.randn(10, 32)
        y = QF.randn(10, 32)
        orth_x = QF.orthogonalize(x, y)

        # Inner products between orthogonalized x and y should be zero (i.e.,
        # orthogonalized x and y should be orthogonal).
        orth_x_dot_y = (orth_x * y).sum(dim=-1)
        np.testing.assert_allclose(orth_x_dot_y, torch.zeros(10), atol=1e-5)

        # (x - orth_x) and y should be on the same axis.
        np.testing.assert_allclose(
            QF.correl(x - orth_x, y, dim=-1).abs(), torch.ones(10), atol=1e-5
        )
