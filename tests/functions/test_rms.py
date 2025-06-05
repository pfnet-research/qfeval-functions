import numpy as np
import torch

import qfeval_functions.functions as QF


def test_rms() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0], [0.1, -0.2, 0.3]])
    np.testing.assert_allclose(
        QF.rms(x, dim=1).numpy(),
        np.array([(14 / 3) ** 0.5, (14 / 3) ** 0.5 / 10]),
    )
