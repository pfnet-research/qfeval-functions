import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_nanmulsum() -> None:
    a = QF.randn(100, 1)
    a = torch.where(a < 0, torch.as_tensor(math.nan), a)
    b = QF.randn(1, 200)
    b = torch.where(b < 0, torch.as_tensor(math.nan), b)
    np.testing.assert_allclose(
        (a * b).nansum().numpy(),
        QF.nanmulsum(a, b).numpy(),
        1e-5,
        1e-5,
    )
    np.testing.assert_allclose(
        (a * b).nansum(dim=1).numpy(),
        QF.nanmulsum(a, b, dim=1).nan_to_num().numpy(),
        1e-5,
        1e-5,
    )
    np.testing.assert_array_equal(
        (a * b).isnan().all(dim=1).numpy(),
        QF.nanmulsum(a, b, dim=1).isnan().numpy(),
    )
    np.testing.assert_allclose(
        (a * b).nansum(dim=-1, keepdim=True).numpy(),
        QF.nanmulsum(a, b, dim=-1, keepdim=True).nan_to_num().numpy(),
        1e-5,
        1e-5,
    )
