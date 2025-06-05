import math

import numpy as np
import pytest
import torch

import qfeval_functions.functions as QF


@pytest.mark.filterwarnings("ignore:Mean of")
def test_nanmulmean() -> None:
    a = QF.randn(100, 1)
    a = torch.where(a < 0, torch.as_tensor(math.nan), a)
    b = QF.randn(1, 200)
    b = torch.where(b < 0, torch.as_tensor(math.nan), b)
    np.testing.assert_allclose(
        np.nanmean((a * b).numpy()),
        QF.nanmulmean(a, b).numpy(),
        1e-5,
        1e-5,
    )
    np.testing.assert_allclose(
        np.nanmean((a * b).numpy(), axis=1),
        QF.nanmulmean(a, b, dim=1).numpy(),
        1e-5,
        1e-5,
    )
    np.testing.assert_array_equal(
        (a * b).isnan().all(dim=1).numpy(),
        QF.nanmulmean(a, b, dim=1).isnan().numpy(),
    )
    np.testing.assert_allclose(
        np.nanmean((a * b).numpy(), axis=-1, keepdims=True),
        QF.nanmulmean(a, b, dim=-1, keepdim=True).numpy(),
        1e-5,
        1e-5,
    )
