import numpy as np

import qfeval_functions.functions as QF


def test_mulmean() -> None:
    a = QF.randn(100, 1)
    b = QF.randn(1, 200)
    np.testing.assert_allclose(
        (a * b).mean().numpy(),
        QF.mulmean(a, b).numpy(),
        1e-5,
        1e-5,
    )
    np.testing.assert_allclose(
        (a * b).mean(dim=1).numpy(),
        QF.mulmean(a, b, dim=1).numpy(),
        1e-5,
        1e-5,
    )
    np.testing.assert_allclose(
        (a * b).mean(dim=-1, keepdim=True).numpy(),
        QF.mulmean(a, b, dim=-1, keepdim=True).numpy(),
        1e-5,
        1e-5,
    )
