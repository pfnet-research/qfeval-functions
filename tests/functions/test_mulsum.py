import numpy as np

import qfeval_functions.functions as QF


def test_mulsum() -> None:
    a = QF.randn(100, 1).double()
    b = QF.randn(1, 200).double()
    np.testing.assert_allclose(
        (a * b).sum().numpy(),
        QF.mulsum(a, b).numpy(),
        1e-5,
        1e-5,
    )
    np.testing.assert_allclose(
        (a * b).sum(dim=1).numpy(),
        QF.mulsum(a, b, dim=1).numpy(),
        1e-5,
        1e-5,
    )
    np.testing.assert_allclose(
        (a * b).sum(dim=-1, keepdim=True).numpy(),
        QF.mulsum(a, b, dim=-1, keepdim=True).numpy(),
        1e-5,
        1e-5,
    )
