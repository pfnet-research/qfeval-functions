import numpy as np
import pandas as pd

import qfeval_functions.functions as QF


def test_mvar() -> None:
    a = QF.randn(100, 10)
    df = pd.DataFrame(a.numpy())
    np.testing.assert_allclose(
        QF.mvar(a, 10, dim=0).numpy(),
        df.rolling(10).var().to_numpy(),
        1e-6,
        1e-6,
    )
    np.testing.assert_allclose(
        QF.mvar(a, 10, dim=0, ddof=0).numpy(),
        df.rolling(10).var(ddof=0).to_numpy(),
        1e-6,
        1e-6,
    )
