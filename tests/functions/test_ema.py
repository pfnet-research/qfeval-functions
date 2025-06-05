import numpy as np
import pandas as pd

import qfeval_functions.functions as QF


def test_ema() -> None:
    a = QF.randn(10, 100)
    df = pd.DataFrame(a.numpy())
    np.testing.assert_allclose(
        QF.ema(a, 0.1, dim=0).numpy(),
        df.ewm(alpha=0.1, axis=0).mean().to_numpy(),
        1e-6,
        1e-6,
    )
