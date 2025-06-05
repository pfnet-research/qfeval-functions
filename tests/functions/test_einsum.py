import numpy as np
import torch

import qfeval_functions.functions as QF


def test_einsum() -> None:
    r"""Test if einsum works.

    NOTE: Having a simple test pattern is enough because the internal logic
    is simple enough.
    """
    a = QF.randn(10, 1, 30)
    b = QF.randn(1, 20, 30)
    np.testing.assert_allclose(
        QF.einsum("abc,adc->ad", a, b),
        torch.einsum("abc,adc->ad", a, b),  # type: ignore
    )
