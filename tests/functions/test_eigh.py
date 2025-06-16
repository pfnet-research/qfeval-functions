import numpy as np
import torch

import qfeval_functions.functions as QF


def test_eigh_cpu() -> None:
    x = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    w, v = QF.eigh(x)
    wn, vn = np.linalg.eigh(x.numpy())
    np.testing.assert_allclose(w.numpy(), wn)
    np.testing.assert_allclose(v.numpy(), vn, atol=1e-6)
