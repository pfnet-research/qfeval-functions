import hashlib

import numpy as np

import qfeval_functions
import qfeval_functions.functions as QF


def test_randperm() -> None:
    RAND_SIZE = 97
    with qfeval_functions.random.seed(1):
        v1a = QF.randperm(RAND_SIZE)
    with qfeval_functions.random.seed(1):
        v1b = QF.randperm(RAND_SIZE)
    with qfeval_functions.random.seed(2):
        v2 = QF.randperm(RAND_SIZE)
    assert v1a.shape == (RAND_SIZE,)
    np.testing.assert_array_equal(
        v1a.sort().values,
        np.arange(RAND_SIZE),
    )
    np.testing.assert_array_equal(v1a.numpy(), v1b.numpy())
    assert np.mean(np.not_equal(v1a.numpy(), v2.numpy())) > 0.95
    m = hashlib.sha1()
    m.update(v1a.numpy().tobytes())
    assert m.hexdigest() == "d995edd95ed72518e52f0e6c49ef8e4ad37b1df7"
