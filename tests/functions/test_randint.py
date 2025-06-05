import hashlib

import numpy as np

import qfeval_functions
import qfeval_functions.functions as QF


def test_randint() -> None:
    RAND_SHAPE = (5, 7, 11)
    with qfeval_functions.random.seed(1):
        v1a = QF.randint(0, 100, RAND_SHAPE)
    with qfeval_functions.random.seed(1):
        v1b = QF.randint(0, 100, RAND_SHAPE)
    with qfeval_functions.random.seed(2):
        v2 = QF.randint(0, 100, RAND_SHAPE)
    assert v1a.shape == RAND_SHAPE
    assert v1a.amax() == 99
    assert v1a.amin() == 0
    np.testing.assert_array_equal(v1a.numpy(), v1b.numpy())
    assert np.mean(np.not_equal(v1a.numpy(), v2.numpy())) > 0.95
    m = hashlib.sha1()
    m.update(v1a.numpy().tobytes())
    assert m.hexdigest() == "b8b48c4017604f3fc2afebad57ad27a88cf33b8a"
