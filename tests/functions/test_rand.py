import hashlib

import numpy as np

import qfeval_functions
import qfeval_functions.functions as QF


def test_rand() -> None:
    with qfeval_functions.random.seed(1):
        v1a = QF.rand(3, 4, 5)
    with qfeval_functions.random.seed(1):
        v1b = QF.rand(3, 4, 5)
    with qfeval_functions.random.seed(2):
        v2 = QF.rand(3, 4, 5)
    np.testing.assert_array_equal(v1a.numpy(), v1b.numpy())
    assert np.all(np.not_equal(v1a.numpy(), v2.numpy()))
    m = hashlib.sha1()
    m.update(v1a.numpy().tobytes())
    assert m.hexdigest() == "4fbb956f90936e3ce6ee85af4d6c18108b3242c4"
