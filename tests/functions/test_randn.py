import hashlib

import numpy as np

import qfeval_functions
import qfeval_functions.functions as QF


def test_randn() -> None:
    with qfeval_functions.random.seed(1):
        v1a = QF.randn(3, 4, 5)
    with qfeval_functions.random.seed(1):
        v1b = QF.randn(3, 4, 5)
    with qfeval_functions.random.seed(2):
        v2 = QF.randn(3, 4, 5)
    np.testing.assert_array_equal(v1a.numpy(), v1b.numpy())
    assert np.all(np.not_equal(v1a.numpy(), v2.numpy()))
    m = hashlib.sha1()
    m.update(v1a.numpy().tobytes())
    assert m.hexdigest() == "e94e5a0bfab0c4fae459f1a2a4b6dea0171c54ea"
