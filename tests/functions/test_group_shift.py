from math import nan

import numpy as np
import torch

import qfeval_functions.functions as QF
from qfeval_functions.functions.group_shift import reduce_nan_patterns


def test_reduce_nan_patterns() -> None:
    # 2-dim input
    x = torch.tensor(
        [
            [1, nan, 2, nan, 3, 4],
            [6, nan, 7, nan, 8, 9],
            [10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, nan],
        ],
        dtype=torch.float,
    )
    np.testing.assert_allclose(
        reduce_nan_patterns(x, 0, 1).numpy(),
        np.array([False, False, True, False]),
    )
    np.testing.assert_allclose(
        # agg_f should not change the result for 2D inputs
        reduce_nan_patterns(x, 0, 1, agg_f="all").numpy(),
        np.array([False, False, True, False]),
    )
    np.testing.assert_allclose(
        reduce_nan_patterns(x, 1, 0).numpy(),
        np.array([True, False, True, False, True, False]),
    )

    # 3-dim input
    y = torch.tensor(
        [
            [[0, 1], [2, nan], [4, nan]],
            [[6, 7], [8, 9], [10, 11]],
            [[12, 13], [nan, nan], [16, 17]],
            [[18, 19], [20, 21], [nan, 23]],
        ],
        dtype=torch.float,
    )
    # If agg_f == "any", an element is evaluated to be True
    # when it contains at least one non-nan values,
    # e.g., [2, nan] is interpreted as True
    np.testing.assert_allclose(
        reduce_nan_patterns(y, 0, 1).numpy(),
        np.array([True, True, False, True]),
    )
    # If agg_f == "all", an element is evaluated to be True
    # only if it does not contain nans,
    # e.g., [2, nan] is interpreted as False
    np.testing.assert_allclose(
        reduce_nan_patterns(y, 0, 1, agg_f="all").numpy(),
        np.array([False, True, False, False]),
    )


def test_group_shift() -> None:
    x = torch.tensor(
        [
            [1, nan, 2, nan, 3, 4],
            [6, nan, 7, nan, 8, 9],
            [10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, nan],
        ],
        dtype=torch.float,
    )

    # Giving refdims
    # The mask should be [T, F, T, F, T, F]

    # even a "zero-shift" will not coincide with the original input
    # due to filling nans
    np.testing.assert_allclose(
        QF.group_shift(x, 0, 1, refdim=0).numpy(),
        np.array(
            [
                [1, nan, 2, nan, 3, nan],
                [6, nan, 7, nan, 8, nan],
                [10, nan, 12, nan, 14, nan],
                [16, nan, 18, nan, 20, nan],
            ],
        ),
    )

    np.testing.assert_allclose(
        QF.group_shift(x, 1, 1, refdim=0).numpy(),
        np.array(
            [
                [nan, nan, 1, nan, 2, nan],
                [nan, nan, 6, nan, 7, nan],
                [nan, nan, 10, nan, 12, nan],
                [nan, nan, 16, nan, 18, nan],
            ]
        ),
    )
    np.testing.assert_allclose(
        QF.group_shift(x, -1, 1, refdim=0).numpy(),
        np.array(
            [
                [2, nan, 3, nan, nan, nan],
                [7, nan, 8, nan, nan, nan],
                [12, nan, 14, nan, nan, nan],
                [18, nan, 20, nan, nan, nan],
            ]
        ),
    )
    np.testing.assert_allclose(
        QF.group_shift(x, -2, 1, refdim=0).numpy(),
        np.array(
            [
                [3, nan, nan, nan, nan, nan],
                [8, nan, nan, nan, nan, nan],
                [14, nan, nan, nan, nan, nan],
                [20, nan, nan, nan, nan, nan],
            ]
        ),
    )

    # when refdim is 1, mask should be [F, F, T, F]
    # so applying more than one shift will get a nan matrix
    np.testing.assert_allclose(
        QF.group_shift(x, 1, 0, refdim=1).numpy(),
        np.array(
            [
                [nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan],
            ]
        ),
    )

    # Giving mask
    mask = torch.tensor([True, False, True, False, True, True])

    np.testing.assert_allclose(
        QF.group_shift(x, -1, 1, mask=mask).numpy(),
        np.array(
            [
                [2, nan, 3, nan, 4, nan],
                [7, nan, 8, nan, 9, nan],
                [12, nan, 14, nan, 15, nan],
                [18, nan, 20, nan, nan, nan],
            ]
        ),
    )
