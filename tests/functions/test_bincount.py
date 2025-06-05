import numpy as np
import pytest
import torch

import qfeval_functions.functions as QF


def test_bincount() -> None:
    np.testing.assert_allclose(
        QF.bincount(torch.tensor([[1, 2, 2], [3, 3, 1]])).numpy(),
        [[0, 1, 2, 0], [0, 1, 0, 2]],
    )
    np.testing.assert_allclose(
        QF.bincount(torch.tensor([1, 2, 2, 3, 3, 1])).numpy(), [0, 2, 2, 2]
    )


@pytest.mark.parametrize(
    "x",
    [
        QF.randint(low=1, high=100, size=(100,)),
        QF.randint(low=1, high=100, size=(0,)),  # edge case
    ],
)
def test_bincount_compared_with_torch(x: torch.Tensor) -> None:
    # expect the same result as torch.bincount when the input is a 1D tensor.
    for minlength in [0, 10, 100, 200]:
        assert torch.allclose(
            QF.bincount(x, dim=-1, minlength=minlength),
            torch.bincount(x, minlength=minlength),
        )
