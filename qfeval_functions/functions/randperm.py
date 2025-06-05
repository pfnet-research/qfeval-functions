import typing

import torch

from qfeval_functions.random import is_fast
from qfeval_functions.random import rng


def randperm(
    n: int,
    *,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> torch.Tensor:
    r"""Returns a random permutation of integers from `0` to `n - 1`."""
    if is_fast():
        return torch.randperm(n, dtype=dtype or torch.int64, device=device)
    v = rng().permutation(n)
    return torch.tensor(v, dtype=dtype or torch.int64, device=device)
