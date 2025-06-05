import typing

import torch

from qfeval_functions.random import is_fast
from qfeval_functions.random import rng


def randint(
    low: int,
    high: int,
    size: typing.Tuple[int, ...],
    *,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> torch.Tensor:
    r"""Returns a tensor filled with random integers generated uniformly
    between low (inclusive) and high (exclusive).
    """
    if is_fast():
        return torch.randint(
            low, high, size, dtype=dtype or torch.int64, device=device
        )
    v = rng().integers(low, high, size)
    return torch.tensor(v, dtype=dtype or torch.int64, device=device)
