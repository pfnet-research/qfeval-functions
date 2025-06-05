import typing

import torch

from qfeval_functions.random import is_fast
from qfeval_functions.random import rng


def randn(
    *size: int,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> torch.Tensor:
    r"""Returns a tensor filled with random numbers from a normal distribution
    with mean 0 and variance 1 (also called the standard normal distribution).
    If the seed is fixed, it must be reproducible in any device.
    """
    if is_fast():
        return torch.randn(*size, dtype=dtype or torch.float32, device=device)
    v = rng().normal(0, 1, size)
    return torch.tensor(v, dtype=dtype or torch.float32, device=device)
