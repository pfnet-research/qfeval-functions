import typing

import torch

from qfeval_functions.random import is_fast
from qfeval_functions.random import rng


def rand(
    *size: int,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> torch.Tensor:
    r"""Returns a tensor filled with random numbers from a uniform distribution
    on the interval [0, 1).  If the seed is fixed, it must be reproducible in
    any device.
    """
    if is_fast():
        return torch.rand(*size, dtype=dtype or torch.float32, device=device)
    v = rng().random(size)
    return torch.tensor(v, dtype=dtype or torch.float32, device=device)
