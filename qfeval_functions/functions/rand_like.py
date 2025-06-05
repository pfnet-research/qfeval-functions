import typing

import torch

from .rand import rand


def rand_like(
    input: torch.Tensor,
    *,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> torch.Tensor:
    r"""Returns a tensor with the same size as input that is filled with random
    numbers from a uniform distribution on the interval [0, 1).  If the seed
    is fixed, it must be reproducible in any device.
    """
    return rand(
        *input.shape, dtype=dtype or input.dtype, device=device or input.device
    )
