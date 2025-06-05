import typing

import torch

from .randn import randn


def randn_like(
    input: torch.Tensor,
    *,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> torch.Tensor:
    r"""Returns a tensor with the same size as input that is filled with
    random numbers from a normal distribution with mean 0 and variance 1.
    If the seed is fixed, it must be reproducible in any device.
    """
    return randn(
        *input.shape, dtype=dtype or input.dtype, device=device or input.device
    )
