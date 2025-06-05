import typing

import torch

from .mulsum import mulsum


def mulmean(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    keepdim: bool = False,
    *,
    _ddof: int = 0,
) -> torch.Tensor:
    r"""Calculate `(x * y).mean(...)` in a memory-efficient way."""
    return mulsum(x, y, dim=dim, keepdim=keepdim, mean=True, _ddof=_ddof)
