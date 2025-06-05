import typing

import torch

from .mulmean import mulmean


def correl(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Returns Pearson correlation coefficient between `x` and `y`."""
    ax = x - x.mean(dim=dim, keepdim=True)
    ay = y - y.mean(dim=dim, keepdim=True)
    axy = mulmean(ax, ay, dim=dim, keepdim=True)
    ax2 = (ax**2).mean(dim=dim, keepdim=True)
    ay2 = (ay**2).mean(dim=dim, keepdim=True)
    result: torch.Tensor = axy / ax2.sqrt() / ay2.sqrt()
    result = result.sum(dim=dim, keepdim=keepdim)
    return result
