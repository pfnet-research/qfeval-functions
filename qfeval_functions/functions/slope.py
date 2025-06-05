import typing

import torch

from .mulmean import mulmean


def slope(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Returns the slope of correlation bewteen `x` and `y`.

    NOTE: This is based on calculation of slope beta at:
    https://en.wikipedia.org/wiki/Simple_linear_regression
    """
    ax = x - x.mean(dim=dim, keepdim=True)
    ay = y - y.mean(dim=dim, keepdim=True)
    axy = mulmean(ax, ay, dim=dim, keepdim=True)
    ax2 = (ax**2).mean(dim=dim, keepdim=True)
    result: torch.Tensor = (axy / ax2).sum(dim=dim, keepdim=keepdim)
    return result
