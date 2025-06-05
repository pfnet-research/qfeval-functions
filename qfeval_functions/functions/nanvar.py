import math
import typing

import torch

from .nanmean import nanmean
from .nansum import nansum


def nanvar(
    x: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    unbiased: bool = True,
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Returns the mean value of each row, ignoring Not a Numbers (NaNs)."""
    n = (~x.isnan()).to(x).sum(dim=dim, keepdim=True)
    if unbiased:
        n = n - 1
        # Prevent gradients from being divided by zeros.
        x = torch.where(n <= 0, torch.as_tensor(math.nan).to(x), x)
    x2 = (x - nanmean(x, dim=dim, keepdim=True)) ** 2
    r = nansum(x2, dim=dim, keepdim=True) / n
    return r.sum(dim=dim, keepdim=keepdim)
