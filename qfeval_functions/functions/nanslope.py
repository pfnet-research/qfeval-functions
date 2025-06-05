import math
import typing

import torch

from .nanmean import nanmean
from .nanmulmean import nanmulmean
from .nansum import nansum


def nanslope(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Returns the slope of correlation bewteen `x` and `y`, ignoring NaNs.

    NOTE: This is based on calculation of slope beta at:
    https://en.wikipedia.org/wiki/Simple_linear_regression
    """
    isnan = x.isnan() | y.isnan()
    x = torch.where(isnan, torch.as_tensor(math.nan).to(x), x)
    y = torch.where(isnan, torch.as_tensor(math.nan).to(y), y)
    ax = x - nanmean(x, dim=dim, keepdim=True)
    ay = y - nanmean(y, dim=dim, keepdim=True)
    axy = nanmulmean(ax, ay, dim=dim, keepdim=True)
    ax2 = nanmean(ax**2, dim=dim, keepdim=True)
    return nansum(axy / ax2, dim=dim, keepdim=keepdim)
