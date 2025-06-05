import math
import typing

import torch

from .nanmean import nanmean
from .nanmulmean import nanmulmean
from .nansum import nansum


def nancorrel(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Returns Pearson correlation coefficient between `x` and `y`, ignoring
    NaNs.
    """
    isnan = x.isnan() | y.isnan()
    x = torch.where(isnan, torch.as_tensor(math.nan).to(x), x)
    y = torch.where(isnan, torch.as_tensor(math.nan).to(y), y)
    ax = x - nanmean(x, dim=dim, keepdim=True)
    ay = y - nanmean(y, dim=dim, keepdim=True)
    axy = nanmulmean(ax, ay, dim=dim, keepdim=True)
    ax2 = nanmean(ax**2, dim=dim, keepdim=True)
    ay2 = nanmean(ay**2, dim=dim, keepdim=True)
    result = axy / ax2.sqrt() / ay2.sqrt()
    return nansum(result, dim=dim, keepdim=keepdim)
