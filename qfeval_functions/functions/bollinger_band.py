import typing

import torch

from .ma import ma
from .mstd import mstd


def bollinger_band(
    x: torch.Tensor,
    window: int = 20,
    sigma: float = 2.0,
    dim: int = -1,
) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns the upper/middle/bottom values of the corresponding Bollinger band.

    Definition: https://www.investopedia.com/terms/b/bollingerbands.asp
    Like Data.bollinger_band
    """
    middle = ma(x, window, dim=dim)
    width = mstd(x, window, dim=dim, ddof=0) * sigma
    return middle + width, middle, middle - width
