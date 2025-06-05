import math
import typing

import torch

from .fillna import fillna
from .mulsum import mulsum


def nanmulsum(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Calculates `QF.nansum(x * y, ...)` in a memory-efficient way."""
    result = mulsum(fillna(x), fillna(y), dim=dim, keepdim=keepdim)
    x_mask = (~x.isnan()).to(result)
    y_mask = (~y.isnan()).to(result)
    count = mulsum(x_mask, y_mask, dim=dim, keepdim=keepdim)
    return torch.where(count > 0, result, torch.as_tensor(math.nan).to(result))
