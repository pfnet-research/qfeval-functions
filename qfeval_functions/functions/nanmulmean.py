import typing

import torch

from .fillna import fillna
from .mulsum import mulsum


def nanmulmean(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    keepdim: bool = False,
    *,
    _ddof: int = 0,
) -> torch.Tensor:
    r"""Calculates `QF.nanmean(x * y, ...)` in a memory-efficient way.

    Parameters:
    - _ddof: Degree of freedom used for statistical metrics such as variance.
    """
    result = mulsum(fillna(x), fillna(y), dim=dim, keepdim=keepdim)
    x_mask = (~x.isnan()).to(result)
    y_mask = (~y.isnan()).to(result)
    return result / (mulsum(x_mask, y_mask, dim=dim, keepdim=keepdim) - _ddof)
