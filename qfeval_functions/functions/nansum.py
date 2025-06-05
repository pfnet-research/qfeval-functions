import math
import typing

import torch


def nansum(
    x: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Returns the sum of all elements, ignoring Not a Numbers (NaNs).

    NOTE: If no valid numbers exist, this returns NaN while torch.nansum
    returns 0.
    """
    is_valid = (~x.isnan()).sum(dim=dim, keepdim=keepdim) > 0
    y = x.nansum(dim=dim, keepdim=keepdim)
    return torch.where(is_valid, y, torch.as_tensor(math.nan).to(y))
