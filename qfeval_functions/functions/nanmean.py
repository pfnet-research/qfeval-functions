import typing

import torch


def nanmean(
    x: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Returns the mean value of each row, ignoring Not a Numbers (NaNs)."""
    count = (~x.isnan()).to(x).sum(dim=dim, keepdim=keepdim)
    return x.nansum(dim=dim, keepdim=keepdim) / count
