import math
import typing

import torch


def nanskew(
    x: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    unbiased: bool = True,
    *,
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Returns the skewness of each row, ignoring Not a Numbers (NaNs)."""
    n = (~x.isnan()).to(x).sum(dim=dim, keepdim=True)
    ddof = 1 if unbiased else 0
    x = torch.where(n <= ddof * 2, torch.as_tensor(math.nan).to(x), x)
    m1 = x.nansum(dim=dim, keepdim=True) / n
    m2 = ((x - m1) ** 2).nansum(dim=dim, keepdim=True)
    m3 = ((x - m1) ** 3).nansum(dim=dim, keepdim=True)
    r = (m3 / m2**1.5) * n * (n - ddof).sqrt() / (n - ddof * 2)
    r = r.sum(dim=dim, keepdim=keepdim)
    return typing.cast(torch.Tensor, r)
