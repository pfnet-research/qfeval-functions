import math
import typing

import torch


def nankurtosis(
    x: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    unbiased: bool = True,
    *,
    fisher: bool = True,
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Returns the kurtosis of each row, ignoring Not a Numbers (NaNs)."""
    n = (~x.isnan()).to(x).sum(dim=dim, keepdim=True)
    ddof = 1 if unbiased else 0
    x = torch.where(n <= ddof * 2, torch.as_tensor(math.nan).to(x), x)
    m1 = x.nansum(dim=dim, keepdim=True) / n
    m2 = ((x - m1) ** 2).nansum(dim=dim, keepdim=True)
    m4 = ((x - m1) ** 4).nansum(dim=dim, keepdim=True)
    r = (m4 / m2**2) * (n + ddof) * n - (3 if fisher else 0) * (n - ddof)
    r = r * (n - ddof) / (n - ddof * 2) / (n - ddof * 3)
    r = r.sum(dim=dim, keepdim=keepdim)
    return typing.cast(torch.Tensor, r)
