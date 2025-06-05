import typing

import torch


# NOTE: This uses the same default value as R's signif function uses.  The
# argument name follows numpy.round and DataFrame.round.
def signif(x: torch.Tensor, decimals: int = 6) -> torch.Tensor:
    r"""Rounds the numbers of the given tensor to the specified number of
    significant digits.
    """
    e = 10 ** (decimals - x.abs().log10().ceil())
    e = torch.where(e.isfinite() & e.ne(0.0), e, torch.tensor(1).to(e))
    return typing.cast(torch.Tensor, (x * e).round() / e)
