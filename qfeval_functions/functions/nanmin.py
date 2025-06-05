import typing

import torch

from .nanmax import nanmax


class NanminResult(typing.NamedTuple):
    values: torch.Tensor
    indices: torch.Tensor


def nanmin(x: torch.Tensor, dim: int, keepdim: bool = False) -> NanminResult:
    r"""Returns a namedtuple `(values, indices)` where values is the minimum
    value of each row of the `input` tensor in the given dimension `dim`,
    ignoring Not a Numbers (NaNs).
    """

    v, idx = nanmax(-x, dim=dim, keepdim=keepdim)
    return NanminResult(-v, idx)
