import typing

import torch


class Result(typing.NamedTuple):
    values: torch.Tensor
    indices: torch.Tensor


def rcummax(x: torch.Tensor, dim: int) -> Result:
    r"""Returns the reversely cumulative max of elements of `input` in the
    dimension `dim`.
    """
    result = torch.cummax(torch.flip(x, [dim]), dim)
    return Result(
        values=torch.flip(result.values, [dim]),
        indices=x.shape[dim] - 1 - torch.flip(result.indices, [dim]),
    )
