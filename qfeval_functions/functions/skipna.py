import functools
import math
import operator
import typing

import torch


def skipna(
    f: typing.Callable[..., torch.Tensor],
    *xs: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Applies the given data to the given function after removing NaNs.  The
    function will take a tensor having non-NaN values and NaN values in the
    order.  The order of non-NaN values must be preserved.
    """
    # Generate an index to convert between a sparse array and a dense array.
    # For the given `x` (sparse): [0, nan, 5, nan, 4, 2, 1], the dense array
    # (the scattered array) should be [0, 5, 4, 2, 1, nan, nan] and `idx` (the
    # mapping form the sparse array to the dense array) should be
    # [0, 5, 1, 6, 2, 3, 4].
    m = functools.reduce(operator.or_, (x.isnan() for x in xs))
    valid_idx = (~m).cumsum(dim) - 1
    invalid_idx = (~m).sum(dim, keepdim=True) + m.cumsum(dim) - 1
    idx = torch.where(m, invalid_idx, valid_idx)
    y = f(*(x.scatter(dim, idx, x) for x in xs)).gather(dim, idx)
    return torch.where(m, torch.as_tensor(math.nan).to(y), y)
