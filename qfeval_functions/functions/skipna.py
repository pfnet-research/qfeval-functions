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
    r"""Applies the given data to the given function after removing NaNs.

    This function makes a NaN-unaware function applicable to data containing
    NaNs.  Along the specified dimension, it first packs the non-NaN values
    to the front (preserving their order) followed by the NaN values, then
    applies the given function ``f`` to the packed tensors, and finally
    restores the results to the original positions of the non-NaN values.
    Positions that had NaN in any of the input tensors are NaN in the
    output.

    CAVEAT: The given function ``f`` must preserve the shape of its inputs
    and must map the :math:`i`-th element of its input to the :math:`i`-th
    element of its output (e.g., cumulative or element-wise operations).

    Args:
        f (Callable[..., Tensor]):
            The function to apply.  It takes as many tensors as ``xs`` and
            returns a tensor of the same shape.
        *xs (Tensor):
            The input tensors.  They must have the same shape.  If multiple
            tensors are given, positions where any tensor has NaN are
            treated as missing in all of them.
        dim (int, optional):
            The dimension along which to pack non-NaN values.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the result
            of ``f`` applied to the non-NaN values, with NaN at positions
            where any input tensor has NaN.

    Example:

        >>> # Cumulative sum, skipping NaNs.
        >>> x = torch.tensor([1.0, nan, 2.0, nan, 3.0])
        >>> QF.skipna(lambda a: a.cumsum(-1), x)
        tensor([1., nan, 3., nan, 6.])

    .. seealso::
        - :func:`apply_for_axis`: Applies a function expecting 2D input
          along a dimension.
        - :func:`fillna`: Replace NaN values instead of removing them.
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
