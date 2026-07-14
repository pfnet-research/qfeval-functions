import math
import typing

import torch


class StableSortResult(typing.NamedTuple):
    values: torch.Tensor
    indices: torch.Tensor


def _unsafe_stable_sort(x: torch.Tensor, dim: int) -> StableSortResult:
    r"""Sorts the given tensor without NaN values preserving the order of
    equivalent elements.
    """

    # 1. First sort values using an unstable algorithm.
    # NOTE: v should be the same as a stable algorithm computes.
    v, idx = x.sort(dim=dim)

    # 2. Computes the rank of v.
    v_rank = torch.ne(v, v.roll(1, dim)).cumsum(dim=dim)

    # 3. Sort idx within a group having the same v's rank.
    idx = v_rank * idx.shape[dim] + idx
    idx = idx.sort(dim=dim).values % idx.shape[dim]

    # 4. Return the result of a stable sorting.
    return StableSortResult(v, idx)


def stable_sort(x: torch.Tensor, dim: int = -1) -> StableSortResult:
    r"""Sorts the given tensor preserving the order of equivalent elements.

    This function sorts elements in ascending order along the specified
    dimension, guaranteeing that elements with equal values keep their
    original relative order (i.e., a stable sort).  NaN values are placed
    at the end, also preserving their original relative order.  Unlike
    :func:`torch.sort`, whose default algorithm does not guarantee
    stability, this function always returns deterministic indices.

    Args:
        x (Tensor):
            The input tensor.
        dim (int, optional):
            The dimension along which to sort.
            Default is -1 (the last dimension).

    Returns:
        StableSortResult: A named tuple of ``(values, indices)``:

            - ``values`` (Tensor): The sorted values, in the same shape as
              the input.
            - ``indices`` (Tensor): The indices of the sorted values in the
              original tensor.

    Example:

        >>> # The two 1.0s keep their original order (index 1 before 3).
        >>> x = torch.tensor([3.0, 1.0, 2.0, 1.0])
        >>> result = QF.stable_sort(x)
        >>> result.values
        tensor([1., 1., 2., 3.])
        >>> result.indices
        tensor([1, 3, 2, 0])

        >>> # NaN values are placed at the end.
        >>> x = torch.tensor([2.0, nan, 1.0])
        >>> result = QF.stable_sort(x)
        >>> result.values
        tensor([1., 2., nan])
        >>> result.indices
        tensor([2, 0, 1])

    .. seealso::
        - ``torch.sort``: PyTorch's built-in sort (also supports
          ``stable=True``).
    """

    # If x has no NaN values, it is okay to apply _unsafe_stable_sort.
    isnan = x.isnan()
    if not isnan.any():
        return _unsafe_stable_sort(x, dim)

    # If x has NaN values, computes the stable result using two results.
    safe_x = x.nan_to_num(math.inf, math.inf, -math.inf)
    result = _unsafe_stable_sort(safe_x, dim)
    # Mapping NaN to 0.0, +inf to -1.0 and other values to -2.0.
    # NOTE: Multiplying -inf to the result should recover NaN and +inf.
    nan_result = _unsafe_stable_sort(
        torch.where(
            safe_x.isposinf(),
            x.nan_to_num(0.0, -1.0),
            torch.full_like(x, -2.0),
        ),
        dim,
    )
    return StableSortResult(
        torch.where(
            result.values.isposinf(),
            nan_result.values * -math.inf,
            result.values,
        ),
        torch.where(
            result.values.isposinf(),
            nan_result.indices,
            result.indices,
        ),
    )
