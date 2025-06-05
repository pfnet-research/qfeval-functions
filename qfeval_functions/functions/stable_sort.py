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
    r"""Sorts the given tensor preserving the order of equivalent elements."""

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
