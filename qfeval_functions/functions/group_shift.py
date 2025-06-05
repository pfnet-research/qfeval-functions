import math
import typing

import torch

from .shift import shift as _shift

AggregateFunction = typing.Literal["any", "all"]


def reduce_nan_patterns(
    x: torch.Tensor,
    dim: int = -1,
    refdim: int = 0,
    agg_f: AggregateFunction = "any",
) -> torch.Tensor:
    r"""Creates a mask for group shift.

    A mask is a one-dimensional boolean tensor that represents the pattern
    of observed values in a reference dimension (`refdim`).
    i.e., `True` values correspond to the locations of non-nan values.

    Args:
        - x: The input tensor. This should have at least 2 dimensions.
        - dim: The dimension along which `x` will be shifted.
        - refdim: The reference dimension to extract a pattern of (non-)nans.
        - agg_f: The function for aggregating all other dimensions.
    Returns:
        - mask: a 1-D boolean tensor

    Examples:
        >>> x = torch.tensor([
            [1.0, 2.0, nan, 1.0],
            [2.0, 4.0, nan, 2.0],
            [3.0, nan, nan, 3.0],
            [1.0, 1.0, 1.0, 1.0],
        ])
        >>> reduce_nan_patterns(x, -1, 0)
        tensor([True, False, False, True])
        # As x.dim() == 2, agg_f does not affect the results
        >>> reduce_nan_patterns(x, -1, 0, agg_f="all")
        tensor([True, False, False, True])
        >>> reduce_nan_patterns(x, 0, 1)
        tensor([False, False, False, True])
    """
    # transpose x so that the first dimension is dim and the second is refdim
    # NOTE: currently, dimensions added here will not be squeezed, since
    # they do not affect "any" and "all" aggregations
    dim = dim % len(x.shape) + 2
    refdim = refdim % len(x.shape) + 2
    x = x[None, None, ...].transpose(0, dim).transpose(1, refdim)

    # aggregate values in all dimensions but dim and refdim
    reduced = (~x.isnan()).flatten(2)
    if agg_f == "any":
        reduced = reduced.any(dim=-1)
    elif agg_f == "all":
        reduced = reduced.all(dim=-1)
    else:
        raise NotImplementedError("Unknown aggregate function.")

    # return True if all (aggregated) values in redim are True
    return reduced.all(dim=1)


def group_shift(
    x: torch.Tensor,
    shift: int = 1,
    dim: int = 0,
    mask: typing.Optional[torch.Tensor] = None,
    refdim: typing.Optional[int] = -1,
    agg_f: AggregateFunction = "any",
) -> torch.Tensor:
    r"""Shifts a tensor along a specified dimension, skipping a given mask.

    This function applies shifts only for locations specified by the mask.
    For example, suppose the mask is `[True, False, True, False, True]`,
    then applying one shift to `[1, nan, 2, nan, 3]` will get
    `[nan, nan, 1, nan, 2]`.

    CAVEAT:
    Before applying shifts, the unmasked values (i.e., mask values are False)
    are filled with nans. So, applying the above masked-shift to
    `[1, 2, 3, 4, 5]` will get `[nan, nan, 1, nan, 3]`, where unmasked values
    (2, 4) are just discarded. Also, zero-shift (`shift == 0`) will not give
    the original input.

    Args:
        - x: The input tensor.
        - shift: The number of places by which the elements of the tensor
            are shifted.
        - dim: The dimension along which `x` will be shifted.
        - mask (optional): A 1D boolean tensor, where `False` values specify
            the indices to be skipped during shifts. The length must be
            equal to `x.shape[dim]`.
        - refdim (optional): If set, the mask is automatically generated.
            See `reduce_nan_patterns` for details.
    Returns:
        - x_shifted: A shifted version of `x`.
    """
    n = x.shape[dim]

    # 1. Create a priority index from the mask
    priority = torch.arange(n, device=x.device)

    if mask is not None:
        mask = mask.squeeze()
        assert mask.shape == (n,)
    elif refdim is not None:
        mask = reduce_nan_patterns(x, dim, refdim, agg_f)
    else:
        raise ValueError("Either mask or refdim must be set.")

    priority = priority + (~mask).int() * n
    index = torch.argsort(priority)
    reversed_index = torch.argsort(index)

    # 2. Move the target dimension to the top to make data manipulation easier.
    x = x.transpose(0, dim)
    shape = x.shape
    x = x.reshape((n, -1))

    # 3. Gather valid values, apply shift, and scatter them.
    # fill unmasked indices with nans
    y = torch.where(
        mask[:, None].expand(n, x.shape[1]),
        x,
        torch.tensor(math.nan).to(x),
    )
    # sort y so that masked values come first
    y = y[index, :]
    # apply shift
    y = torch.where(
        mask[index, None].expand(n, x.shape[1]),
        _shift(y, shift, 0),
        y,
    )
    x = y[reversed_index, :]

    # 4. Restore the shape and the order of dimensions.
    return x.reshape(shape).transpose(0, dim)
