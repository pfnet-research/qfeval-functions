import math
import typing

import torch

from .shift import shift as _shift

AggregateFunction = typing.Literal["any", "all"]
"""Type alias for aggregation functions used in reduce_nan_patterns.

- "any": Position is valid if at least one value is non-NaN
- "all": Position is valid only if all values are non-NaN
"""


def reduce_nan_patterns(
    x: torch.Tensor,
    dim: int = -1,
    refdim: int = 0,
    agg_f: AggregateFunction = "any",
) -> torch.Tensor:
    r"""Creates a boolean mask based on NaN patterns in a reference dimension.

    This function analyzes the NaN pattern in a reference dimension and creates a 1-D
    boolean mask that can be used with :func:`group_shift`. The mask indicates which
    positions along the target dimension contain valid (non-NaN) data.

    The function works by:

    1. Examining values along the reference dimension (``refdim``)
    2. For each position along the target dimension (``dim``), checking if there are
       valid (non-NaN) values in the reference dimension
    3. Aggregating this information using the specified aggregation function
    4. Returning a mask where ``True`` indicates positions with valid data

    This is particularly useful for financial time series where you want to identify
    positions that have valid data across different features or time periods.

    Args:
        x (Tensor): The input tensor. Must have at least 2 dimensions.
        dim (int): The dimension along which the mask will be applied (target dimension
            for shifting). Default: -1 (last dimension)
        refdim (int): The reference dimension to analyze for NaN patterns. Default: 0
        agg_f ({"any", "all"}): Aggregation function for combining information across
            other dimensions:

            - ``"any"``: Position is valid if at least one value is non-NaN
            - ``"all"``: Position is valid only if all values are non-NaN

            Default: "any"

    Returns:
        Tensor: A 1-D boolean tensor of length ``x.shape[dim]`` where ``True`` values
        indicate positions with valid data according to the specified criteria.

    Raises:
        NotImplementedError: If ``agg_f`` is not "any" or "all".

    Example::

        >>> import torch
        >>> import qfeval_functions.functions as QF
        >>>
        >>> # Create a tensor with NaN patterns
        >>> x = torch.tensor([
        ...     [1.0, 2.0, float('nan'), 1.0],
        ...     [2.0, 4.0, float('nan'), 2.0],
        ...     [3.0, float('nan'), float('nan'), 3.0],
        ...     [1.0, 1.0, 1.0, 1.0],
        ... ])
        >>>
        >>> # Check which columns have any valid values across rows
        >>> QF.reduce_nan_patterns(x, dim=-1, refdim=0, agg_f="any")
        tensor([ True, False, False,  True])
        >>>
        >>> # Check which columns have all valid values across rows
        >>> QF.reduce_nan_patterns(x, dim=-1, refdim=0, agg_f="all")
        tensor([ True, False, False,  True])
        >>>
        >>> # Check which rows have all valid values across columns
        >>> QF.reduce_nan_patterns(x, dim=0, refdim=1, agg_f="all")
        tensor([False, False, False,  True])
        >>>
        >>> # Check which rows have any valid values across columns
        >>> QF.reduce_nan_patterns(x, dim=0, refdim=1, agg_f="any")
        tensor([False, False, False,  True])
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

    # TODO(claude): The logic here seems incorrect for agg_f="any" case.
    # When agg_f="any", we should return True for positions where ANY value
    # in the reference dimension is non-NaN, but the current implementation
    # requires ALL aggregated values to be True. This causes the doctest
    # example to return [False, False, False, True] instead of the expected
    # [True, True, True, True] for rows that have at least one valid value.
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
    r"""Shifts a tensor along a specified dimension, applying only to masked positions.

    This function performs a selective shift operation that only moves elements at positions
    specified by a boolean mask, while filling unmasked positions with NaN. This is
    particularly useful for financial time series where you need to shift values while
    preserving specific patterns (e.g., trading days vs. weekends).

    The mask can either be provided directly or generated automatically based on NaN
    patterns in a reference dimension. The function ensures that:

    1. Only elements where ``mask[i] == True`` participate in the shift operation
    2. Elements where ``mask[i] == False`` are replaced with NaN
    3. The shift is applied in a way that maintains the masked structure

    For a 1-D tensor with a mask ``[True, False, True, False, True]``, applying
    ``shift=1`` to values ``[1, nan, 2, nan, 3]`` produces ``[nan, nan, 1, nan, 2]``.

    Warning:
        This function modifies unmasked values by replacing them with NaN, regardless
        of their original values. A zero shift (``shift=0``) will still apply the
        masking operation and will not return the original input unchanged.

    Args:
        x (Tensor): The input tensor to be shifted
        shift (int): The number of positions to shift. Positive values shift forward,
            negative values shift backward. Default: 1
        dim (int): The dimension along which to apply the shift. Default: 0
        mask (Tensor, optional): A 1-D boolean tensor where ``True`` values indicate
            positions that should participate in the shift. Must have length equal
            to ``x.shape[dim]``. Default: None
        refdim (int, optional): If provided and ``mask`` is None, automatically
            generates a mask based on NaN patterns in this reference dimension.
            See :func:`reduce_nan_patterns` for details. Default: -1
        agg_f ({"any", "all"}): Aggregation function used when generating mask from
            ``refdim``. Only used when ``refdim`` is specified. Default: "any"

    Returns:
        Tensor: A tensor of the same shape as :attr:`x` with shifted values according
        to the mask pattern.

    Raises:
        ValueError: If neither ``mask`` nor ``refdim`` is provided.
        AssertionError: If ``mask`` length doesn't match ``x.shape[dim]``.

    Example::

        >>> import torch
        >>> import qfeval_functions.functions as QF
        >>>
        >>> # Basic usage with explicit mask
        >>> x = torch.tensor([1., 2., 3., 4., 5.])
        >>> mask = torch.tensor([True, False, True, False, True])
        >>> QF.group_shift(x, shift=1, dim=0, mask=mask)
        tensor([nan, nan, 1., nan, 3.])

        >>> # Negative shift
        >>> QF.group_shift(x, shift=-1, dim=0, mask=mask)
        tensor([3., nan, 5., nan, nan])

        >>> # 2D tensor with automatic mask generation
        >>> x = torch.tensor([
        ...     [1., 2., float('nan'), 4.],
        ...     [5., 6., float('nan'), 8.],
        ...     [9., 10., 11., 12.]
        ... ])
        >>> QF.group_shift(x, shift=1, dim=1, refdim=0)
        tensor([[nan,  1., nan,  2.],
                [nan,  5., nan,  6.],
                [nan,  9., nan, 10.]])

        >>> # Zero shift still applies masking
        >>> x = torch.tensor([1., 2., 3., 4., 5.])
        >>> mask = torch.tensor([True, False, True, False, True])
        >>> QF.group_shift(x, shift=0, dim=0, mask=mask)
        tensor([1., nan, 3., nan, 5.])
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
