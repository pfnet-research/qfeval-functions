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
        ...     [1.0, 2.0, nan, 1.0],
        ...     [2.0, 4.0, nan, 2.0],
        ...     [3.0, nan, nan, 3.0],
        ...     [1.0, 1.0, 1.0, 1.0],
        ... ])
        >>> reduce_nan_patterns(x, -1, 0)
        tensor([ True, False, False,  True])
        >>> # As x.dim() == 2, agg_f does not affect the results
        >>> reduce_nan_patterns(x, -1, 0, agg_f="all")
        tensor([ True, False, False,  True])
        >>> reduce_nan_patterns(x, 0, 1)
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
    r"""Shift tensor elements along a dimension, skipping masked positions.

    This function performs a selective shift operation where only elements at
    positions marked as ``True`` in the mask are shifted, while positions
    marked as ``False`` are skipped. This is particularly useful for time
    series data where certain time points should be excluded from the shift
    operation, such as weekends in financial data or missing observations.

    The function works by reordering elements based on the mask, applying the
    shift only to valid positions, and then restoring the original order.
    Elements at masked-out positions are replaced with NaN values.

    .. warning::
        Unmasked values (where mask is ``False``) are replaced with NaN before
        shifting. This means that even a zero shift (``shift=0``) will not
        return the original input, as unmasked values will be NaN in the
        output.

    Args:
        x (Tensor):
            The input tensor to be shifted.
        shift (int, optional):
            The number of positions to shift. Positive values shift forward
            (toward higher indices), negative values shift backward.
            Default is 1.
        dim (int, optional):
            The dimension along which to perform the shift.
            Default is 0.
        mask (Tensor, optional):
            A 1D boolean tensor where ``True`` indicates positions to include
            in the shift, and ``False`` indicates positions to skip. Length
            must equal ``x.shape[dim]``. If not provided, :attr:`refdim` must
            be specified.
        refdim (int, optional):
            If specified, automatically generates a mask using
            :func:`reduce_nan_patterns`. This identifies valid positions based
            on non-NaN patterns in the reference dimension. Default is -1.
        agg_f (str, optional):
            Aggregation function used when generating mask from :attr:`refdim`.
            Can be "any" or "all".
            Default is "any".

    Returns:
        Tensor:
            A tensor of the same shape as the input, with elements shifted
            according to the mask. Unmasked positions contain NaN.

    Example:

        >>> # Basic masked shift
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> mask = torch.tensor([True, False, True, False, True])
        >>> shifted = QF.group_shift(x, shift=1, dim=0, mask=mask)
        >>> shifted
        tensor([nan, nan, 1., nan, 3.])

        >>> # Shift with existing NaN values
        >>> x = torch.tensor([1.0, nan, 2.0, nan, 3.0])
        >>> mask = torch.tensor([True, False, True, False, True])
        >>> shifted = QF.group_shift(x, shift=1, dim=0, mask=mask)
        >>> shifted
        tensor([nan, nan, 1., nan, 2.])

        >>> # 2D tensor with automatic mask generation
        >>> x = torch.tensor([[1.0, 2.0, nan, 4.0],
        ...                   [5.0, 6.0, nan, 8.0],
        ...                   [9.0, 10., nan, 12.]])
        >>> # Use refdim=0 to generate mask from first row's NaN pattern
        >>> shifted = QF.group_shift(x, shift=1, dim=1, refdim=0)
        >>> shifted
        tensor([[nan,  1., nan,  2.],
                [nan,  5., nan,  6.],
                [nan,  9., nan, 10.]])

    .. seealso::
        :func:`reduce_nan_patterns`: For understanding mask generation from
        reference dimensions.
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
