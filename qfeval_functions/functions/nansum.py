import math
import typing

import torch


def nansum(
    x: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Compute the sum of tensor elements along specified dimensions, ignoring
    NaN values.

    This function calculates the sum of all valid (non-NaN) elements in a tensor
    along the specified dimension(s). Unlike PyTorch's ``torch.nansum``, this
    function returns NaN when no valid elements are found along a dimension,
    rather than returning 0. This behavior is more mathematically consistent
    for statistical operations where the absence of data should be explicitly
    represented as NaN.

    The NaN-aware sum is computed as:

    .. math::
        \text{nansum}(X) = \sum_{i \text{ valid}} X_i

    where the sum is over all valid (non-NaN) values.

    Args:
        x (Tensor):
            The input tensor containing values.
        dim (int or tuple of ints, optional):
            The dimension(s) along which to compute the sum. If not specified
            (default is empty tuple), the sum is computed over all dimensions.
        keepdim (bool, optional):
            Whether the output tensor has :attr:`dim`
            retained or not. Default is False.

    Returns:
        Tensor:
            The sum values computed only over valid (non-NaN) values.
            When no valid values exist along a dimension, the result is NaN
            (unlike ``torch.nansum`` which returns 0). The shape depends on
            the input dimensions, :attr:`dim`, and :attr:`keepdim` parameters.

    Example:

        >>> # Simple sum with NaN values
        >>> x = torch.tensor([1.0, 2.0, nan, 4.0, 5.0])
        >>> QF.nansum(x)
        tensor(12.)

        >>> # Compare with torch.nansum (returns 0 for all-NaN)
        >>> all_nan = torch.tensor([nan, nan, nan])
        >>> QF.nansum(all_nan)
        tensor(nan)
        >>> torch.nansum(all_nan)
        tensor(0.)

        >>> # 2D tensor with sum along columns
        >>> x = torch.tensor([[1.0, nan, 3.0],
        ...                   [4.0, 5.0, nan]])
        >>> QF.nansum(x, dim=0)
        tensor([5., 5., 3.])

        >>> # Sum along rows
        >>> QF.nansum(x, dim=1)
        tensor([4., 9.])

        >>> # All dimensions
        >>> x = torch.tensor([[1.0, 2.0],
        ...                   [nan, 4.0]])
        >>> QF.nansum(x)
        tensor(7.)

        >>> # With keepdim
        >>> x = torch.tensor([[1.0, nan, 3.0],
        ...                   [4.0, 5.0, nan]])
        >>> QF.nansum(x, dim=1, keepdim=True)
        tensor([[4.],
                [9.]])

        >>> # All NaN slice returns NaN
        >>> x = torch.tensor([[1.0, 2.0],
        ...                   [nan, nan]])
        >>> QF.nansum(x, dim=1)
        tensor([3., nan])

        >>> # Multiple dimensions
        >>> x = torch.tensor([[[1.0, nan], [3.0, 4.0]],
        ...                   [[nan, 6.0], [7.0, nan]]])
        >>> QF.nansum(x, dim=(1, 2))
        tensor([ 8., 13.])

    .. note::
        This function provides more mathematically consistent behavior than
        ``torch.nansum`` by returning NaN when no valid values are present,
        rather than 0. This is particularly important for statistical
        computations where the absence of data should be distinguished
        from a sum of zero.

    .. warning::
        When all values along a dimension are NaN, this function returns NaN
        (not 0 like ``torch.nansum``). This behavior difference should be
        considered when replacing ``torch.nansum`` with this function.

    .. seealso::
        :func:`nanmean`: NaN-aware mean function.
        :func:`nanstd`: NaN-aware standard deviation function.
        :func:`nanvar`: NaN-aware variance function.
        ``torch.nansum``: PyTorch's built-in NaN-aware sum (returns 0 for all-NaN).
    """
    is_valid = (~x.isnan()).sum(dim=dim, keepdim=keepdim) > 0
    y = x.nansum(dim=dim, keepdim=keepdim)
    return torch.where(is_valid, y, torch.as_tensor(math.nan).to(y))
