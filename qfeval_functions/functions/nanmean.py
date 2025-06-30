import typing

import torch


def nanmean(
    x: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Compute the arithmetic mean along specified dimensions, ignoring NaN values.

    This function calculates the mean (average) of tensor elements along the
    specified dimension(s), excluding NaN values from the computation. NaN
    values are treated as missing data and do not contribute to either the
    sum or the count used in the mean calculation.

    The NaN-aware mean is computed as:

    .. math::
        \text{nanmean}(X) = \frac{\sum_{i \text{ valid}} X_i}{N_{\text{valid}}}

    where the sum is over valid (non-NaN) values and :math:`N_{\text{valid}}`
    is the number of valid values.

    Args:
        x (Tensor):
            The input tensor containing values.
        dim (int or tuple of ints, optional):
            The dimension(s) along which to compute the mean. If not specified
            (default is empty tuple), the mean is computed over all dimensions.
        keepdim (bool, optional):
            Whether the output tensor has :attr:`dim` retained or not.
            Default is False.

    Returns:
        Tensor:
            The mean values computed only over valid (non-NaN) values.
            The shape depends on the input dimensions, :attr:`dim`, and
            :attr:`keepdim` parameters.

    Example:
        >>> # Simple mean with NaN values
        >>> x = torch.tensor([1.0, 2.0, nan, 4.0, 5.0])
        >>> QF.nanmean(x)
        tensor(3.)

        >>> # 2D tensor with NaN values
        >>> x = torch.tensor([[1.0, nan, 3.0],
        ...                   [4.0, 5.0, nan]])
        >>> QF.nanmean(x, dim=1)
        tensor([2.0000, 4.5000])

        >>> # All dimensions
        >>> x = torch.tensor([[1.0, 2.0],
        ...                   [nan, 4.0]])
        >>> QF.nanmean(x)
        tensor(2.3333)

        >>> # With keepdim
        >>> x = torch.tensor([[1.0, nan, 3.0],
        ...                   [4.0, 5.0, nan]])
        >>> QF.nanmean(x, dim=1, keepdim=True)
        tensor([[2.0000],
                [4.5000]])

        >>> # All NaN slice
        >>> x = torch.tensor([[1.0, 2.0],
        ...                   [nan, nan]])
        >>> QF.nanmean(x, dim=1)
        tensor([1.5000,    nan])

    .. seealso::
        :func:`nansum`: NaN-aware sum function.
        :func:`nanstd`: NaN-aware standard deviation function.
        :func:`nanvar`: NaN-aware variance function.
        ``torch.nanmean``: PyTorch's built-in NaN-aware mean function.
    """
    count = (~x.isnan()).to(x).sum(dim=dim, keepdim=keepdim)
    return x.nansum(dim=dim, keepdim=keepdim) / count
