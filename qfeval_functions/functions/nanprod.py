import math
import typing

import torch


def nanprod(
    x: torch.Tensor,
    dim: typing.Union[None, int, typing.Tuple[int, ...]] = None,
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Compute the product of tensor elements along specified dimensions,
    ignoring NaN values.

    This function calculates the product of all valid (non-NaN) elements
    in a tensor along the specified dimension(s). Unlike ``numpy.nanprod``,
    which returns 1 when no valid elements are found, this function
    returns NaN for slices that contain no valid values, following the
    convention of :func:`nansum`. This behavior is more mathematically
    consistent for statistical operations where the absence of data
    should be explicitly represented as NaN.

    The NaN-aware product is computed as:

    .. math::
        \text{nanprod}(X) = \prod_{i \text{ valid}} X_i

    where the product is over all valid (non-NaN) values.

    Args:
        x (Tensor):
            The input tensor containing values.
        dim (None, int, or tuple of ints, optional):
            The dimension(s) along which to compute the product. If None
            (default), the product is computed over all dimensions.
            A tuple must contain at least one dimension.
        keepdim (bool, optional):
            Whether the output tensor has :attr:`dim`
            retained or not. Default is False.

    Returns:
        Tensor:
            The product values computed only over valid (non-NaN) values.
            When no valid values exist along a dimension, the result is
            NaN (unlike ``numpy.nanprod`` which returns 1). The shape
            depends on the input dimensions, :attr:`dim`, and
            :attr:`keepdim` parameters.

    Example:

        >>> # Simple product with NaN values
        >>> x = torch.tensor([2.0, nan, 3.0])
        >>> QF.nanprod(x)
        tensor(6.)

        >>> # All-NaN slices return NaN (numpy.nanprod would return 1)
        >>> x = torch.tensor([[2.0, 3.0],
        ...                   [nan, nan]])
        >>> QF.nanprod(x, dim=1)
        tensor([6., nan])

        >>> # 2D tensor with product along columns
        >>> x = torch.tensor([[1.0, nan, 3.0],
        ...                   [4.0, 5.0, nan]])
        >>> QF.nanprod(x, dim=0)
        tensor([4., 5., 3.])

        >>> # With keepdim
        >>> QF.nanprod(x, dim=1, keepdim=True)
        tensor([[ 3.],
                [20.]])

        >>> # Multiple dimensions
        >>> x = torch.tensor([[[1.0, nan], [3.0, 4.0]],
        ...                   [[nan, 6.0], [7.0, nan]]])
        >>> QF.nanprod(x, dim=(1, 2))
        tensor([12., 42.])

    .. note::
        Zeros and infinities are valid values: a slice containing both
        ``0`` and ``inf`` yields NaN because ``0 * inf`` is NaN under
        IEEE 754 arithmetic. Only NaN values are skipped.

    .. warning::
        When all values along a dimension are NaN, this function returns
        NaN (not 1 like ``numpy.nanprod``). This behavior difference
        should be considered when replacing ``numpy.nanprod`` with this
        function.

    .. seealso::
        - :func:`nansum`: NaN-aware sum function.
        - :func:`nancumprod`: NaN-aware cumulative product function.
        - :func:`nanmean`: NaN-aware mean function.
    """
    valid = ~x.isnan()
    filled = torch.where(valid, x, torch.ones_like(x))
    if dim is None:
        p = filled.prod()
        if keepdim:
            p = p.reshape((1,) * x.dim())
    elif isinstance(dim, int):
        p = filled.prod(dim=dim, keepdim=keepdim)
    else:
        ndim = max(x.dim(), 1)
        p = filled
        for d in sorted((d % ndim for d in dim), reverse=True):
            p = p.prod(dim=d, keepdim=keepdim)
    is_valid = valid.sum(dim=dim, keepdim=keepdim) > 0
    return torch.where(is_valid, p, torch.as_tensor(math.nan).to(p))
