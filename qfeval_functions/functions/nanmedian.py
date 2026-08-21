import typing

import torch

from .nanquantile import _reduce_over_dims


def nanmedian(
    x: torch.Tensor,
    dim: typing.Union[None, int, typing.Tuple[int, ...]] = None,
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Compute the median of a tensor, ignoring NaN values.

    This function calculates the median of tensor elements along the
    specified dimension(s) while excluding NaN values from the
    computation.  It extends ``torch.nanmedian`` with support for
    reducing over multiple dimensions at once, matching the unified
    :attr:`dim` semantics of the other NaN-aware reductions in this
    package (e.g., :func:`nanmean`), and it always returns median
    values only (never indices).  If all values of a reduced slice are
    NaN, the result for that slice is NaN.

    Args:
        x (Tensor):
            The input tensor containing values.
        dim (None, int, or tuple of ints, optional):
            The dimension(s) along which to compute the median.  If
            None (default), the median is computed over all elements.
        keepdim (bool, optional):
            Whether the output tensor has :attr:`dim` retained or not.
            Default is False.

    Returns:
        Tensor:
            The median values computed only over valid (non-NaN)
            values.  The shape depends on the input dimensions,
            :attr:`dim`, and :attr:`keepdim` parameters.

    Example:

        >>> # For an even number of valid values, the lower of the two
        >>> # middle values (2.0, not 3.0) is returned.
        >>> x = torch.tensor([2.0, 1.0, nan, 5.0, 4.0])
        >>> QF.nanmedian(x)
        tensor(2.)

        >>> # nanquantile averages the two middle values instead.
        >>> QF.nanquantile(x, 0.5)
        tensor(3.)

        >>> # 2D tensor with dimension specification
        >>> x = torch.tensor([[1.0, 2.0, nan],
        ...                   [4.0, nan, 6.0]])
        >>> QF.nanmedian(x, dim=1)
        tensor([1., 4.])

        >>> # Keep dimensions
        >>> QF.nanmedian(x, dim=1, keepdim=True)
        tensor([[1.],
                [4.]])

        >>> # Multiple dimensions
        >>> x = torch.tensor([[[1.0, 2.0], [3.0, nan]],
        ...                   [[5.0, 6.0], [7.0, 8.0]]])
        >>> QF.nanmedian(x, dim=(1, 2))
        tensor([2., 6.])

        >>> # All-NaN slice yields NaN
        >>> x = torch.tensor([[1.0, 2.0],
        ...                   [nan, nan]])
        >>> QF.nanmedian(x, dim=1)
        tensor([1., nan])

    .. note::
        Like ``torch.nanmedian`` (and unlike ``numpy.nanmedian``), when
        a slice has an even number of valid values, this function
        returns the lower of the two middle values instead of their
        average.  As a consequence, the result is always one of the
        input values.  For the interpolated (averaging) behavior, use
        ``QF.nanquantile(x, 0.5)``.

    .. seealso::
        - :func:`nanquantile`: NaN-aware quantile function
          (``q=0.5`` averages the two middle values).
        - :func:`mmedian`: Moving (sliding window) median function.
        - ``torch.nanmedian``: PyTorch's built-in NaN-aware median
          (single-dimension reduction only).
    """
    if dim is None:
        result = torch.nanmedian(x)
        if keepdim:
            result = result.reshape((1,) * x.dim())
        return result
    if isinstance(dim, int):
        return torch.nanmedian(x, dim=dim, keepdim=keepdim).values
    return _reduce_over_dims(
        x, dim, keepdim, lambda y: torch.nanmedian(y, dim=-1).values
    )
