import typing

import torch


def _reduce_over_dims(
    x: torch.Tensor,
    dim: typing.Tuple[int, ...],
    keepdim: bool,
    op: typing.Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    r"""Apply a single-dimension reduction over multiple dimensions.

    This normalizes the given (possibly negative) dimensions,
    deduplicates and sorts them, permutes them to the end of the tensor,
    and flattens them into a single trailing dimension.  The operation
    :attr:`op`, which must reduce the last dimension of its input, is
    then applied.  If :attr:`keepdim` is True, each reduced dimension is
    restored as a size-1 dimension at its original position.
    """
    dims = sorted({d % x.dim() for d in dim})
    kept = [d for d in range(x.dim()) if d not in dims]
    result = op(x.permute(kept + dims).flatten(start_dim=len(kept)))
    if keepdim:
        for d in dims:
            result = result.unsqueeze(d)
    return result


def nanquantile(
    x: torch.Tensor,
    q: float,
    dim: typing.Union[None, int, typing.Tuple[int, ...]] = None,
    keepdim: bool = False,
    interpolation: str = "linear",
) -> torch.Tensor:
    r"""Compute the ``q``-th quantile of a tensor, ignoring NaN values.

    This function calculates the quantile of tensor elements along the
    specified dimension(s) while excluding NaN values from the
    computation.  It extends ``torch.nanquantile`` with support for
    reducing over multiple dimensions at once, matching the unified
    :attr:`dim` semantics of the other NaN-aware reductions in this
    package (e.g., :func:`nanmean`).  NaN values are treated as missing
    data; if all values of a reduced slice are NaN, the result for that
    slice is NaN.

    Args:
        x (Tensor):
            The input tensor containing values.
        q (float):
            The quantile to compute, which must be in the range
            ``[0, 1]``.  Only a scalar Python float is supported, which
            keeps the output shape identical to that of the other
            NaN-aware reductions.
        dim (None, int, or tuple of ints, optional):
            The dimension(s) along which to compute the quantile.  If
            None (default), the quantile is computed over all elements.
        keepdim (bool, optional):
            Whether the output tensor has :attr:`dim` retained or not.
            Default is False.
        interpolation (str, optional):
            The interpolation method to use when the desired quantile
            lies between two data points.  One of ``"linear"``
            (default), ``"lower"``, ``"higher"``, ``"nearest"``, or
            ``"midpoint"``.

    Returns:
        Tensor:
            The quantile values computed only over valid (non-NaN)
            values.  The shape depends on the input dimensions,
            :attr:`dim`, and :attr:`keepdim` parameters.

    Raises:
        ValueError: If ``q`` is outside the range ``[0, 1]``.
        RuntimeError: If ``interpolation`` is not one of the supported
            methods (raised by ``torch.nanquantile``).

    Example:

        >>> # Simple quantile with NaN values
        >>> x = torch.tensor([1.0, 2.0, nan, 4.0, 5.0])
        >>> QF.nanquantile(x, 0.5)
        tensor(3.)

        >>> # 2D tensor with dimension specification
        >>> x = torch.tensor([[1.0, 2.0, nan],
        ...                   [4.0, nan, 6.0]])
        >>> QF.nanquantile(x, 0.5, dim=1)
        tensor([1.5000, 5.0000])

        >>> # Keep dimensions
        >>> QF.nanquantile(x, 0.5, dim=1, keepdim=True)
        tensor([[1.5000],
                [5.0000]])

        >>> # Interpolation modes
        >>> QF.nanquantile(x, 0.5, dim=1, interpolation="lower")
        tensor([1., 4.])

        >>> # Multiple dimensions
        >>> x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
        ...                   [[5.0, 6.0], [7.0, nan]]])
        >>> QF.nanquantile(x, 0.5, dim=(1, 2))
        tensor([2.5000, 6.0000])

        >>> # All-NaN slice yields NaN
        >>> x = torch.tensor([[1.0, 2.0],
        ...                   [nan, nan]])
        >>> QF.nanquantile(x, 0.5, dim=1)
        tensor([1.5000,    nan])

    .. note::
        A slice whose values are all NaN yields NaN, consistent with
        ``numpy.nanquantile``.  Unlike ``torch.nanquantile``, this
        function accepts a tuple of dimensions, in which case the
        quantile is computed jointly over all specified dimensions.

    .. seealso::
        - :func:`nanmedian`: NaN-aware median function.
        - :func:`winsorize`: Quantile-based clipping built on this
          function.
        - :func:`mquantile`: Moving (sliding window) quantile function.
        - ``torch.nanquantile``: PyTorch's built-in NaN-aware quantile
          (single-dimension reduction only).
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in the range [0, 1], but got {q}.")
    if dim is None:
        result = torch.nanquantile(x, q, interpolation=interpolation)
        if keepdim:
            result = result.reshape((1,) * x.dim())
        return result
    if isinstance(dim, int):
        return torch.nanquantile(
            x, q, dim=dim, keepdim=keepdim, interpolation=interpolation
        )
    return _reduce_over_dims(
        x,
        dim,
        keepdim,
        lambda y: torch.nanquantile(y, q, dim=-1, interpolation=interpolation),
    )
