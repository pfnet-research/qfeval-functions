import typing

import torch

from .nanmean import nanmean


def nanrms(
    x: torch.Tensor,
    dim: typing.Union[None, int, typing.Tuple[int, ...]] = None,
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Compute the root mean square of a tensor, ignoring NaN values.

    This function calculates the root mean square (RMS) of tensor
    elements along the specified dimension(s), excluding NaN values from
    the computation. NaN values are treated as missing data and
    contribute to neither the sum of squares nor the count, so this
    function relates to :func:`rms` exactly as :func:`nanmean` relates to
    the ordinary mean. The RMS measures the magnitude of values
    regardless of their signs, and is commonly used to quantify the size
    of signals or returns.

    The NaN-aware root mean square is computed as:

    .. math::
        \text{nanrms}(X) = \sqrt{\frac{1}{N_{\text{valid}}}
            \sum_{i \text{ valid}} X_i^2}

    where the sum is over valid (non-NaN) values and
    :math:`N_{\text{valid}}` is the number of valid values.

    Args:
        x (Tensor):
            The input tensor containing values.
        dim (None, int, or tuple of ints, optional):
            The dimension(s) along which to compute the root mean square.
            If ``None`` (default), reduces over all dimensions.
        keepdim (bool, optional):
            Whether the output tensor has :attr:`dim` retained or not.
            Default is ``False``.

    Returns:
        Tensor:
            The root mean square computed only over valid (non-NaN)
            values. When no valid values exist along a dimension, the
            result is NaN. The shape depends on the input dimensions,
            :attr:`dim`, and :attr:`keepdim` parameters.

    Example:

        >>> # Simple root mean square with NaN values
        >>> x = torch.tensor([3.0, nan, -4.0])
        >>> QF.nanrms(x)
        tensor(3.5355)

        >>> # Matches rms() when no NaN values are present
        >>> x = torch.tensor([1.0, -2.0, 3.0, -4.0])
        >>> QF.nanrms(x)
        tensor(2.7386)

        >>> # 2D tensor with NaN values
        >>> x = torch.tensor([[1.0, -2.0, nan],
        ...                   [3.0, nan, -4.0]])
        >>> QF.nanrms(x, dim=1)
        tensor([1.5811, 3.5355])

        >>> # With keepdim
        >>> QF.nanrms(x, dim=1, keepdim=True)
        tensor([[1.5811],
                [3.5355]])

        >>> # All NaN slice returns NaN
        >>> x = torch.tensor([[3.0, -4.0],
        ...                   [nan, nan]])
        >>> QF.nanrms(x, dim=1)
        tensor([3.5355,    nan])

    .. note::
        Infinite values are valid values: any slice containing ``inf`` or
        ``-inf`` yields ``inf`` because squaring makes it positive
        infinity. Only NaN values are skipped.

    .. seealso::
        - :func:`rms`: Root mean square function (NaN propagates).
        - :func:`nanmean`: NaN-aware mean function.
        - :func:`nanstd`: NaN-aware standard deviation function.
    """
    return nanmean(x.square(), dim=dim, keepdim=keepdim).sqrt()
