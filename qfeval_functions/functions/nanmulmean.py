import typing

import torch

from .fillna import fillna
from .mulsum import mulsum


def nanmulmean(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    keepdim: bool = False,
    *,
    _ddof: int = 0,
) -> torch.Tensor:
    r"""Compute the mean of element-wise product, ignoring NaN values, in a
    memory-efficient way.

    This function calculates the mean of the element-wise product of two tensors
    ``nanmean(x * y, dim)`` without creating the intermediate product tensor in
    memory, while also excluding NaN values from the computation. If either
    :attr:`x` or :attr:`y` has a NaN at a given position, that pair is excluded
    from the mean calculation.

    The function is mathematically equivalent to ``nanmean(x * y, dim)`` but
    uses a more memory-efficient implementation that avoids materializing
    the full product tensor, making it suitable for large tensor operations
    and complex broadcasting patterns.

    The NaN-aware mean is computed as:

    .. math::
        \text{nanmulmean}(X, Y) = \frac{\sum_{i \text{ valid}} X_i \cdot Y_i}{
            N_{\text{valid}} - \text{ddof}}

    where the sum is over valid (non-NaN) pairs and :math:`N_{\text{valid}}`
    is the number of valid pairs.

    Args:
        x (Tensor):
            The first input tensor.
        y (Tensor):
            The second input tensor. Must be broadcastable with :attr:`x`.
        dim (int or tuple of ints, optional):
            The dimension(s) along which to
            compute the mean. If not specified (default is empty tuple), the
            mean is computed over all dimensions.
        keepdim (bool, optional):
            Whether the output tensor has :attr:`dim`
            retained or not. Default is False.
        _ddof (int, optional):
            Delta degrees of freedom for internal calculations.
            The divisor used is ``N_valid - _ddof``, where ``N_valid`` is the
            number of valid (non-NaN) pairs. Default is 0. This is an internal
            parameter.

    Returns:
        Tensor:
            The mean of the element-wise product computed only over valid
            (non-NaN) pairs. The shape depends on the input dimensions,
            :attr:`dim`, and :attr:`keepdim` parameters.

    Example:

        >>> # Simple element-wise product mean with NaN
        >>> x = torch.tensor([1.0, 2.0, nan, 4.0])
        >>> y = torch.tensor([2.0, nan, 4.0, 5.0])
        >>> QF.nanmulmean(x, y)
        tensor(11.)

        >>> # 2D tensors with NaN values
        >>> x = torch.tensor([[1.0, nan, 3.0],
        ...                   [4.0, 5.0, nan]])
        >>> y = torch.tensor([[2.0, 4.0, nan],
        ...                   [nan, 10.0, 12.0]])
        >>> QF.nanmulmean(x, y, dim=1)
        tensor([ 2., 50.])

        >>> # Broadcasting with NaN handling
        >>> x = torch.tensor([[1.0], [nan], [3.0]])
        >>> y = torch.tensor([2.0, 4.0, nan])
        >>> QF.nanmulmean(x, y)
        tensor(6.)

        >>> # With keepdim
        >>> x = torch.tensor([[1.0, nan], [3.0, 4.0]])
        >>> y = torch.tensor([[2.0, 4.0], [nan, 5.0]])
        >>> QF.nanmulmean(x, y, dim=1, keepdim=True)
        tensor([[ 2.],
                [20.]])

    .. seealso::
        :func:`mulmean`: Memory-efficient product mean without NaN handling.
        :func:`nanmean`: NaN-aware mean function.
        :func:`mulsum`: Memory-efficient product sum function.
    """
    result = mulsum(fillna(x), fillna(y), dim=dim, keepdim=keepdim)
    x_mask = (~x.isnan()).to(result)
    y_mask = (~y.isnan()).to(result)
    return result / (mulsum(x_mask, y_mask, dim=dim, keepdim=keepdim) - _ddof)
