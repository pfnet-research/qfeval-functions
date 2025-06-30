import math
import typing

import torch

from .fillna import fillna
from .mulsum import mulsum


def nanmulsum(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Compute the sum of element-wise product, ignoring NaN values, in a
    memory-efficient way.

    This function calculates the sum of the element-wise product of two tensors
    ``nansum(x * y, dim)`` without creating the intermediate product tensor in
    memory, while also excluding NaN values from the computation. If either
    :attr:`x` or :attr:`y` has a NaN at a given position, that pair is excluded
    from the sum calculation.

    The function is mathematically equivalent to ``nansum(x * y, dim)`` but
    uses a more memory-efficient implementation that avoids materializing
    the full product tensor, making it suitable for large tensor operations
    and complex broadcasting patterns.

    The NaN-aware sum is computed as:

    .. math::
        \text{nanmulsum}(X, Y) = \sum_{i \text{ valid}} X_i \cdot Y_i

    where the sum is over valid (non-NaN) pairs only.

    Args:
        x (Tensor):
            The first input tensor.
        y (Tensor):
            The second input tensor. Must be broadcastable with :attr:`x`.
        dim (int or tuple of ints, optional):
            The dimension(s) along which to compute the sum. If not specified
            (default is empty tuple), the sum is computed over all dimensions.
        keepdim (bool, optional):
            Whether the output tensor has :attr:`dim`
            retained or not. Default is False.

    Returns:
        Tensor:
            The sum of the element-wise product computed only over valid
            (non-NaN) pairs. If no valid pairs exist along a dimension, the
            result is NaN. The shape depends on the input dimensions,
            :attr:`dim`, and :attr:`keepdim` parameters.

    Example:

        >>> # Simple element-wise product sum with NaN
        >>> x = torch.tensor([1.0, 2.0, nan, 4.0])
        >>> y = torch.tensor([2.0, nan, 4.0, 5.0])
        >>> QF.nanmulsum(x, y)
        tensor(22.)

        >>> # 2D tensors with NaN values
        >>> x = torch.tensor([[1.0, nan, 3.0],
        ...                   [4.0, 5.0, nan]])
        >>> y = torch.tensor([[2.0, 4.0, nan],
        ...                   [nan, 10.0, 12.0]])
        >>> QF.nanmulsum(x, y, dim=1)
        tensor([ 2., 50.])

        >>> # Broadcasting with NaN handling
        >>> x = torch.tensor([[1.0], [nan], [3.0]])
        >>> y = torch.tensor([2.0, 4.0, nan])
        >>> QF.nanmulsum(x, y)
        tensor(24.)

        >>> # With keepdim
        >>> x = torch.tensor([[1.0, nan], [3.0, 4.0]])
        >>> y = torch.tensor([[2.0, 4.0], [nan, 5.0]])
        >>> QF.nanmulsum(x, y, dim=1, keepdim=True)
        tensor([[ 2.],
                [20.]])

        >>> # All NaN pairs
        >>> x = torch.tensor([nan, nan])
        >>> y = torch.tensor([1.0, 2.0])
        >>> QF.nanmulsum(x, y)
        tensor(nan)

        >>> # Mixed valid and invalid pairs
        >>> x = torch.tensor([1.0, nan, 3.0, 4.0])
        >>> y = torch.tensor([2.0, 5.0, nan, 6.0])
        >>> QF.nanmulsum(x, y)
        tensor(26.)

    .. warning::
        If all pairs along a dimension contain at least one NaN value, the
        result for that dimension is NaN. This differs from standard summation
        where NaN values would propagate through the entire calculation.

    .. seealso::
        :func:`mulsum`: Memory-efficient product sum without NaN handling.
        :func:`nansum`: NaN-aware sum function.
        :func:`nanmulmean`: NaN-aware memory-efficient product mean.
    """
    result = mulsum(fillna(x), fillna(y), dim=dim, keepdim=keepdim)
    x_mask = (~x.isnan()).to(result)
    y_mask = (~y.isnan()).to(result)
    count = mulsum(x_mask, y_mask, dim=dim, keepdim=keepdim)
    return torch.where(count > 0, result, torch.as_tensor(math.nan).to(result))
