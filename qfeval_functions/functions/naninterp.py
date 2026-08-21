import math

import torch

from .bfill import bfill
from .ffill import ffill


def naninterp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    r"""Linearly interpolate NaN values between valid values.

    This function replaces each NaN value lying strictly between two
    valid (non-NaN) values along the specified dimension with its linear
    interpolation.  For a NaN at position :math:`t` whose nearest valid
    neighbors are at position :math:`p` (before, with value :math:`v_p`)
    and position :math:`n` (after, with value :math:`v_n`), the result
    is:

    .. math::
        y[t] = v_p + (v_n - v_p) \cdot \frac{t - p}{n - p}

    Leading and trailing NaN values, which lack a valid neighbor on one
    side, remain NaN.  Valid values are returned unchanged.  This matches
    ``Series.interpolate(method="linear", limit_area="inside")`` of
    pandas and differs from pandas' default ``interpolate()``, which also
    forward-fills trailing NaN values.

    Args:
        x (Tensor):
            The input tensor containing values to be interpolated.  Must
            be a floating point tensor.
        dim (int, optional):
            The dimension along which to interpolate.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape as the input, with interior NaN
            values replaced by linear interpolations.  Leading and
            trailing NaN values remain NaN.

    Raises:
        TypeError: If ``x`` is not a floating point tensor.

    Example:

        >>> QF.naninterp(torch.tensor([0.0, nan, nan, 3.0]))
        tensor([0., 1., 2., 3.])

        >>> # Leading and trailing NaNs remain NaN.
        >>> QF.naninterp(torch.tensor([nan, 1.0, nan, 2.0, nan]))
        tensor([   nan, 1.0000, 1.5000, 2.0000,    nan])

        >>> # 2D example with dim=1
        >>> x = torch.tensor([[1.0, nan, 3.0],
        ...                   [10.0, nan, 20.0]])
        >>> QF.naninterp(x, dim=1)
        tensor([[ 1.,  2.,  3.],
                [10., 15., 20.]])

    .. note::
        Infinite values are treated as valid values and are kept as is.
        A gap adjacent to a ±inf neighbor is filled following IEEE 754
        arithmetic; e.g., interpolating between a finite value and inf
        yields inf or NaN depending on the direction.

    .. seealso::
        - :func:`ffill`: Forward fill missing values.
        - :func:`bfill`: Backward fill missing values.
        - :func:`fillna`: Replace NaN/infinity values with fixed numbers.
        - :func:`nanema`: Exponential moving average skipping NaN values.
    """
    if not x.is_floating_point():
        raise TypeError(
            "naninterp only supports floating point tensors, "
            f"but got {x.dtype}."
        )

    # 1. Build positions along `dim`, broadcastable against `x`.
    shape = [1] * x.dim()
    shape[dim] = x.shape[dim]
    t = torch.arange(x.shape[dim], dtype=x.dtype, device=x.device)
    t = t.reshape(shape)

    # 2. Find the positions and values of the previous/next valid values.
    nan_value = torch.as_tensor(math.nan).to(x)
    position = torch.where(x.isnan(), nan_value, t)
    prev_position = ffill(position, dim)
    next_position = bfill(position, dim)
    prev_value = ffill(x, dim)
    next_value = bfill(x, dim)

    # 3. Interpolate linearly.  Where a valid neighbor is missing (i.e.,
    # leading/trailing NaNs), the neighbor position is NaN, so the result
    # remains NaN.  At valid positions, the weight may be 0/0 = NaN, but
    # those positions take `x` directly below.
    weight = (t - prev_position) / (next_position - prev_position)
    interp = prev_value + (next_value - prev_value) * weight
    return torch.where(x.isnan(), interp, x)
