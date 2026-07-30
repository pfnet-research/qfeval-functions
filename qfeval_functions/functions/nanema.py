import math

import torch

from .ema import _exponential_weighted_sum


def nanema(x: torch.Tensor, alpha: float, dim: int = -1) -> torch.Tensor:
    r"""Compute exponential moving average, skipping NaN values.

    This function calculates the exponential moving average (EMA) of a
    tensor along the specified dimension while ignoring NaN values.  At
    each valid (non-NaN) position :math:`i`, only the valid values are
    averaged, but the weight of each value still decays with the absolute
    distance in positions (i.e., calendar time), not with the number of
    valid values in between:

    .. math::
        \text{nanema}[i] = \frac
            {\sum_{j \le i,\ x[j]\ \text{valid}} x[j] \cdot
                (1-\alpha)^{i-j}}
            {\sum_{j \le i,\ x[j]\ \text{valid}} (1-\alpha)^{i-j}}

    At valid positions, this matches
    ``Series.ewm(alpha=alpha, adjust=True, ignore_na=False).mean()`` of
    pandas.  Unlike pandas, which carries the last mean forward at NaN
    positions, this function returns NaN there, preserving the
    missingness of the input.  Positions before the first valid value are
    also NaN.  On NaN-free input, this function is identical to
    :func:`ema`.

    Args:
        x (Tensor):
            The input tensor containing values to be averaged.  It may
            contain NaN values, which are skipped.
        alpha (float):
            The smoothing factor, must be in the range (0, 1).  Smaller
            values result in more smoothing (slower decay).
        dim (int, optional):
            The dimension along which to compute the exponential moving
            average.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the
            exponential moving average of the valid values.  NaN input
            positions and positions before the first valid value are NaN.

    Example:

        >>> x = torch.tensor([1.0, nan, 3.0, 4.0])
        >>> QF.nanema(x, alpha=0.5)
        tensor([1.0000,    nan, 2.6000, 3.4615])

        >>> # Positions before the first valid value remain NaN.
        >>> QF.nanema(torch.tensor([nan, nan, 2.0, nan]), alpha=0.5)
        tensor([nan, nan, 2., nan])

        >>> # 2D example with dim=1
        >>> x = torch.tensor([[1.0, nan, 3.0],
        ...                   [4.0, 5.0, nan]])
        >>> QF.nanema(x, alpha=0.5, dim=1)
        tensor([[1.0000,    nan, 2.6000],
                [4.0000, 4.6667,    nan]])

    .. note::
        This function shares the efficient doubling-based algorithm of
        :func:`ema` (:math:`O(\log n)` doubling steps along the
        dimension).  Infinite values are treated as valid values: once a
        ±inf value enters the weighted sums, subsequent outputs become
        inf (or NaN if infinities of both signs are mixed).

    .. seealso::
        - :func:`ema`: Exponential moving average without NaN handling.
        - :func:`nanmean`: Mean of valid values.
        - :func:`fillna`: Replace NaN/infinity values with fixed numbers.
        - :func:`naninterp`: Linear interpolation of NaN values.
    """
    valid = ~x.isnan()
    # NOTE: torch.where is used instead of nan_to_num to preserve ±inf.
    filled = torch.where(valid, x, torch.zeros_like(x))
    num = _exponential_weighted_sum(filled, alpha=alpha, dim=dim)
    den = _exponential_weighted_sum(valid.to(x), alpha=alpha, dim=dim)
    # NOTE: den is 0 only where no valid value has appeared yet, so 0/0
    # yields NaN there; at any valid position, den >= 1.
    return torch.where(valid, num / den, torch.as_tensor(math.nan).to(x))
