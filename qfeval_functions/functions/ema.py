import torch


def _exponential_weighted_sum(
    x: torch.Tensor, alpha: float, dim: int = -1
) -> torch.Tensor:
    r"""This returns :math:`ews[i]=\sum_{j=0}^{i}v[j]*(1-\alpha)^(i-j)` over a
    given dimension `dim`.

    NOTE: This uses a nature of a geometrical progression:
    :math:`a[i]/a[i-d]==(1-\alpha)^d`.  In each iteration, this calculates a
    geometrical progression of `2^i` elements internally for each original
    element.
    """

    # 1. Move the target dimension to the top to make data manipulation easier.
    x = x.transpose(0, dim)

    # 2. In i-th step, exponential weighted sum with the window size of
    # (2 ** i) is calculated.  This enables to calculate exponential weighted
    # sum in O(log n) where n is the length of the array.
    shift = 1
    decay = 1 - alpha
    while shift < len(x) and abs(decay) > 1e-8:
        x = torch.cat((x[:shift], x[shift:] + x[:-shift] * decay), dim=0)
        shift *= 2
        decay = decay**2

    # 3. Restore the order of dimensions.
    return x.transpose(0, dim)


def ema(x: torch.Tensor, alpha: float, dim: int = -1) -> torch.Tensor:
    r"""Compute exponential moving average along a specified dimension.

    This function calculates the exponential moving average (EMA) of a tensor
    along the specified dimension. EMA is a type of weighted moving average
    where more recent values have higher weights. The weight of each value
    decreases exponentially as you go back in time.

    For each position :math:`i` along the specified dimension, the EMA is
    computed as:

    .. math::
        \text{EMA}[i] = \frac{\sum_{j=0}^{i} x[j] \cdot (1-\alpha)^{i-j}}
                             {\sum_{j=0}^{i} (1-\alpha)^{i-j}}

    where :math:`\alpha` is the smoothing factor (0 < :math:`\alpha` < 1).
    Smaller values of :math:`\alpha` give more weight to recent observations.

    Args:
        x (Tensor):
            The input tensor containing values to be averaged.
        alpha (float):
            The smoothing factor, must be in the range (0, 1). Smaller values
            result in more smoothing (slower decay).
        dim (int, optional):
            The dimension along which to compute the exponential moving
            average.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the exponential
            moving average values.

    Example:

        >>> # Simple 1D example
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> ema_result = QF.ema(x, alpha=0.5)
        >>> ema_result
        tensor([1.0000, 1.6667, 2.4286, 3.2667, 4.1613])

        >>> # 2D example with dim=1
        >>> x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
        ...                   [4.0, 3.0, 2.0, 1.0]])
        >>> ema_result = QF.ema(x, alpha=0.3, dim=1)
        >>> ema_result
        tensor([[1.0000, 1.5882, 2.2329, 2.9305],
                [4.0000, 3.4118, 2.7671, 2.0695]])

    .. note::
        The implementation uses an efficient :math:`O(\log n)` algorithm based on
        geometric progression properties, making it suitable for long sequences.
    """
    ew_weight = _exponential_weighted_sum(
        torch.ones_like(x), alpha=alpha, dim=dim
    )
    ew_sum = _exponential_weighted_sum(x, alpha=alpha, dim=dim)
    return ew_sum / ew_weight
