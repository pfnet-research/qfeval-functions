import torch

from .ewmvar import ewmvar


def ewmstd(
    x: torch.Tensor, alpha: float, dim: int = -1, bias: bool = False
) -> torch.Tensor:
    r"""Compute the exponentially weighted moving standard deviation.

    This function calculates the standard deviation of all elements up to
    each position along the specified dimension, giving exponentially
    larger weights to more recent values.  It is the square root of
    :func:`ewmvar` and follows the ``adjust=True`` convention of pandas,
    so the result matches
    ``pandas.DataFrame.ewm(alpha=alpha, adjust=True).std(bias=bias)``.

    With the weights :math:`w_{ij} = (1 - \alpha)^{i-j}`, the result is:

    .. math::
        \text{Std}[i] = \sqrt{\text{Var}[i]},
        \qquad
        \text{Var}_b[i] =
        \frac{\sum_{j=0}^{i} w_{ij} (x[j] - \mu[i])^2}
             {\sum_{j=0}^{i} w_{ij}},

    where :math:`\mu[i]` is the exponentially weighted mean and, unless
    ``bias=True``, :math:`\text{Var}_b[i]` is corrected by the factor
    :math:`W_1[i]^2 / (W_1[i]^2 - W_2[i])` with
    :math:`W_1[i] = \sum_j w_{ij}` and :math:`W_2[i] = \sum_j w_{ij}^2`.

    Args:
        x (Tensor):
            The input tensor containing values.
        alpha (float):
            The smoothing factor, must be in the range (0, 1). Smaller
            values result in more smoothing (slower decay).
        dim (int, optional):
            The dimension along which to compute the exponentially
            weighted moving standard deviation. Default is -1 (the last
            dimension).
        bias (bool, optional):
            If True, take the square root of the biased weighted
            variance. If False (default), apply the same bias correction
            as pandas first.

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the
            exponentially weighted moving standard deviation values.

    Example:

        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> QF.ewmstd(x, alpha=0.5)
        tensor([   nan, 0.7071, 0.9636, 1.1772, 1.3452])

        >>> # The biased standard deviation starts at zero.
        >>> QF.ewmstd(x, alpha=0.5, bias=True)
        tensor([0.0000, 0.4714, 0.7284, 0.9286, 1.0805])

        >>> # 2D example along dim=0
        >>> x = torch.tensor([[1.0, 8.0],
        ...                   [2.0, 4.0],
        ...                   [3.0, 2.0]])
        >>> QF.ewmstd(x, alpha=0.5, dim=0)
        tensor([[   nan,    nan],
                [0.7071, 2.8284],
                [0.9636, 2.7255]])

    .. note::
        Since :func:`ewmvar` clamps tiny negative rounding results to
        zero, this function never produces spurious NaN values from the
        square root.  The first element along :attr:`dim` is NaN with
        ``bias=False`` (matching pandas ``ewm().std()``) and 0 with
        ``bias=True``.

    .. note::
        Like :func:`ema`, a NaN value contaminates all subsequent outputs
        along :attr:`dim`.  This differs from pandas, which skips NaN
        values instead.

    .. seealso::
        - :func:`ewmvar`: Exponentially weighted moving variance.
        - :func:`ewmcovar`: Exponentially weighted moving covariance.
        - :func:`ema`: Exponential moving average (the matching mean).
        - :func:`mstd`: Moving standard deviation over a fixed-size
          window.
    """
    result: torch.Tensor = ewmvar(x, alpha=alpha, dim=dim, bias=bias) ** 0.5
    return result
