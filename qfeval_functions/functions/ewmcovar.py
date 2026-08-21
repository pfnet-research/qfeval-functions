import torch

from .ema import _exponential_weighted_sum
from .nanmean import nanmean


def ewmcovar(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    dim: int = -1,
    bias: bool = False,
) -> torch.Tensor:
    r"""Compute the exponentially weighted moving covariance of two tensors.

    This function calculates the covariance between :attr:`x` and
    :attr:`y` over all elements up to each position along the specified
    dimension, giving exponentially larger weights to more recent pairs.
    It follows the ``adjust=True`` convention of pandas, so the result
    matches ``x.ewm(alpha=alpha, adjust=True).cov(y, bias=bias)`` for
    pandas series/frames.

    At position :math:`i`, each pair :math:`(x[j], y[j])` contributes
    with the weight :math:`w_{ij} = (1 - \alpha)^{i-j}`, and the biased
    covariance is:

    .. math::
        \text{Cov}_b[i] =
        \frac{\sum_{j=0}^{i}
              w_{ij} (x[j] - \mu_x[i]) (y[j] - \mu_y[i])}{W_1[i]},

    where :math:`\mu_x[i]` and :math:`\mu_y[i]` are the exponentially
    weighted means of :attr:`x` and :attr:`y`, and
    :math:`W_1[i] = \sum_{j=0}^{i} w_{ij}`.  With ``bias=False`` (the
    default), the result is corrected for the effective sample size
    implied by the weights:

    .. math::
        \text{Cov}[i] = \text{Cov}_b[i] \cdot
        \frac{W_1[i]^2}{W_1[i]^2 - W_2[i]},
        \qquad
        W_2[i] = \sum_{j=0}^{i} w_{ij}^2.

    Args:
        x (Tensor):
            The first input tensor.
        y (Tensor):
            The second input tensor. Must be broadcastable with
            :attr:`x`.
        alpha (float):
            The smoothing factor, must be in the range (0, 1). Smaller
            values result in more smoothing (slower decay).
        dim (int, optional):
            The dimension along which to compute the exponentially
            weighted moving covariance. Default is -1 (the last
            dimension).
        bias (bool, optional):
            If True, return the biased (population-style) weighted
            covariance. If False (default), apply the same bias
            correction as pandas.

    Returns:
        Tensor:
            A tensor of the broadcast shape of the inputs, containing the
            exponentially weighted moving covariance values.

    Example:

        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y = torch.tensor([2.0, 1.0, 4.0, 3.0, 6.0])
        >>> QF.ewmcovar(x, y, alpha=0.5)
        tensor([    nan, -0.5000,  1.3571,  0.6714,  2.3710])

        >>> QF.ewmcovar(x, y, alpha=0.5, bias=True)
        tensor([ 0.0000, -0.2222,  0.7755,  0.4178,  1.5297])

        >>> # The covariance of a tensor with itself is its variance.
        >>> QF.ewmcovar(x, x, alpha=0.5)
        tensor([   nan, 0.5000, 0.9286, 1.3857, 1.8097])

        >>> # Negating one input negates the covariance.
        >>> QF.ewmcovar(x, -x, alpha=0.5)
        tensor([    nan, -0.5000, -0.9286, -1.3857, -1.8097])

    .. note::
        With ``bias=False``, the covariance of a single pair is undefined
        (:math:`W_1^2 - W_2 = 0`), so the first element along :attr:`dim`
        is NaN, exactly as in pandas ``ewm().cov()``.  With ``bias=True``,
        the first element is 0.

    .. note::
        For numerical stability, each input is centered on its own
        NaN-aware mean (:func:`nanmean`) before any moment is
        accumulated; the covariance is invariant to such shifts.  This
        keeps the result accurate even for values with large offsets
        relative to their variability, though extremely long drifting
        series may still lose some precision.  Unlike :func:`ewmvar`, no
        clamping is applied because a covariance may be negative.

    .. note::
        Like :func:`ema`, a NaN value in either input contaminates all
        subsequent outputs along :attr:`dim`.  This differs from pandas,
        which skips NaN values instead.

    .. seealso::
        - :func:`ewmcorrel`: Exponentially weighted moving correlation.
        - :func:`ewmvar`: Exponentially weighted moving variance.
        - :func:`covar`: Covariance over a whole dimension.
    """
    x, y = torch.broadcast_tensors(x, y)
    cx = x - nanmean(x, dim=dim, keepdim=True)
    cy = y - nanmean(y, dim=dim, keepdim=True)
    ones = torch.ones_like(cx)
    w1 = _exponential_weighted_sum(ones, alpha=alpha, dim=dim)
    mean_x = _exponential_weighted_sum(cx, alpha=alpha, dim=dim) / w1
    mean_y = _exponential_weighted_sum(cy, alpha=alpha, dim=dim) / w1
    xy_sum = _exponential_weighted_sum(cx * cy, alpha=alpha, dim=dim)
    cov = xy_sum / w1 - mean_x * mean_y
    if bias:
        return cov
    w2 = _exponential_weighted_sum(ones, alpha=alpha * (2 - alpha), dim=dim)
    result: torch.Tensor = cov * w1**2 / (w1**2 - w2)
    return result
