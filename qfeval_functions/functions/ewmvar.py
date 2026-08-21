import torch

from .ema import _exponential_weighted_sum
from .nanmean import nanmean


def ewmvar(
    x: torch.Tensor, alpha: float, dim: int = -1, bias: bool = False
) -> torch.Tensor:
    r"""Compute the exponentially weighted moving variance of a tensor.

    This function calculates the variance of all elements up to each
    position along the specified dimension, giving exponentially larger
    weights to more recent values.  It follows the ``adjust=True``
    convention of pandas, so the result matches
    ``pandas.DataFrame.ewm(alpha=alpha, adjust=True).var(bias=bias)``.

    At position :math:`i`, each value :math:`x[j]` (:math:`j \le i`)
    contributes with the weight :math:`w_{ij} = (1 - \alpha)^{i-j}`, and
    the biased variance is the weighted average of squared deviations
    around the weighted mean :math:`\mu[i]`:

    .. math::
        \mu[i] = \frac{\sum_{j=0}^{i} w_{ij} x[j]}{W_1[i]},
        \qquad
        \text{Var}_b[i] =
        \frac{\sum_{j=0}^{i} w_{ij} (x[j] - \mu[i])^2}{W_1[i]},

    where :math:`W_1[i] = \sum_{j=0}^{i} w_{ij}`.  With ``bias=False``
    (the default), the result is corrected for the effective sample size
    implied by the weights:

    .. math::
        \text{Var}[i] = \text{Var}_b[i] \cdot
        \frac{W_1[i]^2}{W_1[i]^2 - W_2[i]},
        \qquad
        W_2[i] = \sum_{j=0}^{i} w_{ij}^2.

    Args:
        x (Tensor):
            The input tensor containing values.
        alpha (float):
            The smoothing factor, must be in the range (0, 1]. Smaller
            values result in more smoothing (slower decay).
        dim (int, optional):
            The dimension along which to compute the exponentially
            weighted moving variance. Default is -1 (the last dimension).
        bias (bool, optional):
            If True, return the biased (population-style) weighted
            variance. If False (default), apply the same bias correction
            as pandas.

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the
            exponentially weighted moving variance values.

    Raises:
        ValueError: If ``alpha`` does not satisfy ``0 < alpha <= 1``.

    Example:

        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> QF.ewmvar(x, alpha=0.5)
        tensor([   nan, 0.5000, 0.9286, 1.3857, 1.8097])

        >>> # The biased variance defines the first element as zero.
        >>> QF.ewmvar(x, alpha=0.5, bias=True)
        tensor([0.0000, 0.2222, 0.5306, 0.8622, 1.1675])

        >>> # 2D example along dim=1
        >>> x = torch.tensor([[1.0, 2.0, 4.0, 8.0],
        ...                   [1.0, 1.0, 1.0, 1.0]])
        >>> QF.ewmvar(x, alpha=0.5, dim=1)
        tensor([[    nan,  0.5000,  2.5000, 11.0714],
                [    nan,  0.0000,  0.0000,  0.0000]])

    .. note::
        With ``bias=False``, the variance of a single observation is
        undefined (:math:`W_1^2 - W_2 = 0`), so the first element along
        :attr:`dim` is NaN, exactly as in pandas ``ewm().var()``.  With
        ``bias=True``, the first element is 0.

    .. note::
        For numerical stability, the input is centered on its NaN-aware
        mean (:func:`nanmean`) before any moment is accumulated; the
        variance is invariant to such a shift.  This keeps the result
        accurate even for values with a large offset relative to their
        variance (e.g., values around ``1e6`` with a variance of ``1``),
        though extremely long drifting series may still lose some
        precision.  Tiny negative results caused by rounding are clamped
        to zero.

    .. note::
        Like :func:`ema`, a NaN value contaminates all subsequent outputs
        along :attr:`dim`.  This differs from pandas, which skips NaN
        values instead.  (:func:`nanema` is the NaN-skipping counterpart
        of :func:`ema`.)

    .. seealso::
        - :func:`ewmstd`: Exponentially weighted moving standard
          deviation (square root of this).
        - :func:`ewmcovar`: Exponentially weighted moving covariance.
        - :func:`ema`: Exponential moving average (the matching mean).
        - :func:`mvar`: Moving variance over a fixed-size window.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must satisfy 0 < alpha <= 1, but got {alpha}.")
    y = x - nanmean(x, dim=dim, keepdim=True)
    ones = torch.ones_like(y)
    w1 = _exponential_weighted_sum(ones, alpha=alpha, dim=dim)
    mean = _exponential_weighted_sum(y, alpha=alpha, dim=dim) / w1
    sq_sum = _exponential_weighted_sum(y * y, alpha=alpha, dim=dim)
    var = (sq_sum / w1 - mean**2).clamp_min(0)
    if bias:
        return var
    w2 = _exponential_weighted_sum(ones, alpha=alpha * (2 - alpha), dim=dim)
    result: torch.Tensor = var * w1**2 / (w1**2 - w2)
    return result
