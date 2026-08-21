import torch

from .ewmcovar import ewmcovar
from .ewmvar import ewmvar


def ewmcorrel(
    x: torch.Tensor, y: torch.Tensor, alpha: float, dim: int = -1
) -> torch.Tensor:
    r"""Compute the exponentially weighted moving correlation of two tensors.

    This function calculates the Pearson correlation coefficient between
    :attr:`x` and :attr:`y` over all elements up to each position along
    the specified dimension, giving exponentially larger weights to more
    recent pairs.  It follows the ``adjust=True`` convention of pandas,
    so the result matches ``x.ewm(alpha=alpha, adjust=True).corr(y)`` for
    pandas series/frames.

    With the weights :math:`w_{ij} = (1 - \alpha)^{i-j}`, the correlation
    is the ratio of the biased exponentially weighted covariance to the
    product of the biased exponentially weighted standard deviations:

    .. math::
        \text{Corr}[i] =
        \frac{\text{Cov}_b[i]}
             {\sqrt{\text{Var}_b^x[i] \cdot \text{Var}_b^y[i]}}.

    The bias-correction factor of :func:`ewmcovar` and :func:`ewmvar`
    cancels between the numerator and the denominator, so the correlation
    has no ``bias`` argument.

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
            weighted moving correlation. Default is -1 (the last
            dimension).

    Returns:
        Tensor:
            A tensor of the broadcast shape of the inputs, containing the
            exponentially weighted moving correlation values.

    Example:

        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y = torch.tensor([2.0, 1.0, 4.0, 3.0, 6.0])
        >>> QF.ewmcorrel(x, y, alpha=0.5)
        tensor([    nan, -1.0000,  0.7856,  0.4845,  0.8512])

        >>> # A positive linear relationship gives a correlation of 1.
        >>> QF.ewmcorrel(x, 2 * x + 1, alpha=0.5)
        tensor([nan, 1., 1., 1., 1.])

        >>> # A negative linear relationship gives a correlation of -1.
        >>> QF.ewmcorrel(x, -x, alpha=0.5)
        tensor([nan, -1., -1., -1., -1.])

    .. note::
        The correlation of a single pair is undefined (0/0), so the first
        element along :attr:`dim` is NaN, exactly as in pandas
        ``ewm().corr()``.  If either input is constant so far, its
        variance is zero and the correlation is NaN as well.  Because of
        floating-point rounding, results may exceed the interval
        :math:`[-1, 1]` by a tiny margin.

    .. note::
        Like :func:`ema`, a NaN value in either input contaminates all
        subsequent outputs along :attr:`dim`.  This differs from pandas,
        which skips NaN values instead.

    .. seealso::
        - :func:`ewmcovar`: Exponentially weighted moving covariance.
        - :func:`ewmvar`: Exponentially weighted moving variance.
        - :func:`correl`: Pearson correlation over a whole dimension.
        - :func:`mcorrel`: Moving correlation over a fixed-size window.
    """
    x, y = torch.broadcast_tensors(x, y)
    cov = ewmcovar(x, y, alpha=alpha, dim=dim, bias=True)
    var_x = ewmvar(x, alpha=alpha, dim=dim, bias=True)
    var_y = ewmvar(y, alpha=alpha, dim=dim, bias=True)
    result: torch.Tensor = cov / (var_x * var_y).sqrt()
    return result
