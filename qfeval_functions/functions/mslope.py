import torch

from .mcovar import mcovar
from .mvar import mvar


def mslope(
    x: torch.Tensor, y: torch.Tensor, span: int, dim: int = -1
) -> torch.Tensor:
    r"""Compute the moving (sliding window) slope of a simple linear
    regression.

    This function calculates, for each sliding window of size :attr:`span`
    along the specified dimension, the ordinary least squares slope
    coefficient :math:`\beta` of the regression :math:`y = \alpha + \beta x`
    over the window, where :attr:`x` holds the explanatory (independent)
    values and :attr:`y` holds the response (dependent) values.  The input
    tensors are broadcast to a common shape, and the output tensor has that
    shape.  For positions where the sliding window cannot fully cover
    preceding elements (i.e., the first ``span - 1`` elements along the
    selected dimension), the result is ``nan``.  This is compatible with
    ``pandas.DataFrame.rolling(span).cov(other)`` divided by
    ``pandas.DataFrame.rolling(span).var()``.

    The moving slope is computed using the formula:

    .. math::
        \text{MSLOPE}[i] = \frac{
            \sum_{j=i-\text{span}+1}^{i}
            \left(x[j] - \mu_x[i]\right)\left(y[j] - \mu_y[i]\right)
        }{
            \sum_{j=i-\text{span}+1}^{i} \left(x[j] - \mu_x[i]\right)^2
        }

    where :math:`\mu_x[i]` and :math:`\mu_y[i]` are the means of ``x`` and
    ``y`` over the same window.  This is a convenience composition of
    :func:`mcovar` and :func:`mvar` (with a consistent ``ddof``, which
    cancels out), so it inherits their ``O(N)`` time complexity and their
    numerical stability against large offsets.

    In quantitative finance, this is used, for example, to estimate the
    rolling beta of an asset's returns against market returns.

    Args:
        x (Tensor):
            The input tensor of explanatory (independent) values.
        y (Tensor):
            The input tensor of response (dependent) values. Must be
            broadcastable with :attr:`x`.
        span (int):
            The size of the sliding window. Must be positive.
        dim (int, optional):
            The dimension along which to compute the moving slope.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the broadcast shape of the inputs, containing the
            moving slope values. The first ``span - 1`` elements along the
            specified dimension are ``nan``.

    Example:

        >>> # A perfectly linear relationship: y = 2 * x + 3
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y = torch.tensor([5.0, 7.0, 9.0, 11.0, 13.0])
        >>> QF.mslope(x, y, span=3)
        tensor([nan, nan, 2., 2., 2.])

        >>> # A noisy relationship
        >>> y = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0])
        >>> QF.mslope(x, y, span=3)
        tensor([   nan,    nan, 0.5000, 1.0000, 1.0000])

    .. note::
        If :attr:`x` is constant within a window (zero variance), the slope
        is mathematically undefined and the result is non-finite (NaN or
        infinity).  If a window contains any NaN value in either input, the
        moving slope for that window is NaN.  Unlike
        ``pandas.DataFrame.rolling``, there is no ``min_periods``-style
        option to skip NaN values.

    .. seealso::
        - :func:`slope`: Simple linear regression slope over an entire
          dimension.
        - :func:`nanslope`: NaN-aware slope function.
        - :func:`mcovar`: Moving covariance function.
        - :func:`mcorrel`: Moving Pearson correlation function.
    """
    x, y = torch.broadcast_tensors(x, y)
    result: torch.Tensor = mcovar(x, y, span, dim=dim, ddof=0) / mvar(
        x, span, dim=dim, ddof=0
    )
    return result
