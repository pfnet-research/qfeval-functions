import torch

from .mcovar import mcovar
from .mstd import mstd


def mcorrel(
    x: torch.Tensor, y: torch.Tensor, span: int, dim: int = -1
) -> torch.Tensor:
    r"""Compute the moving (sliding window) Pearson correlation of two
    tensors.

    This function calculates the Pearson correlation coefficient between
    elements of :attr:`x` and :attr:`y` within a sliding window of size
    :attr:`span` along the specified dimension.  The input tensors are
    broadcast to a common shape, and the output tensor has that shape.  For
    positions where the sliding window cannot fully cover preceding elements
    (i.e., the first ``span - 1`` elements along the selected dimension),
    the result is ``nan``.  This is compatible with
    ``pandas.DataFrame.rolling(span).corr(other)``.

    The moving correlation is computed using the formula:

    .. math::
        \text{MCORREL}[i] = \frac{
            \sum_{j=i-\text{span}+1}^{i}
            \left(x[j] - \mu_x[i]\right)\left(y[j] - \mu_y[i]\right)
        }{
            \sqrt{\sum_{j=i-\text{span}+1}^{i}
            \left(x[j] - \mu_x[i]\right)^2}
            \sqrt{\sum_{j=i-\text{span}+1}^{i}
            \left(y[j] - \mu_y[i]\right)^2}
        }

    where :math:`\mu_x[i]` and :math:`\mu_y[i]` are the means of ``x`` and
    ``y`` over the same window.  This is a convenience composition of
    :func:`mcovar` and :func:`mstd` (with a consistent ``ddof``, which
    cancels out), so it inherits their ``O(N)`` time complexity and their
    numerical stability against large offsets.

    Args:
        x (Tensor):
            The first input tensor.
        y (Tensor):
            The second input tensor. Must be broadcastable with :attr:`x`.
        span (int):
            The size of the sliding window. Must be positive.
        dim (int, optional):
            The dimension along which to compute the moving correlation.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the broadcast shape of the inputs, containing the
            moving correlation values. The first ``span - 1`` elements along
            the specified dimension are ``nan``.

    Example:

        >>> # Perfectly positively correlated series
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
        >>> QF.mcorrel(x, y, span=3)
        tensor([nan, nan, 1., 1., 1.])

        >>> # Perfectly negatively correlated series
        >>> y = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        >>> QF.mcorrel(x, y, span=3)
        tensor([nan, nan, -1., -1., -1.])

        >>> # Partially correlated series
        >>> y = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0])
        >>> QF.mcorrel(x, y, span=3)
        tensor([   nan,    nan, 0.5000, 0.6547, 0.6547])

    .. note::
        If either series is constant within a window (zero variance), the
        correlation is mathematically undefined and the result is NaN.
        Due to floating-point rounding, results may exceed the interval
        ``[-1, 1]`` by a tiny error.  If a window contains any NaN value in
        either input, the moving correlation for that window is NaN.
        Unlike ``pandas.DataFrame.rolling``, there is no
        ``min_periods``-style option to skip NaN values.

    .. seealso::
        - :func:`correl`: Pearson correlation over an entire dimension.
        - :func:`mcovar`: Moving covariance function.
        - :func:`nancorrel`: NaN-aware Pearson correlation function.
    """
    x, y = torch.broadcast_tensors(x, y)
    result: torch.Tensor = mcovar(x, y, span, dim=dim, ddof=0) / (
        mstd(x, span, dim=dim, ddof=0) * mstd(y, span, dim=dim, ddof=0)
    )
    return result
