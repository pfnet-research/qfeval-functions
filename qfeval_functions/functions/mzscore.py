import torch

from .ma import ma
from .mstd import mstd


def mzscore(
    x: torch.Tensor, span: int, dim: int = -1, ddof: int = 1
) -> torch.Tensor:
    r"""Compute the moving (sliding window) z-score of a tensor.

    This function standardizes each element against its own trailing window:
    it subtracts the moving average over the ``span`` elements ending at the
    position and divides by the moving standard deviation of the same
    window.  The output tensor has the same shape as the input tensor.  For
    positions where the sliding window cannot fully cover preceding elements
    (i.e., the first ``span - 1`` elements along the selected dimension),
    the result is ``nan``.  This is equivalent to the pandas expression
    ``(df - df.rolling(span).mean()) / df.rolling(span).std(ddof=ddof)``.

    The moving z-score is computed using the formula:

    .. math::
        \text{MZSCORE}[i] = \frac{x[i] - \mu[i]}{\sigma[i]}

    where :math:`\mu[i]` is the moving average and :math:`\sigma[i]` is the
    moving standard deviation of the window ending at position :math:`i`.
    This is a convenience composition of :func:`ma` and :func:`mstd`, so it
    runs in ``O(N)`` time independent of :attr:`span`.  Note that, unlike
    :func:`mstd` itself, the numerator :math:`x[i] - \mu[i]` cancels values
    of the input's magnitude, so the absolute precision of the z-score is
    bounded by the floating-point resolution of inputs with a large offset
    relative to their standard deviation.

    Args:
        x (Tensor):
            The input tensor containing values.
        span (int):
            The size of the sliding window. Must be positive.
        dim (int, optional):
            The dimension along which to compute the moving z-score.
            Default is -1 (the last dimension).
        ddof (int, optional):
            Delta degrees of freedom used for the moving standard deviation.
            The divisor used in the variance calculation is
            ``span - ddof``. Must be less than ``span``; otherwise the
            result is NaN. Default is 1 (sample standard deviation).

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the moving
            z-score values. The first ``span - 1`` elements along the
            specified dimension are ``nan``.

    Example:

        >>> # A linear ramp has a constant moving z-score
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> QF.mzscore(x, span=3)
        tensor([nan, nan, 1., 1., 1.])

        >>> # Population statistics (ddof=0)
        >>> QF.mzscore(x, span=3, ddof=0)
        tensor([   nan,    nan, 1.2247, 1.2247, 1.2247])

        >>> # The latest element is standardized against its own window
        >>> x = torch.tensor([1.0, 2.0, 4.0, 0.0])
        >>> QF.mzscore(x, span=2)
        tensor([    nan,  0.7071,  0.7071, -0.7071])

        >>> # A constant window yields NaN
        >>> x = torch.tensor([1.0, 3.0, 3.0, 3.0])
        >>> QF.mzscore(x, span=3)
        tensor([   nan,    nan, 0.5774,    nan])

    .. note::
        If a window is constant (zero standard deviation), the moving
        z-score is undefined and the result is NaN; in particular,
        ``span=1`` yields NaN everywhere.  If a window contains any NaN
        value, the moving z-score for that window is NaN.  Unlike
        ``pandas.DataFrame.rolling``, there is no ``min_periods``-style
        option to skip NaN values.

    .. seealso::
        - :func:`ma`: Moving average function.
        - :func:`mstd`: Moving standard deviation function.
        - :func:`zscore`: Cross-sectional z-score over an entire dimension.
    """
    result: torch.Tensor = (x - ma(x, span, dim=dim)) / mstd(
        x, span, dim=dim, ddof=ddof
    )
    return result
