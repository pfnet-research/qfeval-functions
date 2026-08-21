import math

import torch

from .apply_for_axis import apply_for_axis
from .mskew import _mmoments


def _mkurtosis(x: torch.Tensor, span: int) -> torch.Tensor:
    """Returns the moving sample excess kurtosis of the given tensor ``x``,
    whose shape is ``(B, N)``, along the 2nd dimension."""

    w = span
    m2, _, m4 = _mmoments(x, w, need_m4=True)
    result: torch.Tensor = (
        (w - 1) / ((w - 2) * (w - 3)) * (w * (w + 1) * m4 / m2**2 - 3 * (w - 1))
    )
    return result


def mkurtosis(x: torch.Tensor, span: int, dim: int = -1) -> torch.Tensor:
    r"""Compute the moving (sliding window) sample excess kurtosis of a
    tensor.

    This function calculates the excess kurtosis of elements within a
    sliding window of size :attr:`span` along the specified dimension. The
    output tensor has the same shape as the input tensor. For positions
    where the sliding window cannot fully cover preceding elements (i.e.,
    the first ``span - 1`` elements along the selected dimension), the
    result is ``nan``.

    The moving kurtosis is the bias-corrected sample excess kurtosis
    :math:`G_2` (0 for normally distributed data), the same estimator as
    ``pandas.DataFrame.rolling(span).kurt()`` and
    ``scipy.stats.kurtosis(..., bias=False, fisher=True)``.  With the
    windowed central moment sums

    .. math::
        M_k[i] = \sum_{j=i-\text{span}+1}^{i}
        \left(x[j] - \mu[i]\right)^k,
        \quad \mu[i] = \frac{1}{\text{span}}
        \sum_{j=i-\text{span}+1}^{i} x[j],

    it is defined as:

    .. math::
        \text{MKURT}[i] =
        \frac{\text{span} - 1}
        {\left(\text{span} - 2\right) \left(\text{span} - 3\right)}
        \left(
        \text{span} \left(\text{span} + 1\right)
        \frac{M_4[i]}{M_2[i]^2}
        - 3 \left(\text{span} - 1\right)
        \right)

    The moving kurtosis is computed in ``O(N)`` time independent of
    :attr:`span`, using cumulative central moment sums of
    :attr:`span`-sized chunks merged by the pairwise update formulas by
    Chan et al. and Terriberry (see `Algorithms for calculating variance
    <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance>`_).
    All sums are taken over deviations from nearby reference values, so the
    result is numerically stable even when the input has a large offset
    relative to its scale (e.g., values around ``1e6`` with a standard
    deviation of ``1e-3``).  See `A Numerically Stable and Fast
    Implementation of Moving Averages and Variances
    <https://imoz.jp/scraps/202607_mvar.en.html>`_ for a detailed
    description of the chunked cumulative-statistics algorithm, which this
    function extends to the 3rd and 4th central moments.

    Args:
        x (Tensor):
            The input tensor containing values.
        span (int):
            The size of the sliding window. If it is less than 4, the
            result is all-NaN because the bias correction is undefined.
        dim (int, optional):
            The dimension along which to compute the moving kurtosis.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the moving
            excess kurtosis values. The first ``span - 1`` elements along
            the specified dimension are ``nan``.

    Example:

        >>> # Simple moving kurtosis with window size 4
        >>> x = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
        >>> QF.mkurtosis(x, span=4)
        tensor([   nan,    nan,    nan, 0.7577, 0.7577])

        >>> # An outlier makes a window heavy-tailed (positive kurtosis)
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        >>> QF.mkurtosis(x, span=5)
        tensor([    nan,     nan,     nan,     nan, -1.2000,  4.9866])

        >>> # 2D tensor with moving kurtosis along rows
        >>> x = torch.tensor([[1.0, 2.0, 4.0, 8.0, 9.0],
        ...                   [0.0, 5.0, 6.0, 7.0, 12.0]])
        >>> QF.mkurtosis(x, span=4, dim=1)
        tensor([[    nan,     nan,     nan,  0.7577, -3.8690],
                [    nan,     nan,     nan,  2.7039,  2.7039]])

        >>> # Constant windows have no defined kurtosis
        >>> x = torch.tensor([2.0, 2.0, 2.0, 2.0, 3.0])
        >>> QF.mkurtosis(x, span=4)
        tensor([   nan,    nan,    nan,    nan, 4.0000])

    .. note::
        If a window contains any NaN value, the moving kurtosis for that
        window is NaN. Unlike ``pandas.DataFrame.rolling``, there is no
        ``min_periods``-style option to skip NaN values.  A window whose
        values are all (nearly) equal has no defined kurtosis because the
        standardization divides by a vanishing variance; the result is NaN
        (``pandas`` instead special-cases perfectly uniform windows to
        ``-3.0``).

    .. seealso::
        - :func:`mskew`: Moving skewness function (3rd standardized
          moment).
        - :func:`nankurtosis`: NaN-aware (non-moving) kurtosis function.
        - :func:`mvar`: Moving variance function, which this algorithm
          extends.
    """
    if span < 4:
        return torch.full_like(x, math.nan)
    return apply_for_axis(lambda x: _mkurtosis(x, span), x, dim)
