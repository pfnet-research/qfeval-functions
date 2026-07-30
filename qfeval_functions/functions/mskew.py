import math
import typing

import torch

from .apply_for_axis import apply_for_axis
from .rcumsum import rcumsum


def _mmoments(
    x: torch.Tensor, span: int, need_m4: bool
) -> typing.Tuple[torch.Tensor, ...]:
    """Returns the moving central moment sums ``(M2, M3)`` of the given
    tensor ``x``, whose shape is ``(B, N)``, along the 2nd dimension,
    followed by ``M4`` if ``need_m4`` is set.

    This extends the chunked cumulative-statistics algorithm of ``_mvar``
    (https://imoz.jp/scraps/202607_mvar.en.html) to the 3rd and 4th central
    moments.  As in ``_mvar``, each part (a chunk prefix or suffix) keeps
    ``n`` (count), ``r`` (local reference value), the power sums ``S1..S4``
    of local deviations ``y = x - r``, ``m_bar`` (relative mean ``S1 / n``)
    and ``Mk`` (central moment sums ``Sum((x - mu)^k)``, with ``mu = r +
    m_bar``).  Parts are merged with the pairwise update formulas by Chan et
    al. and Terriberry (see "Algorithms for calculating variance",
    Wikipedia).
    """

    w = span

    # 1. Reshape the target dimension into length-`w` chunks with prepended
    # NaNs (the same chunk layout as `_mvar`).
    pad_len = w * 2 - x.shape[1] % w
    x = torch.nn.functional.pad(x, (pad_len, 0), value=math.nan)
    x = x.reshape((x.shape[0], x.shape[1] // w, w))

    # 2. Compute the per-part statistics for every chunk prefix (part B) and
    # chunk suffix (part A), locally centered on a reference value `r` (the
    # chunk's first element for a prefix, its last element for a suffix) to
    # avoid catastrophic cancellation (see `_mvar`).  The central moment
    # sums follow from the power sums of `y = x - r`:
    # `M2 = S2 - S1 * m_bar`,
    # `M3 = S3 - 3 * m_bar * S2 + 2 * m_bar^2 * S1`, and
    # `M4 = S4 - 4 * m_bar * S3 + 6 * m_bar^2 * S2 - 3 * m_bar^3 * S1`.
    #
    # Prefixes (part B): counts n = 1..w, reference r_p = first chunk element.
    n_p = torch.arange(1, w + 1, dtype=x.dtype, device=x.device)
    r_p = x[:, :, :1]
    y_p = x - r_p
    S1_p = y_p.cumsum(dim=2)
    S2_p = (y_p**2).cumsum(dim=2)
    S3_p = (y_p**3).cumsum(dim=2)
    m_bar_p = S1_p / n_p
    M2_p = S2_p - S1_p * m_bar_p
    M3_p = S3_p - 3 * m_bar_p * S2_p + 2 * m_bar_p**2 * S1_p
    if need_m4:
        S4_p = (y_p**4).cumsum(dim=2)
        M4_p = (
            S4_p
            - 4 * m_bar_p * S3_p
            + 6 * m_bar_p**2 * S2_p
            - 3 * m_bar_p**3 * S1_p
        )
    #
    # Suffixes (part A): counts n = w..1, reference r_s = last chunk element.
    n_s = torch.arange(w, 0, -1, dtype=x.dtype, device=x.device)
    r_s = x[:, :, -1:]
    y_s = x - r_s
    S1_s = rcumsum(y_s, dim=2)
    S2_s = rcumsum(y_s**2, dim=2)
    S3_s = rcumsum(y_s**3, dim=2)
    m_bar_s = S1_s / n_s
    M2_s = S2_s - S1_s * m_bar_s
    M3_s = S3_s - 3 * m_bar_s * S2_s + 2 * m_bar_s**2 * S1_s
    if need_m4:
        S4_s = rcumsum(y_s**4, dim=2)
        M4_s = (
            S4_s
            - 4 * m_bar_s * S3_s
            + 6 * m_bar_s**2 * S2_s
            - 3 * m_bar_s**3 * S1_s
        )

    # 3. Each window is a suffix of the previous chunk (part A, count `n_a`)
    # followed by a prefix of the current chunk (part B, count `n_b`); an
    # aligned window is just a whole chunk prefix.  Merge the two parts with
    # the pairwise update formulas by Chan et al. and Terriberry, where the
    # mean gap `delta = mu_B - mu_A` never forms a large offset because the
    # means are stored relative to their references (see `_mvar`).  With
    # `n_a + n_b = w`:
    # `M2 = M2_A + M2_B + delta^2 * n_a * n_b / w`,
    # `M3 = M3_A + M3_B + delta^3 * n_a * n_b * (n_a - n_b) / w^2
    #       + 3 * delta * (n_a * M2_B - n_b * M2_A) / w`, and
    # `M4 = M4_A + M4_B
    #       + delta^4 * n_a * n_b * (n_a^2 - n_a * n_b + n_b^2) / w^3
    #       + 6 * delta^2 * (n_a^2 * M2_B + n_b^2 * M2_A) / w^2
    #       + 4 * delta * (n_a * M3_B - n_b * M3_A) / w`.
    delta = (r_p[:, 1:] - r_s[:, :-1]) + (
        m_bar_p[:, 1:, :-1] - m_bar_s[:, :-1, 1:]
    )
    n_a, n_b = n_s[1:], n_p[:-1]
    M2_a, M2_b = M2_s[:, :-1, 1:], M2_p[:, 1:, :-1]
    M3_a, M3_b = M3_s[:, :-1, 1:], M3_p[:, 1:, :-1]
    M2 = M2_a + M2_b + delta**2 * (n_a * n_b / w)
    M3 = (
        M3_a
        + M3_b
        + delta**3 * (n_a * n_b * (n_a - n_b) / w**2)
        + 3 * delta * (n_a * M2_b - n_b * M2_a) / w
    )
    moments = [(M2, M2_p), (M3, M3_p)]
    if need_m4:
        M4 = (
            M4_s[:, :-1, 1:]
            + M4_p[:, 1:, :-1]
            + delta**4 * (n_a * n_b * (n_a**2 - n_a * n_b + n_b**2) / w**3)
            + 6 * delta**2 * (n_a**2 * M2_b + n_b**2 * M2_a) / w**2
            + 4 * delta * (n_a * M3_b - n_b * M3_a) / w
        )
        moments.append((M4, M4_p))

    # 4. Append the aligned windows (whole chunk prefixes), restore the
    # original layout, and drop the columns coming from the padding.
    return tuple(
        torch.cat((m, m_p[:, 1:, -1:]), dim=2).flatten(start_dim=1)[
            :, pad_len - w :
        ]
        for m, m_p in moments
    )


def _mskew(x: torch.Tensor, span: int) -> torch.Tensor:
    """Returns the moving sample skewness of the given tensor ``x``, whose
    shape is ``(B, N)``, along the 2nd dimension."""

    w = span
    m2, m3 = _mmoments(x, w, need_m4=False)
    g1 = math.sqrt(w) * m3 / m2**1.5
    result: torch.Tensor = math.sqrt(w * (w - 1)) / (w - 2) * g1
    return result


def mskew(x: torch.Tensor, span: int, dim: int = -1) -> torch.Tensor:
    r"""Compute the moving (sliding window) sample skewness of a tensor.

    This function calculates the skewness of elements within a sliding
    window of size :attr:`span` along the specified dimension. The output
    tensor has the same shape as the input tensor. For positions where the
    sliding window cannot fully cover preceding elements (i.e., the first
    ``span - 1`` elements along the selected dimension), the result is
    ``nan``.

    The moving skewness is the adjusted Fisher-Pearson standardized moment
    coefficient :math:`G_1`, the same estimator as
    ``pandas.DataFrame.rolling(span).skew()`` and
    ``scipy.stats.skew(..., bias=False)``.  With the windowed central
    moment sums

    .. math::
        M_k[i] = \sum_{j=i-\text{span}+1}^{i}
        \left(x[j] - \mu[i]\right)^k,
        \quad \mu[i] = \frac{1}{\text{span}}
        \sum_{j=i-\text{span}+1}^{i} x[j],

    it is defined as:

    .. math::
        \text{MSKEW}[i] =
        \frac{\sqrt{\text{span} \left(\text{span} - 1\right)}}
        {\text{span} - 2}
        \cdot \frac{\sqrt{\text{span}} \, M_3[i]}{M_2[i]^{3/2}}

    The moving skewness is computed in ``O(N)`` time independent of
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
    function extends to the 3rd central moment.

    Args:
        x (Tensor):
            The input tensor containing values.
        span (int):
            The size of the sliding window. If it is less than 3, the
            result is all-NaN because the bias correction is undefined.
        dim (int, optional):
            The dimension along which to compute the moving skewness.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the moving
            skewness values. The first ``span - 1`` elements along the
            specified dimension are ``nan``.

    Example:

        >>> # Simple moving skewness with window size 3
        >>> x = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
        >>> QF.mskew(x, span=3)
        tensor([   nan,    nan, 0.9352, 0.9352, 0.9352])

        >>> # Symmetric windows have zero skewness
        >>> x = torch.tensor([3.0, 1.0, 2.0, 3.0, 1.0, 2.0])
        >>> QF.mskew(x, span=3)
        tensor([nan, nan, 0., 0., 0., 0.])

        >>> # 2D tensor with moving skewness along rows
        >>> x = torch.tensor([[1.0, 2.0, 4.0, 8.0],
        ...                   [1.0, 4.0, 5.0, 6.0]])
        >>> QF.mskew(x, span=3, dim=1)
        tensor([[    nan,     nan,  0.9352,  0.9352],
                [    nan,     nan, -1.2933,  0.0000]])

        >>> # Constant windows have no defined skewness
        >>> x = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0])
        >>> QF.mskew(x, span=4)
        tensor([nan, nan, nan, nan, 2.])

    .. note::
        If a window contains any NaN value, the moving skewness for that
        window is NaN. Unlike ``pandas.DataFrame.rolling``, there is no
        ``min_periods``-style option to skip NaN values.  A window whose
        values are all (nearly) equal has no defined skewness because the
        standardization divides by a vanishing variance; the result is NaN
        (``pandas`` instead special-cases perfectly uniform windows to
        ``0.0``).

    .. seealso::
        - :func:`mkurtosis`: Moving kurtosis function (4th standardized
          moment).
        - :func:`nanskew`: NaN-aware (non-moving) skewness function.
        - :func:`mvar`: Moving variance function, which this algorithm
          extends.
    """
    if span < 3:
        return torch.full_like(x, math.nan)
    return apply_for_axis(lambda x: _mskew(x, span), x, dim)
