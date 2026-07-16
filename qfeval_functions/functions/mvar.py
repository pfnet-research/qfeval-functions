import math

import torch

from .apply_for_axis import apply_for_axis
from .rcumsum import rcumsum


def _mvar(x: torch.Tensor, span: int, ddof: int) -> torch.Tensor:
    """Returns the moving variance of the given tensor ``x``, whose shape is
    ``(B, N)``, along the 2nd dimension.

    This follows the chunked cumulative-statistics algorithm described in
    https://imoz.jp/scraps/202607_mvar.en.html .  The notation below matches
    the article: ``w`` is the window size, and each part (a chunk prefix or
    suffix) keeps ``n`` (count), ``r`` (local reference value), ``S`` (sum of
    local deviations ``y = x - r``), ``m_bar`` (relative mean ``S / n``) and
    ``M`` (sum of squared deviations ``Sum(x - mu)^2``, with ``mu = r +
    m_bar``).
    """

    w = span

    # 1. Reshape the target dimension into length-`w` chunks with prepended
    # NaNs (the same chunk layout as `msum`).
    pad_len = w * 2 - x.shape[1] % w
    x = torch.nn.functional.pad(x, (pad_len, 0), value=math.nan)
    x = x.reshape((x.shape[0], x.shape[1] // w, w))

    # 2. Compute the per-part statistics `(n, S, m_bar, M)` for every chunk
    # prefix (part B) and chunk suffix (part A).  Each part is locally
    # centered on a reference value `r` (the chunk's first element for a
    # prefix, its last element for a suffix), so that `S`, `m_bar` and `M`
    # are all accumulated over small local deviations `y = x - r`.  This
    # avoids catastrophic cancellation, and lets NaN/inf contaminate exactly
    # the parts that contain them.
    #
    # Prefixes (part B): counts n = 1..w, reference r_p = first chunk element.
    n_p = torch.arange(1, w + 1, dtype=x.dtype, device=x.device)
    r_p = x[:, :, :1]
    y_p = x - r_p
    S_p = y_p.cumsum(dim=2)
    m_bar_p = S_p / n_p
    M_p = (y_p * y_p).cumsum(dim=2) - S_p * m_bar_p
    #
    # Suffixes (part A): counts n = w..1, reference r_s = last chunk element.
    n_s = torch.arange(w, 0, -1, dtype=x.dtype, device=x.device)
    r_s = x[:, :, -1:]
    y_s = x - r_s
    S_s = rcumsum(y_s, dim=2)
    m_bar_s = S_s / n_s
    M_s = rcumsum(y_s * y_s, dim=2) - S_s * m_bar_s

    # 3. Each window is a suffix of the previous chunk (part A) followed by a
    # prefix of the current chunk (part B); an aligned window is just a whole
    # chunk prefix.  Merge the two parts with Chan et al.'s parallel formula
    # `M = M_A + M_B + n_A * n_B / (n_A + n_B) * delta^2`, where the mean gap
    # is `delta = mu_B - mu_A`.  Since the means are stored relative to their
    # references, `delta = (r_B - r_A) + (m_bar_B - m_bar_A)` never forms a
    # large offset.  Here `n_A + n_B = w`, so the weight is `n_A * n_B / w`.
    delta = (r_p[:, 1:] - r_s[:, :-1]) + (
        m_bar_p[:, 1:, :-1] - m_bar_s[:, :-1, 1:]
    )
    weight = n_s[1:] * n_p[:-1] / w
    M = M_s[:, :-1, 1:] + M_p[:, 1:, :-1] + delta * delta * weight
    M = torch.cat((M, M_p[:, 1:, -1:]), dim=2)

    # 4. Var = M / (w - ddof).
    return M.flatten(start_dim=1)[:, pad_len - w :] / max(0, w - ddof)


def mvar(
    x: torch.Tensor, span: int, dim: int = -1, ddof: int = 1
) -> torch.Tensor:
    r"""Compute the moving (sliding window) variance of a tensor.

    This function calculates the variance of elements within a sliding window of
    size :attr:`span` along the specified dimension. The output tensor has the
    same shape as the input tensor. For positions where the sliding window
    cannot fully cover preceding elements (i.e., the first ``span - 1`` elements
    along the selected dimension), the result is ``nan``.

    The moving variance is computed using the formula:

    .. math::
        \text{MVAR}[i] = \frac{1}{\text{span} - \text{ddof}}
        \sum_{j=i-\text{span}+1}^{i} \left(x[j] - \mu[i]\right)^2,
        \quad \mu[i] = \frac{1}{\text{span}}
        \sum_{j=i-\text{span}+1}^{i} x[j]

    The moving variance is computed in ``O(N)`` time independent of
    :attr:`span`, using cumulative statistics of :attr:`span`-sized chunks
    merged by Chan's parallel algorithm.  All sums are taken over deviations
    from nearby reference values, so the result is numerically stable even
    when the input has a large offset relative to its variance (e.g., values
    around ``1e6`` with a variance of ``1e-6``).

    See `A Numerically Stable and Fast Implementation of Moving Averages and
    Variances <https://imoz.jp/scraps/202607_mvar.en.html>`_ for a detailed
    description of the chunked cumulative-statistics algorithm.

    Args:
        x (Tensor):
            The input tensor containing values.
        span (int):
            The size of the sliding window. Must be positive.
        dim (int, optional):
            The dimension along which to compute the moving variance.
            Default is -1 (the last dimension).
        ddof (int, optional):
            Delta degrees of freedom. The divisor used in the calculation is
            ``span - ddof``. Use 0 for population variance. Must be less than
            ``span``; otherwise the result is ``inf`` or ``nan``.
            Default is 1 (sample variance).

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the moving
            variance values. The first ``span - 1`` elements along the specified
            dimension are ``nan``.

    Example:

        >>> # Simple moving variance with window size 3
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> QF.mvar(x, span=3)
        tensor([nan, nan, 1., 1., 1.])

        >>> # 2D tensor with moving variance along columns
        >>> x = torch.tensor([[1.0, 2.0, 1.0, 3.0],
        ...                   [4.0, 5.0, 4.0, 6.0],
        ...                   [2.0, 3.0, 2.0, 4.0]])
        >>> QF.mvar(x, span=2, dim=1)
        tensor([[   nan, 0.5000, 0.5000, 2.0000],
                [   nan, 0.5000, 0.5000, 2.0000],
                [   nan, 0.5000, 0.5000, 2.0000]])

        >>> # Population variance (ddof=0)
        >>> x = torch.tensor([1.0, 3.0, 5.0, 7.0])
        >>> QF.mvar(x, span=2, ddof=0)
        tensor([nan, 1., 1., 1.])

        >>> # Sample variance (ddof=1, default)
        >>> QF.mvar(x, span=2, ddof=1)
        tensor([nan, 2., 2., 2.])

        >>> # Moving variance along rows
        >>> x = torch.tensor([[1.0, 2.0],
        ...                   [3.0, 4.0],
        ...                   [5.0, 6.0]])
        >>> QF.mvar(x, span=2, dim=0)
        tensor([[nan, nan],
                [2., 2.],
                [2., 2.]])

    .. note::
        If a window contains any NaN value, the moving variance for that
        window is NaN. Unlike ``pandas.DataFrame.rolling``, there is no
        ``min_periods``-style option to skip NaN values.

    .. seealso::
        - :func:`mstd`: Moving standard deviation function (square root of
          this).
        - :func:`msum`: Moving sum function.
        - :func:`ma`: Moving average function.
    """
    return apply_for_axis(lambda x: _mvar(x, span, ddof), x, dim)
