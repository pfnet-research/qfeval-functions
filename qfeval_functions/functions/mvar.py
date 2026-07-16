import math

import torch

from .apply_for_axis import apply_for_axis
from .rcumsum import rcumsum


def _mvar(x: torch.Tensor, span: int, ddof: int) -> torch.Tensor:
    """Returns the moving variance of the given tensor ``x``, whose shape is
    ``(B, N)``, along the 2nd dimension."""

    # 1. Reshape the target dimension into `(*, span)` with prepending NaNs
    # (the same chunk layout as `msum`).
    pad_len = span * 2 - x.shape[1] % span
    x = torch.nn.functional.pad(x, (pad_len, 0), value=math.nan)
    x = x.reshape((x.shape[0], x.shape[1] // span, span))

    # 2. Compute the mean and the sum of squared deviations (M2) of every
    # chunk prefix and suffix.  Each part is centered on one of its own
    # elements (the chunk's first/last element), so all sums are taken over
    # small local deviations, which avoids catastrophic cancellation, and
    # NaN/inf values contaminate exactly the parts containing them.
    n_p = torch.arange(1, span + 1, dtype=x.dtype, device=x.device)
    y_p = x - x[:, :, :1]
    s_p = y_p.cumsum(dim=2)
    mean_p = s_p / n_p
    m2_p = (y_p * y_p).cumsum(dim=2) - s_p * mean_p

    n_s = torch.arange(span, 0, -1, dtype=x.dtype, device=x.device)
    y_s = x - x[:, :, -1:]
    s_s = rcumsum(y_s, dim=2)
    mean_s = s_s / n_s
    m2_s = rcumsum(y_s * y_s, dim=2) - s_s * mean_s

    # 3. Each window is a chunk suffix followed by a chunk prefix (the
    # prefix is the whole chunk for aligned windows).  Merge the two parts
    # with Chan's parallel algorithm:
    # `M2 = M2_s + M2_p + delta^2 * n_s * n_p / (n_s + n_p)`, where `delta`
    # is the difference of the part means.  The part means are represented
    # relative to the part centers, so `delta` is also computed without a
    # large offset.
    delta = (x[:, 1:, :1] - x[:, :-1, -1:]) + (
        mean_p[:, 1:, :-1] - mean_s[:, :-1, 1:]
    )
    weight = n_s[1:] * n_p[:-1] / span
    m2 = m2_s[:, :-1, 1:] + m2_p[:, 1:, :-1] + delta * delta * weight
    m2 = torch.cat((m2, m2_p[:, 1:, -1:]), dim=2)
    return m2.flatten(start_dim=1)[:, pad_len - span :] / max(0, span - ddof)


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
