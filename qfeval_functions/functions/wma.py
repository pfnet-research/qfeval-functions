import math

import torch

from .apply_for_axis import apply_for_axis
from .rcumsum import rcumsum


def _wma(x: torch.Tensor, span: int) -> torch.Tensor:
    """Returns the linearly-weighted moving average of the given tensor
    ``x``, whose shape is ``(B, N)``, along the 2nd dimension."""

    w = span

    # 1. Reshape the target dimension into length-`w` chunks with prepended
    # NaNs (the same chunk layout as `msum`).
    pad_len = w * 2 - x.shape[1] % w
    x = torch.nn.functional.pad(x, (pad_len, 0), value=math.nan)
    x = x.reshape((x.shape[0], x.shape[1] // w, w))

    # 2. Compute per-chunk prefix (cumsum) and suffix (rcumsum) cumulatives
    # of `x` and of `k * x`, where `k` is the local index within a chunk.
    # All sums are local to a single chunk and `k` is bounded by `w`, which
    # keeps the arithmetic numerically stable regardless of the position in
    # the series.
    k = torch.arange(w, dtype=x.dtype, device=x.device)
    xk = x * k
    p1 = x.cumsum(dim=2)
    pk = xk.cumsum(dim=2)
    s1 = rcumsum(x, dim=2)
    sk = rcumsum(xk, dim=2)

    # 3. A window ending at offset `e` (`0 <= e <= w - 2`) of a chunk joins
    # the previous chunk's suffix starting at `e + 1` with the current
    # chunk's prefix up to `e`.  The element at local index `k` of the
    # current chunk has weight `k + (w - e)`, and the element at local index
    # `m` of the previous chunk has weight `m - e`, so the weighted sum is
    # `(Pk[e] + (w - e) * P1[e]) + (Sk[e + 1] - e * S1[e + 1])`.  An aligned
    # window (`e = w - 1`) is a whole chunk with weights `k + 1`.
    e = torch.arange(w - 1, dtype=x.dtype, device=x.device)
    mixed = (pk[:, 1:, :-1] + (w - e) * p1[:, 1:, :-1]) + (
        sk[:, :-1, 1:] - e * s1[:, :-1, 1:]
    )
    x = torch.cat((mixed, pk[:, 1:, -1:] + p1[:, 1:, -1:]), dim=2)

    # 4. Divide by the total weight `1 + 2 + ... + w`.
    return x.flatten(start_dim=1)[:, pad_len - w :] / (w * (w + 1) / 2)


def wma(x: torch.Tensor, span: int, dim: int = -1) -> torch.Tensor:
    r"""Compute the linearly-weighted moving average (WMA) of a tensor.

    This function calculates the weighted average of elements within a
    sliding window of size :attr:`span` along the specified dimension, where
    weights increase linearly from ``1`` for the oldest element to
    :attr:`span` for the newest element. The output tensor has the same
    shape as the input tensor. For positions where the sliding window cannot
    fully cover preceding elements (i.e., the first ``span - 1`` elements
    along the selected dimension), the result is ``nan``.

    The weighted moving average is computed using the formula:

    .. math::
        \text{WMA}[i] = \frac{\sum_{k=1}^{\text{span}}
        k \cdot x[i - \text{span} + k]}
        {\text{span} \cdot (\text{span} + 1) / 2}

    The weighted moving average is computed in ``O(N)`` time independent of
    :attr:`span`, by decomposing each window into a chunk suffix and a chunk
    prefix obtained from cumulative sums of ``x`` and ``k * x``, where ``k``
    is the index local to a :attr:`span`-sized chunk.  All partial sums are
    local to a two-chunk range and no global position index is multiplied
    in, so the computation stays numerically stable even for long series
    with a large offset or drift.  See `A Numerically Stable and Fast
    Implementation of Moving Averages and Variances
    <https://imoz.jp/scraps/202607_mvar.en.html>`_ for a detailed
    description of the chunked algorithm.

    In terms of weighting, WMA sits between :func:`ma`, which weights all
    window elements uniformly, and :func:`ema`, which decays weights
    exponentially: WMA reacts to recent changes faster than the simple
    moving average while still dropping old elements completely.

    Args:
        x (Tensor):
            The input tensor.
        span (int):
            The size of the sliding window. Must be positive.
        dim (int, optional):
            The dimension along which to compute the weighted moving
            average. Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the weighted
            moving averages. The first ``span - 1`` elements along the
            specified dimension are ``nan``.

    Example:

        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> QF.wma(x, span=3)
        tensor([   nan,    nan, 2.3333, 3.3333, 4.3333])

        >>> # A constant input stays constant.
        >>> x = torch.tensor([7.0, 7.0, 7.0, 7.0])
        >>> QF.wma(x, span=2)
        tensor([nan, 7., 7., 7.])

        >>> x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
        ...                   [5.0, 6.0, 7.0, 8.0]])
        >>> QF.wma(x, span=2, dim=1)
        tensor([[   nan, 1.6667, 2.6667, 3.6667],
                [   nan, 5.6667, 6.6667, 7.6667]])

    .. note::
        If a window contains any NaN value, the weighted moving average for
        that window is NaN. Unlike ``pandas.DataFrame.rolling``, there is no
        ``min_periods``-style option to skip NaN values.

    .. seealso::
        - :func:`ma`: Moving average function (uniform weights).
        - :func:`ema`: Exponential moving average function.
        - :func:`msum`: Moving sum function (the same chunked algorithm).
    """
    return apply_for_axis(lambda x: _wma(x, span), x, dim)
