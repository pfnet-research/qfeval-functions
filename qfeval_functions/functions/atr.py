import math

import torch

from .apply_for_axis import apply_for_axis
from .ma import ma
from .rsi import _ema_2dim_recursive
from .shift import shift


def _wilder_atr(tr: torch.Tensor, span: int) -> torch.Tensor:
    # Computing one ATR value requires `span` true ranges, and the first
    # true range (at index 0) is NaN because it has no previous close, so
    # a series with `span` or fewer elements has no valid output position.
    if tr.shape[1] <= span:
        return torch.full_like(tr, math.nan)
    # Seed Wilder's smoothing with the simple average of the first `span`
    # true ranges (at indices 1 to `span`), then apply the recursion
    # ATR[i] = (ATR[i-1] * (span - 1) + TR[i]) / span, following TA-Lib:
    # https://github.com/TA-Lib/ta-lib/blob/f393d2af97e5526a34b2e3f4bdad25d9e44f83ac/src/ta_func/ta_ATR.c # NOQA
    seed = tr[:, 1 : span + 1].mean(dim=1, keepdim=True)
    smoothed = _ema_2dim_recursive(
        torch.cat((seed, tr[:, span + 1 :]), dim=1), alpha=1.0 / span
    )
    return torch.cat(
        (tr.new_full((tr.shape[0], span), math.nan), smoothed), dim=1
    )


def atr(
    high: torch.Tensor,
    low: torch.Tensor,
    close: torch.Tensor,
    span: int = 14,
    use_sma: bool = False,
    dim: int = -1,
) -> torch.Tensor:
    r"""Compute the Average True Range (ATR) along the specified dimension.

    ATR is a volatility indicator that averages the true range, i.e., the
    greatest of the current bar's high-low range and the absolute gaps
    from the previous close:

    .. math::
        \text{TR}[i] = \max\bigl(\text{high}[i] - \text{low}[i],
        \lvert\text{high}[i] - \text{close}[i-1]\rvert,
        \lvert\text{low}[i] - \text{close}[i-1]\rvert\bigr)

    The three input tensors are broadcast against each other, and the true
    range is computed elementwise on the broadcast shape.

    Two averaging methods are supported:

    - ``use_sma=False`` (default): Wilder's smoothing (an exponential
      moving average), compatible with TA-Lib except for NaN handling (see
      the notes below).  The output at index ``span`` along the dimension
      is the simple average of the first ``span`` true ranges, and
      subsequent values follow

      .. math::
          \text{ATR}[i] = \frac{\text{ATR}[i-1] \times (\text{span} - 1)
          + \text{TR}[i]}{\text{span}}

      See https://www.investopedia.com/terms/a/atr.asp
    - ``use_sma=True``: a simple moving average of the true range over
      the trailing ``span`` elements, a variant commonly used in Japanese
      technical analysis references.

    The first element along the dimension has no previous close, so its
    true range is undefined, and the first ``span`` elements of the output
    are NaN.  If the dimension has ``span`` or fewer elements, all output
    values are NaN while the output shape still matches the broadcast
    input shape, because computing one ATR value requires ``span`` true
    ranges, i.e., ``span + 1`` bars.

    NaN values in the input propagate to the output instead of being
    converted to valid ATR values:

    - ``use_sma=False`` (Wilder's smoothing): once a NaN true range enters
      the initial average or the recursive smoothing, all subsequent
      outputs are NaN.
    - ``use_sma=True`` (simple moving average): a NaN price makes up to
      two adjacent true ranges NaN, and only the outputs whose windows
      contain them are NaN.

    Operations made undefined by infinite prices also result in NaN.

    Args:
        high (Tensor):
            The input tensor containing high prices.
        low (Tensor):
            The input tensor containing low prices.
        close (Tensor):
            The input tensor containing closing prices.
        span (int, optional):
            The window size used to average true ranges.
            Default is 14.
        use_sma (bool, optional):
            If ``True``, use a simple moving average instead of Wilder's
            smoothing.  Default is ``False``.
        dim (int, optional):
            The dimension along which to compute ATR.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the broadcast input shape, containing ATR values
            (the first ``span`` elements along the dimension are NaN).

    Raises:
        ValueError: If ``span`` is not positive.
        TypeError: If ``span`` is not an integer (``bool`` is rejected).
        TypeError: If any of ``high``, ``low``, and ``close`` is not a
            floating point tensor.

    Example:

        >>> high = torch.tensor([10.0, 12.0, 11.0, 14.0])
        >>> low = torch.tensor([9.0, 10.0, 9.0, 11.0])
        >>> close = torch.tensor([9.5, 11.0, 10.0, 13.0])
        >>> QF.atr(high, low, close, span=2)
        tensor([   nan,    nan, 2.2500, 3.1250])

        >>> QF.atr(high, low, close, span=2, use_sma=True)
        tensor([   nan,    nan, 2.2500, 3.0000])

        >>> # Constant prices yield an ATR of zero after the warm-up.
        >>> c = torch.full((5,), 3.0)
        >>> QF.atr(c, c, c, span=2)
        tensor([nan, nan, 0., 0., 0.])

    .. note::
        The inputs are not validated against the OHLC invariants (such as
        ``high >= low`` and ``low <= close <= high``); each output is
        simply the result of the formulas above for the given inputs.  For
        inputs satisfying the invariants, the true range and hence the ATR
        are always non-negative, and constant prices yield an ATR of zero.

    .. seealso::
        - :func:`rsi`: Relative Strength Index, which shares Wilder's
          smoothing.
        - :func:`ma`: Simple moving average, used when ``use_sma=True``.
        - :func:`bollinger_band`: Another volatility-based indicator.
    """
    # NOTE: bool is a subclass of int, so it must be rejected explicitly.
    if isinstance(span, bool) or not isinstance(span, int):
        raise TypeError(f"span must be an integer, but got {span!r}.")
    if span <= 0:
        raise ValueError(f"span must be a positive integer, but got {span}.")
    for name, tensor in (("high", high), ("low", low), ("close", close)):
        if not tensor.is_floating_point():
            raise TypeError(
                f"atr only supports floating point tensors, but "
                f"{name} has dtype {tensor.dtype}."
            )
    high, low, close = torch.broadcast_tensors(high, low, close)
    prev_close = shift(close, 1, dim)
    tr = torch.maximum(
        high - low,
        torch.maximum((high - prev_close).abs(), (low - prev_close).abs()),
    )
    if use_sma:
        return ma(tr, span, dim)
    return apply_for_axis(lambda t: _wilder_atr(t, span), tr, dim)
