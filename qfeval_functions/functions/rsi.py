import math

import torch

from .apply_for_axis import apply_for_axis
from .msum import msum


def _ema_2dim_recursive(x: torch.Tensor, alpha: float) -> torch.Tensor:
    assert x.dim() == 2
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html
    # Like ewm(*, adjust=False) behavior
    #
    # definition:
    #   y_0 = x_0
    #   y_t = alpha x_t + (1-alpha) y_{t-1}
    x = torch.transpose(x, 0, 1)
    decay = 1 - alpha

    y = torch.zeros_like(x)

    y[0] = x[0]  # set initial value

    # Compute the EMA for the rest of the elements
    for n in range(1, len(x)):
        y[n] = y[n - 1] * decay + x[n] * (1 - decay)
    return torch.transpose(y, 0, 1)


def _rsi(
    x: torch.Tensor, span: int = 14, use_sma: bool = False
) -> torch.Tensor:
    # Computing one RSI value requires `span` differences, i.e., `span + 1`
    # prices, so a shorter series has no valid output position.
    if x.shape[1] <= span:
        return torch.full_like(x, math.nan)
    # Ignore metastock compatible mode: https://github.com/TA-Lib/ta-lib/blob/f393d2af97e5526a34b2e3f4bdad25d9e44f83ac/src/ta_func/ta_RSI.c#L270C1-L321C1 # NOQA
    delta = x.diff(1)
    # for i<span, prevLoss and prevGain is mean gain.
    # https://github.com/TA-Lib/ta-lib/blob/f393d2af97e5526a34b2e3f4bdad25d9e44f83ac/src/ta_func/ta_RSI.c#L323-L348
    # for i>=span, use EMA
    # https://github.com/TA-Lib/ta-lib/blob/f393d2af97e5526a34b2e3f4bdad25d9e44f83ac/src/ta_func/ta_RSI.c#L373-L385

    if use_sma:
        gain = msum(torch.relu(delta[:, :]), span=span, dim=-1)[:, span - 1 :]
        loss = msum(torch.relu(-delta[:, :]), span=span, dim=-1)[:, span - 1 :]
    else:
        initial_gain = torch.mean(
            torch.relu(delta[:, :span]), dim=-1, keepdim=True
        )
        initial_loss = torch.mean(
            torch.relu(-delta[:, :span]),
            dim=-1,
            keepdim=True,
        )
        gain = _ema_2dim_recursive(
            torch.cat((initial_gain, torch.relu(delta[:, span:])), dim=1),
            alpha=1 / span,
        )
        loss = _ema_2dim_recursive(
            torch.cat((initial_loss, torch.relu(-delta[:, span:])), dim=1),
            alpha=1 / span,
        )

    # A window with no gains and no losses (i.e., a flat series) yields 0/0.
    # RSI is defined to be the neutral value 50 there.  NOTE: This
    # intentionally differs from TA-Lib, which returns 0 for flat series.
    # Any other NaN (e.g., caused by NaN or infinite inputs) is propagated
    # as is instead of being converted to a valid RSI value.
    ratio = gain / (gain + loss) * 100
    res_not_padded = torch.where(
        (gain == 0) & (loss == 0),
        ratio.new_tensor(50.0),
        ratio,
    )
    res = torch.concat(
        (
            res_not_padded.new_full((res_not_padded.shape[0], span), math.nan),
            res_not_padded,
        ),
        dim=1,
    )
    return res


def rsi(
    x: torch.Tensor, span: int = 14, use_sma: bool = False, dim: int = -1
) -> torch.Tensor:
    r"""Compute the Relative Strength Index (RSI) along the specified
    dimension.

    RSI is a momentum oscillator that measures the magnitude of recent gains
    relative to recent losses, ranging from 0 to 100:

    .. math::
        \text{RSI} = \frac{\text{gain}}{\text{gain} + \text{loss}}
        \times 100

    where gain and loss are averages of the upward and downward price
    changes over the trailing window of ``span`` elements.  Values above 70
    are conventionally considered overbought, and values below 30 oversold.

    Two averaging methods are supported:

    - ``use_sma=False`` (default): Wilder's smoothing (an exponential moving
      average), compatible with TA-Lib.
      See https://www.investopedia.com/terms/r/rsi.asp
    - ``use_sma=True``: a simple moving average.
      See https://info.monex.co.jp/technical-analysis/indicators/005.html

    The first ``span`` elements along the dimension are filled with NaN
    because they do not have enough preceding elements.  If the dimension
    has ``span`` or fewer elements, all output values are NaN while the
    output shape still matches the input shape, because computing one RSI
    value requires ``span`` price changes, i.e., ``span + 1`` prices.

    If a window contains no price changes at all (both the average gain and
    the average loss are zero), the RSI is defined to be the neutral value
    50.  NOTE: This intentionally differs from TA-Lib, which returns 0 for
    flat series.

    NaN values in the input propagate to the output instead of being
    converted to valid RSI values:

    - ``use_sma=False`` (Wilder's smoothing): once a NaN price change enters
      the initial average or the recursive smoothing, all subsequent outputs
      are NaN.
    - ``use_sma=True`` (simple moving average): a NaN price makes up to two
      adjacent price changes NaN, and only the outputs whose windows contain
      them (at most ``span + 1`` positions) are NaN.

    Operations made undefined by infinite prices also result in NaN.

    Args:
        x (Tensor):
            The input tensor containing prices.
        span (int, optional):
            The window size used to average gains and losses.
            Default is 14.
        use_sma (bool, optional):
            If ``True``, use a simple moving average instead of Wilder's
            smoothing.  Default is ``False``.
        dim (int, optional):
            The dimension along which to compute RSI.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing RSI values
            in :math:`[0, 100]` (the first ``span`` elements along the
            dimension are NaN).

    Raises:
        ValueError: If ``span`` is not positive.
        TypeError: If ``span`` is not an integer (``bool`` is rejected).
        TypeError: If ``x`` is not a floating point tensor.

    Example:

        >>> x = torch.tensor([1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0])
        >>> QF.rsi(x, span=3)
        tensor([    nan,     nan,     nan, 66.6667, 77.7778, 85.1852, 56.7901])

        >>> QF.rsi(x, span=3, use_sma=True)
        tensor([    nan,     nan,     nan, 66.6667, 66.6667, 66.6667, 66.6667])

        >>> # A flat series yields the neutral value 50.
        >>> QF.rsi(torch.full((6,), 5.0), span=3)
        tensor([nan, nan, nan, 50., 50., 50.])

    .. seealso::
        - :func:`rci`: Rank Correlation Index, another momentum indicator.
        - :func:`ma`: Simple moving average function.
        - :func:`ema`: Exponential moving average function.
    """
    # NOTE: bool is a subclass of int, so it must be rejected explicitly.
    if isinstance(span, bool) or not isinstance(span, int):
        raise TypeError(f"span must be an integer, but got {span!r}.")
    if span <= 0:
        raise ValueError(f"span must be a positive integer, but got {span}.")
    if not x.is_floating_point():
        raise TypeError(
            f"rsi only supports floating point tensors, but got {x.dtype}."
        )
    return apply_for_axis(lambda x: _rsi(x, span, use_sma), x, dim)
