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

    res_not_padded = torch.nan_to_num(
        gain / (gain + loss) * 100, 0
    )  # if gain=0 and loss=0, expect 100
    res = torch.concat(
        (
            torch.full((res_not_padded.shape[0], span), torch.nan),
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
    because they do not have enough preceding elements.

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

    Example:

        >>> x = torch.tensor([1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0])
        >>> QF.rsi(x, span=3)
        tensor([    nan,     nan,     nan, 66.6667, 77.7778, 85.1852, 56.7901])

        >>> QF.rsi(x, span=3, use_sma=True)
        tensor([    nan,     nan,     nan, 66.6667, 66.6667, 66.6667, 66.6667])

    .. seealso::
        - :func:`rci`: Rank Correlation Index, another momentum indicator.
        - :func:`ma`: Simple moving average function.
        - :func:`ema`: Exponential moving average function.
    """
    return apply_for_axis(lambda x: _rsi(x, span, use_sma), x, dim)
