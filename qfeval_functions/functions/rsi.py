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
    r"""Compute Relative Strength Index (RSI) technical indicator.

    The Relative Strength Index is a momentum oscillator that measures the
    speed and change of price movements. RSI oscillates between 0 and 100,
    where values above 70 are typically considered overbought and values
    below 30 are considered oversold.

    The RSI is calculated using the formula:

    .. math::
        \text{RSI} = 100 - \frac{100}{1 + \text{RS}}

    where :math:`\text{RS} = \frac{\text{Average Gain}}{\text{Average Loss}}`

    When :attr:`use_sma` is False (default), the implementation follows the
    standard Wilder's smoothing method (exponential moving average) which is
    compatible with TA-Lib. When :attr:`use_sma` is True, simple moving
    averages are used instead.

    Args:
        x (Tensor):
            Input tensor containing price data. The RSI is computed along
            the specified dimension.
        span (int, optional):
            The number of periods to use for RSI calculation. Standard
            value is 14. Must be greater than 1. Default is 14.
        use_sma (bool, optional):
            If True, uses Simple Moving Average for gain/loss smoothing.
            If False, uses Exponential Moving Average (Wilder's method).
            Default is False.
        dim (int, optional):
            The dimension along which to compute the RSI. Default is -1
            (the last dimension).

    Returns:
        Tensor:
            The RSI values with the same shape as the input. The first
            :attr:`span` values along the specified dimension are set
            to NaN since they cannot be computed.

    Example:
        >>> # Simple price series
        >>> from qfeval_functions.functions.rsi import rsi
        >>> prices = torch.tensor([[10.0, 10.5, 10.2, 10.8, 11.0, 10.7, 11.2, 11.5, 11.1, 11.8]])
        >>> rsi_values = rsi(prices, span=5, dim=1)
        >>> rsi_values.shape
        torch.Size([1, 10])
        >>> # First 5 values should be NaN
        >>> torch.isnan(rsi_values[0, :5]).all()
        tensor(True)

        >>> # Multi-dimensional example
        >>> batch_prices = torch.tensor([[[10.0, 10.5, 10.2, 11.0, 11.5],
        ...                               [20.0, 19.5, 20.2, 19.8, 20.5]],
        ...                              [[15.0, 15.3, 14.8, 15.5, 15.1],
        ...                               [25.0, 25.2, 24.8, 25.5, 25.3]]])
        >>> rsi_batch = rsi(batch_prices, span=3, dim=-1)
        >>> rsi_batch.shape
        torch.Size([2, 2, 5])

        >>> # Using SMA instead of EMA
        >>> rsi_sma = rsi(prices, span=5, use_sma=True, dim=1)
        >>> rsi_sma.shape
        torch.Size([1, 10])

        >>> # Compare EMA vs SMA methods
        >>> rsi_ema = rsi(prices, span=5, use_sma=False, dim=1)
        >>> # Results will be different due to smoothing method

    .. seealso::
        - :func:`ema`: Exponential moving average used in RSI calculation.
        - :func:`msum`: Moving sum used in SMA-based RSI calculation.

    .. note::
        The RSI is widely used in technical analysis for:

        - Identifying overbought/oversold conditions (RSI > 70 or RSI < 30)
        - Detecting divergences between price and momentum
        - Generating buy/sell signals at extreme RSI levels
        - Confirming trend reversals and continuations
        - Multi-timeframe momentum analysis

    .. note::
        In quantitative finance applications, RSI can be used for:

        - Momentum factor construction in multi-factor models
        - Mean reversion strategy development
        - Risk management through momentum regime detection
        - Portfolio optimization based on momentum characteristics
        - Systematic trading signal generation

    .. warning::
        RSI is a lagging indicator and can remain in overbought/oversold
        conditions for extended periods during strong trends. Consider
        combining with other indicators for better signal quality.

    References:
        - Default method (use_sma=False): https://www.investopedia.com/terms/r/rsi.asp
        - SMA method (use_sma=True): https://info.monex.co.jp/technical-analysis/indicators/005.html
        - TA-Lib compatibility: https://github.com/TA-Lib/ta-lib
    """
    return apply_for_axis(lambda x: _rsi(x, span, use_sma), x, dim)
