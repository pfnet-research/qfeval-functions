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
    """
    Definition(use_sma=False):
    - https://www.investopedia.com/terms/r/rsi.asp
    - Compatible with TA-lib
    Definition(use_sma=True):
    - https://info.monex.co.jp/technical-analysis/indicators/005.html
    """
    return apply_for_axis(lambda x: _rsi(x, span, use_sma), x, dim)
