import torch

from .apply_for_axis import apply_for_axis


def _rci(x: torch.Tensor, period: int) -> torch.Tensor:
    batch_index = torch.arange(x.shape[0], device=x.device)
    time_index = torch.arange(x.shape[1], device=x.device)
    period_index = torch.arange(period, device=x.device)
    prices = x[
        batch_index[:, None, None],
        (time_index[None, :, None] - period_index[None, None, :]).relu(),
    ]  # no meaning for the prices[:, span-1]
    price_rank = (-prices).argsort().argsort()
    d = (period_index[None, None] - price_rank).square().sum(dim=-1)
    denominator = period * (period**2 - 1)
    v: torch.Tensor = (1 - 6 * d / denominator) * 100
    v[:, : period - 1] = torch.nan  # fill first span-1 elements to nan
    return v


def rci(x: torch.Tensor, period: int = 9, dim: int = -1) -> torch.Tensor:
    """
    https://kabu.com/investment/guide/technical/14.html
    """
    return apply_for_axis(lambda x: _rci(x, period), x, dim)
