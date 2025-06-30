import typing

import torch

from .ma import ma
from .mstd import mstd


def bollinger_band(
    x: torch.Tensor,
    window: int = 20,
    sigma: float = 2.0,
    dim: int = -1,
) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Compute Bollinger Bands for a tensor.

    Bollinger Bands are a technical analysis tool consisting of a middle band
    (moving average) and two outer bands (upper and lower). The outer bands
    are placed at a distance of :attr:`sigma` standard deviations from the
    middle band. This indicator is commonly used in financial analysis to
    identify potential support and resistance levels.

    Args:
        x (Tensor):
            The input tensor containing time series data.
        window (int, optional):
            The size of the moving window for calculating the moving average
            and standard deviation.
            Default is 20.
        sigma (float, optional):
            The number of standard deviations for the upper and lower bands.
            Default is 2.0.
        dim (int, optional):
            The dimension along which to compute the Bollinger Bands.
            Default is -1 (the last dimension).

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            A tuple containing three tensors of the same shape as the input:

            - Upper band: middle band + (sigma * standard deviation)
            - Middle band: moving average
            - Lower band: middle band - (sigma * standard deviation)

    Example:

        >>> x = torch.tensor([[100., 102., 101., 103., 105.],
        ...                   [50., 48., 49., 51., 52.]])
        >>> upper, middle, lower = QF.bollinger_band(x, window=3, sigma=1.0, dim=1)
        >>> upper
        tensor([[     nan,      nan, 101.8165, 102.8165, 104.6330],
                [     nan,      nan,  49.8165,  50.5805,  51.9139]])
        >>> middle
        tensor([[     nan,      nan, 101.0000, 102.0000, 103.0000],
                [     nan,      nan,  49.0000,  49.3333,  50.6667]])
        >>> lower
        tensor([[     nan,      nan, 100.1835, 101.1835, 101.3670],
                [     nan,      nan,  48.1835,  48.0861,  49.4195]])
    """
    middle = ma(x, window, dim=dim)
    width = mstd(x, window, dim=dim, ddof=0) * sigma
    return middle + width, middle, middle - width
