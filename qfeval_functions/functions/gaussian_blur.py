import math

import torch
import torch.nn.functional as F

from .apply_for_axis import apply_for_axis


def _gaussian_filter(n: int, sigma: float) -> torch.Tensor:
    r"""Returns a symmetric Gaussian window, with parameter sigma, as a 1D
    tensor with n elements.
    """

    # Integral of the Gaussian function, whose sigma is 1.
    def f(x: torch.Tensor) -> torch.Tensor:
        return (x / math.sqrt(2)).erf() / 2

    a = torch.arange(n, dtype=torch.float64) - (n - 1) / 2
    return f((a + 0.5) / sigma) - f((a - 0.5) / sigma)


def gaussian_blur(x: torch.Tensor, sigma: float, dim: int = -1) -> torch.Tensor:
    r"""Applies a Gaussian filter with the given `sigma` parameter to `x` along
    the specified axis `dim`.

    Specifically, this assigns the weighted mean of valid values with a
    Gaussian filter to each element.  This makes values outside the range have
    no weight, so it works well even for biased values, while zero padding
    brings the surrounding values closer to zero.  Additionally, as it makes
    NaN values have no weight, it also works well for biased values that
    include NaNs.

    NOTE: This function uses interval averages of a Gaussian function instead
    of point-sampling for its discretized window function.  Typical
    implementations of Gaussian filters use point-sampling (e.g.,
    `scipy.ndimage.gaussian_filter1d`).  However, they have an undersampling
    issue for small $\sigma$ (c.f., https://bartwronski.com/2021/10/31/,
    Implementation section in https://en.wikipedia.org/wiki/Gaussian_blur).
    This calculates the interval averages using the integral of a Gaussian
    function.
    """

    def _blur(x: torch.Tensor) -> torch.Tensor:
        # Apply convolution with x and a Gaussian filter.
        w = _gaussian_filter(x.shape[-1] * 2 + 1, sigma).to(x.device)
        a = F.conv1d(x.to(w)[:, None], w[None, None], padding="same")
        count = F.conv1d(
            (~x.isnan()).to(w)[:, None], w[None, None], padding="same"
        )
        return (a / count)[:, 0].to(x)

    return apply_for_axis(_blur, x, dim)
