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
    r"""Apply Gaussian blur to a tensor along a specified dimension.

    This function applies a one-dimensional Gaussian filter to smooth data
    along the specified dimension. The Gaussian blur operation computes a
    weighted average of neighboring values, where weights follow a Gaussian
    (normal) distribution centered at each point. This is commonly used for
    noise reduction, data smoothing, and signal processing.

    Unlike typical implementations that use point-sampling (such as
    ``scipy.ndimage.gaussian_filter1d``), this function uses interval averages
    of the Gaussian function for improved accuracy, especially for small
    :attr:`sigma` values. This approach avoids undersampling issues and
    provides more accurate results.

    Args:
        x (Tensor):
            The input tensor to be blurred.
        sigma (float):
            The standard deviation of the Gaussian kernel. Larger values
            produce more smoothing. Must be positive.
        dim (int, optional):
            The dimension along which to apply the Gaussian blur.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the
            Gaussian-blurred values.

    Example:

        >>> # Simple 1D Gaussian blur
        >>> x = torch.tensor([0., 0., 0., 10., 0., 0., 0.])
        >>> QF.gaussian_blur(x, sigma=1.0)
        tensor([0.0864, 0.6494, 2.4324, 3.8310, 2.4324, 0.6494, 0.0864])

        >>> # 2D tensor: blur along different dimensions
        >>> x = torch.zeros(3, 5)
        >>> x[1, 2] = 10.0
        >>> QF.gaussian_blur(x, sigma=0.8, dim=0)  # blur along rows
        tensor([[0.0000, 0.0000, 3.2135, 0.0000, 0.0000],
                [0.0000, 0.0000, 4.9832, 0.0000, 0.0000],
                [0.0000, 0.0000, 3.2135, 0.0000, 0.0000]])
        >>> QF.gaussian_blur(x, sigma=0.8, dim=1)  # blur along columns
        tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.4020, 2.4298, 4.6886, 2.4298, 0.4020],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])

    .. seealso::
        - https://en.wikipedia.org/wiki/Gaussian_blur
        - https://bartwronski.com/2021/10/31/gaussian-blur-corrected-improved-and-optimized/
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
