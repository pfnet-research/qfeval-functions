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
    d = f((a + 0.5) / sigma) - f((a - 0.5) / sigma)
    return d.clamp(torch.finfo(torch.float64).eps)


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

    NaN values are excluded from the weighted average: at each output
    position, the Gaussian weights are renormalized over the valid
    (non-NaN) values only, and positions that are NaN in the input remain
    NaN in the output.  All other output positions are finite as long as at
    least one valid value exists along the target dimension.

    Args:
        x (Tensor):
            The input tensor to be blurred.
        sigma (float):
            The standard deviation of the Gaussian kernel. Larger
            magnitudes produce more smoothing; only the magnitude matters,
            so the sign is ignored. ``sigma=0`` applies no smoothing and
            returns the input unchanged, while a very large (or infinite)
            :attr:`sigma` approaches a uniform average over the valid
            values along the target dimension. A ``nan`` :attr:`sigma`
            produces an all-``nan`` output.
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
        >>> QF.gaussian_blur(x, sigma=0.2, dim=0)  # blur along rows
        tensor([[0.0000, 0.0000, 0.0625, 0.0000, 0.0000],
                [0.0000, 0.0000, 9.8758, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0625, 0.0000, 0.0000]])
        >>> QF.gaussian_blur(x, sigma=0.2, dim=1)  # blur along columns
        tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0621, 9.8758, 0.0621, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])

    .. seealso::
        - :func:`ma`: Simple moving average, another smoothing function.
        - :func:`ema`: Exponential moving average function.
        - https://en.wikipedia.org/wiki/Gaussian_blur
        - https://bartwronski.com/2021/10/31/gaussian-blur-corrected-improved-and-optimized/
    """

    def _blur(x: torch.Tensor) -> torch.Tensor:
        # Apply convolution with x and a Gaussian filter, excluding NaNs.
        # Only the magnitude of sigma matters, so its sign is ignored.
        w = _gaussian_filter(x.shape[-1] * 2 + 1, abs(sigma)).to(x.device)
        m = ~x.isnan()
        xf = torch.where(m, x, torch.zeros_like(x))
        a = F.conv1d(xf.to(w)[:, None], w[None, None], padding="same")
        # `weight` accumulates the Gaussian weights of the valid (non-NaN)
        # values only, so dividing by it both corrects the boundary effect
        # (weights cut off outside the tensor) and renormalizes the weights
        # over non-NaN values.
        weight = F.conv1d(m.to(w)[:, None], w[None, None], padding="same")
        # Clamp the accumulated weight away from zero to avoid dividing by
        # zero where no valid value exists (all NaN); such positions are
        # restored to NaN by the mask below anyway.
        eps = torch.finfo(weight.dtype).eps
        out = (a / weight.clamp(min=eps))[:, 0].to(x)
        return torch.where(m, out, torch.as_tensor(math.nan).to(out))

    return apply_for_axis(_blur, x, dim)
