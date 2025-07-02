import typing

import torch

from qfeval_functions.random import is_fast
from qfeval_functions.random import rng


def randn(
    *size: int,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> torch.Tensor:
    r"""Generate a tensor filled with random numbers from a standard normal distribution.

    This function creates a tensor of the specified size filled with random
    numbers sampled from a normal (Gaussian) distribution with mean 0 and
    standard deviation 1. The function ensures reproducibility across different
    devices when a seed is fixed, making it ideal for deterministic simulations
    and statistical modeling.

    When fast mode is enabled, this function delegates to PyTorch's native
    :func:`torch.randn` for optimal performance. Otherwise, it uses a custom
    random number generator to ensure cross-device reproducibility.

    The standard normal distribution has the probability density function:

    .. math::
        f(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}

    Args:
        *size (int):
            Sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a single argument
            that is a sequence.
        dtype (torch.dtype, optional):
            The desired data type of the returned tensor. If not specified,
            defaults to ``torch.float32``.
        device (torch.device, optional):
            The desired device of the returned tensor. If not specified,
            uses the default device.

    Returns:
        Tensor:
            A tensor of shape ``size`` filled with random numbers from
            the standard normal distribution N(0, 1).

    Example:
        >>> # Generate a 1D tensor from standard normal distribution
        >>> x = QF.randn(5)
        >>> x.shape
        torch.Size([5])
        >>> # Values should be roughly centered around 0
        >>> # abs(x.mean()) < 1.0  # Statistical test (may rarely fail)

        >>> # Generate a 2D tensor with specific dtype
        >>> y = QF.randn(3, 4, dtype=torch.float64)
        >>> y.shape
        torch.Size([3, 4])
        >>> y.dtype
        torch.float64

        >>> # Generate on specific device
        >>> z = QF.randn(2, 3, device='cpu')
        >>> z.device
        device(type='cpu')

        >>> # Generate with unpacked tuple
        >>> shape = (2, 3, 4)
        >>> w = QF.randn(*shape)
        >>> w.shape
        torch.Size([2, 3, 4])

        >>> # Generate random walk increments
        >>> steps = QF.randn(1000)
        >>> random_walk = torch.cumsum(steps, dim=0)
        >>> random_walk.shape
        torch.Size([1000])

    .. seealso::
        - :func:`randn_like`: Generate normal random tensor with same shape as input.
        - :func:`rand`: Generate random tensor from uniform distribution.
        - :func:`torch.randn`: PyTorch's native normal random tensor generation.

    .. note::
        To generate samples from N(μ, σ²), use: ``μ + σ * QF.randn(...)``.

    .. warning::
        In financial applications, ensure that the normality assumption is
        appropriate for your use case, as financial returns often exhibit
        fat tails and skewness not captured by the normal distribution.
    """
    if is_fast():
        return torch.randn(*size, dtype=dtype or torch.float32, device=device)
    v = rng().normal(0, 1, size)
    return torch.tensor(v, dtype=dtype or torch.float32, device=device)
